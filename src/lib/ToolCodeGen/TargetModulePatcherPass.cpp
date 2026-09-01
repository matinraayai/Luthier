//===-- TargetModulePatcherPass.cpp -----------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
/// \file
/// Implements the \c TargetModulePatcherPass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetModulePatcherPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/Cloning.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
#include <MCTargetDesc/AMDGPUMCExpr.h>
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include "luthier/ToolCodeGen/StateValueArrayStorage.h"
#include "luthier/ToolCodeGen/TargetModuleBranchRelaxation.h"
#include <AMDGPU.h>
#include <AMDGPUTargetMachine.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalIFunc.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCExpr.h>
#include <llvm/MC/MCSymbol.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-target-module-patcher"

namespace luthier {

namespace {

/// Per-wave scratch reservation (in bytes) for the injected payloads'
/// own stack when at least one payload uses dynamic-stack (var-sized
/// alloca / recursion / indirect calls). Used only in the
/// (dynamic-app, dynamic-payload) case; static-payload cases use the
/// payload MFs' actual fixed frame sizes instead.
static llvm::cl::opt<unsigned> LuthierInstrumentationStackSize(
    "luthier-instrumentation-stack-size", llvm::cl::init(4096),
    llvm::cl::desc(
        "Per-wave scratch reservation (bytes) for the injected payloads' "
        "own stack when at least one payload uses dynamic stack."));

/// Access-injection tag + explicit-instantiation trick that lets us reach
/// through \c GCNUserSGPRUsageInfo's private data members. The private
/// booleans are set once, in the ctor, from function attributes and there is
/// no public setter to flip them afterwards. We need them flipped from a
/// mid-pipeline pass, so we form pointer-to-members inside an explicit
/// instantiation of \c PrivateAccessor: forming a member pointer inside an
/// explicit instantiation bypasses access checks (per
/// [temp.spec]), and the friend declaration then hoists a namespace-scope
/// \c get(Tag) that returns the member pointer for anyone to use.
template <typename Tag, typename Tag::MemberT M> struct PrivateAccessor {
  friend typename Tag::MemberT get(Tag) { return M; }
};

struct FlatScratchInitTag {
  using MemberT = bool llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(FlatScratchInitTag);
};
template struct PrivateAccessor<FlatScratchInitTag,
                                &llvm::GCNUserSGPRUsageInfo::FlatScratchInit>;

struct PrivateSegmentBufferTag {
  using MemberT = bool llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(PrivateSegmentBufferTag);
};
template struct PrivateAccessor<
    PrivateSegmentBufferTag, &llvm::GCNUserSGPRUsageInfo::PrivateSegmentBuffer>;

struct KernargSegmentPtrTag {
  using MemberT = bool llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(KernargSegmentPtrTag);
};
template struct PrivateAccessor<KernargSegmentPtrTag,
                                &llvm::GCNUserSGPRUsageInfo::KernargSegmentPtr>;

struct QueuePtrTag {
  using MemberT = bool llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(QueuePtrTag);
};
template struct PrivateAccessor<QueuePtrTag,
                                &llvm::GCNUserSGPRUsageInfo::QueuePtr>;

struct PrivateSegmentSizeTag {
  using MemberT = bool llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(PrivateSegmentSizeTag);
};
template struct PrivateAccessor<
    PrivateSegmentSizeTag, &llvm::GCNUserSGPRUsageInfo::PrivateSegmentSize>;

struct DispatchPtrTag {
  using MemberT = bool llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(DispatchPtrTag);
};
template struct PrivateAccessor<DispatchPtrTag,
                                &llvm::GCNUserSGPRUsageInfo::DispatchPtr>;

struct DispatchIDTag {
  using MemberT = bool llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(DispatchIDTag);
};
template struct PrivateAccessor<DispatchIDTag,
                                &llvm::GCNUserSGPRUsageInfo::DispatchID>;

struct ImplicitBufferPtrTag {
  using MemberT = bool llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(ImplicitBufferPtrTag);
};
template struct PrivateAccessor<ImplicitBufferPtrTag,
                                &llvm::GCNUserSGPRUsageInfo::ImplicitBufferPtr>;

struct NumUsedUserSGPRsTag {
  using MemberT = unsigned llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(NumUsedUserSGPRsTag);
};
template struct PrivateAccessor<NumUsedUserSGPRsTag,
                                &llvm::GCNUserSGPRUsageInfo::NumUsedUserSGPRs>;

struct NumKernargPreloadSGPRsTag {
  using MemberT = unsigned llvm::GCNUserSGPRUsageInfo::*;
  friend MemberT get(NumKernargPreloadSGPRsTag);
};
template struct PrivateAccessor<
    NumKernargPreloadSGPRsTag,
    &llvm::GCNUserSGPRUsageInfo::NumKernargPreloadSGPRs>;

// The MFI's add{PrivateSegmentBuffer,FlatScratchInit} routines call
// getNextUserSGPR, which asserts NumSystemSGPRs == 0. By the time we run
// (post-codegen, pre-AsmPrinter), system SGPRs have already been added. We
// temporarily zero NumSystemSGPRs across the add* call and restore afterward.
struct NumSystemSGPRsTag {
  using MemberT = unsigned llvm::SIMachineFunctionInfo::*;
  friend MemberT get(NumSystemSGPRsTag);
};
template struct PrivateAccessor<NumSystemSGPRsTag,
                                &llvm::SIMachineFunctionInfo::NumSystemSGPRs>;

struct SIMFI_NumUserSGPRsTag {
  using MemberT = unsigned llvm::SIMachineFunctionInfo::*;
  friend MemberT get(SIMFI_NumUserSGPRsTag);
};
template struct PrivateAccessor<SIMFI_NumUserSGPRsTag,
                                &llvm::SIMachineFunctionInfo::NumUserSGPRs>;

// Reach into \c llvm::AnalysisManager<Function>'s private results storage
// so we can splice a cached \c MachineFunctionAnalysis::Result from the
// IModule's FAM into the target module's FAM without deep-cloning the
// underlying \c MachineFunction. See \c movePayloadMFIntoTarget below.
//
// The two private members we need (declared at
// \c llvm/IR/PassManager.h:571,575):
//   * \c AnalysisResultLists — \c DenseMap<Function*, ResultListT>, where
//     \c ResultListT is a \c std::list of
//     \c pair<AnalysisKey*, unique_ptr<ResultConceptT>>.
//   * \c AnalysisResults — \c DenseMap<pair<AnalysisKey*, Function*>,
//     ResultListT::iterator>, indexing into the above list.
//
// The \c ResultConceptT template argument set (Function + FAM's public
// nested \c Invalidator class) is spelled out here because both the
// \c AnalysisResultListT / \c AnalysisResultListMapT / \c AnalysisResultMapT
// typedefs are private inside \c AnalysisManager. Keeping these local
// aliases in sync with LLVM's PassManager.h is a maintenance edge — if
// the upstream layout ever changes, this whole helper needs revisiting.
using FAM = llvm::FunctionAnalysisManager;
using FAMResultListT =
    std::list<std::pair<llvm::AnalysisKey *,
                        std::unique_ptr<llvm::detail::AnalysisResultConcept<
                            llvm::Function, FAM::Invalidator>>>>;
using FAMResultListMapT = llvm::DenseMap<llvm::Function *, FAMResultListT>;
using FAMResultMapT =
    llvm::DenseMap<std::pair<llvm::AnalysisKey *, llvm::Function *>,
                   FAMResultListT::iterator>;

struct FAMResultListsTag {
  using MemberT = FAMResultListMapT FAM::*;
  friend MemberT get(FAMResultListsTag);
};
template struct PrivateAccessor<FAMResultListsTag, &FAM::AnalysisResultLists>;

struct FAMResultsTag {
  using MemberT = FAMResultMapT FAM::*;
  friend MemberT get(FAMResultsTag);
};
template struct PrivateAccessor<FAMResultsTag, &FAM::AnalysisResults>;

/// Force \c GCNUserSGPRUsageInfo::FlatScratchInit true. AMDGPUAsmPrinter
/// reads that flag to decide the KD's
/// \c KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT bit, so flipping it
/// here is what makes hardware actually preload FS_INIT on the instrumented
/// dispatch. \c NumUsedUserSGPRs is bumped to keep the two counters
/// consistent with what the ctor would have set on first pass.
void forceFlatScratchInit(llvm::GCNUserSGPRUsageInfo &Info) {
  if (Info.hasFlatScratchInit())
    return;
  Info.*get(FlatScratchInitTag{}) = true;
  Info.*get(NumUsedUserSGPRsTag{}) +=
      llvm::GCNUserSGPRUsageInfo::getNumUserSGPRForField(
          llvm::GCNUserSGPRUsageInfo::FlatScratchInitID);
}

/// Same as \c forceFlatScratchInit for PRIVATE_SEGMENT_BUFFER.
void forcePrivateSegmentBuffer(llvm::GCNUserSGPRUsageInfo &Info) {
  if (Info.hasPrivateSegmentBuffer())
    return;
  Info.*get(PrivateSegmentBufferTag{}) = true;
  Info.*get(NumUsedUserSGPRsTag{}) +=
      llvm::GCNUserSGPRUsageInfo::getNumUserSGPRForField(
          llvm::GCNUserSGPRUsageInfo::PrivateSegmentBufferID);
}

/// Same as \c forceFlatScratchInit for KERNARG_SEGMENT_PTR.
void forceKernargSegmentPtr(llvm::GCNUserSGPRUsageInfo &Info) {
  if (Info.hasKernargSegmentPtr())
    return;
  Info.*get(KernargSegmentPtrTag{}) = true;
  Info.*get(NumUsedUserSGPRsTag{}) +=
      llvm::GCNUserSGPRUsageInfo::getNumUserSGPRForField(
          llvm::GCNUserSGPRUsageInfo::KernargSegmentPtrID);
}

/// Same as \c forceFlatScratchInit for QUEUE_PTR. Enables the queue-ptr
/// preload SGPR pair on the instrumented kernel so the AMDGPUAsmPrinter
/// emits the corresponding \c ENABLE_SGPR_QUEUE_PTR bit and the
/// hidden_queue_ptr kernarg metadata record.
void forceQueuePtr(llvm::GCNUserSGPRUsageInfo &Info) {
  if (Info.hasQueuePtr())
    return;
  Info.*get(QueuePtrTag{}) = true;
  Info.*get(NumUsedUserSGPRsTag{}) +=
      llvm::GCNUserSGPRUsageInfo::getNumUserSGPRForField(
          llvm::GCNUserSGPRUsageInfo::QueuePtrID);
}

/// Same as \c forceFlatScratchInit for PRIVATE_SEGMENT_SIZE. Enables the
/// preload of the per-wave private-segment-size scalar (a single 32-bit
/// SGPR) so \c AMDGPUAsmPrinter emits the corresponding
/// \c ENABLE_SGPR_PRIVATE_SEGMENT_SIZE bit and the runtime provisions
/// the SGPR with the total per-wave scratch size at dispatch time.
/// Used by the dynamic-stack branch of the SP setup to compute the
/// instrumentation SP as \c PSS - Reservation at runtime.
void forcePrivateSegmentSize(llvm::GCNUserSGPRUsageInfo &Info) {
  if (Info.hasPrivateSegmentSize())
    return;
  Info.*get(PrivateSegmentSizeTag{}) = true;
  Info.*get(NumUsedUserSGPRsTag{}) +=
      llvm::GCNUserSGPRUsageInfo::getNumUserSGPRForField(
          llvm::GCNUserSGPRUsageInfo::PrivateSegmentSizeID);
}

/// Same as \c forceFlatScratchInit for DISPATCH_PTR.
void forceDispatchPtr(llvm::GCNUserSGPRUsageInfo &Info) {
  if (Info.hasDispatchPtr())
    return;
  Info.*get(DispatchPtrTag{}) = true;
  Info.*get(NumUsedUserSGPRsTag{}) +=
      llvm::GCNUserSGPRUsageInfo::getNumUserSGPRForField(
          llvm::GCNUserSGPRUsageInfo::DispatchPtrID);
}

/// Same as \c forceFlatScratchInit for DISPATCH_ID.
void forceDispatchID(llvm::GCNUserSGPRUsageInfo &Info) {
  if (Info.hasDispatchID())
    return;
  Info.*get(DispatchIDTag{}) = true;
  Info.*get(NumUsedUserSGPRsTag{}) +=
      llvm::GCNUserSGPRUsageInfo::getNumUserSGPRForField(
          llvm::GCNUserSGPRUsageInfo::DispatchIdID);
}

/// Same as \c forceFlatScratchInit for IMPLICIT_BUFFER_PTR.
void forceImplicitBufferPtr(llvm::GCNUserSGPRUsageInfo &Info) {
  if (Info.hasImplicitBufferPtr())
    return;
  Info.*get(ImplicitBufferPtrTag{}) = true;
  Info.*get(NumUsedUserSGPRsTag{}) +=
      llvm::GCNUserSGPRUsageInfo::getNumUserSGPRForField(
          llvm::GCNUserSGPRUsageInfo::ImplicitBufferPtrID);
}

/// Map a Luthier \c ScalarValueArgument (the thing the instrumentation
/// pipeline records as "this payload wanted X") to the AMDGPU user-SGPR
/// preload it corresponds to, when there is one. Returns \c nullopt for
/// SVs backed by system SGPRs (workgroup ids), VGPRs (workitem ids), or
/// non-preload sources.
static std::optional<llvm::AMDGPUFunctionArgInfo::PreloadedValue>
userSGPRPreloadForSV(ScalarValueArgument SV) {
  using PV = llvm::AMDGPUFunctionArgInfo::PreloadedValue;
  switch (SV) {
  case WAVEFRONT_PRIVATE_SEGMENT_BUFFER:
    return PV::PRIVATE_SEGMENT_BUFFER;
  case KERNEL_ARG_PTR:
  case IMPLICIT_ARG_BUFFER:
    return PV::KERNARG_SEGMENT_PTR;
  case DISPATCH_ID:
    return PV::DISPATCH_ID;
  case FLAT_SCRATCH:
    return PV::FLAT_SCRATCH_INIT;
  case DISPATCH_PTR:
    return PV::DISPATCH_PTR;
  case QUEUE_PTR:
    return PV::QUEUE_PTR;
  case WORK_ITEM_PRIVATE_SEGMENT_SIZE:
    return PV::PRIVATE_SEGMENT_SIZE;
  default:
    return std::nullopt;
  }
}

/// For a given user-SGPR preload, the pair of (flag-flip helper,
/// \c SIMachineFunctionInfo add method) needed to force-enable it.
/// Returns a pair of null functions for preloads that aren't user
/// SGPRs (system SGPRs, preloaded VGPRs, spilled values) — the caller
/// falls through and dispatches those separately.
static std::pair<void (*)(llvm::GCNUserSGPRUsageInfo &),
                 llvm::Register (llvm::SIMachineFunctionInfo::*)(
                     const llvm::SIRegisterInfo &)>
userSGPRPreloadForceOps(llvm::AMDGPUFunctionArgInfo::PreloadedValue PV) {
  using AMDPV = llvm::AMDGPUFunctionArgInfo;
  switch (PV) {
  case AMDPV::PRIVATE_SEGMENT_BUFFER:
    return {&forcePrivateSegmentBuffer,
            &llvm::SIMachineFunctionInfo::addPrivateSegmentBuffer};
  case AMDPV::DISPATCH_PTR:
    return {&forceDispatchPtr, &llvm::SIMachineFunctionInfo::addDispatchPtr};
  case AMDPV::QUEUE_PTR:
    return {&forceQueuePtr, &llvm::SIMachineFunctionInfo::addQueuePtr};
  case AMDPV::KERNARG_SEGMENT_PTR:
    return {&forceKernargSegmentPtr,
            &llvm::SIMachineFunctionInfo::addKernargSegmentPtr};
  case AMDPV::DISPATCH_ID:
    return {&forceDispatchID, &llvm::SIMachineFunctionInfo::addDispatchID};
  case AMDPV::FLAT_SCRATCH_INIT:
    return {&forceFlatScratchInit,
            &llvm::SIMachineFunctionInfo::addFlatScratchInit};
  case AMDPV::PRIVATE_SEGMENT_SIZE:
    return {&forcePrivateSegmentSize,
            &llvm::SIMachineFunctionInfo::addPrivateSegmentSize};
  case AMDPV::IMPLICIT_BUFFER_PTR:
    return {&forceImplicitBufferPtr,
            &llvm::SIMachineFunctionInfo::addImplicitBufferPtr};
  default:
    return {nullptr, nullptr};
  }
}

/// Every AMDGPU \c PreloadedValue the aggregator knows how to clear and
/// re-materialize. Used both for the pre-clear snapshot in
/// \c emitInitialEntryKernelSetup and for the deterministic iteration
/// order that \c forceEnableRequestedUserSGPRPreloads walks when adding
/// preloads back and when reporting \c NewPositions to the caller.
static constexpr llvm::AMDGPUFunctionArgInfo::PreloadedValue
    AllPreloadedValues[] = {
        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER,
        llvm::AMDGPUFunctionArgInfo::DISPATCH_PTR,
        llvm::AMDGPUFunctionArgInfo::QUEUE_PTR,
        llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR,
        llvm::AMDGPUFunctionArgInfo::DISPATCH_ID,
        llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT,
        llvm::AMDGPUFunctionArgInfo::WORKGROUP_ID_X,
        llvm::AMDGPUFunctionArgInfo::WORKGROUP_ID_Y,
        llvm::AMDGPUFunctionArgInfo::WORKGROUP_ID_Z,
        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
        llvm::AMDGPUFunctionArgInfo::IMPLICIT_BUFFER_PTR,
        llvm::AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR,
        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_SIZE,
        llvm::AMDGPUFunctionArgInfo::WORKITEM_ID_X,
        llvm::AMDGPUFunctionArgInfo::WORKITEM_ID_Y,
        llvm::AMDGPUFunctionArgInfo::WORKITEM_ID_Z,
    };

/// True if \p PV lives in the user-SGPR block (the SGPRs the HW
/// preloads into \c s[0 .. NumUserSGPRs-1]).
static bool isUserSGPRPreload(llvm::AMDGPUFunctionArgInfo::PreloadedValue PV) {
  using AMDPV = llvm::AMDGPUFunctionArgInfo;
  switch (PV) {
  case AMDPV::PRIVATE_SEGMENT_BUFFER:
  case AMDPV::DISPATCH_PTR:
  case AMDPV::QUEUE_PTR:
  case AMDPV::KERNARG_SEGMENT_PTR:
  case AMDPV::DISPATCH_ID:
  case AMDPV::FLAT_SCRATCH_INIT:
  case AMDPV::IMPLICIT_BUFFER_PTR:
  case AMDPV::PRIVATE_SEGMENT_SIZE:
    return true;
  default:
    return false;
  }
}

/// Result of \c forceEnableRequestedUserSGPRPreloads. \c NewPositions
/// records the physreg each enabled \c PreloadedValue landed on after
/// the aggregator cleared and re-added everything in the union of
/// (instrumentation-required, already-preloaded). The caller drives the
/// restore-move loop off of this map instead of re-querying
/// \c SIMFI.getPreloadedReg after the fact. The kernarg-preload anchors
/// and \c KernargPreloadDisabled flag control the two branches of the
/// kernarg handling step (see \c emitInitialEntryKernelSetup).
struct ForceEnableResult {
  llvm::SmallVector<
      std::pair<llvm::AMDGPUFunctionArgInfo::PreloadedValue, llvm::MCRegister>,
      16>
      NewPositions;

  /// The original SGPR range \c [OrigPreloadStartSGPR .. OrigPreloadEndSGPR]
  /// where the app kernel's kernarg-preload block sat before the
  /// aggregator ran. Zero when \c OrigPreloadLength was 0. Used by the
  /// caller both to emit the \c S_LOAD_DWORD_IMM fallback (writing into
  /// this range) and to emit the S_MOV shuffle when preload stays
  /// enabled.
  llvm::MCRegister OrigPreloadStartSGPR;
  llvm::MCRegister OrigPreloadEndSGPR;

  /// True iff \c ST.hasKernargPreload() is true, \c OrigPreloadLength
  /// was > 0, and the aggregator determined the (new fixed user SGPRs)
  /// + (preload) total exceeded \c ST.getMaxNumUserSGPRs(). The
  /// aggregator has already zeroed \c NumKernargPreloadSGPRs — the
  /// caller must emit the \c S_LOAD_DWORD_IMM fallback.
  bool KernargPreloadDisabled = false;
};

/// Aggregator that collapses the four historical preload force-enable
/// sites (top-level user-SGPR walker, \c emitCodeToSetupScratch,
/// per-SVA inline switch, kernarg-preload ceiling branch) into one
/// pass. Clears every \c ArgDescriptor field on \c SIMFI.ArgInfo, zeros
/// every relevant \c GCNUserSGPRUsageInfo bool + the \c NumUsedUserSGPRs
/// / \c NumUserSGPRs / \c NumSystemSGPRs counters (preserving
/// \c NumKernargPreloadSGPRs), then re-materializes the union of
/// (\p RequiredPreloads, \p PreloadedArgSnapshot) at canonical physreg
/// positions in \c AllPreloadedValues order.
///
/// Ordering guarantees:
///   * All user SGPRs are added first, with \c NumSystemSGPRs == 0
///     (the \c getNextUserSGPR precondition holds trivially post-clear).
///   * The kernarg preload block is either re-applied (bumping
///     \c NumUserSGPRs / \c NumUsedUserSGPRs) or disabled BEFORE any
///     system SGPRs are added, so system SGPRs land at
///     \c s[FixedAfterForce + PreloadLen + N_sys].
///   * System SGPRs (\c WORKGROUP_ID_X/Y/Z, PSWO) are added next.
///   * Preloaded VGPRs (\c WORKITEM_ID_X/Y/Z) are set last.
///
/// \c amdgpu-no-* fn-attrs on \p KernelF are stripped for every
/// PreloadedValue in the union that has such an attr — the
/// AMDGPUAsmPrinter reads those attrs to decide which KD enable-bits
/// and metadata records to emit.
static ForceEnableResult forceEnableRequestedUserSGPRPreloads(
    llvm::SIMachineFunctionInfo &SIMFI, const llvm::GCNSubtarget &ST,
    const llvm::SIRegisterInfo &TRI, llvm::Function &KernelF,
    const llvm::SmallSet<llvm::AMDGPUFunctionArgInfo::PreloadedValue, 16>
        &RequiredPreloads,
    llvm::ArrayRef<std::pair<llvm::AMDGPUFunctionArgInfo::PreloadedValue,
                             llvm::MCRegister>>
        PreloadedArgSnapshot,
    unsigned OrigPreloadLength) {
  using AMDPV = llvm::AMDGPUFunctionArgInfo;
  ForceEnableResult FR;

  // 1. Aggregate the union of instrumentation-required preloads and
  //    the app kernel's already-preloaded set.
  llvm::SmallSet<AMDPV::PreloadedValue, 16> Union;
  for (AMDPV::PreloadedValue PV : RequiredPreloads)
    Union.insert(PV);
  for (const auto &[PV, Reg] : PreloadedArgSnapshot)
    Union.insert(PV);

  // 2. Snapshot the preloaded kernarg block anchors BEFORE we mutate
  //    any SIMFI state. The block sits at the tail of the app kernel's
  //    used user SGPRs.
  auto &Info = SIMFI.getUserSGPRInfo();
  const unsigned OrigNumUsedUserSGPRs = Info.getNumUsedUserSGPRs();
  if (OrigPreloadLength > 0 && OrigNumUsedUserSGPRs >= OrigPreloadLength) {
    FR.OrigPreloadStartSGPR = llvm::MCRegister::from(
        llvm::AMDGPU::SGPR0 + OrigNumUsedUserSGPRs - OrigPreloadLength);
    FR.OrigPreloadEndSGPR = llvm::MCRegister::from(
        FR.OrigPreloadStartSGPR.id() + OrigPreloadLength - 1);
  }

  // 3. Clear ArgInfo + SIMFI counters (preserve NumKernargPreloadSGPRs).
  llvm::AMDGPUFunctionArgInfo &ArgInfo = SIMFI.getArgInfo();
  ArgInfo.PrivateSegmentBuffer = llvm::ArgDescriptor{};
  ArgInfo.DispatchPtr = llvm::ArgDescriptor{};
  ArgInfo.QueuePtr = llvm::ArgDescriptor{};
  ArgInfo.KernargSegmentPtr = llvm::ArgDescriptor{};
  ArgInfo.DispatchID = llvm::ArgDescriptor{};
  ArgInfo.FlatScratchInit = llvm::ArgDescriptor{};
  ArgInfo.PrivateSegmentSize = llvm::ArgDescriptor{};
  ArgInfo.ImplicitBufferPtr = llvm::ArgDescriptor{};
  ArgInfo.WorkGroupIDX = llvm::ArgDescriptor{};
  ArgInfo.WorkGroupIDY = llvm::ArgDescriptor{};
  ArgInfo.WorkGroupIDZ = llvm::ArgDescriptor{};
  ArgInfo.WorkGroupInfo = llvm::ArgDescriptor{};
  ArgInfo.PrivateSegmentWaveByteOffset = llvm::ArgDescriptor{};
  ArgInfo.WorkItemIDX = llvm::ArgDescriptor{};
  ArgInfo.WorkItemIDY = llvm::ArgDescriptor{};
  ArgInfo.WorkItemIDZ = llvm::ArgDescriptor{};
  ArgInfo.ImplicitArgPtr = llvm::ArgDescriptor{};

  Info.*get(PrivateSegmentBufferTag{}) = false;
  Info.*get(DispatchPtrTag{}) = false;
  Info.*get(QueuePtrTag{}) = false;
  Info.*get(KernargSegmentPtrTag{}) = false;
  Info.*get(DispatchIDTag{}) = false;
  Info.*get(FlatScratchInitTag{}) = false;
  Info.*get(PrivateSegmentSizeTag{}) = false;
  Info.*get(ImplicitBufferPtrTag{}) = false;
  Info.*get(NumUsedUserSGPRsTag{}) = 0;
  SIMFI.*get(SIMFI_NumUserSGPRsTag{}) = 0;
  SIMFI.*get(NumSystemSGPRsTag{}) = 0;

  // 4. Strip amdgpu-no-* fn-attrs on \p KernelF for anything in the
  //    Union whose enablement is gated by such an attr. Metadata-only
  //    attrs (e.g. amdgpu-no-implicitarg-ptr) are left to the
  //    kernarg-buffer expansion step.
  for (AMDPV::PreloadedValue PV : AllPreloadedValues) {
    if (!Union.contains(PV))
      continue;
    switch (PV) {
    case AMDPV::DISPATCH_PTR:
      KernelF.removeFnAttr("amdgpu-no-dispatch-ptr");
      break;
    case AMDPV::DISPATCH_ID:
      KernelF.removeFnAttr("amdgpu-no-dispatch-id");
      break;
    case AMDPV::IMPLICIT_BUFFER_PTR:
      KernelF.removeFnAttr("amdgpu-no-implicit-buffer-ptr");
      break;
    case AMDPV::WORKGROUP_ID_X:
      KernelF.removeFnAttr("amdgpu-no-workgroup-id-x");
      KernelF.removeFnAttr("amdgpu-no-cluster-id-x");
      break;
    case AMDPV::WORKGROUP_ID_Y:
      KernelF.removeFnAttr("amdgpu-no-workgroup-id-y");
      KernelF.removeFnAttr("amdgpu-no-cluster-id-y");
      break;
    case AMDPV::WORKGROUP_ID_Z:
      KernelF.removeFnAttr("amdgpu-no-workgroup-id-z");
      KernelF.removeFnAttr("amdgpu-no-cluster-id-z");
      break;
    case AMDPV::WORKITEM_ID_X:
      KernelF.removeFnAttr("amdgpu-no-workitem-id-x");
      break;
    case AMDPV::WORKITEM_ID_Y:
      KernelF.removeFnAttr("amdgpu-no-workitem-id-y");
      break;
    case AMDPV::WORKITEM_ID_Z:
      KernelF.removeFnAttr("amdgpu-no-workitem-id-z");
      break;
    default:
      break;
    }
  }

  // 5a. Re-add all user SGPRs first (canonical AMDGPU sub-order via
  //     \c AllPreloadedValues). The clear guarantees \c NumSystemSGPRs
  //     == 0, so the \c getNextUserSGPR precondition holds and no
  //     save/zero/restore workaround is needed.
  for (AMDPV::PreloadedValue PV : AllPreloadedValues) {
    if (!Union.contains(PV) || !isUserSGPRPreload(PV))
      continue;
    auto [Force, Add] = userSGPRPreloadForceOps(PV);
    if (!Force || !Add)
      continue;
    Force(Info);
    (SIMFI.*Add)(TRI);
  }

  // 6. Kernarg-preload ceiling check. Only meaningful on subtargets
  //    that support kernarg preload and when the app kernel had a
  //    non-empty preload block. If the new fixed-user-SGPR count plus
  //    the original preload length exceeds the HW ceiling, drop the
  //    preload block entirely (\c AMDGPUAsmPrinter will emit
  //    \c kernarg_preload_length=0 in the KD and the caller falls
  //    back to \c S_LOAD_DWORD_IMM). Otherwise, re-apply the block by
  //    bumping the user-SGPR counters — this MUST happen before system
  //    SGPRs are added below so the system SGPRs land at
  //    \c s[FixedAfterForce + OrigPreloadLength ..], matching the KD's
  //    dispatch-time layout.
  if (ST.hasKernargPreload() && OrigPreloadLength > 0) {
    const unsigned NewFixedUserSGPRs = Info.getNumUsedUserSGPRs();
    if (NewFixedUserSGPRs + OrigPreloadLength > ST.getMaxNumUserSGPRs()) {
      Info.*get(NumKernargPreloadSGPRsTag{}) = 0;
      FR.KernargPreloadDisabled = true;
    } else {
      SIMFI.*get(SIMFI_NumUserSGPRsTag{}) += OrigPreloadLength;
      Info.*get(NumUsedUserSGPRsTag{}) += OrigPreloadLength;
    }
  }

  // 5b. Re-add system SGPRs (WorkGroupIDs, PSWO).
  for (AMDPV::PreloadedValue PV : AllPreloadedValues) {
    if (!Union.contains(PV))
      continue;
    switch (PV) {
    case AMDPV::WORKGROUP_ID_X:
      (void)SIMFI.addWorkGroupIDX();
      break;
    case AMDPV::WORKGROUP_ID_Y:
      (void)SIMFI.addWorkGroupIDY();
      break;
    case AMDPV::WORKGROUP_ID_Z:
      (void)SIMFI.addWorkGroupIDZ();
      break;
    case AMDPV::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET:
      (void)SIMFI.addPrivateSegmentWaveByteOffset();
      break;
    default:
      break;
    }
  }

  // 5c. Set preloaded VGPRs (WorkItemIDs). Mirrors the packed-TID mask
  //     branch of the previous inline switch.
  const bool HasPacked = ST.hasFeature(llvm::AMDGPU::FeaturePackedTID);
  for (AMDPV::PreloadedValue PV : AllPreloadedValues) {
    if (!Union.contains(PV))
      continue;
    switch (PV) {
    case AMDPV::WORKITEM_ID_X:
      SIMFI.setWorkItemIDX(llvm::ArgDescriptor::createRegister(
          llvm::AMDGPU::VGPR0, HasPacked ? 0x3ffu : ~0u));
      break;
    case AMDPV::WORKITEM_ID_Y:
      SIMFI.setWorkItemIDY(llvm::ArgDescriptor::createRegister(
          HasPacked ? llvm::AMDGPU::VGPR0 : llvm::AMDGPU::VGPR1,
          HasPacked ? (0x3ffu << 10) : ~0u));
      break;
    case AMDPV::WORKITEM_ID_Z:
      SIMFI.setWorkItemIDZ(llvm::ArgDescriptor::createRegister(
          HasPacked ? llvm::AMDGPU::VGPR0 : llvm::AMDGPU::VGPR2,
          HasPacked ? (0x3ffu << 20) : ~0u));
      break;
    default:
      break;
    }
  }

  // 7. Record NewPositions in \c AllPreloadedValues order.
  for (AMDPV::PreloadedValue PV : AllPreloadedValues) {
    if (!Union.contains(PV))
      continue;
    if (llvm::MCRegister R = SIMFI.getPreloadedReg(PV))
      FR.NewPositions.push_back({PV, R});
  }

  return FR;
}

/// Turn off the kernel-argument preload feature on the instrumented
/// kernel. Zeroes \c NumKernargPreloadSGPRs in \c GCNUserSGPRUsageInfo
/// so \c AMDGPUAsmPrinter emits \c kernarg_preload_length=0 in the KD,
/// and decrements \c SIMachineFunctionInfo::NumUserSGPRs by the
/// original preload length so the KD's user-SGPR count drops back to
/// just the system-user-SGPR count (avoiding a HW-limit violation
/// when the force-enabled system SGPRs plus preload would exceed the
/// 16-user-SGPR ceiling).
void disableKernargPreload(llvm::SIMachineFunctionInfo &MFI,
                           unsigned OrigPreloadLength) {
  MFI.getUserSGPRInfo().*get(NumKernargPreloadSGPRsTag{}) = 0;
  unsigned &NumUserSGPRs = MFI.*get(SIMFI_NumUserSGPRsTag{});
  assert(NumUserSGPRs >= OrigPreloadLength &&
         "NumUserSGPRs must include the preload region we're removing");
  NumUserSGPRs -= OrigPreloadLength;
}

/// Emits the per-wave scratch setup at the kernel entry: spills the
/// kernarg-derived PSB.sub0/sub1 and FLAT_SCRATCH_INIT lo/hi into SVA
/// lanes, adds PRIVATE_SEGMENT_WAVE_BYTE_OFFSET to compute the wave's
/// scratch base, stores SGPR32 to the instrumentation-stack-start lane,
/// and reads the spilled kernarg values back into SGPR0/1/FS_LO/HI so
/// the application's prolog still sees them.
///
/// The instrumentation SP is derived from four inputs:
///   * \p AppUsesDynamicStack — the app kernel's MFI.hasVarSizedObjects().
///   * \p AppPrivateSegmentFixedSize — the app kernel's
///     MFI.getStackSize() (the static top of the app's stack).
///   * \p PayloadUsesDynamicStack — set if any attached injected-payload
///     MF has var-sized stack objects.
///   * \p PayloadMaxFixedStackSize — the max MFI.getStackSize() across
///     all attached payload MFs.
/// The instrumentation SP is saved into the SVA's StackPointerStoreLane
/// so the payload prologue can pick it up on entry. SGPR0 is used as the
/// scratch register to materialize the value; it is spilled to the
/// frame-pointer spill lane before use and restored from it after the
/// V_WRITELANE that stores the SP.
///
/// The instrumentation frame reserves 8 bytes immediately below the
/// initial SP for two 32-bit slots used by the partial-callgraph V0
/// handoff protocol: emergency VGPR spill at SP-8 and SVA spill at
/// SP-4. This carve-out is applied in both paths:
///   * app static: SP = \p AppPrivateSegmentFixedSize + 8 (top of the
///     app's static frame, plus the 8-byte slot region). Materialized
///     with \c S_MOV_B32 SGPR0, SP.
///   * app dynamic: SP = PRIVATE_SEGMENT_SIZE - Reservation where
///     Reservation = payload budget + 8. Queried at runtime via the
///     preloaded PSS SGPR (force-enabled here if the app didn't
///     already request it) and materialized with
///     \c S_SUB_U32 SGPR0, PSS_reg, Reservation. Payload budget is
///     \p PayloadMaxFixedStackSize for static payloads and
///     \p LuthierInstrumentationStackSize for dynamic payloads.
llvm::Error emitCodeToSetupScratch(llvm::MachineInstr &EntryInstr,
                                   llvm::MCRegister SVSStorageVGPR,
                                   bool AppUsesDynamicStack,
                                   unsigned AppPrivateSegmentFixedSize,
                                   bool PayloadUsesDynamicStack,
                                   unsigned PayloadMaxFixedStackSize,
                                   const StateValueArraySpecs &Specs) {
  auto &MF = *EntryInstr.getMF();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  const auto &TRI = *ST.getRegisterInfo();
  auto &MFI = *MF.getInfo<llvm::SIMachineFunctionInfo>();
  // On architected-flat-scratch subtargets (gfx9-CDNA, gfx10.3+, gfx11+,
  // gfx12) FLAT_SCRATCH_INIT is not a preloaded SGPR — the hardware sets
  // up the flat-scratch state into the architectural register directly,
  // so the kernel prolog has no SGPR pair to spill into the SVA. The
  // matching slot lookup returns nullopt, and the corresponding spill/
  // restore step must be skipped. \c InjectedPayloadPEIPass already
  // gates its FrameSpillSlots on this same predicate.
  const bool ArchitectedFS = ST.hasArchitectedFlatScratch();
  bool HasFS = ST.enableFlatScratch();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   emitCodeToSetupScratch MF='"
             << MF.getName()
             << "' SVSVGPR=" << llvm::printReg(SVSStorageVGPR, &TRI)
             << " appDynStack=" << AppUsesDynamicStack
             << " appPrivSegFixedSize=" << AppPrivateSegmentFixedSize
             << " payloadDynStack=" << PayloadUsesDynamicStack
             << " payloadMaxFixedSize=" << PayloadMaxFixedStackSize
             << " archFS=" << ArchitectedFS << "\n");

  if (!ArchitectedFS) {

    /// Get the private wave byte offset. The aggregator in
    /// \c emitInitialEntryKernelSetup installs PSWO up front whenever
    /// \c RequiresScratchAndStackSetup is true on a non-arch-FS target;
    /// its absence here is a bug in the caller's \c RequiredPreloads
    /// computation, not an operating condition to paper over.
    llvm::MCRegister PSWO = MFI.getPreloadedReg(
        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);
    if (!PSWO)
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: kernel '{0}' requires scratch/stack "
          "setup on a non-architected-FS target but "
          "PRIVATE_SEGMENT_WAVE_BYTE_OFFSET was not force-enabled by the "
          "preload aggregator",
          MF.getName()));

    auto EmitScratchPSBInit = [&](llvm::MCRegister Lo, llvm::MCRegister Hi,
                                  uint8_t Lane) {
      // 1. Spill orig Lo/Hi to the SP/FP spill lanes.
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32),
                          SVSStorageVGPR)
          .addReg(Lo)
          .addImm(Specs.getStackPointerRegSpillLane())
          .addReg(SVSStorageVGPR);
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32),
                          SVSStorageVGPR)
          .addReg(Hi)
          .addImm(Specs.getFramePointerRegSpillLane())
          .addReg(SVSStorageVGPR);
      // 2. Compute per-wave Lo/Hi.
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::S_ADD_U32))
          .addReg(Lo, llvm::RegState::Define)
          .addReg(Lo, llvm::RegState::Kill)
          .addReg(PSWO);
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::S_ADDC_U32))
          .addReg(Hi, llvm::RegState::Define)
          .addReg(Hi, llvm::RegState::Kill)
          .addImm(0);
      // 3. Save per-wave Lo/Hi to the instrumentation home lanes.
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32),
                          SVSStorageVGPR)
          .addReg(Lo)
          .addImm(Lane)
          .addReg(SVSStorageVGPR);
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32),
                          SVSStorageVGPR)
          .addReg(Hi)
          .addImm(Lane + 1)
          .addReg(SVSStorageVGPR);
      // 4. Restore orig Lo/Hi from the spill lanes.
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_READLANE_B32), Lo)
          .addReg(SVSStorageVGPR)
          .addImm(Specs.getStackPointerRegSpillLane());
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_READLANE_B32), Hi)
          .addReg(SVSStorageVGPR)
          .addImm(Specs.getFramePointerRegSpillLane());
    };

    if (!HasFS) {
      // PSB was force-enabled up front in \c emitInitialEntryKernelSetup 's
      // post-snapshot user-SGPR block when the scratch-and-stack setup
      // requires it.
      llvm::MCRegister PSB = MFI.getPreloadedReg(
          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER);
      if (!PSB)
        return LUTHIER_MAKE_GENERIC_ERROR(
            "PRIVATE_SEGMENT_BUFFER preload was not enabled for the "
            "non-arch-FS buffer-scratch path");

      auto PSBLane = Specs.findArgumentLane(WAVEFRONT_PRIVATE_SEGMENT_BUFFER);
      if (PSBLane == Specs.argument_lane_end()) {
        return LUTHIER_MAKE_GENERIC_ERROR(
            "Non-architected FS target with no FS enabled doesn't have private "
            "segment enabled for scratch");
      }

      // PSB initialization.
      EmitScratchPSBInit(TRI.getSubReg(PSB, llvm::AMDGPU::sub0),
                         TRI.getSubReg(PSB, llvm::AMDGPU::sub1),
                         PSBLane->second);

      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32),
                          SVSStorageVGPR)
          .addReg(TRI.getSubReg(PSB, llvm::AMDGPU::sub2))
          .addImm(PSBLane->second + 2)
          .addReg(SVSStorageVGPR);
      (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32),
                          SVSStorageVGPR)
          .addReg(TRI.getSubReg(PSB, llvm::AMDGPU::sub3))
          .addImm(PSBLane->second + 3)
          .addReg(SVSStorageVGPR);
    }

    // FS_INIT was force-enabled up front in the post-snapshot user-SGPR
    // block when RequiresScratchAndStackSetup was true on a non-arch-FS
    // target — so it should be available here.
    llvm::MCRegister FSInit =
        MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT);
    if (!FSInit)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "FLAT_SCRATCH_INIT preload was not enabled for the non-arch-FS "
          "target");

    auto FSLane = Specs.findArgumentLane(FLAT_SCRATCH);
    if (FSLane == Specs.argument_lane_end()) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Non-architected FS target has not FS enabled");
    }

    // FS initialization.
    EmitScratchPSBInit(TRI.getSubReg(FSInit, llvm::AMDGPU::sub0),
                       TRI.getSubReg(FSInit, llvm::AMDGPU::sub1),
                       FSLane->second);
  }

  // Compute the instrumentation SP and stash it in the SVA's
  // StackPointerStoreLane. The instrumentation frame reserves two
  // 32-bit slots immediately below SP for the partial-callgraph V0
  // handoff protocol:
  //   * [SP-8, SP-4) — emergency VGPR spill slot (holds V0's app
  //     value while the SVA is loaded into V0 across an unresolved-
  //     edge call).
  //   * [SP-4, SP)   — SVA spill slot (used by the spilled SVS
  //     schemes to hold the SVA itself when V0 must be repurposed).
  // \c SVSSlotsReservation captures that 8-byte carve-out; the
  // initial SP is always \c Base + SVSSlotsReservation so the two
  // slots live below SP at fixed offsets and the payload's own
  // growth region begins at SP itself.
  //
  // Setup steps:
  //   1. Spill SGPR0 to the frame-pointer spill lane of the SVA
  //      (SGPR0 is the scratch register we use to materialize the SP
  //      value; we restore it at the end).
  //   2. Materialize the instrumentation SP value in SGPR0:
  //        * app static:  SGPR0 = AppPrivateSegmentFixedSize +
  //                                 SVSSlotsReservation.
  //        * app dynamic: SGPR0 = PRIVATE_SEGMENT_SIZE - Reservation
  //          (queried at runtime via the preloaded PSS SGPR;
  //          Reservation = payload budget + SVSSlotsReservation).
  //          Force-enable the PSS preload if the app didn't request
  //          it.
  //   3. Save SGPR0 (the instrumentation SP) into the SVA's
  //      StackPointerStoreLane — this is where the payload prologue
  //      picks it up on entry.
  //   4. Restore SGPR0 from the frame-pointer spill lane.
  static constexpr unsigned SVSSlotsReservation = 8;

  // 1. Spill SGPR0 to the FP spill lane.
  (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(llvm::AMDGPU::SGPR0)
      .addImm(Specs.getFramePointerRegSpillLane())
      .addReg(SVSStorageVGPR);

  // 2. Materialize the SP value in SGPR0.
  if (!AppUsesDynamicStack) {
    const unsigned SP = AppPrivateSegmentFixedSize + SVSSlotsReservation;
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]     "
                  "InstrumentationStackStart(static)="
               << SP << " (= AppPrivateSegmentFixedSize("
               << AppPrivateSegmentFixedSize << ") + SVSSlotsReservation("
               << SVSSlotsReservation << "))\n");
    (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_MOV_B32), llvm::AMDGPU::SGPR0)
        .addImm(SP);
  } else {
    // The PRIVATE_SEGMENT_SIZE preload is installed up front by the
    // aggregator in \c emitInitialEntryKernelSetup whenever the app
    // kernel uses dynamic stack; its absence here is a bug in the
    // caller's \c RequiredPreloads computation.
    llvm::MCRegister PSS =
        MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_SIZE);
    if (!PSS)
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: kernel '{0}' uses dynamic stack but "
          "PRIVATE_SEGMENT_SIZE was not force-enabled by the preload "
          "aggregator",
          MF.getName()));

    const unsigned PayloadBudget =
        PayloadUsesDynamicStack
            ? static_cast<unsigned>(LuthierInstrumentationStackSize)
            : PayloadMaxFixedStackSize;
    const unsigned Reservation = PayloadBudget + SVSSlotsReservation;
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]     "
                  "InstrumentationStackStart(dynamic)= PSS("
               << llvm::printReg(PSS, &TRI) << ") - " << Reservation
               << " (= PayloadBudget(" << PayloadBudget
               << ") + SVSSlotsReservation(" << SVSSlotsReservation << "))\n");
    // SGPR0 = PSS - (PayloadBudget + SVSSlotsReservation).
    (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_SUB_U32), llvm::AMDGPU::SGPR0)
        .addReg(PSS)
        .addImm(Reservation);
  }

  // 3. Save the instrumentation SP (SGPR0) into the SVA's
  //    StackPointerStoreLane.
  (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(llvm::AMDGPU::SGPR0)
      .addImm(Specs.getStackPointerStoreLane())
      .addReg(SVSStorageVGPR);

  // 4. Restore SGPR0 from the FP spill lane.
  (void)llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::V_READLANE_B32),
                      llvm::AMDGPU::SGPR0)
      .addReg(SVSStorageVGPR)
      .addImm(Specs.getFramePointerRegSpillLane());

  return llvm::Error::success();
}

llvm::Error emitCodeToStoreSGPRKernelArg(llvm::MachineInstr &InsertionPoint,
                                         llvm::MCRegister SrcSGPR,
                                         llvm::MCRegister SVSVGPR,
                                         int SpillSlotStart, int NumSlots,
                                         bool KillAfterUse) {
  const auto &TRI = *InsertionPoint.getMF()->getSubtarget().getRegisterInfo();
  const auto &TII = *InsertionPoint.getMF()->getSubtarget().getInstrInfo();
  size_t Size = TRI.getRegSizeInBits(*TRI.getPhysRegBaseClass(SrcSGPR));
  auto &InsertionPointMBB = *InsertionPoint.getParent();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]     emitCodeToStoreSGPRKernelArg "
                "src="
             << llvm::printReg(SrcSGPR, &TRI)
             << " SVS=" << llvm::printReg(SVSVGPR, &TRI) << " size=" << Size
             << "b slotStart=" << SpillSlotStart << " numSlots=" << NumSlots
             << " kill=" << KillAfterUse << "\n");
  if (Size == 32) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NumSlots == 1, "Mismatch between number of SGPRs in the argument and "
                       "save slot lanes."));
    (void)llvm::BuildMI(InsertionPointMBB, InsertionPoint, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSVGPR)
        .addReg(SrcSGPR, llvm::getKillRegState(KillAfterUse))
        .addImm(SpillSlotStart)
        .addReg(SVSVGPR);
  } else {
    size_t NumChannels = Size / 32;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NumSlots == NumChannels,
        "Mismatch between number of SGPRs in the argument and "
        "save slot lanes."));
    for (int i = 0; i < NumSlots; i++) {
      auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
      (void)llvm::BuildMI(InsertionPointMBB, InsertionPoint, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSVGPR)
          .addReg(TRI.getSubReg(SrcSGPR, SubIdx),
                  llvm::getKillRegState(KillAfterUse))
          .addImm(SpillSlotStart + i)
          .addReg(SVSVGPR);
    }
  }
  return llvm::Error::success();
}

/// Emit the extended kernarg buffer expansion at the initial-entry
/// kernel's entry. Only called when \c IMPLICIT_ARG_BUFFER is in the
/// aggregate SVA arg set.
///
/// The loader has arranged for \c KERNARG_SEGMENT_PTR to point at an
/// extended kernarg buffer laid out as
///   \verbatim
///   [app_kernarg_ptr : 64][impl_args ...]
///   \endverbatim
/// (implicit args packed inline immediately after the first 8 bytes).
/// Two shapes depending on whether the app kernel has an
/// explicit kernarg pointer at all:
///
///  * If the app kernel's IR takes a first kernarg (i64 pointer):
///    save \c KERNARG_SEGMENT_PTR through two free SVA lanes,
///    \c S_ADD it by 8 in place to derive the impl-args base
///    pointer, \c V_WRITELANE that pointer into the
///    IMPLICIT_ARG_BUFFER SVA lanes, restore \c KERNARG_SEGMENT_PTR
///    from the saved SVA lanes, then \c S_LOAD_DWORDX2 the first 8
///    bytes of the extended buffer back into \c KERNARG_SEGMENT_PTR
///    so the app kernel prolog reads the app's original kernarg
///    address from that SGPR pair.
///
///  * If the app kernel has no explicit kernarg (\c KernelF.arg_size
///    is 0): the extended kernarg is just impl args starting at
///    offset 0. \c KERNARG_SEGMENT_PTR is already the impl-args
///    base — copy its two halves directly into the SVA lanes with
///    no arithmetic, no save/restore, and no \c S_LOAD. The app
///    kernel doesn't consume \c KERNARG_SEGMENT_PTR, so its final
///    value is clobbered.
llvm::Error emitKernargBufferExpansion(llvm::MachineInstr &EntryInstr,
                                       llvm::MCRegister SVSStorageVGPR,
                                       const StateValueArraySpecs &Specs,
                                       bool HasAppKernarg) {
  auto &MF = *EntryInstr.getMF();
  auto &MFI = *MF.getInfo<llvm::SIMachineFunctionInfo>();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  const auto &TRI = *ST.getRegisterInfo();
  auto &MBB = *EntryInstr.getParent();

  llvm::MCRegister KernargPtr =
      MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);
  if (!KernargPtr)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "TargetModulePatcherPass: kernel '{0}' needs IMPLICIT_ARG_BUFFER "
        "kernarg expansion but KERNARG_SEGMENT_PTR was not enabled",
        MF.getName()));

  auto ImplLane = Specs.findArgumentLane(IMPLICIT_ARG_BUFFER);
  if (ImplLane == Specs.argument_lane_end())
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "TargetModulePatcherPass: kernel '{0}' gated on IMPLICIT_ARG_BUFFER "
        "but the SVA specs did not assign it a lane",
        MF.getName()));

  llvm::DebugLoc DL;
  llvm::MCRegister KernSub0 = TRI.getSubReg(KernargPtr, llvm::AMDGPU::sub0);
  llvm::MCRegister KernSub1 = TRI.getSubReg(KernargPtr, llvm::AMDGPU::sub1);

  if (!HasAppKernarg) {
    // Extended kernarg is just impl args starting at offset 0.
    // KERNARG_SEGMENT_PTR already IS the impl-args base pointer.
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]     emitKernargBufferExpansion "
                  "MF='"
               << MF.getName() << "' KernargPtr="
               << llvm::printReg(KernargPtr, &TRI) << " (no app kernarg)"
               << " ImplLaneBase=" << static_cast<unsigned>(ImplLane->second)
               << "\n");
    (void)llvm::BuildMI(MBB, EntryInstr, DL,
                        TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
        .addReg(KernSub0)
        .addImm(ImplLane->second + 0)
        .addReg(SVSStorageVGPR);
    (void)llvm::BuildMI(MBB, EntryInstr, DL,
                        TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
        .addReg(KernSub1)
        .addImm(ImplLane->second + 1)
        .addReg(SVSStorageVGPR);
    return llvm::Error::success();
  }

  // App-has-kernargs path. Save KERNARG_SEGMENT_PTR into the
  // dedicated stack-pointer and frame-pointer spill lanes (SVA lanes
  // 0 and 1 — reserved by \c StateValueArraySpecs for exactly this
  // "save app-live SGPR across setup" pattern). Lane 2 is the
  // instrumentation stack-pointer store slot and is used by
  // \c emitCodeToSetupScratch; leave it alone here.
  const uint8_t SPSpillLane = Specs.getStackPointerRegSpillLane();
  const uint8_t FPSpillLane = Specs.getFramePointerRegSpillLane();

  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]     emitKernargBufferExpansion "
                "MF='"
             << MF.getName()
             << "' KernargPtr=" << llvm::printReg(KernargPtr, &TRI)
             << " SaveLanes=" << SPSpillLane << "," << FPSpillLane
             << " ImplLaneBase=" << ImplLane->second << "\n");

  // 1. Save KERNARG_SEGMENT_PTR's two halves into the SP + FP spill
  //    lanes.
  (void)llvm::BuildMI(MBB, EntryInstr, DL,
                      TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(KernSub0)
      .addImm(SPSpillLane)
      .addReg(SVSStorageVGPR);
  (void)llvm::BuildMI(MBB, EntryInstr, DL,
                      TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(KernSub1)
      .addImm(FPSpillLane)
      .addReg(SVSStorageVGPR);

  // 2. KERNARG_SEGMENT_PTR += 8 in place — the pair now holds the
  //    impl-args base pointer.
  (void)llvm::BuildMI(MBB, EntryInstr, DL, TII.get(llvm::AMDGPU::S_ADD_U32),
                      KernSub0)
      .addReg(KernSub0)
      .addImm(8);
  (void)llvm::BuildMI(MBB, EntryInstr, DL, TII.get(llvm::AMDGPU::S_ADDC_U32),
                      KernSub1)
      .addReg(KernSub1)
      .addImm(0);

  // 3. Write the impl-args base into IMPLICIT_ARG_BUFFER SVA lanes.
  (void)llvm::BuildMI(MBB, EntryInstr, DL,
                      TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(KernSub0)
      .addImm(ImplLane->second + 0)
      .addReg(SVSStorageVGPR);
  (void)llvm::BuildMI(MBB, EntryInstr, DL,
                      TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(KernSub1)
      .addImm(ImplLane->second + 1)
      .addReg(SVSStorageVGPR);

  // 4. Restore KERNARG_SEGMENT_PTR from the SP + FP spill lanes.
  (void)llvm::BuildMI(MBB, EntryInstr, DL,
                      TII.get(llvm::AMDGPU::V_READLANE_B32), KernSub0)
      .addReg(SVSStorageVGPR)
      .addImm(SPSpillLane);
  (void)llvm::BuildMI(MBB, EntryInstr, DL,
                      TII.get(llvm::AMDGPU::V_READLANE_B32), KernSub1)
      .addReg(SVSStorageVGPR)
      .addImm(FPSpillLane);

  // 5. Load app_kernarg_ptr (bytes +0..+7) INTO KernargPtr itself.
  //    S_LOAD reads SBASE before writing SDST, so SBASE == SDST is
  //    well-defined. Wait on the load before the SVA-arg loop (which
  //    reads KernargPtr) and the app prolog run.
  unsigned EncodedOffset0 =
      llvm::AMDGPU::convertSMRDOffsetUnits(ST, /*ByteOffset=*/0);
  (void)llvm::BuildMI(MBB, EntryInstr, DL,
                      TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM), KernargPtr)
      .addReg(KernargPtr)
      .addImm(EncodedOffset0)
      .addImm(/*cpol=*/0);
  (void)llvm::BuildMI(MBB, EntryInstr, DL, TII.get(llvm::AMDGPU::S_WAITCNT))
      .addImm(0);

  return llvm::Error::success();
}

/// Walk the target MF's per-MBB storage intervals from
/// \c SVStorageAndLoadLocations and emit
/// \c currentSVS.emitCodeToSwitchSVS(MI, nextSVS) at every boundary. This
/// makes the SVA actually migrate between storage schemes across the
/// target's control flow — without this, the load plan exists but the
/// runtime state never matches it.
void emitSVSSwitchesForMF(llvm::MachineFunction &MF,
                          const SVStorageAndLoadLocations &SVLocations,
                          const StateValueArraySpecs &Specs,
                          const llvm::SlotIndexes &SI) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   emitSVSSwitchesForMF "
                "MF='"
             << MF.getName() << "' (" << MF.size() << " MBB(s))\n");
  unsigned SwitchesEmitted = 0;
  for (llvm::MachineBasicBlock &MBB : MF) {
    llvm::ArrayRef<StateValueStorageSegment> Segments =
        SVLocations.getStorageIntervals(MBB);
    if (Segments.size() < 2)
      continue;
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                               << llvm::printMBBReference(MBB) << " has "
                               << Segments.size() << " segments\n");
    for (unsigned I = 0, E = Segments.size() - 1; I < E; ++I) {
      const StateValueArrayStorage &Curr = Segments[I].getSVS();
      const StateValueArrayStorage &Next = Segments[I + 1].getSVS();
      if (Curr == Next)
        continue;
      ++SwitchesEmitted;
      // Resolve the boundary slot index → MI inside MBB. The switch
      // must be emitted before the MI at Segments[I+1].begin(); that's
      // where the runtime location of the SVA changes. SlotIndexes maps
      // a slot to its owning MI (or returns null at non-MI slots like
      // block-end), so we use getInstructionFromIndex and fall back to
      // walking forward to the next instruction within MBB.
      llvm::SlotIndex Boundary = Segments[I + 1].begin();
      llvm::MachineInstr *MI = SI.getInstructionFromIndex(Boundary);
      llvm::MachineBasicBlock::iterator InsertPt;
      if (MI && MI->getParent() == &MBB) {
        InsertPt = MI->getIterator();
      } else {
        // No MI is directly anchored at this slot (e.g., the slot
        // corresponds to a block boundary / deleted MI). Fall back to
        // the first terminator so the switch still happens before any
        // control-flow leaves MBB.
        InsertPt = MBB.getFirstTerminator();
        if (InsertPt == MBB.end())
          InsertPt = MBB.end();
      }
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]       "
                                    "emit SVS switch at segment boundary "
                                 << I << " -> " << (I + 1) << "\n");
      Curr.emitCodeToSwitchSVS(InsertPt, Next, Specs);
    }
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   emitted " << SwitchesEmitted
             << " within-MBB SVS switch(es) for MF '" << MF.getName() << "'\n");

  // Cross-MBB SVS join reconciliation. The analysis threads a single
  // \c SVS in program-iteration order, so at CFG joins two predecessors
  // may have different tail-SVSes reaching a common successor. Walk
  // every MBB M, and for each predecessor P of M whose tail SVS != M's
  // head SVS, emit \c TailSVS.emitCodeToSwitchSVS(<anchor>, HeadSVS)
  // to migrate the SVA on the P → M edge. Anchor selection:
  //   * If P has one successor (M), emit at P's terminator — the
  //     switch runs on every P → M traversal because P has no other
  //     successors to disturb.
  //   * Else if M has one predecessor (P), emit at M's first MI —
  //     safe because M is only entered via P.
  //   * Else the edge is critical. Split it by creating a fresh MBB
  //     \c Split under P: rewrite P's terminator to reference \c Split
  //     instead of M (or append an explicit S_BRANCH if M was P's
  //     fall-through), route \c Split to M via an unconditional
  //     S_BRANCH, and emit the SVS switch inside \c Split before its
  //     terminator. This preserves the branch's condition while
  //     interposing the switch on that single edge only.
  //
  // Both loops snapshot their targets first because splitting mutates
  // MBB/CFG state (MF's MBB list, M's predecessor list).
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  llvm::SmallVector<llvm::MachineBasicBlock *, 32> MBBSnapshot;
  for (llvm::MachineBasicBlock &MBB : MF)
    MBBSnapshot.push_back(&MBB);
  unsigned CrossMBBJoins = 0;
  unsigned CriticalEdgeSplits = 0;
  for (llvm::MachineBasicBlock *MBBPtr : MBBSnapshot) {
    llvm::MachineBasicBlock &MBB = *MBBPtr;
    auto Segs = SVLocations.getStorageIntervals(MBB);
    if (Segs.empty())
      continue;
    const StateValueArrayStorage &HeadSVS = Segs.front().getSVS();
    llvm::SmallVector<llvm::MachineBasicBlock *, 4> Preds(
        MBB.predecessors().begin(), MBB.predecessors().end());
    for (llvm::MachineBasicBlock *Pred : Preds) {
      auto PredSegs = SVLocations.getStorageIntervals(*Pred);
      if (PredSegs.empty())
        continue;
      const StateValueArrayStorage &TailSVS = PredSegs.back().getSVS();
      if (TailSVS == HeadSVS)
        continue;
      if (Pred->succ_size() == 1) {
        LLVM_DEBUG(luthier::dbgs()
                   << "[TargetModulePatcherPass]     "
                      "cross-MBB SVS reconcile at pred-terminator "
                   << llvm::printMBBReference(*Pred) << " -> "
                   << llvm::printMBBReference(MBB) << "\n");
        TailSVS.emitCodeToSwitchSVS(Pred->getFirstTerminator(), HeadSVS, Specs);
        ++CrossMBBJoins;
      } else if (MBB.pred_size() == 1) {
        LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                      "cross-MBB SVS reconcile at succ-head "
                                   << llvm::printMBBReference(*Pred) << " -> "
                                   << llvm::printMBBReference(MBB) << "\n");
        TailSVS.emitCodeToSwitchSVS(MBB.begin(), HeadSVS, Specs);
        ++CrossMBBJoins;
      } else {
        // Critical edge: interpose a fresh MBB on the P → M edge.
        auto *Split = MF.CreateMachineBasicBlock();
        MF.insert(MF.end(), Split);
        // Rewrite any explicit reference to M in P's terminators to
        // Split. If M was P's fall-through (no explicit operand),
        // append an unconditional S_BRANCH to Split so control flows
        // into Split whenever P would have fallen through to M.
        bool Rewrote = false;
        for (auto &Term : Pred->terminators()) {
          for (auto &MO : Term.operands()) {
            if (MO.isMBB() && MO.getMBB() == &MBB) {
              MO.setMBB(Split);
              Rewrote = true;
            }
          }
        }
        if (!Rewrote) {
          llvm::BuildMI(*Pred, Pred->end(), llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_BRANCH))
              .addMBB(Split);
        }
        Pred->replaceSuccessor(&MBB, Split);
        Split->addSuccessor(&MBB);
        // Seed Split's liveins with M's expected entry regs plus
        // every storage register the source TailSVS and target
        // HeadSVS touch — the switch code below reads from TailSVS's
        // regs and writes to HeadSVS's regs, so both must be
        // considered live at Split's entry.
        for (const auto &LI : MBB.liveins())
          Split->addLiveIn(LI.PhysReg, LI.LaneMask);
        {
          llvm::SmallVector<llvm::MCRegister, 4> SVSRegs;
          TailSVS.getAllStorageRegisters(SVSRegs);
          HeadSVS.getAllStorageRegisters(SVSRegs);
          for (llvm::MCRegister R : SVSRegs)
            if (!Split->isLiveIn(R))
              Split->addLiveIn(R);
        }
        Split->sortUniqueLiveIns();
        // Emit the SVS switch inside Split, then close with an
        // unconditional branch to M.
        TailSVS.emitCodeToSwitchSVS(Split->end(), HeadSVS, Specs);
        llvm::BuildMI(*Split, Split->end(), llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_BRANCH))
            .addMBB(&MBB);
        LLVM_DEBUG(luthier::dbgs()
                   << "[TargetModulePatcherPass]     "
                      "cross-MBB SVS reconcile via critical-edge split "
                   << llvm::printMBBReference(*Pred) << " -> "
                   << llvm::printMBBReference(*Split) << " -> "
                   << llvm::printMBBReference(MBB) << "\n");
        ++CrossMBBJoins;
        ++CriticalEdgeSplits;
      }
    }
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   emitted " << CrossMBBJoins
             << " cross-MBB SVS reconciliation(s) (" << CriticalEdgeSplits
             << " via critical-edge split) for MF '" << MF.getName() << "'\n");
}

/// Emit the partial-callgraph V0-courier handoff for a target MF:
///
///  1. Fast path — if \c SVLoc reports a single fixed SVS across every
///     target MF, no handoff is needed (the SVA VGPR is unused by all
///     functions and survives across calls). Skip.
///
///  2. Callee side — at the entry of a device function (non-kernel),
///     emit \c EntrySVS.pickOffSVA(firstMI, Specs). \c V0 arrives
///     holding the SVA (from the caller's \c handOffSVA); pickOff
///     stores it into the entry-block SVS and restores \c V0's app
///     value from the SVS's emergency slot.
///
///  3. Caller side — every MBB whose \c PMBB has unresolved edges
///     may cross into an unknown callee. Find the last call or
///     indirect-branch MI in the MBB and emit
///     \c BlockSVS.handOffSVA(MI, Specs) before it. handOff spills
///     \c V0's app value to the SVS's emergency slot and loads the
///     SVA into \c V0 so the callee sees \c V0 == SVA.
void emitPartialCallgraphSVSHandoffWraps(llvm::MachineFunction &MF,
                                         const IPPredicatedCFG &IPCFG,
                                         const SVStorageAndLoadLocations &SVLoc,
                                         const StateValueArraySpecs &Specs) {
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   "
                                "emitPartialCallgraphSVSHandoffWraps MF='"
                             << MF.getName() << "'\n");
  if (SVLoc.hasFixedStorageAcrossAllFunctions()) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "SVS is fixed across all functions; "
                                  "skipping handoff for MF '"
                               << MF.getName() << "'\n");
    return;
  }

  const llvm::Function &F = MF.getFunction();
  // 2. Callee side: pickOffSVA at entry of every device function.
  if (F.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL && !MF.empty()) {
    llvm::MachineBasicBlock &EntryMBB = MF.front();
    if (!EntryMBB.empty()) {
      auto Segs = SVLoc.getStorageIntervals(EntryMBB);
      if (!Segs.empty()) {
        const StateValueArrayStorage &EntrySVS = Segs.front().getSVS();
        LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                      "pickOffSVA at entry of device fn '"
                                   << MF.getName() << "'\n");
        EntrySVS.pickOffSVA(EntryMBB.front(), Specs);
      }
    }
  }

  // 3. Caller side: handOffSVA before the last call/indirect-branch MI
  //    in every MBB whose PMBB has unresolved edges.
  unsigned HandOffsEmitted = 0;
  for (llvm::MachineBasicBlock &MBB : MF) {
    llvm::MachineInstr *TargetMI = nullptr;
    for (auto It = MBB.rbegin(), End = MBB.rend(); It != End; ++It) {
      if (It->isCall() || It->isIndirectBranch()) {
        TargetMI = &*It;
        break;
      }
    }
    if (!TargetMI) {
      LLVM_DEBUG(luthier::dbgs()
                 << "[TargetModulePatcherPass]     "
                    "MBB has no call/indirect-branch "
                 << llvm::printMBBReference(MBB) << "; skipping\n");
      continue;
    }
    auto Segs = SVLoc.getStorageIntervals(MBB);
    assert(!Segs.empty() && "Empty SVStorage Segment");

    const StateValueArrayStorage &BlockSVS = Segs.back().getSVS();
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "handOffSVA before call/indirect-branch in "
                               << llvm::printMBBReference(MBB) << "\n");
    BlockSVS.handOffSVA(*TargetMI, Specs);
    ++HandOffsEmitted;
  }
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   emitted "
                             << HandOffsEmitted << " partial-callgraph SVS "
                             << "handOff(s) for MF '" << MF.getName() << "'\n");
}

/// Scratch registers scavenged at a patchpoint.
///
/// \c Pair is the \c SReg_64 the emitted \c SI_CALL will use as both its
/// return-address destination and its call-target source. \c SCCSave is
/// a 32-bit SGPR reserved for holding \c $scc across the injection when
/// \c $scc is live at the site; it is invalid (default-constructed
/// \c MCRegister) when \c $scc is dead at the site and no save/restore
/// is required. The two are always non-overlapping.
struct ScavengedPatchpointRegs {
  llvm::MCRegister Pair;
  llvm::MCRegister SCCSave;
};

/// Pick an \c SReg_64 pair at \p MI that we can hand to the site's
/// \c S_SWAPPC_B64 as both the return-address destination and the
/// call-target source, and — when \c $scc is live at \p MI — additionally
/// pick a 32-bit SGPR to hold \c $scc across the S_GETPC / S_ADD_U32 /
/// S_ADDC_U32 sequence the site expands to (all three of those defs
/// clobber \c $scc, and the S_ADD_U64 gfx12 path does the same). Every
/// picked register must satisfy three constraints:
///   1. Dead at \p MI — otherwise the swap clobbers a live app value.
///      We seed \c LivePhysRegs from the enclosing MBB's stock live-outs
///      (populated upstream via \c IPPredicatedLiveness / \c IPPredCFG)
///      and step backward to \p MI.
///   2. Not overlapping any SVA-storage register at \p MI's segment —
///      resolved by walking \c SVLocations.getStorageIntervals for the
///      containing MBB and finding the segment that covers \p MI's slot,
///      then unioning that SVS's \c getAllStorageRegisters. Overlap
///      is tested via \c TRI.regsOverlap so paired candidates whose sub-
///      regs alias individually-reserved SGPRs are rejected.
///   3. Not reserved by \c MRI — the \c LivePhysRegs::available check
///      subsumes this.
///
/// \c SCCSave must additionally not overlap the picked \c Pair.
static llvm::Expected<ScavengedPatchpointRegs>
scavengeSGPRsAtSite(const llvm::MachineInstr &MI,
                    const SVStorageAndLoadLocations &SVLocations,
                    const llvm::SlotIndexes &SI) {
  const llvm::MachineFunction &MF = *MI.getMF();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto &TRI = *ST.getRegisterInfo();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  const llvm::MachineBasicBlock &MBB = *MI.getParent();

  // Collect the SVA-owned regs active at this MI's segment. A spilled
  // SVS scheme returns up to three SGPRs (FS_hi, FS_lo, instrumentation-
  // stack-pointer); a VGPR-backed scheme returns one VGPR. We disallow
  // paired candidates that overlap any of them.
  llvm::SmallVector<llvm::MCRegister, 4> SVAReserved;
  {
    const llvm::SlotIndex MISlot = SI.getInstructionIndex(MI);
    for (const StateValueStorageSegment &Seg :
         SVLocations.getStorageIntervals(MBB)) {
      if (Seg.begin() <= MISlot && MISlot < Seg.end()) {
        Seg.getSVS().getAllStorageRegisters(SVAReserved);
        break;
      }
    }
  }

  // Live-at-MI: start from MBB live-outs and walk backward to \p MI.
  llvm::LivePhysRegs Live(TRI);
  Live.addLiveOuts(MBB);
  for (auto It = MBB.rbegin(); It != MBB.rend() && &*It != &MI; ++It)
    Live.stepBackward(*It);

  auto OverlapsSVA = [&](llvm::MCPhysReg R) {
    for (llvm::MCRegister SVAR : SVAReserved)
      if (TRI.regsOverlap(R, SVAR))
        return true;
    return false;
  };

  ScavengedPatchpointRegs Out;
  for (llvm::MCPhysReg Reg : llvm::AMDGPU::SReg_64RegClass) {
    if (OverlapsSVA(Reg))
      continue;
    if (!Live.available(MRI, Reg))
      continue;
    Out.Pair = llvm::MCRegister(Reg);
    break;
  }
  if (!Out.Pair)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "TargetModulePatcherPass: could not scavenge SReg_64 for SI_CALL at "
        "PATCHPOINT in MF '{0}' MBB {1}: no pair is simultaneously dead at "
        "the site and free of SVA-storage overlap.",
        MF.getName(), MBB.getNumber()));

  // If $scc is live across the patchpoint, the S_GETPC + S_ADD_U32 /
  // S_ADDC_U32 sequence (or the S_ADD_U64 on gfx12) will clobber it, so
  // we need a 32-bit SGPR to spill $scc into via S_CSELECT_B32 and
  // restore it via S_CMP_LG_U32.
  if (Live.contains(llvm::AMDGPU::SCC)) {
    for (llvm::MCPhysReg Reg : llvm::AMDGPU::SGPR_32RegClass) {
      if (OverlapsSVA(Reg))
        continue;
      if (TRI.regsOverlap(Reg, Out.Pair))
        continue;
      if (!Live.available(MRI, Reg))
        continue;
      Out.SCCSave = llvm::MCRegister(Reg);
      break;
    }
    if (!Out.SCCSave)
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: $scc is live at the PATCHPOINT in MF "
          "'{0}' MBB {1} but no free SGPR_32 could be scavenged for the "
          "SCC save slot (disjoint from the SReg_64 pair {2} and any "
          "SVA-storage regs at the site).",
          MF.getName(), MBB.getNumber(),
          llvm::printReg(Out.Pair, &TRI)));
  }
  return Out;
}

/// Replace \p PatchpointMI with an outlined-payload call sequence:
///   S_GETPC_B64 $pair
///   S_ADD_U32  $pair.sub0, $pair.sub0, @callee@rel32@lo
///   S_ADDC_U32 $pair.sub1, $pair.sub1, @callee@rel32@hi
///   SI_CALL    $pair, $pair, @callee
/// This mirrors the direct-call sequence stock \c SITargetLowering emits.
///
/// The pair MUST already be verified dead at \p PatchpointMI and non-
/// overlapping with any SVA-storage reg (see \c scavengeSGPRPairAtSite).
/// A per-call-site continuation \c MCSymbol is attached as the post-instr
/// symbol of the emitted \c SI_CALL and returned to the caller —
/// \c rewritePayloadReturn uses it to materialize the return address
/// via \c S_GETPC when the payload clobbers the scavenged pair.
///
/// Both the \c PATCHPOINT MI and \p ExternHandle (the target-
/// module extern declaration the marker references) are erased here as
/// a pair.
static llvm::MCSymbol *emitSICallAtPatchpoint(llvm::MachineInstr &PatchpointMI,
                                              llvm::Function &PayloadFn,
                                              llvm::Function &ExternHandle,
                                              llvm::MCRegister ScavengedPair,
                                              llvm::MCRegister SCCSaveSGPR) {
  assert(PatchpointMI.getOpcode() == llvm::TargetOpcode::PATCHPOINT &&
         "emitSICallAtPatchpoint expects a PATCHPOINT MI");
  auto &MBB = *PatchpointMI.getParent();
  auto &MF = *MBB.getParent();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto *TII = ST.getInstrInfo();
  const auto *TRI = ST.getRegisterInfo();
  const llvm::DebugLoc DL;

  // If $scc is live across the patchpoint, snapshot it into SCCSaveSGPR
  // before the S_ADD(C) sequence clobbers it. S_CSELECT_B32 reads $scc
  // but does not modify it, so this is a pure spill:
  //   SCCSaveSGPR = ($scc ? 1 : 0)
  if (SCCSaveSGPR) {
    (void)llvm::BuildMI(MBB, PatchpointMI, DL,
                        TII->get(llvm::AMDGPU::S_CSELECT_B32), SCCSaveSGPR)
        .addImm(1)
        .addImm(0);
  }

  // Materialize the callee address into ScavengedPair. Emitted as
  // top-level MIs (not a bundle) so any post-instr symbols added later
  // are visible to AsmPrinter's per-instruction emission loop, and to
  // match the canonical unbundled pattern used by upstream
  // SIInstrInfo::insertIndirectBranch.
  llvm::BuildMI(MBB, PatchpointMI, DL, TII->get(llvm::AMDGPU::S_GETPC_B64),
                ScavengedPair);
  if (ST.has64BitLiterals()) {
    // +4 compensates for S_GETPC_B64 returning PC-of-next-instr
    (void)llvm::BuildMI(MBB, PatchpointMI, DL,
                        TII->get(llvm::AMDGPU::S_ADD_U64), ScavengedPair)
        .addReg(ScavengedPair)
        .addGlobalAddress(&PayloadFn, /*Offset=*/4,
                          llvm::SIInstrInfo::MO_REL32);
  } else {
    llvm::MCRegister Sub0 = TRI->getSubReg(ScavengedPair, llvm::AMDGPU::sub0);
    llvm::MCRegister Sub1 = TRI->getSubReg(ScavengedPair, llvm::AMDGPU::sub1);
    // On hardware where S_GETPC_B64 zero-extends the 48-bit PC instead of
    // sign-extending it, the high half comes back with bits [63:48] cleared.
    // S_SEXT_I32_I16 re-derives them from bit 47 (i.e. bit 15 of the high
    // word). It is a plain SOP1 that does not touch $scc, so it is safe
    // inside the SCCSaveSGPR-protected region. Adding it also pushes the
    // S_ADD_U32 4 bytes further from the S_GETPC, which both REL32 addends
    // below have to absorb.
    int64_t Adjust = 0;
    if (ST.hasGetPCZeroExtension()) {
      (void)llvm::BuildMI(MBB, PatchpointMI, DL,
                          TII->get(llvm::AMDGPU::S_SEXT_I32_I16), Sub1)
          .addReg(Sub1);
      Adjust = 4;
    }
    // S_GETPC_B64 returns the address of the instruction after it, but each
    // REL32 relocation is resolved against the address of its own literal
    // operand: 4 bytes into the S_ADD_U32, and 12 bytes into the S_ADD_U32
    // for the S_ADDC_U32's literal. Both halves must encode the *same*
    // 64-bit delta, so the lo/hi addends differ. These are the same
    // constants upstream SIInstrInfo::expandPostRAPseudo applies when it
    // lowers SI_PC_ADD_REL_OFFSET.
    (void)llvm::BuildMI(MBB, PatchpointMI, DL,
                        TII->get(llvm::AMDGPU::S_ADD_U32), Sub0)
        .addReg(Sub0)
        .addGlobalAddress(&PayloadFn, /*Offset=*/Adjust + 4,
                          llvm::SIInstrInfo::MO_REL32);
    (void)llvm::BuildMI(MBB, PatchpointMI, DL,
                        TII->get(llvm::AMDGPU::S_ADDC_U32), Sub1)
        .addReg(Sub1)
        .addGlobalAddress(&PayloadFn, /*Offset=*/Adjust + 12,
                          llvm::SIInstrInfo::MO_REL32 + 1);
  }

  // Restore $scc from SCCSaveSGPR before the SI_CALL.
  if (SCCSaveSGPR) {
    (void)llvm::BuildMI(MBB, PatchpointMI, DL,
                        TII->get(llvm::AMDGPU::S_CMP_LG_U32))
        .addReg(SCCSaveSGPR)
        .addImm(0);
  }

  // SI_CALL — same pair as dst and src, so post-swap the pair holds
  // the return address for the payload's S_SETPC_B64_return.
  auto CallMI = llvm::BuildMI(MBB, PatchpointMI, DL,
                              TII->get(llvm::AMDGPU::SI_CALL), ScavengedPair)
                    .addReg(ScavengedPair)
                    .addGlobalAddress(&PayloadFn);

  // Mark the caller's frame as containing calls now that we've inserted a
  // SI_CALL. Without this, AMDGPUResourceUsageAnalysis takes its no-calls
  // early-return path
  MF.getFrameInfo().setHasCalls(true);

  // Continuation symbol pinned at the point control returns to. Used by
  // rewritePayloadReturn's Case-B trampoline when the payload has
  // clobbered ScavengedPair before its return terminator.  Must be a
  // *named* temp symbol: the Case-B trampoline references it via
  // \c ContSym@rel32@lo / \c ContSym@rel32@hi (see \c rewritePayloadReturn)
  // and MC only emits an R_AMDGPU_REL32_LO/HI relocation when the target
  // symbol has an ELF entry — anonymous temp symbols
  // (\c createTempSymbol under \c UseNamesOnTempLabels=false) get folded
  // away with no name, so the reloc points at \c "" and the linker
  // rejects the object.
  llvm::MCSymbol *ContSym = MF.getContext().createNamedTempSymbol(
      "luthier_call_ret");
  CallMI->setPostInstrSymbol(MF, ContSym);

  // Drop the MI-level use first, then rewire the surviving IR-level
  // ones. Erasing the PATCHPOINT MI accounts for the MachineOperand
  // reference; the extern handle can still show up as an IR use because
  // TraceFunctionTranslator (\c TraceFunctionTranslator.cpp:2165 —
  // PATCHPOINT case) lifted every marker into a
  // \c call @luthier.patchpoint.N(ptr @ExternHandle, ...) inside the
  // target-module Function's trace IR body, and those trace calls are
  // real \c llvm::Use s that outlive the PATCHPOINT MI. RAUW them to
  // \p PayloadFn: types match (both the extern and the definition are
  // \c void()) and \c movePayloadMFIntoTarget is about to install
  // \p PayloadFn in the target module under this same name, so the
  // trace IR ends up pointing at the payload's real definition. That
  // also matters for the move itself — leaving the extern in place
  // would collide with \p PayloadFn 's name on the module and LLVM
  // would auto-rename it, silently breaking the
  // \c getGlobalAddress(&PayloadFn) references we just baked into the
  // \c SI_CALL sequence above.
  PatchpointMI.eraseFromParent();
  if (!ExternHandle.use_empty())
    ExternHandle.replaceAllUsesWith(&PayloadFn);
  ExternHandle.eraseFromParent();
  return ContSym;
}

/// Move the payload's IR \c Function from \p IModule into \p TargetModule
/// and steal its cached \c MachineFunctionAnalysis::Result from \p IFAM
/// into \p TargetFAM without deep-cloning the underlying
/// \c MachineFunction.
///
/// The steal is done via the \c FAMResults / \c FAMResultLists ADL tags
/// declared above. \c std::list::splice is used so the \c unique_ptr owning the
/// \c MachineFunction migrates atomically between the two FAMs' storage.
///
/// Fails if \p IFAM has no cached \c MachineFunctionAnalysis result for
/// \p PayloadFn.
llvm::Error movePayloadMFIntoTarget(llvm::Function &PayloadFn,
                                    llvm::Module &TargetModule,
                                    llvm::FunctionAnalysisManager &IFAM,
                                    llvm::FunctionAnalysisManager &TargetFAM,
                                    const llvm::ValueToValueMapTy &VMap) {
  llvm::Module &IModule = *PayloadFn.getParent();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   movePayloadMFIntoTarget '"
             << PayloadFn.getName() << "'\n");

  // 1. Detach the IR Function from IModule and attach to TargetModule.
  // Identity is preserved — existing references remain valid.
  IModule.getFunctionList().remove(PayloadFn.getIterator());
  TargetModule.getFunctionList().push_back(&PayloadFn);

  // Remap any \c MO_GlobalAddress operand in the payload's MIR that still
  // names an IModule-side stand-in. \c moveIModuleIntoTarget's Pass 1 did
  // an IR-level \c replaceAllUsesWith when it merged the placeholder
  // declaration with the target-module definition and populated \p VMap
  // with the survivor, but that only fixes IR \c Use edges — MI operands
  // reference their \c GlobalValue directly and are invisible to IR RAUW.
  // Left un-remapped they turn into dangling pointers the moment Pass 1
  // erases the placeholder in \c DeclsToErase, and the assembly printer
  // crashes in \c Mangler::getNameWithPrefix on the freed GV. The full
  // MF-clone path already does this in \c cloneMFInto (\c Cloning.cpp:390);
  // the splice path here has to open-code it because there's no clone
  // pass to catch it.
  if (auto *MFRes =
          IFAM.getCachedResult<llvm::MachineFunctionAnalysis>(PayloadFn)) {
    for (llvm::MachineBasicBlock &MBB : MFRes->getMF()) {
      for (llvm::MachineInstr &MI : MBB.instrs()) {
        for (llvm::MachineOperand &MO : MI.operands()) {
          if (!MO.isGlobal())
            continue;
          auto It = VMap.find(MO.getGlobal());
          if (It == VMap.end())
            continue;
          auto *NewGV = llvm::cast<llvm::GlobalValue>(It->second);
          if (NewGV == MO.getGlobal())
            continue;
          MO.ChangeToGA(NewGV, MO.getOffset(), MO.getTargetFlags());
        }
      }
    }
  }

  // 2. Splice the MFAnalysis result entry between the two FAMs.
  llvm::AnalysisKey *const ID = llvm::MachineFunctionAnalysis::ID();
  auto &IResults = IFAM.*get(FAMResultsTag{});
  auto &ILists = IFAM.*get(FAMResultListsTag{});
  auto &TResults = TargetFAM.*get(FAMResultsTag{});
  auto &TLists = TargetFAM.*get(FAMResultListsTag{});

  auto ResIt = IResults.find({ID, &PayloadFn});
  if (ResIt == IResults.end())
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "TargetModulePatcherPass: payload function '{0}' has no cached "
        "MachineFunctionAnalysis result in the IModule's FAM — the "
        "ISel/machine-passes stage must have failed to lower it.",
        PayloadFn.getName()));

  FAMResultListT::iterator ListIt = ResIt->second;
  FAMResultListT &ISlot = ILists[&PayloadFn];
  FAMResultListT &TSlot = TLists[&PayloadFn];

  // std::list::splice(pos, other, it) moves a single node from `other`
  // to `*this` before `pos`. The node's iterator (`ListIt`) stays valid
  // and now refers to the moved element in TSlot.
  TSlot.splice(TSlot.end(), ISlot, ListIt);

  // 3. Update the index maps. Erase from IFAM, insert into TargetFAM
  // with the now-migrated iterator.
  IResults.erase(ResIt);
  TResults[{ID, &PayloadFn}] = ListIt;

  // 4. Clean up the IModule's per-function list slot if it became empty
  // (mirrors AnalysisManager::clear behavior).
  if (ISlot.empty())
    ILists.erase(&PayloadFn);

  return llvm::Error::success();
}

/// Rewrite every return terminator in \p PayloadMF so control lands back
/// at the caller's SI_CALL continuation \p ContSym via \p ScavengedPair.
///
/// Our SI_CALL emits into \p ScavengedPair via
/// \c S_SWAPPC_B64, so post-swap the pair holds the return address the
/// caller expects the payload to jump through. Two cases:
///
///   * **Case A** — no MI in the payload writes any sub-reg overlapping
///     \p ScavengedPair. The pair still holds the return address at every
///     return site, so we simply repoint each \c S_SETPC_B64_return's
///     operand to \p ScavengedPair.
///
///   * **Case B** — some MI clobbers \p ScavengedPair between entry and
///     the return. The pair's contents are stale, so we re-materialize
///     \p ContSym's address into it at every return site via the same
///     \c S_GETPC_B64 + \c S_ADD_U32 / \c S_ADDC_U32 pattern
llvm::Error rewritePayloadReturn(llvm::MachineFunction &PayloadMF,
                                 llvm::MCRegister ScavengedPair,
                                 llvm::MCSymbol *ContSym,
                                 bool PreserveSCCInCaseB) {
  const auto &ST = PayloadMF.getSubtarget<llvm::GCNSubtarget>();
  const auto *TII = ST.getInstrInfo();
  const auto *TRI = ST.getRegisterInfo();
  const auto &MRI = PayloadMF.getRegInfo();
  auto &MCCtx = PayloadMF.getContext();
  const llvm::MCRegister Sub0 =
      TRI->getSubReg(ScavengedPair, llvm::AMDGPU::sub0);
  const llvm::MCRegister Sub1 =
      TRI->getSubReg(ScavengedPair, llvm::AMDGPU::sub1);

  // Detect whether ANY MI in the payload defines any sub-reg overlapping
  // ScavengedPair. Return terminators themselves are uses; their operand
  // is not a def, so scanning all_defs is safe against false positives.
  bool ScavClobbered = false;
  for (const llvm::MachineBasicBlock &MBB : PayloadMF) {
    for (const llvm::MachineInstr &MI : MBB) {
      for (const llvm::MachineOperand &MO : MI.all_defs()) {
        if (!MO.isReg())
          continue;
        llvm::Register R = MO.getReg();
        if (!R.isPhysical())
          continue;
        if (TRI->regsOverlap(R, ScavengedPair)) {
          ScavClobbered = true;
          break;
        }
      }
      if (ScavClobbered)
        break;
    }
    if (ScavClobbered)
      break;
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   rewritePayloadReturn '"
             << PayloadMF.getName()
             << "' ScavengedPair=" << llvm::printReg(ScavengedPair, TRI)
             << " clobbered=" << ScavClobbered << "\n");

  // Collect return terminators. Snapshot before mutating so we don't
  // invalidate the MBB iterator while inserting into the same MBB.
  llvm::SmallVector<llvm::MachineInstr *, 4> Returns;
  for (llvm::MachineBasicBlock &MBB : PayloadMF)
    for (llvm::MachineInstr &MI : MBB)
      if (MI.isReturn())
        Returns.push_back(&MI);

  for (llvm::MachineInstr *RetMI : Returns) {
    if (RetMI->getOpcode() != llvm::AMDGPU::S_SETPC_B64_return)
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: unexpected return opcode {0} in "
          "payload '{1}' — expected S_SETPC_B64_return post-PEI.",
          RetMI->getOpcode(), PayloadMF.getName()));
    auto &MBB = *RetMI->getParent();
    const llvm::DebugLoc DL;

    if (!ScavClobbered) {
      // Case A: repoint the setpc's use to ScavengedPair.
      llvm::MachineOperand &RAOp = RetMI->getOperand(0);
      RAOp.setReg(ScavengedPair);
      RAOp.setIsUndef(false);
      continue;
    }

    // Case B: rematerialize ContSym's address into ScavengedPair.
    // S_GETPC + S_ADD(C) are emitted as top-level MIs, not bundled: the
    // post-instr symbol on the S_GETPC has to be a definition AsmPrinter
    // actually emits, and MachineInstrBundleIterator hides bundle-interior
    // MIs from the emitFunctionBody loop (AsmPrinter.cpp), so a label on a
    // bundled MI is silently dropped and the (ContSym - PostGetPCLabel)
    // fixup fails MC's relocatable-expression check. Upstream
    // SIInstrInfo::insertIndirectBranch emits this same triple unbundled.
    const bool Has64BitLiterals =
        PayloadMF.getSubtarget<llvm::GCNSubtarget>().has64BitLiterals();

    // If the caller had $scc live across the patchpoint,
    // InjectedPayloadPreserveLiveRegsPass has already emitted a
    //   $scc = COPY vregN
    // right before RetMI to restore the caller's SCC value into $scc.
    // Our S_ADD_U32 / S_ADDC_U32 (or S_ADD_U64) below will then clobber
    // that just-restored SCC, so we need to spill $scc into a scratch
    // SGPR right before the trampoline and re-prime it right before
    // RetMI. Scavenge a dead SGPR at the return point: nothing except
    // the payload's live-out set is live here, and the pair itself is
    // being fully redefined by S_GETPC.
    llvm::MCRegister SCCTrampSave;
    if (PreserveSCCInCaseB) {
      // We step backward THROUGH RetMI so \c Live picks up the
      // terminator's implicit uses. InjectedPayloadPreserveLiveRegsPass
      // attaches an implicit-use of every payload-preserved physreg to
      // the return terminator (see
      // InjectedPayloadPreserveLiveRegsPass.cpp:266) so RA sees them
      // as live-out; without stepping into RetMI here we would miss
      // exactly those regs and could clobber a caller-visible value.
      llvm::LivePhysRegs Live(*TRI);
      Live.addLiveOuts(MBB);
      for (auto It = MBB.rbegin(); It != MBB.rend(); ++It) {
        Live.stepBackward(*It);
        if (&*It == RetMI)
          break;
      }
      for (llvm::MCPhysReg Reg : llvm::AMDGPU::SGPR_32RegClass) {
        if (TRI->regsOverlap(Reg, ScavengedPair))
          continue;
        if (!Live.available(MRI, Reg))
          continue;
        SCCTrampSave = llvm::MCRegister(Reg);
        break;
      }
      if (!SCCTrampSave)
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "TargetModulePatcherPass: could not scavenge a free SGPR_32 in "
            "payload '{0}' MBB {1} to preserve $scc across the Case B "
            "return trampoline.",
            PayloadMF.getName(), MBB.getNumber()));
      // Spill $scc into SCCTrampSave: SCCTrampSave = ($scc ? 1 : 0).
      // S_CSELECT_B32 reads $scc without modifying it.
      (void)llvm::BuildMI(MBB, RetMI, DL,
                          TII->get(llvm::AMDGPU::S_CSELECT_B32), SCCTrampSave)
          .addImm(1)
          .addImm(0);
    }

    // Emit the same PC-relative sequence \c SI_PC_ADD_REL_OFFSET would
    // expand into for a direct branch to \p ContSym. Handing the offset
    // to MC as a symbol-difference wrapped in AND/ASHR (the previous
    // shape) fails \c MCExpr::evaluateAsRelocatable — the AMDGPU reloc
    // set has no "low/high 32 bits of a symbol difference" primitive,
    // and MC never gets to the point of folding the inner
    // \c (ContSym - PostGetPCLabel) to a constant. Using
    // \c ContSym@rel32@lo / \c ContSym@rel32@hi with the same +4/+12
    // addends the call-site half of this pass uses (see
    // \c emitSICallAtPatchpoint above) lowers to plain
    // \c R_AMDGPU_REL32_LO / \c R_AMDGPU_REL32_HI relocations MC knows
    // how to encode, and the linker resolves them against \p ContSym 's
    // in-section address for free.
    (void)llvm::BuildMI(MBB, RetMI, DL, TII->get(llvm::AMDGPU::S_GETPC_B64),
                        ScavengedPair);
    int64_t Adjust = 0;
    if (ST.hasGetPCZeroExtension()) {
      // Sign-extend the high half — same rationale as in the call-site
      // sequence: hardware whose \c S_GETPC_B64 zero-extends the 48-bit
      // PC needs the high word re-derived from bit 47 before the
      // \c REL32_HI add.  Pushes the \c S_ADD_U32 4 bytes further from
      // the \c S_GETPC, which both REL32 addends below absorb.
      (void)llvm::BuildMI(MBB, RetMI, DL,
                          TII->get(llvm::AMDGPU::S_SEXT_I32_I16), Sub1)
          .addReg(Sub1);
      Adjust = 4;
    }
    // Wrapping \p ContSym in an \c MCSymbolRefExpr with the
    // \c AMDGPUMCExpr::S_REL32_LO / \c S_REL32_HI specifier reaches
    // \c AMDGPUELFObjectWriter as \c R_AMDGPU_REL32_LO / \c R_AMDGPU_REL32_HI
    // (see \c AMDGPUELFObjectWriter.cpp:57-60); the +4/+12 addend is the
    // standard offset-from-S_GETPC-return-address to each add's literal
    // operand, matching \c SIInstrInfo::expandPostRAPseudo 's
    // \c SI_PC_ADD_REL_OFFSET expansion.
    auto EquateOffset = [&](llvm::StringRef Prefix,
                            llvm::AMDGPUMCExpr::Specifier Spec,
                            int64_t Addend) {
      llvm::MCSymbol *Sym = MCCtx.createTempSymbol(Prefix,
                                                   /*AlwaysAddSuffix=*/true);
      Sym->setVariableValue(llvm::MCBinaryExpr::createAdd(
          llvm::MCSymbolRefExpr::create(ContSym, Spec, MCCtx),
          llvm::MCConstantExpr::create(Addend, MCCtx), MCCtx));
      return Sym;
    };
    llvm::MCSymbol *OffsetLo = EquateOffset(
        "luthier_payload_ret_lo", llvm::AMDGPUMCExpr::S_REL32_LO, Adjust + 4);
    llvm::MCSymbol *OffsetHi = EquateOffset(
        "luthier_payload_ret_hi", llvm::AMDGPUMCExpr::S_REL32_HI, Adjust + 12);
    (void)llvm::BuildMI(MBB, RetMI, DL, TII->get(llvm::AMDGPU::S_ADD_U32))
        .addReg(Sub0, llvm::RegState::Define)
        .addReg(Sub0)
        .addSym(OffsetLo, llvm::SIInstrInfo::MO_FAR_BRANCH_OFFSET);
    (void)llvm::BuildMI(MBB, RetMI, DL, TII->get(llvm::AMDGPU::S_ADDC_U32))
        .addReg(Sub1, llvm::RegState::Define)
        .addReg(Sub1)
        .addSym(OffsetHi, llvm::SIInstrInfo::MO_FAR_BRANCH_OFFSET);
    (void)Has64BitLiterals;

    // Restore $scc right before the return terminator, after the
    // trampoline's S_ADD(C) has finished clobbering it. S_CMP_LG_U32
    // sets $scc = (SCCTrampSave != 0), the exact round-trip of the
    // S_CSELECT_B32 save above. S_SETPC_B64_return does not read $scc,
    // so this becomes the live value the caller sees at ContSym.
    if (SCCTrampSave) {
      (void)llvm::BuildMI(MBB, RetMI, DL,
                          TII->get(llvm::AMDGPU::S_CMP_LG_U32))
          .addReg(SCCTrampSave)
          .addImm(0);
    }

    llvm::MachineOperand &RAOp = RetMI->getOperand(0);
    RAOp.setReg(ScavengedPair);
    RAOp.setIsUndef(false);
  }
  return llvm::Error::success();
}

/// Map a \c ScalarValueArgument to the AMDGPU
/// \c PreloadedValue that supplies its bits from the HSA kernarg preload.
/// Returns \c std::nullopt for SVA entries that have no kernarg source
/// (e.g., USER_ARG_PTR / IMPLICIT_ARG_BUFFER, which are filled in
/// elsewhere). Used by the initial-entry-kernel setup to find the source
/// SGPR for each requested kernarg spill.
static std::optional<llvm::AMDGPUFunctionArgInfo::PreloadedValue>
preloadedValueForSVA(ScalarValueArgument SA) {
  switch (SA) {
  case WAVEFRONT_PRIVATE_SEGMENT_BUFFER:
    return llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER;
  case KERNEL_ARG_PTR:
    return llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR;
  case DISPATCH_ID:
    return llvm::AMDGPUFunctionArgInfo::DISPATCH_ID;
  case FLAT_SCRATCH:
    return llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT;
  case DISPATCH_PTR:
    return llvm::AMDGPUFunctionArgInfo::DISPATCH_PTR;
  case QUEUE_PTR:
    return llvm::AMDGPUFunctionArgInfo::QUEUE_PTR;
  case WORK_ITEM_PRIVATE_SEGMENT_SIZE:
    return llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_SIZE;
  case WORKGROUP_ID_X:
    return llvm::AMDGPUFunctionArgInfo::WORKGROUP_ID_X;
  case WORKGROUP_ID_Y:
    return llvm::AMDGPUFunctionArgInfo::WORKGROUP_ID_Y;
  case WORKGROUP_ID_Z:
    return llvm::AMDGPUFunctionArgInfo::WORKGROUP_ID_Z;
  case WORKITEM_ID_X:
    return llvm::AMDGPUFunctionArgInfo::WORKITEM_ID_X;
  case WORKITEM_ID_Y:
    return llvm::AMDGPUFunctionArgInfo::WORKITEM_ID_Y;
  case WORKITEM_ID_Z:
    return llvm::AMDGPUFunctionArgInfo::WORKITEM_ID_Z;
  case IMPLICIT_ARG_BUFFER:
    return std::nullopt;
  }
  static_assert(SCALAR_VALUE_ARGUMENT_LAST == WORKITEM_ID_Z,
                "add case to preloadedValueForSVA");
  return std::nullopt;
}

/// Per-target-kernel preamble info computed inline from IPIP, SVLocations,
/// and each payload MF's frame info. Replaces the fields of
/// \c FunctionPreambleDescriptor::KernelPreambleSpecs that the patcher
/// actually reads: the "any attached payload needs scratch" flag and the
/// union of ScalarValueArguments the attached payloads request.
struct SVAScratchSetupInfo {
  bool RequiresScratchAndStackSetup{false};
  /// Set if any attached payload MF has var-sized stack objects
  /// (\c MachineFrameInfo::hasVarSizedObjects). Selects the
  /// dynamic-payload branch of the SP setup logic.
  bool AnyPayloadUsesDynamicStack{false};
  /// Maximum static frame size across all attached payload MFs
  /// (\c MachineFrameInfo::getStackSize). Used as the reservation for
  /// the dynamic-app + static-payload case.
  unsigned PayloadMaxFixedStackSize{0};
  llvm::SmallDenseSet<ScalarValueArgument, 8> RequestedKernelArguments{};
  llvm::SmallDenseSet<amdgpu::hsamd::ValueKind, 32>
      ImplicitArgsExplicitlyRequested{};

  [[nodiscard]] bool usesSVA() const {
    return RequiresScratchAndStackSetup || !RequestedKernelArguments.empty();
  }
};

/// Fold every injected-payload function targeting \p KernelMF into a
/// single \c SVAScratchSetupInfo describing what the kernel's prologue must
/// set up: scratch/stack enablement (per \c InjectedPayloadPEIPass's
/// rule) and the union of \c ScalarValueArgument sources the payloads
/// request. Payloads targeting a different MF are skipped.
///
/// The target module only ever carries one initial-entry-point kernel
/// (attributed with \c InitialEntryPointAttr by \c CodeDiscoveryPass),
/// and prologue emission is only meaningful there — device-function
/// target MFs never get a kernel prologue. Callers pass that single
/// kernel MF in and receive the prologue spec for it.
SVAScratchSetupInfo
computeInitialEntryKernelSVAInfo(const llvm::MachineFunction &KernelMF,
                                 llvm::Module &IModule,
                                 llvm::FunctionAnalysisManager &IFAM,
                                 const InjectedPayloadAndInstPoint &IPIP,
                                 const SVStorageAndLoadLocations &SVLocations,
                                 const IPPredicatedLiveness &IPLiveness) {
  SVAScratchSetupInfo Info;

  // Partial-callgraph fallback. When the prototype callgraph isn't
  // fully recovered, some payloads/trace functions may be reached via
  // unresolved edges and their side-effects aren't visible in the
  // per-payload scan below. Conservatively force every implicit-arg
  // pointer on: request IMPLICIT_ARG_BUFFER (drives the kernarg
  // buffer expansion) and mark every impl-arg opt-out attr for
  // clearing on the instrumented kernel, so the AMDGPUAttributor
  // can't drop any of them.
  if (!IPLiveness.isFullyDiscovered()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]     "
                  "IPLiveness not fully discovered; forcing all "
                  "implicit args on for kernel '"
               << KernelMF.getName() << "'\n");
    Info.RequestedKernelArguments.insert(IMPLICIT_ARG_BUFFER);
    for (auto Attr : {amdgpu::hsamd::ValueKind::HiddenHostcallBuffer,
                      amdgpu::hsamd::ValueKind::HiddenHeapV1,
                      amdgpu::hsamd::ValueKind::HiddenMultiGridSyncArg,
                      amdgpu::hsamd::ValueKind::HiddenDefaultQueue,
                      amdgpu::hsamd::ValueKind::HiddenCompletionAction,
                      amdgpu::hsamd::ValueKind::HiddenQueuePtr,
                      amdgpu::hsamd::ValueKind::HiddenPrintfBuffer})
      Info.ImplicitArgsExplicitlyRequested.insert(Attr);
  }

  for (llvm::Function &PayloadFn : IModule) {
    if (!PayloadFn.hasFnAttribute(InjectedPayloadAttribute))
      continue;
    if (!IPIP.contains(PayloadFn))
      continue;
    const llvm::MachineInstr *AppMI = IPIP.at(PayloadFn);
    if (AppMI->getMF() != &KernelMF)
      continue;

    // Mirror InjectedPayloadPEIPass's RequiresScratchAndStackSetup rule:
    //   * SVA storage for this IP is spilled (load VGPR unavailable), OR
    //   * the payload MF itself has stack objects or calls.
    if (const auto *LoadPlan =
            SVLocations.getStateValueArrayLoadPlanForInstPoint(*AppMI))
      if (!LoadPlan->StateValueArrayLoadVGPR)
        Info.RequiresScratchAndStackSetup = true;
    if (auto *MFRes =
            IFAM.getCachedResult<llvm::MachineFunctionAnalysis>(PayloadFn)) {
      const llvm::MachineFrameInfo &MFI = MFRes->getMF().getFrameInfo();
      if (MFI.hasStackObjects() || MFI.hasCalls())
        Info.RequiresScratchAndStackSetup = true;
      if (MFI.hasVarSizedObjects())
        Info.AnyPayloadUsesDynamicStack = true;
      Info.PayloadMaxFixedStackSize =
          std::max<unsigned>(Info.PayloadMaxFixedStackSize,
                             static_cast<unsigned>(MFI.getStackSize()));
    }

    // Union the payload's requested SVA scalar-value arguments and
    // its transitively-used implicit args into the kernel's entry
    // set.
    const InjectedPayloadSideEffects &SE =
        IFAM.getResult<InjectedPayloadSideEffectsAnalysis>(PayloadFn);
    for (ScalarValueArgument SA : SE.svas())
      Info.RequestedKernelArguments.insert(SA);
    for (amdgpu::hsamd::ValueKind Attr : SE.implicit_args())
      Info.ImplicitArgsExplicitlyRequested.insert(Attr);
  }
  // The scratch/stack setup path in \c emitCodeToSetupScratch consumes
  // PSB and FLAT_SCRATCH_INIT — one of them per branch on the target's
  // FS mode — so record those SVs alongside the payload's own reads.
  // This lets \c forceEnableRequestedUserSGPRPreloads make ONE pass over
  // the SV set and force-enable every user-SGPR preload the downstream
  // emit steps need, rather than each emit step force-enabling its own.
  if (Info.RequiresScratchAndStackSetup) {
    const auto &ST = KernelMF.getSubtarget<llvm::GCNSubtarget>();
    if (!ST.hasArchitectedFlatScratch()) {
      Info.RequestedKernelArguments.insert(FLAT_SCRATCH);
      if (!ST.enableFlatScratch())
        Info.RequestedKernelArguments.insert(WAVEFRONT_PRIVATE_SEGMENT_BUFFER);
    }
  }
  return Info;
}

/// Emit the initial-entry kernel's SVA preamble at its first instruction:
/// per-wave scratch/stack setup (when requested) plus an
/// \c emitCodeToStoreSGPRKernelArg for each \c ScalarValueArgument in
/// \c KernelInfo.RequestedKernelArguments. The SVA storage register for
/// the entry MBB comes from \c SVLocations.
///
/// Called once per pass — only the target module's initial-entry-point
/// kernel gets a prologue; other target functions (device functions the
/// kernel calls) never need this setup, so the caller only invokes this
/// when the initial entry point is a kernel.
llvm::Error
emitInitialEntryKernelSetup(llvm::MachineFunction &KernelMF,
                            const SVAScratchSetupInfo &KernelInfo,
                            const SVStorageAndLoadLocations &SVLocations,
                            const StateValueArraySpecs &Specs) {
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] "
                                "emitInitialEntryKernelSetup for kernel '"
                             << KernelMF.getName() << "'\n");
  if (!KernelInfo.usesSVA()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   kernel '" << KernelMF.getName()
               << "' does not use SVA, skipping\n");
    return llvm::Error::success();
  }
  if (KernelMF.empty())
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("TargetModulePatcherPass: kernel '{0}' has no MBBs; "
                      "cannot insert SVA setup",
                      KernelMF.getName()));
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   kernel '" << KernelMF.getName()
             << "' uses SVA; setup begin\n");
  llvm::MachineBasicBlock &EntryMBB = KernelMF.front();
  if (EntryMBB.empty())
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "TargetModulePatcherPass: kernel '{0}' has an empty entry MBB; "
        "cannot insert SVA setup",
        KernelMF.getName()));
  llvm::MachineInstr &EntryInstr = EntryMBB.front();

  llvm::ArrayRef<StateValueStorageSegment> EntrySegments =
      SVLocations.getStorageIntervals(EntryMBB);
  if (EntrySegments.empty())
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "TargetModulePatcherPass: kernel '{0}' has no SVA storage "
        "segment at entry; SV-load-locations analysis is inconsistent",
        KernelMF.getName()));
  const StateValueArrayStorage &EntrySVS = EntrySegments.front().getSVS();
  llvm::MCRegister EntrySVSStorageReg = EntrySVS.getStateValueStorageReg();
  const auto *TRI0 = KernelMF.getSubtarget().getRegisterInfo();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]     entry SVS storage reg="
             << llvm::printReg(EntrySVSStorageReg, TRI0) << "\n");

  // Setup VGPR: SVA construction always happens in a VGPR (WRITELANE /
  // READLANE are how args flow in/out). Prefer the SVS analysis's
  // chosen storage when it's already a VGPR — the setup writes land
  // directly into the final storage and the final-move step below is
  // a no-op. If the analysis picked a spilled scheme (storage reg 0)
  // or an AGPR (non-\c VGPR_32RegClass reg), fall back to \c VGPR3
  // and, at the tail, emit a move into the permanent storage via
  // \c EntrySVS.emitCodeToStoreSVA. V3 is dead at kernel entry across
  // every supported target.
  llvm::MCRegister SVSStorageReg;
  bool SetupIsFinalStorage;
  if (EntrySVSStorageReg &&
      llvm::AMDGPU::VGPR_32RegClass.contains(EntrySVSStorageReg)) {
    SVSStorageReg = EntrySVSStorageReg;
    SetupIsFinalStorage = true;
  } else {
    SVSStorageReg = llvm::AMDGPU::VGPR3;
    SetupIsFinalStorage = false;
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]     setup SVA VGPR="
             << llvm::printReg(SVSStorageReg, TRI0)
             << (SetupIsFinalStorage ? " (== final storage)"
                                     : " (V3; moved to final at tail)")
             << "\n");

  // Snapshot every preloaded arg's physical position BEFORE the
  // aggregator below tears the ArgInfo down and re-materializes at
  // canonical positions. The restore-move loop later uses this to
  // shuffle preloaded args back to the OldReg positions the app
  // kernel's already-codegened MIR reads.
  auto &SIMFI = *KernelMF.getInfo<llvm::SIMachineFunctionInfo>();
  llvm::SmallVector<
      std::pair<llvm::AMDGPUFunctionArgInfo::PreloadedValue, llvm::MCRegister>,
      16>
      PreloadedArgSnapshot;
  for (auto PV : AllPreloadedValues) {
    if (llvm::MCRegister R = SIMFI.getPreloadedReg(PV))
      PreloadedArgSnapshot.push_back({PV, R});
  }

  // Compute stack setup inputs up front so we can decide whether the
  // aggregator needs to force-enable PSS (dynamic stack) or PSWO (non-
  // arch-FS scratch base).
  auto &ST = KernelMF.getSubtarget<llvm::GCNSubtarget>();
  const auto &SITRI = *ST.getRegisterInfo();
  const llvm::MachineFrameInfo &AppMFI = KernelMF.getFrameInfo();
  const bool AppUsesDynamicStack = AppMFI.hasVarSizedObjects();
  const unsigned AppPrivateSegmentFixedSize =
      static_cast<unsigned>(AppMFI.getStackSize());

  const unsigned OrigPreloadLength =
      SIMFI.getUserSGPRInfo().getNumKernargPreloadSGPRs();

  // Build the union of preloads the instrumentation setup needs.
  //   * Every SVA the payloads requested maps to a preload via
  //     \c preloadedValueForSVA (IMPLICIT_ARG_BUFFER returns nullopt and
  //     is filled by the kernarg-buffer expansion instead).
  //   * PSWO is consumed by \c emitCodeToSetupScratch's non-arch-FS
  //     branch.
  //   * PSS is consumed by \c emitCodeToSetupScratch's dynamic-stack
  //     branch.
  //   * KERNARG_SEGMENT_PTR is consumed by the kernarg-preload fallback
  //     ( \c S_LOAD_DWORD_IMM at kernel entry ) whenever the app kernel
  //     had a preload block on a subtarget that supports one.
  llvm::SmallSet<llvm::AMDGPUFunctionArgInfo::PreloadedValue, 16>
      RequiredPreloads;
  for (ScalarValueArgument SA : KernelInfo.RequestedKernelArguments) {
    if (auto PV = preloadedValueForSVA(SA))
      RequiredPreloads.insert(*PV);
  }
  if (KernelInfo.RequiresScratchAndStackSetup &&
      !ST.hasArchitectedFlatScratch() && !ST.enableFlatScratch())
    RequiredPreloads.insert(
        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);
  if (KernelInfo.RequiresScratchAndStackSetup && AppUsesDynamicStack)
    RequiredPreloads.insert(
        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_SIZE);
  if (OrigPreloadLength > 0 && ST.hasKernargPreload())
    RequiredPreloads.insert(
        llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);

  // Single aggregated force-enable: clears the ArgInfo + counters and
  // re-materializes (RequiredPreloads ∪ PreloadedArgSnapshot) at
  // canonical physreg positions in \c AllPreloadedValues order. All
  // \c amdgpu-no-* fn-attrs gating preload enable-bits are stripped in
  // there too.
  llvm::Function &KernelF = KernelMF.getFunction();
  ForceEnableResult FR = forceEnableRequestedUserSGPRPreloads(
      SIMFI, ST, SITRI, KernelF, RequiredPreloads, PreloadedArgSnapshot,
      OrigPreloadLength);
  llvm::DenseMap<llvm::AMDGPUFunctionArgInfo::PreloadedValue, llvm::MCRegister>
      NewPosMap(FR.NewPositions.begin(), FR.NewPositions.end());

  if (KernelInfo.RequiresScratchAndStackSetup) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "RequiresScratchAndStackSetup; emitting\n");
    if (auto Err = emitCodeToSetupScratch(
            EntryInstr, SVSStorageReg, AppUsesDynamicStack,
            AppPrivateSegmentFixedSize, KernelInfo.AnyPayloadUsesDynamicStack,
            KernelInfo.PayloadMaxFixedStackSize, Specs))
      return Err;
  }

  // Kernarg buffer expansion (only when IMPLICIT_ARG_BUFFER is
  // requested by some payload). Runs BEFORE the SVA-arg loop below so
  // the loop sees KERNARG_SEGMENT_PTR holding orig_kernarg_ptr, not
  // wrapper_addr.
  if (KernelInfo.RequestedKernelArguments.contains(IMPLICIT_ARG_BUFFER)) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "needsKernargBufferExpansion; emitting\n");
    /// Adjust the instrumented kernel's fn attrs so AMDGPUAsmPrinter
    /// emits the right metadata for the extended kernarg buffer:
    /// * Clear each \c amdgpu-no-<impl_arg> attr in the aggregated
    ///   set so hidden implicit-arg records get emitted after the
    ///   explicit app_kernarg_ptr arg. When the callgraph wasn't
    ///   fully recovered the aggregator seeded the full set, so
    ///   every impl-arg is forced on.
    /// * When the app kernel has an explicit kernarg arg 0, strip
    ///   \c byref  \c align-N off it so AMDGPUAsmPrinter emits it
    ///   as an address-typed record (size 8) instead of the
    ///   \c by_value inline buffer CodeDiscoveryPass initially set.

    /// Clear implicit arg disable-ing attributes if instrumentation requested
    /// hidden kernel args
    if (!KernelInfo.ImplicitArgsExplicitlyRequested.empty()) {
      KernelF.removeFnAttr("amdgpu-no-implicitarg-ptr");
      KernelF.removeFnAttr("amdgpu-implicitarg-num-bytes");
    }
    for (amdgpu::hsamd::ValueKind Attr :
         KernelInfo.ImplicitArgsExplicitlyRequested) {
      LLVM_DEBUG(luthier::dbgs()
                 << "[TargetModulePatcherPass]       clearing fn-attr for "
                    "ValueKind="
                 << static_cast<unsigned>(Attr) << " on kernel '"
                 << KernelF.getName() << "'\n");
      switch (Attr) {
      case amdgpu::hsamd::ValueKind::HiddenHostcallBuffer:
        KernelF.removeFnAttr("amdgpu-no-hostcall-ptr");
        break;
      case amdgpu::hsamd::ValueKind::HiddenHeapV1:
        KernelF.removeFnAttr("amdgpu-no-heap-ptr");
        break;
      case amdgpu::hsamd::ValueKind::HiddenMultiGridSyncArg:
        KernelF.removeFnAttr("amdgpu-no-multigrid-sync-arg");
        break;
      case amdgpu::hsamd::ValueKind::HiddenDefaultQueue:
        KernelF.removeFnAttr("amdgpu-no-default-queue");
        break;
      case amdgpu::hsamd::ValueKind::HiddenCompletionAction:
        KernelF.removeFnAttr("amdgpu-no-completion-action");
        break;
      case amdgpu::hsamd::ValueKind::HiddenQueuePtr:
        // \c hidden_queue_ptr is a hidden kernarg the launcher writes
        // into the extended kernarg buffer — payloads consume it via
        // kernarg-load at its metadata-declared offset, not via a
        // preloaded SGPR. Force-enabling the QueuePtr HW preload here
        // would shift every subsequent user SGPR (e.g. Kernarg from
        // s[4:5] to s[6:7]) at dispatch time, invalidating every use
        // of those SGPRs the app kernel's already-codegened MIR
        // references. Only the fn-attr clear is needed so the streamer
        // emits the metadata record.
        KernelF.removeFnAttr("amdgpu-no-queue-ptr");
        break;
      default:
        break;
      }
    }
    const bool HasAppKernarg = KernelF.arg_size() >= 1;
    if (HasAppKernarg) {
      llvm::Argument &Arg0 = *KernelF.arg_begin();
      Arg0.removeAttr(llvm::Attribute::ByRef);
      Arg0.removeAttr(llvm::Attribute::Alignment);
    }
    if (auto Err = emitKernargBufferExpansion(EntryInstr, SVSStorageReg, Specs,
                                              HasAppKernarg))
      return Err;
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]     "
                "RequestedKernelArguments count="
             << KernelInfo.RequestedKernelArguments.size() << "\n");
  for (ScalarValueArgument SA : KernelInfo.RequestedKernelArguments) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]       SVA arg="
                               << static_cast<unsigned>(SA) << "\n");
    auto LaneIt = Specs.findArgumentLane(SA);
    if (LaneIt == Specs.argument_lane_end())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: kernel '{0}' requests SVA arg {1} "
          "but the SVA specs do not assign it a lane",
          KernelMF.getName(), static_cast<unsigned>(SA)));
    std::optional<llvm::AMDGPUFunctionArgInfo::PreloadedValue> PV =
        preloadedValueForSVA(SA);
    if (!PV) {
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]         "
                                    "no preloaded value (filled elsewhere)\n");
      continue; // USER_ARG_PTR / IMPLICIT_ARG_BUFFER: filled in elsewhere.
    }
    // Source physreg comes from the aggregator's returned map. If the
    // SVA's PV isn't in the map, the caller under-populated
    // \c RequiredPreloads — that's a bug, not a runtime condition.
    auto NewIt = NewPosMap.find(*PV);
    if (NewIt == NewPosMap.end())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: kernel '{0}' requests SVA arg {1} "
          "but the preload aggregator did not install the corresponding "
          "PreloadedValue (missing from RequiredPreloads)",
          KernelMF.getName(), static_cast<unsigned>(SA)));
    llvm::MCRegister SrcReg = NewIt->second;
    const auto *ArgDesc = std::get<0>(SIMFI.getPreloadedValue(*PV));
    if (!ArgDesc || !ArgDesc->isRegister())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: kernel '{0}' requests SVA arg {1} "
          "but the source preloaded reg is not enabled on the MF",
          KernelMF.getName(), static_cast<unsigned>(SA)));
    unsigned Mask = ArgDesc->getMask();
    int NumSlots =
        static_cast<int>(StateValueArraySpecs::getArgumentLaneSize(SA));

    // WORKITEM_ID_{X,Y,Z} come as PRELOADED VGPRs (V0/V1/V2 on
    // non-packed subtargets; all V0 on packed-TID):
    //   1. Save SGPR0 (the temp) through the SP spill lane.
    //   2. \c V_READFIRSTLANE reads lane 0 of the workitem VGPR
    //      into SGPR0.
    //   3. On packed-TID (Mask != ~0u) the ID occupies a specific
    //      10-bit sub-range of the VGPR (X: bits 0..9, Y: 10..19,
    //      Z: 20..29). \c S_BFE_U32 extracts the sub-range with
    //      immediate = (Width << 16) | Offset — Offset from
    //      \c countr_zero(Mask), Width from \c popcount(Mask).
    //      On non-packed (Mask == ~0u) no extraction needed.
    //   4. \c V_WRITELANE deposits the extracted value into the
    //      SVA lane assigned to this workitem-id.
    //   5. Restore SGPR0 from the SP spill lane so downstream
    //      code / the app prolog sees its pre-clobber value.
    if (llvm::AMDGPU::VGPR_32RegClass.contains(SrcReg)) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          NumSlots == 1,
          "WORKITEM_ID_* SVA arg must occupy exactly one lane."));
      const uint8_t SPSpillLane = Specs.getStackPointerRegSpillLane();
      const llvm::MCRegister TempSGPR = llvm::AMDGPU::SGPR0;
      const auto &TII = *KernelMF.getSubtarget().getInstrInfo();
      // 1. Save TempSGPR into the SP spill lane.
      (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageReg)
          .addReg(TempSGPR)
          .addImm(SPSpillLane)
          .addReg(SVSStorageReg);
      // 2. Read lane 0 of the workitem VGPR into TempSGPR.
      (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_READFIRSTLANE_B32), TempSGPR)
          .addReg(SrcReg);
      // 3. Packed-TID extraction. See comment block above.
      if (Mask != ~0u) {
        unsigned Offset = llvm::countr_zero(Mask);
        unsigned Width = llvm::popcount(Mask);
        (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                            TII.get(llvm::AMDGPU::S_BFE_U32), TempSGPR)
            .addReg(TempSGPR)
            .addImm((Width << 16) | Offset);
      }
      // 4. Write TempSGPR into the workitem's SVA lane.
      (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageReg)
          .addReg(TempSGPR)
          .addImm(LaneIt->second)
          .addReg(SVSStorageReg);
      // 5. Restore TempSGPR from the SP spill lane.
      (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::V_READLANE_B32), TempSGPR)
          .addReg(SVSStorageReg)
          .addImm(SPSpillLane);
      continue;
    }

    if (auto Err =
            emitCodeToStoreSGPRKernelArg(EntryInstr, SrcReg, SVSStorageReg,
                                         /*SpillSlotStart=*/LaneIt->second,
                                         NumSlots, /*KillAfterUse=*/false))
      return Err;
  }

  // Restore each shifted preloaded arg from its new (post-aggregator)
  // physical position back to the SGPR / VGPR the lifted kernel body
  // reads it from. Classes not preloaded originally aren't in the
  // snapshot and are skipped; classes whose position didn't change
  // compare equal and are skipped. Drive off of the map the aggregator
  // returned, not off of \c SIMFI.getPreloadedReg (both map back to the
  // same physreg but the map is the aggregator's contract).
  const auto &TII = *KernelMF.getSubtarget().getInstrInfo();
  const auto &TRIR = *KernelMF.getSubtarget().getRegisterInfo();
  for (const auto &[Class, OldReg] : PreloadedArgSnapshot) {
    auto NewIt = NewPosMap.find(Class);
    if (NewIt == NewPosMap.end())
      continue; // aggregator dropped it (should not happen — snapshot
                // is unioned in — but leaves the old reg as-is).
    llvm::MCRegister NewReg = NewIt->second;
    if (!NewReg || NewReg == OldReg)
      continue;
    const llvm::TargetRegisterClass *RC = TRIR.getPhysRegBaseClass(OldReg);
    unsigned NumChannels = TRIR.getRegSizeInBits(*RC) / 32;
    const bool IsVGPR = llvm::SIRegisterInfo::isVGPRClass(RC);
    const unsigned MoveOpc =
        IsVGPR ? llvm::AMDGPU::V_MOV_B32_e32 : llvm::AMDGPU::S_MOV_B32;
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]     restore preload class "
               << unsigned(Class) << ": " << llvm::printReg(NewReg, &TRIR)
               << " -> " << llvm::printReg(OldReg, &TRIR) << " (" << NumChannels
               << " x " << (IsVGPR ? "v32" : "s32") << ")\n");
    for (unsigned I = 0; I < NumChannels; ++I) {
      llvm::MCRegister OldSub, NewSub;
      if (NumChannels == 1) {
        OldSub = OldReg;
        NewSub = NewReg;
      } else {
        unsigned SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(I);
        OldSub = TRIR.getSubReg(OldReg, SubIdx);
        NewSub = TRIR.getSubReg(NewReg, SubIdx);
      }
      (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                          TII.get(MoveOpc), OldSub)
          .addReg(NewSub);
    }
  }

  // Kernarg-preload handling. Split on \c FR.KernargPreloadDisabled:
  //   * true  — the aggregator determined the (new fixed user SGPRs) +
  //             (preload) total exceeded the HW 16-SGPR ceiling and
  //             already zeroed \c NumKernargPreloadSGPRs. Emit the
  //             manual \c S_LOAD_DWORD_IMM fallback that writes the
  //             original preload dwords into the range the lifted app
  //             kernel's MIR reads them from.
  //   * false — HW preload stays enabled. If the aggregator's re-
  //             materialization landed the preload block at a different
  //             SGPR range than the original, emit the S_MOV shuffle
  //             back to the range the lifted app expects.
  // Emitted at the very end of the position-correcting section so any
  // KERNARG_SEGMENT_PTR shuffle emitted above has already put the base
  // pointer at its final position.
  if (OrigPreloadLength > 0) {
    if (FR.KernargPreloadDisabled) {
      LLVM_DEBUG(
          luthier::dbgs()
          << "[TargetModulePatcherPass]     "
             "preload disabled: (new fixed user SGPRs) + preload > HW ceiling; "
             "emitting manual S_LOAD_DWORD for "
          << OrigPreloadLength << " preload dword(s)\n");
      llvm::MCRegister KernargSegPtr = SIMFI.getPreloadedReg(
          llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);
      if (!KernargSegPtr)
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "TargetModulePatcherPass: kernel '{0}' needs manual kernarg "
            "S_LOAD_DWORD fallback but KERNARG_SEGMENT_PTR was not "
            "installed by the aggregator",
            KernelMF.getName()));
      // Read the original preload dword offset from the KD attr
      // (populated by CodeDiscoveryPass). A missing attr means the
      // original KD had offset 0.
      unsigned PreloadOffsetDwords = 0;
      if (KernelF.hasFnAttribute("amdgpu.kd.kernarg_preload_offset")) {
        KernelF.getFnAttribute("amdgpu.kd.kernarg_preload_offset")
            .getValueAsString()
            .getAsInteger(10, PreloadOffsetDwords);
      }
      for (unsigned I = 0; I < OrigPreloadLength; ++I) {
        llvm::MCRegister Dst =
            llvm::MCRegister::from(FR.OrigPreloadStartSGPR.id() + I);
        unsigned ByteOff = (PreloadOffsetDwords + I) * 4;
        unsigned EncOff = llvm::AMDGPU::convertSMRDOffsetUnits(ST, ByteOff);
        (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                            TII.get(llvm::AMDGPU::S_LOAD_DWORD_IMM), Dst)
            .addReg(KernargSegPtr)
            .addImm(EncOff)
            .addImm(/*cpol=*/0);
      }
      (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::S_WAITCNT))
          .addImm(0);
    } else {
      // HW preload stays on; compute the new destination range from
      // the post-aggregator user-SGPR count (block sits at the tail of
      // \c NumUsedUserSGPRs). Skip when it hasn't moved.
      const unsigned NewNumUsedUserSGPRs =
          SIMFI.getUserSGPRInfo().getNumUsedUserSGPRs();
      if (NewNumUsedUserSGPRs < OrigPreloadLength)
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "TargetModulePatcherPass: kernel '{0}' post-aggregator "
            "NumUsedUserSGPRs ({1}) is smaller than the preload length ({2})",
            KernelMF.getName(), NewNumUsedUserSGPRs, OrigPreloadLength));
      const llvm::MCRegister NewPreloadStartSGPR = llvm::MCRegister::from(
          llvm::AMDGPU::SGPR0 + NewNumUsedUserSGPRs - OrigPreloadLength);
      if (NewPreloadStartSGPR != FR.OrigPreloadStartSGPR) {
        LLVM_DEBUG(luthier::dbgs()
                   << "[TargetModulePatcherPass]     "
                      "preload shuffle: "
                   << OrigPreloadLength << " dword(s) "
                   << llvm::printReg(NewPreloadStartSGPR, &TRIR) << " -> "
                   << llvm::printReg(FR.OrigPreloadStartSGPR, &TRIR) << "\n");
        for (unsigned I = 0; I < OrigPreloadLength; ++I) {
          llvm::MCRegister Dst =
              llvm::MCRegister::from(FR.OrigPreloadStartSGPR.id() + I);
          llvm::MCRegister Src =
              llvm::MCRegister::from(NewPreloadStartSGPR.id() + I);
          (void)llvm::BuildMI(EntryMBB, EntryInstr, llvm::DebugLoc(),
                              TII.get(llvm::AMDGPU::S_MOV_B32), Dst)
              .addReg(Src);
        }
      }
    }
  }

  // Move the SVA from the setup VGPR (V3 in the fallback path) into
  // the final storage picked by \c SVStorageAndLoadLocations. When
  // \c SetupIsFinalStorage is true, the setup writes already landed
  // in the picked VGPR — no move needed. Otherwise delegate to the
  // SVS's \c emitCodeToStoreSVA, which handles spilled / AGPR schemes
  // (scratch save-restore dance, AGPR write) uniformly.
  if (!SetupIsFinalStorage) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "move SVA "
                               << llvm::printReg(SVSStorageReg, &TRIR)
                               << " -> final SVS storage\n");
    EntrySVS.emitCodeToStoreSVA(EntryInstr, SVSStorageReg);
  }

  return llvm::Error::success();
}

/// Moves every non-payload global object -- globals, hooks, helper functions --
/// out of the IModule and into the target module, and re-homes each moved
/// definition's MIR into \p TargetFAM so the asm printer can find it. Injected
/// payloads are the only things left behind; Phase B.3 moves those into the
/// target module individually via \c movePayloadMFIntoTarget when it emits
/// each SI_CALL. \p VMap comes back populated so downstream operand-remap
/// consumers resolve correctly: a moved object maps to itself, and a
/// declaration that the target module already had maps to that one.
llvm::Error moveIModuleIntoTarget(llvm::Module &IModule,
                                  llvm::Module &TargetModule,
                                  llvm::FunctionAnalysisManager &IFAM,
                                  llvm::FunctionAnalysisManager &TargetFAM,
                                  llvm::ValueToValueMapTy &VMap) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] moveIModuleIntoTarget: "
             << "IModule has "
             << std::distance(IModule.global_begin(), IModule.global_end())
             << " global(s), " << IModule.size() << " function(s)\n");
  // Pass 0 — move, rather than copy, every tool global into the target module.
  // A moved GlobalVariable keeps its identity, so every reference to it — from
  // the IModule MIR that pass 2 clones and Phase B.3's moved payload MFs, and
  // from the target module afterwards — stays valid without remapping, and the
  // target
  // binary ends up with exactly one definition instead of a definition plus an
  // orphaned original. Copying instead left the definition behind in the
  // IModule and produced a second, separately-attributed GlobalVariable whose
  // initializer had to be re-derived.
  llvm::SmallVector<llvm::GlobalVariable *, 8> MovedGVs;
  for (auto &GV : IModule.globals()) {
    // The llvm.* metadata globals (llvm.used, llvm.compiler.used,
    // llvm.global.annotations, ...) list payloads and hooks, neither of which
    // is cloned into the target module. Moving them would drag those
    // references across, so they stay behind in the IModule.
    if (GV.getName().starts_with("llvm."))
      continue;
    MovedGVs.push_back(&GV);
  }
  for (llvm::GlobalVariable *GV : MovedGVs) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   move GV '"
                               << GV->getName() << "'\n");
    IModule.removeGlobalVariable(GV);
    TargetModule.insertGlobalVariable(GV);
    // cloneMFInto requires a VMap entry for every global operand it walks; a
    // moved global maps to itself.
    VMap[GV] = GV;
  }
  const unsigned MovedGVCount = MovedGVs.size();

  // Pass 1 — move every non-payload Function into the target module.
  //
  // Injected payloads stay behind: Phase B.3 inlines each of them at its
  // instrumentation point, so a copy here would be dead. Everything a payload
  // can reach — the hooks it calls, their helpers, and the declarations those
  // use — has to end up in the target module for the emitted object to be
  // self-contained.
  //
  // Moving rather than copying, exactly as for the globals above: the Function
  // keeps its identity, so every reference to it stays valid with no remapping,
  // and no second body-less definition is left behind. The previous copy left
  // the target module holding a fresh Function whose IR body was a lone
  // `unreachable`, with the real code only reachable through a cloned
  // MachineFunction.
  //
  // Declarations are the exception. The target module already declares the
  // AMDGPU intrinsics tool code calls, and moving a same-named Function in
  // would make LLVM rename it (`llvm.amdgcn.ballot.i32.1`) — invalid for an
  // intrinsic. Those are redirected to the target module's own declaration.
  unsigned SkippedPayloads = 0;
  unsigned RedirectedDecls = 0;
  llvm::SmallVector<llvm::Function *, 16> FuncsToMove;
  llvm::SmallVector<llvm::Function *, 8> DeclsToErase;
  for (llvm::Function &F : IModule.functions()) {
    if (F.hasFnAttribute(InjectedPayloadAttribute)) {
      ++SkippedPayloads;
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   skip payload '"
                                 << F.getName() << "'\n");
      continue;
    }
    if (F.isDeclaration()) {
      if (llvm::Function *Existing = TargetModule.getFunction(F.getName());
          Existing && Existing->getFunctionType() == F.getFunctionType()) {
        // Merge the IModule declaration with the TargetModule definition.
        // Redirect the IR as well as the VMap. The functions moved below keep
        // their bodies, and a body still calling a Function the IModule owns
        // leaves a cross-module edge in the target module's call graph —
        // CallGraph::getOrInsertFunction asserts on exactly that ("Function not
        // in current module!") as soon as the asm printer's legacy pipeline
        // builds one. The VMap entry alone only fixes MIR global operands.
        //
        // \c PatchPCUsagesPass::emitEntryPointSeed plants an extern
        // declaration of every trace-relevant target-module Function in
        // IModule and threads it into the seed array's initializer, keeping
        // the initializer's Function references inside IModule so
        // \c LazyCallGraphAnalysis on IModule does not see a cross-module
        // Node (see \c CGSCCPassManager.cpp:683). Here is where those
        // placeholders finally rejoin the definitions they stood in for.
        F.replaceAllUsesWith(Existing);
        VMap[&F] = Existing;
        DeclsToErase.push_back(&F);
        ++RedirectedDecls;
        continue;
      }
    }
    FuncsToMove.push_back(&F);
  }
  for (llvm::Function *F : FuncsToMove) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   move Fn '" << F->getName()
               << "' (decl=" << F->isDeclaration() << ")\n");
    IModule.getFunctionList().remove(F->getIterator());
    TargetModule.getFunctionList().push_back(F);
    // Moved, not cloned: it maps to itself.
    VMap[F] = F;
  }
  // Rewrite MI operands in payloads (still resident in IModule/IFAM at
  // this point — Phase B.3 is what moves them) whose \c MO_GlobalAddress
  // names one of the placeholder declarations we're about to drop. The
  // IR-level \c replaceAllUsesWith above only touched \c llvm::Use edges,
  // and \c MachineOperand::getGlobal() is not one of those. Once the
  // placeholder is freed by the erase loop below, \c ValueToValueMapTy's
  // \c CallbackVH removes the corresponding \c VMap entry automatically,
  // so any deferred remap (e.g. inside \c movePayloadMFIntoTarget) would
  // look up a stale pointer that no longer maps to anything. Fixing the
  // operands here — while both the source key and the survivor are
  // still live — is the only correct spot.
  for (llvm::Function &PayloadFn : IModule) {
    if (!PayloadFn.hasFnAttribute(InjectedPayloadAttribute))
      continue;
    auto *MFRes =
        IFAM.getCachedResult<llvm::MachineFunctionAnalysis>(PayloadFn);
    if (!MFRes)
      continue;
    for (llvm::MachineBasicBlock &MBB : MFRes->getMF()) {
      for (llvm::MachineInstr &MI : MBB.instrs()) {
        for (llvm::MachineOperand &MO : MI.operands()) {
          if (!MO.isGlobal())
            continue;
          auto It = VMap.find(MO.getGlobal());
          if (It == VMap.end())
            continue;
          auto *NewGV = llvm::cast<llvm::GlobalValue>(It->second);
          if (NewGV == MO.getGlobal())
            continue;
          MO.ChangeToGA(NewGV, MO.getOffset(), MO.getTargetFlags());
        }
      }
    }
  }
  // Drop the now-dead extern placeholders. RAUW above stripped their only
  // uses, so this cannot invalidate any operand of a still-live IR value —
  // and leaving them behind would ship an IModule holding declarations of
  // every target-module symbol that appeared in the seed.
  for (llvm::Function *F : DeclsToErase) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   erase merged extern decl '"
               << F->getName() << "'\n");
    F->eraseFromParent();
  }

  // A moved global brings its initializer with it, and a moved function keeps
  // its body, so the only remapping left is for references that resolved to a
  // target-module declaration above. Redo the initializers through VMap; a
  // no-op for the common case of a tool counter initialized to zero.
  for (llvm::GlobalVariable *GV : MovedGVs) {
    if (!GV->hasInitializer())
      continue;
    GV->setInitializer(llvm::MapValue(GV->getInitializer(), VMap));
  }

  // Comdat re-homing.
  unsigned ReHomedComdats = 0;
  auto ReHomeComdat = [&](llvm::GlobalObject *GO) {
    const llvm::Comdat *SrcC = GO->getComdat();
    if (!SrcC)
      return;
    llvm::Comdat *DstC = TargetModule.getOrInsertComdat(SrcC->getName());
    DstC->setSelectionKind(SrcC->getSelectionKind());
    GO->setComdat(DstC);
    ++ReHomedComdats;
  };
  for (llvm::GlobalVariable *GV : MovedGVs)
    ReHomeComdat(GV);
  for (llvm::Function *F : FuncsToMove)
    ReHomeComdat(F);

  // Move global aliases.
  llvm::SmallVector<llvm::GlobalAlias *, 4> MovedAliases;
  for (llvm::GlobalAlias &GA : IModule.aliases())
    if (!GA.getName().starts_with("llvm."))
      MovedAliases.push_back(&GA);
  for (llvm::GlobalAlias *GA : MovedAliases) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   move alias '"
                               << GA->getName() << "'\n");
    IModule.removeAlias(GA);
    TargetModule.insertAlias(GA);
    VMap[GA] = GA;
  }

  // Move global ifuncs.
  llvm::SmallVector<llvm::GlobalIFunc *, 4> MovedIFuncs;
  for (llvm::GlobalIFunc &GI : IModule.ifuncs())
    MovedIFuncs.push_back(&GI);
  for (llvm::GlobalIFunc *GI : MovedIFuncs) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   move ifunc '"
                               << GI->getName() << "'\n");
    IModule.removeIFunc(GI);
    TargetModule.insertIFunc(GI);
    VMap[GI] = GI;
  }

  // Module-level inline asm.
  if (!IModule.getModuleInlineAsm().empty()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   append module inline asm ("
               << IModule.getModuleInlineAsm().size() << " chars)\n");
    TargetModule.appendModuleInlineAsm(IModule.getModuleInlineAsm());
    IModule.setModuleInlineAsm("");
  }

  // Pass 2 — re-home each moved definition's MIR.
  //
  // Moving the Function does not move its MachineFunction: that lives in the
  // instrumentation module's FunctionAnalysisManager, keyed by this Function,
  // while NewPMAsmPrinter looks the target module's MIR up in the target FAM.
  // Create the MF there and deep-clone the body across. This has to run after
  // the move loop, both so F->getParent() is the target module when
  // MachineFunctionAnalysis::run builds the destination, and so every global
  // operand cloneMFInto walks already has its VMap entry.
  unsigned ClonedMFs = 0;
  for (llvm::Function *F : FuncsToMove) {
    if (F->isDeclaration())
      continue;
    auto *SrcMFRes = IFAM.getCachedResult<llvm::MachineFunctionAnalysis>(*F);
    if (SrcMFRes == nullptr) {
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   no MF for '"
                                 << F->getName() << "', skip MF re-home\n");
      continue; // Definition without lifted MIR — rare; nothing to re-home.
    }
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   re-home MF '" << F->getName()
               << "' (" << SrcMFRes->getMF().size() << " MBB(s))\n");
    llvm::MachineFunction &DstMF =
        TargetFAM.getResult<llvm::MachineFunctionAnalysis>(*F).getMF();
    if (auto Err = cloneMFInto(SrcMFRes->getMF(), VMap, DstMF))
      return Err;
    ++ClonedMFs;
  }

  // Pass 3 — move named metadata (\c llvm.module.flags,
  // \c llvm.dbg.cu, \c llvm.ident, \c llvm.linker.options, and any
  // custom named MD the IModule carries) into the target module.
  unsigned MovedNamedMDs = 0;
  unsigned MergedNamedMDs = 0;
  llvm::SmallVector<llvm::NamedMDNode *, 8> NamedMDs;
  for (llvm::NamedMDNode &NMD : IModule.named_metadata())
    NamedMDs.push_back(&NMD);
  for (llvm::NamedMDNode *SrcNMD : NamedMDs) {
    if (llvm::NamedMDNode *ExistingDst =
            TargetModule.getNamedMetadata(SrcNMD->getName())) {
      LLVM_DEBUG(luthier::dbgs()
                 << "[TargetModulePatcherPass]   merge named MD '"
                 << SrcNMD->getName() << "' (" << SrcNMD->getNumOperands()
                 << " operand(s))\n");
      for (llvm::MDNode *Op : SrcNMD->operands())
        ExistingDst->addOperand(Op);
      ++MergedNamedMDs;
    } else {
      LLVM_DEBUG(luthier::dbgs()
                 << "[TargetModulePatcherPass]   move named MD '"
                 << SrcNMD->getName() << "' (" << SrcNMD->getNumOperands()
                 << " operand(s))\n");
      llvm::NamedMDNode *DstNMD =
          TargetModule.getOrInsertNamedMetadata(SrcNMD->getName());
      for (llvm::MDNode *Op : SrcNMD->operands())
        DstNMD->addOperand(Op);
      ++MovedNamedMDs;
    }
    IModule.eraseNamedMetadata(SrcNMD);
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] moveIModuleIntoTarget "
                "done: "
             << MovedGVCount << " GV(s) moved, " << FuncsToMove.size()
             << " Fn(s) moved (" << SkippedPayloads << " payload(s) skipped, "
             << RedirectedDecls << " decl(s) redirected), " << ClonedMFs
             << " MF(s) re-homed, " << ReHomedComdats << " comdat(s) re-homed, "
             << MovedAliases.size() << " alias(es) moved, "
             << MovedIFuncs.size() << " ifunc(s) moved, " << MovedNamedMDs
             << " named MD(s) moved, " << MergedNamedMDs
             << " named MD(s) merged\n");
  return llvm::Error::success();
}

/// Strip the `amdgpu-num-vgpr` and `amdgpu-num-sgpr` attributes from every
/// function in \p TargetModule. These were set by CodeDiscoveryPass based
/// on the original (pre-instrumentation) code-object's register usage,
/// but are no longer accurate after our instrumentation extends the
/// register footprint. Leaving them present would mislead the LLVM
/// AMDGPU backend's register-pressure heuristics on the next codegen
/// pass over the target.
void stripStaleNumRegsAttrs(llvm::Module &TargetModule) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] stripStaleNumRegsAttrs "
                "over "
             << TargetModule.size() << " function(s)\n");
  for (llvm::Function &F : TargetModule.functions()) {
    F.removeFnAttr("amdgpu-num-vgpr");
    F.removeFnAttr("amdgpu-num-sgpr");
  }
}

/// Walk every non-indirect branch in \p MF, sum byte sizes via
/// \c TII.getInstSizeInBytes to estimate per-MBB layout offsets, and
/// report any branch whose target lies beyond \c s_branch's signed
/// 16-bit-word range (±131,068 bytes). When out-of-range branches are
/// found we currently emit a diagnostic and return their count — the
/// actual relax-to-\c s_setpc_b64 rewrite (with SGPR scavenging from
/// per-MBB live-ins and SVA-lane spill fallback per
/// \c StateValueArraySpecs::findLowestFreeLanes) is the next phase.
///
/// Why we need this even though stock LLVM \c BranchRelaxationPass
/// exists: \c CodeGenerator::printAssemblyFile invokes
/// \c TM.addAsmPrinter directly, skipping the standard post-RA
/// machine-pass chain. So no LLVM pass runs between TargetModulePatcher
/// and AsmPrinter; out-of-range branches would be silently emitted
/// with truncated displacements.
/// One entry per direct branch whose target lies beyond \c s_branch's
/// signed 16-bit-word range. Populated by \c detectOutOfRangeBranches
/// and consumed by the eventual \c s_setpc_b64 rewriter (task #26
/// rewrite phase, not yet implemented).
struct OutOfRangeBranchRecord {
  const llvm::MachineFunction *MF;
  const llvm::MachineInstr *Branch;
  const llvm::MachineBasicBlock *Target;
  int64_t BranchOffset;
  int64_t Delta;
};

unsigned
detectOutOfRangeBranches(const llvm::MachineFunction &MF,
                         llvm::SmallVectorImpl<OutOfRangeBranchRecord> &Out) {
  static constexpr int64_t kSBranchMaxBytes = (1LL << 18) - 4;
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  llvm::DenseMap<const llvm::MachineBasicBlock *, int64_t> MBBOffset;
  int64_t Cursor = 0;
  for (const auto &MBB : MF) {
    MBBOffset[&MBB] = Cursor;
    for (const auto &MI : MBB)
      Cursor += TII.getInstSizeInBytes(MI);
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   detectOutOfRangeBranches MF='"
             << MF.getName() << "' totalSize=" << Cursor << " bytes\n");
  unsigned NumOutOfRange = 0;
  Cursor = 0;
  for (const auto &MBB : MF) {
    int64_t MIOffset = Cursor;
    for (const auto &MI : MBB) {
      if (MI.isBranch() && !MI.isIndirectBranch()) {
        if (auto *TargetMBB = TII.getBranchDestBlock(MI)) {
          int64_t TgtOff = MBBOffset[TargetMBB];
          int64_t Delta = TgtOff - (MIOffset + TII.getInstSizeInBytes(MI));
          if (Delta > kSBranchMaxBytes || Delta < -kSBranchMaxBytes) {
            ++NumOutOfRange;
            LLVM_DEBUG(luthier::dbgs()
                       << "[TargetModulePatcherPass]     out-of-range branch "
                          "at 0x"
                       << llvm::Twine::utohexstr(MIOffset) << " -> "
                       << llvm::printMBBReference(*TargetMBB)
                       << " delta=" << Delta << "B\n");
            Out.push_back({&MF, &MI, TargetMBB, MIOffset, Delta});
          }
        }
      }
      MIOffset += TII.getInstSizeInBytes(MI);
    }
    Cursor = MIOffset;
  }
  return NumOutOfRange;
}

} // namespace

llvm::PreservedAnalyses
TargetModulePatcherPass::run(Prototype &IP, PrototypeAnalysisManager &IPAM) {
  LLVM_DEBUG(luthier::dbgs() << "=== Luthier Target Module Patcher Pass ===\n");

  llvm::Module &IModule = IP.getInstrumentationModule();
  llvm::Module &TargetModule = IP.getTargetModule();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] IModule='" << IModule.getName()
             << "' (" << IModule.size() << " function(s))\n");
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] TargetModule='"
                             << TargetModule.getName() << "' ("
                             << TargetModule.size() << " function(s))\n");

  llvm::LLVMContext &Ctx = IModule.getContext();

  // This pass reads both halves of the prototype, and each has its own
  // managers.
  llvm::FunctionAnalysisManager &TargetFAM =
      IPAM.getResult<TargetFunctionAnalysisManagerPrototypeProxy>(IP)
          .getManager();
  llvm::FunctionAnalysisManager &IFAM =
      IPAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(IP)
          .getManager();
  auto getTargetMF = [&](llvm::Function &F) -> llvm::MachineFunction & {
    return TargetFAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();
  };

  const SVStorageAndLoadLocations &SVLocations =
      IPAM.getResult<SVStorageAndLoadLocationsAnalysis>(IP);

  const StateValueArraySpecs &SVASpecs =
      IPAM.getResult<StateValueArraySpecsAnalysis>(IP);
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] SVASpecs resolved\n");

  const InjectedPayloadAndInstPoint &IPIP =
      IPAM.getResult<InjectedPayloadAndInstPointAnalysis>(IP);

  const IPPredicatedCFG &IPCFG =
      IPAM.getResult<IPPredCFGAnalysis>(IP).getVecCFG();
  const IPPredicatedLiveness &IPLiveness =
      IPAM.getResult<IPPredicatedLivenessAnalysis>(IP);

  /// Find the initial kernel function entry point
  llvm::MachineFunction *InitialEntryKernelMF = nullptr;
  for (llvm::Function &F : TargetModule) {
    if (!F.hasFnAttribute(InitialEntryPointAttr))
      continue;
    assert(!InitialEntryKernelMF &&
           "target module has multiple initial-entry-point functions");
    if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
      InitialEntryKernelMF = &getTargetMF(F);
  }

  llvm::MachineFunctionAnalysisManager &TargetMFAM =
      IPAM.getResult<TargetMachineFunctionAnalysisManagerPrototypeProxy>(IP)
          .getManager();

  /// SVA Setup & Storage Code Emission
  /// For each target MF, walk SVStorageAndLoadLocations'
  /// StateValueStorageIntervals and emit the SVS switch code at each
  /// interval boundary.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === "
                                "Emit SVS Switches For MF ===\n");
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;
    llvm::MachineFunction &MF =
        TargetFAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();
    emitSVSSwitchesForMF(MF, SVLocations, SVASpecs,
                         TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(MF));
  }

  /// Partial-callgraph SVS handoff. Wrap the last call in every MBB
  /// whose PMBB.hasUnresolvedEdges() with the V0-courier protocol
  /// (BlockSVS.emitCodeToLoadSVA / emitCodeToStoreSVA around the call
  /// site into a VGPRStateValueArrayStorage(V0)). Depends on the SVS
  /// switches being in place first — this pass reads the last segment
  /// of the MBB's storage intervals to pick the right BlockSVS at the
  /// call site.
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] === "
                "Emit Partial-Callgraph SVS Handoff Wraps ===\n");
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;
    llvm::MachineFunction &MF =
        TargetFAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();
    emitPartialCallgraphSVSHandoffWraps(MF, IPCFG, SVLocations, SVASpecs);
  }

  // Emit the SVA preload setup (scratch + kernarg spills into
  // SVA lanes) at the initial-entry kernel's first instruction, when
  // there is one. \c InitialEntryKernelMF is null if the target module's
  // initial entry point is a device function rather than a kernel — no
  // kernel prologue is emitted in that case. Other target functions (the
  // helpers cloned in later during Phase B.1) never need this setup
  // either, so the pass emits at most one prologue per invocation.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase A.2: "
                                "emitInitialEntryKernelSetup ===\n");
  if (InitialEntryKernelMF) {
    SVAScratchSetupInfo InitialEntryKernelInfo =
        computeInitialEntryKernelSVAInfo(*InitialEntryKernelMF, IModule, IFAM,
                                         IPIP, SVLocations, IPLiveness);
    if (auto Err = emitInitialEntryKernelSetup(*InitialEntryKernelMF,
                                               InitialEntryKernelInfo,
                                               SVLocations, SVASpecs)) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::none();
    }
  } else {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   no "
                                  "initial-entry kernel; skipping prologue\n");
  }

  // ============= Phase B: Target Patching ===============================
  //
  // Step 1: Clone IModule globals + non-payload non-hook Functions into
  // the target module. The returned VMap is used by the inliner so
  // cross-module operands resolve. Helper MFs are constructed in the
  // target FAM (via MachineFunctionAnalysis::run) and populated with
  // cloneMFInto, so they're visible to subsequent target-side loops
  // and to NewPMAsmPrinter's per-Function MF lookup.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase B.1: "
                                "moveIModuleIntoTarget ===\n");
  llvm::ValueToValueMapTy VMap;
  if (auto Err =
          moveIModuleIntoTarget(IModule, TargetModule, IFAM, TargetFAM, VMap)) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return llvm::PreservedAnalyses::none();
  }

  // Step 2: Strip stale num-{vgpr,sgpr} attrs — CodeDiscoveryPass set
  // these from the original code object, and they're wrong now that
  // we've inlined payloads + cloned helpers.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase B.2: "
                                "stripStaleNumRegsAttrs ===\n");
  stripStaleNumRegsAttrs(TargetModule);

  // Step 3: Outline every injected payload behind an SI_CALL at its
  // target PATCHPOINT. Ordering:
  //   3a. Seed each touched target MF's MBB.liveins from the cached
  //       IPPredicatedLiveness so `scavengeSGPRPairAtSite`'s
  //       `LivePhysRegs::addLiveOuts` sees a correct successor-liveness
  //       union. The branch-relaxation phase later re-seeds; that pass
  //       is idempotent against a clean state.
  //   3b. Scavenge an SReg_64 pair for every PATCHPOINT up front (fail
  //       fast — no mutation yet).
  //   3c. Per payload: emit SI_CALL (+ callee-address materialization)
  //       at the PATCHPOINT (erasing the marker AND the extern-handle
  //       Function decl as a spec-mandated pair), move the payload MF
  //       ownership from IFAM to TargetFAM (moving the IR Function
  //       from IModule to TargetModule), and rewrite the payload's
  //       return terminators to jump back via the scavenged pair
  //       (Case A) or via an MCSymbol trampoline (Case B).
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] === Phase B.3: outline "
                "injected payloads via SI_CALL ===\n");

  // 3a. Seed liveins on every target MF that hosts a PATCHPOINT.
  // Mirrors TargetModuleBranchRelaxation's PMBB-seeded liveness setup —
  // one source of truth for target-module liveness across the whole
  // patcher.
  llvm::DenseSet<llvm::MachineFunction *> TargetMFsTouchedByPayloads;
  for (const auto &[InjectedPayloadFunc, InsertionPointMI] :
       IPIP.payload_mi()) {
    TargetMFsTouchedByPayloads.insert(InsertionPointMI->getMF());
  }
  for (llvm::MachineFunction *MF : TargetMFsTouchedByPayloads) {
    MF->getProperties().setTracksLiveness();
    for (llvm::MachineBasicBlock &MBB : *MF) {
      if (MBB.empty())
        continue;
      // Skip blocks that were introduced after IPCFG was built
      if (!IPCFG.contains(MBB))
        continue;
      const PredicatedMachineBasicBlock &PMBB =
          const_cast<IPPredicatedCFG &>(IPCFG).getPredMBB(MBB.front());
      // Seed liveins with the union of Active + Inactive PMBB live-ins
      const llvm::LivePhysRegs *ActiveLI =
          IPLiveness.getPMBBActiveLiveIns(PMBB);
      const llvm::LivePhysRegs *InactiveLI =
          IPLiveness.getPMBBInactiveLiveIns(PMBB);
      if (!ActiveLI && !InactiveLI)
        continue;
      MBB.clearLiveIns();
      if (ActiveLI)
        for (llvm::MCPhysReg R : *ActiveLI)
          MBB.addLiveIn(R);
      if (InactiveLI)
        for (llvm::MCPhysReg R : *InactiveLI)
          MBB.addLiveIn(R);
      MBB.sortUniqueLiveIns();
    }
  }

  // 3b. Scavenge SGPRs. Snapshotted into `ScavengedByPayload` before
  // any orchestration mutation so a scavenge failure aborts cleanly.
  // Each entry captures the SReg_64 pair the SI_CALL will use plus,
  // when $scc is live across the patchpoint, an extra SGPR_32 to spill
  // $scc into for the duration of the call-setup sequence.
  llvm::MapVector<llvm::Function *, ScavengedPatchpointRegs>
      ScavengedByPayload;
  for (const auto &[InjectedPayloadFunc, InsertionPointMI] :
       IPIP.payload_mi()) {
    const llvm::MachineFunction &TargetHostMF = *InsertionPointMI->getMF();
    const auto &SI = TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(
        const_cast<llvm::MachineFunction &>(TargetHostMF));
    auto ScavOrErr = scavengeSGPRsAtSite(*InsertionPointMI, SVLocations, SI);
    if (!ScavOrErr) {
      Ctx.emitError(llvm::toString(ScavOrErr.takeError()));
      return llvm::PreservedAnalyses::none();
    }
    ScavengedByPayload[InjectedPayloadFunc] = *ScavOrErr;
  }

  // 3c. Per-payload orchestration.
  unsigned PayloadCount = 0;
  for (const auto &[InjectedPayloadFunc, InsertionPointMI] :
       IPIP.payload_mi()) {
    ++PayloadCount;
    const ScavengedPatchpointRegs &Regs =
        ScavengedByPayload[InjectedPayloadFunc];
    const llvm::MCRegister Scav = Regs.Pair;
    const llvm::MCRegister SCCSave = Regs.SCCSave;

    auto *InjectedPayloadMFRes =
        IFAM.getCachedResult<llvm::MachineFunctionAnalysis>(
            *InjectedPayloadFunc);
    if (!InjectedPayloadMFRes) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: payload function '{0}' has no "
          "MachineFunction in the instrumentation module",
          InjectedPayloadFunc->getName()))));
      return llvm::PreservedAnalyses::none();
    }
    llvm::MachineFunction &PayloadMF = InjectedPayloadMFRes->getMF();

    // Emit the SI_CALL at the PATCHPOINT and capture the continuation
    // symbol before we mutate the payload MF (order matters: the setpc
    // rewrite Case-B branch uses this symbol). The helper also erases
    // the PATCHPOINT MI and the extern-handle Function decl atomically
    llvm::Function *ExternHandle =
        IPIP.getExternHandleFromInjectedPayload(*InjectedPayloadFunc);
    assert(ExternHandle && "every PATCHPOINT must have an associated payload");
    llvm::MCSymbol *ContSym = emitSICallAtPatchpoint(
        *InsertionPointMI, *InjectedPayloadFunc, *ExternHandle, Scav,
        SCCSave);

    // Move the payload MF from IFAM to TargetFAM (and the IR Function
    // from IModule to TargetModule). After this call, PayloadMF is
    // owned by TargetFAM; the reference stays valid because splice
    // keeps the list node — and thus the Result's unique_ptr — in
    // place.
    if (auto Err = movePayloadMFIntoTarget(*InjectedPayloadFunc, TargetModule,
                                           IFAM, TargetFAM, VMap)) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::none();
    }

    // Rewrite payload returns to land back at ContSym via Scav.
    if (auto Err = rewritePayloadReturn(PayloadMF, Scav, ContSym,
                                        /*PreserveSCCInCaseB=*/bool(SCCSave))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::none();
    }
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   Phase B.3 outlined "
             << PayloadCount << " payload(s)\n");

  /// Relax short branches in the target module that don't make their targets
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === "
                                "Branch relaxation per kernel ===\n");
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration() || !getFunctionEntryPoint(F).has_value())
      continue;
    llvm::MachineFunction &MF =
        TargetFAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   relaxer for function '"
               << F.getName() << "' (" << MF.size() << " MBB(s))\n");

    TargetModuleBranchRelaxation BR(IPCFG, IPLiveness, SVLocations, SVASpecs);
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     running "
                                  "TargetModuleBranchRelaxation on '"
                               << F.getName() << "'\n");
    BR.run(MF);
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "TargetModuleBranchRelaxation done for '"
                               << F.getName() << "'\n");
  }

  // Sanity-check: re-run the detector and hard-error if anything
  // remains out of range.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase B.5: "
                                "post-relax out-of-range sanity check ===\n");
  llvm::SmallVector<OutOfRangeBranchRecord, 4> OutOfRange;
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration() || !getFunctionEntryPoint(F).has_value())
      continue;
    detectOutOfRangeBranches(getTargetMF(F), OutOfRange);
  }
  if (!OutOfRange.empty()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   " << OutOfRange.size()
               << " branch(es) still over-range; failing\n");
    std::string Detail;
    llvm::raw_string_ostream OS(Detail);
    OS << "TargetModulePatcherPass: " << OutOfRange.size()
       << " branch(es) remain over-range after BranchRelaxationPass; "
       << "branches:";
    for (const auto &R : OutOfRange) {
      OS << "\n  - " << R.MF->getName() << " offset 0x";
      OS.write_hex(static_cast<uint64_t>(R.BranchOffset));
      OS << " → " << R.Target->getName() << " (delta " << R.Delta << " bytes)";
    }
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(Detail)));
    return llvm::PreservedAnalyses::none();
  }
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] run complete; "
                                "target module is patched and verified\n");

  // Preserve the outer MAM proxy so the Prototype adaptor doesn't
  // wipe every cached module-level analysis for both modules on the way out
  // — downstream consumers (notably the AsmPrinter driver) still need the
  // cached MachineFunctionAnalysis results for the target module we just
  // mutated.
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  // The patcher clones globals into the target module, strips stale register
  // attributes and splices payload MIR into target MFs. It edits those MFs in
  // place — none are created or destroyed — so the cached
  // MachineFunctionAnalysis results stay live, which the asm printer scheduled
  // after this pass depends on. Prototype-level analyses are all dropped.
  PA.preserve<TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  return PA;
}

} // namespace luthier
