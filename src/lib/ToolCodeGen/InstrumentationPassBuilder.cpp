//===-- InstrumentationPassBuilder.cpp ------------------------------------===//
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
/// Implements \c luthier::InstrumentationPassBuilder, including
/// \c buildInstrumentationPipeline and the Luthier-owned AMDGPU codegen
/// pass builder used to splice \c InjectedPayloadPEIPass into the machine
/// pipeline.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InstrumentationPassBuilder.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/CodeDiscoveryPass.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadPEIPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadPreserveLiveRegsPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
#include "luthier/ToolCodeGen/InstructionTracesAnalysis.h"
#include "luthier/ToolCodeGen/IntrinsicMIRLoweringPass.h"
#include "luthier/ToolCodeGen/NewPMAsmPrinter.h"
#include "luthier/ToolCodeGen/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/PrototypeCallGraph.h"
#include "luthier/ToolCodeGen/SVAPhysVGPRPinPass.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include "luthier/ToolCodeGen/TargetModulePatcherPass.h"
#include "luthier/ToolCodeGen/TraceFunctionTranslationAnalysis.h"

#include <AMDGPU.h>
#include <AMDGPUCtorDtorLowering.h>
#include <AMDGPUExportKernelRuntimeHandles.h>
#include <AMDGPUISelDAGToDAG.h>
#include <AMDGPULowerVGPREncoding.h>
#include <AMDGPUPerfHintAnalysis.h>
#include <AMDGPUPreloadKernArgProlog.h>
#include <AMDGPUPrepareAGPRAlloc.h>
#include <AMDGPURemoveIncompatibleFunctions.h>
#include <AMDGPUReserveWWMRegs.h>
#include <AMDGPUTargetMachine.h>
#include <AMDGPUUnifyDivergentExitNodes.h>
#include <AMDGPUWaitSGPRHazards.h>
#include <GCNDPPCombine.h>
#include <GCNNSAReassign.h>
#include <GCNPreRALongBranchReg.h>
#include <GCNPreRAOptimizations.h>
#include <GCNRewritePartialRegUses.h>
#include <SIFixSGPRCopies.h>
#include <SIFixVGPRCopies.h>
#include <SIFoldOperands.h>
#include <SIFormMemoryClauses.h>
#include <SILoadStoreOptimizer.h>
#include <SILowerControlFlow.h>
#include <SILowerSGPRSpills.h>
#include <SILowerWWMCopies.h>
#include <SIMachineFunctionInfo.h>
#include <SIOptimizeExecMasking.h>
#include <SIOptimizeExecMaskingPreRA.h>
#include <SIOptimizeVGPRLiveRange.h>
#include <SIPeepholeSDWA.h>
#include <SIPostRABundler.h>
#include <SIPreAllocateWWMRegs.h>
#include <SIShrinkInstructions.h>
#include <SIWholeQuadMode.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/UniformityAnalysis.h>
#include <llvm/CodeGen/AtomicExpand.h>
#include <llvm/CodeGen/BranchRelaxation.h>
#include <llvm/CodeGen/DeadMachineInstructionElim.h>
#include <llvm/CodeGen/EarlyIfConversion.h>
#include <llvm/CodeGen/LibcallLoweringInfo.h>
#include <llvm/CodeGen/MachineCSE.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineLICM.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/PEI.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/PostRAHazardRecognizer.h>
#include <llvm/CodeGen/RegAllocFast.h>
#include <llvm/CodeGen/RegAllocGreedyPass.h>
#include <llvm/CodeGen/RegAllocRegistry.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassManagerInternal.h>
#include <llvm/IR/PrintPasses.h>
#include <llvm/Passes/CodeGenPassBuilder.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <atomic>
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/CGPassBuilderOption.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/ExpandVariadics.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/FlattenCFG.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/LICM.h>
#include <llvm/Transforms/Scalar/LoopDataPrefetch.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>
#include <llvm/Transforms/Scalar/NaryReassociate.h>
#include <llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h>
#include <llvm/Transforms/Scalar/Sink.h>
#include <llvm/Transforms/Scalar/StraightLineStrengthReduce.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>
#include <llvm/Transforms/Utils/FixIrreducible.h>
#include <llvm/Transforms/Utils/LCSSA.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/UnifyLoopExits.h>
#include <llvm/Transforms/Vectorize/LoadStoreVectorizer.h>

using llvm::Error;
using llvm::ModulePassManager;
using llvm::PassBuilder;
using llvm::PassInstrumentationCallbacks;
using llvm::StringRef;

//===----------------------------------------------------------------------===//
// PassManagerWrapper access shim.
//
// llvm::PassManagerWrapper has a private constructor and befriends only
//   template <typename D, typename T> class llvm::CodeGenPassBuilder;
// so every specialization of CodeGenPassBuilder is a friend of PMW, but
// friendship does not propagate to *derived* classes. Luthier's
// AMDGPUCodeGenPassBuilder (a subclass of CodeGenPassBuilder<...>) therefore
// cannot construct a PassManagerWrapper from its own buildPipeline().
//
// We work around this without touching the LLVM header by declaring an
// explicit specialization of llvm::CodeGenPassBuilder for a Luthier-owned
// dummy type pair. Because the friend declaration on PMW uses the
// unrestricted template form (`friend class CodeGenPassBuilder;`), this
// specialization is a friend and its members can invoke PMW's private
// constructor. C++17 guaranteed copy elision means callers do not need any
// access to PMW's move/copy constructors either.
namespace llvm {

// Dummy tags used solely to instantiate a Luthier-owned specialization of
// llvm::CodeGenPassBuilder<>. They are never used as CodeGenPassBuilder's
// real template arguments elsewhere, so this specialization cannot collide
// with any other instantiation.
struct LuthierPMWAccessTag {};
struct LuthierPMWAccessTMTag {};

template <>
class CodeGenPassBuilder<LuthierPMWAccessTag, LuthierPMWAccessTMTag> {
public:
  /// Construct a \c llvm::PassManagerWrapper wrapping \p MPM. Callable from
  /// anywhere; the friend relationship on this specialization is what makes
  /// PMW's private constructor reachable.
  static PassManagerWrapper make(ModulePassManager &MPM) {
    return PassManagerWrapper(MPM);
  }
};

} // namespace llvm

namespace luthier {

/// \returns a freshly constructed \c llvm::PassManagerWrapper wrapping
/// \p MPM. See the "PassManagerWrapper access shim" comment above for why
/// this indirection exists.
static inline llvm::PassManagerWrapper
makePassManagerWrapper(llvm::ModulePassManager &MPM) {
  return llvm::CodeGenPassBuilder<llvm::LuthierPMWAccessTag,
                                  llvm::LuthierPMWAccessTMTag>::make(MPM);
}

} // namespace luthier

namespace luthier {

//===----------------------------------------------------------------------===//
// LLVM_UPSTREAM_SYNC — clone of AMDGPUTargetMachine.cpp's AMDGPU codegen pass
// builder infrastructure.
//
// This block is a literal clone of the anonymous-namespace codegen pass
// builder machinery from llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp so
// that Luthier can construct an AMDGPUCodeGenPassBuilder directly and splice
// its own passes into a pipeline it owns end-to-end. The AMDGPU translation
// unit already registers every cl::opt<...> this code depends on; to avoid
// duplicate registration when this TU is linked alongside AMDGPU, we look
// each option up by name in LLVM's global CL argument map (see
// llvm::cl::getRegisteredOptions()) rather than declaring a fresh
// cl::opt<...> here. Everything else is a byte-for-byte transliteration; it
// will be pruned and adapted in a follow-up.
//===----------------------------------------------------------------------===//

namespace {

using namespace llvm;

//=== CL option accessors: read AMDGPU's already-registered opts. ==========//

template <typename OptT> OptT &lookupRegisteredOpt(StringRef Name) {
  auto *Raw = llvm::cl::getRegisteredOptions().lookup(Name);
  assert(Raw && "AMDGPU cl::opt not registered — has the AMDGPU target been "
                "initialized?");
  return *static_cast<OptT *>(Raw);
}

#define LUTHIER_AMDGPU_LOOKUP_BOOL(VarName, CLName)                            \
  inline cl::opt<bool> &VarName() {                                            \
    return lookupRegisteredOpt<cl::opt<bool>>(CLName);                         \
  }

#define LUTHIER_AMDGPU_LOOKUP_BOOL_EXT(VarName, CLName)                        \
  inline cl::opt<bool, true> &VarName() {                                      \
    return lookupRegisteredOpt<cl::opt<bool, true>>(CLName);                   \
  }

LUTHIER_AMDGPU_LOOKUP_BOOL(EnableEarlyIfConversion, "amdgpu-early-ifcvt")
LUTHIER_AMDGPU_LOOKUP_BOOL(OptExecMaskPreRA, "amdgpu-opt-exec-mask-pre-ra")
LUTHIER_AMDGPU_LOOKUP_BOOL(LowerCtorDtor, "amdgpu-lower-global-ctor-dtor")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableLoadStoreVectorizer,
                           "amdgpu-load-store-vectorizer")
LUTHIER_AMDGPU_LOOKUP_BOOL(ScalarizeGlobal, "amdgpu-scalarize-global-loads")
LUTHIER_AMDGPU_LOOKUP_BOOL(InternalizeSymbols, "amdgpu-internalize-symbols")
LUTHIER_AMDGPU_LOOKUP_BOOL(EarlyInlineAll, "amdgpu-early-inline-all")
LUTHIER_AMDGPU_LOOKUP_BOOL(RemoveIncompatibleFunctions,
                           "amdgpu-enable-remove-incompatible-functions")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableSDWAPeephole, "amdgpu-sdwa-peephole")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableDPPCombine, "amdgpu-dpp-combine")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableAMDGPUAliasAnalysis, "enable-amdgpu-aa")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableLibCallSimplify, "amdgpu-simplify-libcall")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableLowerKernelArguments,
                           "amdgpu-ir-lower-kernel-arguments")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableRegReassign, "amdgpu-reassign-regs")
LUTHIER_AMDGPU_LOOKUP_BOOL(OptVGPRLiveRange, "amdgpu-opt-vgpr-liverange")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableSIModeRegisterPass, "amdgpu-mode-register")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableInsertDelayAlu, "amdgpu-enable-delay-alu")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableVOPD, "amdgpu-enable-vopd")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableDCEInRA, "amdgpu-dce-in-ra")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableSetWavePriority, "amdgpu-set-wave-priority")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableScalarIRPasses, "amdgpu-scalar-ir-passes")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableLowerExecSync, "amdgpu-enable-lower-exec-sync")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableSwLowerLDS, "amdgpu-enable-sw-lower-lds")
LUTHIER_AMDGPU_LOOKUP_BOOL_EXT(EnableObjectLinking,
                               "amdgpu-enable-object-linking")
LUTHIER_AMDGPU_LOOKUP_BOOL_EXT(EnableLowerModuleLDS,
                               "amdgpu-enable-lower-module-lds")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnablePreRAOptimizations,
                           "amdgpu-enable-pre-ra-optimizations")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnablePromoteKernelArguments,
                           "amdgpu-enable-promote-kernel-arguments")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableImageIntrinsicOptimizer,
                           "amdgpu-enable-image-intrinsic-optimizer")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableLoopPrefetch, "amdgpu-loop-prefetch")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableRewritePartialRegUses,
                           "amdgpu-enable-rewrite-partial-reg-uses")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableHipStdPar, "amdgpu-enable-hipstdpar")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableAMDGPUAttributor, "amdgpu-attributor-enable")
LUTHIER_AMDGPU_LOOKUP_BOOL(NewRegBankSelect, "new-reg-bank-select")
LUTHIER_AMDGPU_LOOKUP_BOOL(HasClosedWorldAssumption,
                           "amdgpu-link-time-closed-world")
LUTHIER_AMDGPU_LOOKUP_BOOL(EnableUniformIntrinsicCombine,
                           "amdgpu-enable-uniform-intrinsic-combine")

inline cl::opt<ScanOptions> &AMDGPUAtomicOptimizerStrategy() {
  return lookupRegisteredOpt<cl::opt<ScanOptions>>(
      "amdgpu-atomic-optimizer-strategy");
}

inline cl::opt<std::string> &AMDGPUSchedStrategy() {
  return lookupRegisteredOpt<cl::opt<std::string>>("amdgpu-sched-strategy");
}

#undef LUTHIER_AMDGPU_LOOKUP_BOOL
#undef LUTHIER_AMDGPU_LOOKUP_BOOL_EXT

//===----------------------------------------------------------------------===//
// AMDGPU CodeGen Pass Builder interface (cloned).
//===----------------------------------------------------------------------===//

class AMDGPUCodeGenPassBuilder
    : public CodeGenPassBuilder<AMDGPUCodeGenPassBuilder, GCNTargetMachine> {
  using Base = CodeGenPassBuilder<AMDGPUCodeGenPassBuilder, GCNTargetMachine>;

public:
  AMDGPUCodeGenPassBuilder(GCNTargetMachine &TM,
                           const CGPassBuilderOption &Opts,
                           PassInstrumentationCallbacks *PIC);

  /// Luthier end-to-end instrumentation codegen pipeline builder.
  ///
  /// Populates \p PPM with the following, in order:
  ///   1. A \c RunOnInstrumentationModuleAdaptor wrapping a
  ///      \c ModulePassManager that carries the ISEL half of the AMDGPU
  ///      codegen pipeline (from \c addISelPasses through
  ///      \c addCoreISelPasses), so instruction selection runs on the
  ///      instrumentation module of the prototype.
  ///   2. \c IntrinsicMIRLoweringPass, added directly at the Prototype level
  ///      so it can see MIR from both modules of the prototype at once.
  ///   3. A second \c RunOnInstrumentationModuleAdaptor wrapping a
  ///      \c ModulePassManager that carries the machine-passes half of the
  ///      AMDGPU pipeline (from \c addMachinePasses through the machine
  ///      verifier), with Luthier's \c InjectedPayloadPEIPass registered via
  ///      \c CodeGenPassBuilder::insertPass so it is inserted immediately
  ///      after LLVM's stock \c PrologEpilogInserterPass.
  ///
  /// No asm-printer streams are taken here: assembly emission in Luthier is
  /// scheduled by a later PPM-level pass that first patches the
  /// instrumentation logic into the target module of the prototype.
  Error buildPipeline(PrototypePassManager &PPM) const;

  void addIRPasses(PassManagerWrapper &PMW) const;
  void addCodeGenPrepare(PassManagerWrapper &PMW) const;
  void addPreISel(PassManagerWrapper &PMW) const;
  void addILPOpts(PassManagerWrapper &PMWM) const;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) const;
  void addAsmPrinter(PassManagerWrapper &PMW) const;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) const;
  Error addInstSelector(PassManagerWrapper &PMW) const;
  void addPreRewrite(PassManagerWrapper &PMW) const;
  void addMachineSSAOptimization(PassManagerWrapper &PMW) const;
  void addPostRegAlloc(PassManagerWrapper &PMW) const;
  void addPreEmitPass(PassManagerWrapper &PMWM) const;
  void addPreEmitRegAlloc(PassManagerWrapper &PMW) const;
  Error addRegAssignmentFast(PassManagerWrapper &PMW) const;
  Error addRegAssignmentOptimized(PassManagerWrapper &PMW) const;
  void addPreRegAlloc(PassManagerWrapper &PMW) const;
  Error addFastRegAlloc(PassManagerWrapper &PMW) const;
  Error addOptimizedRegAlloc(PassManagerWrapper &PMW) const;
  void addPreSched2(PassManagerWrapper &PMW) const;
  void addPostBBSections(PassManagerWrapper &PMW) const;

private:
  Error validateRegAllocOptions() const;

public:
  /// Check if a pass is enabled given \p Opt option. The option always
  /// overrides defaults if explicitly used. Otherwise its default will be used
  /// given that a pass shall work at an optimization \p Level minimum.
  bool isPassEnabled(const cl::opt<bool> &Opt,
                     CodeGenOptLevel Level = CodeGenOptLevel::Default) const;
  void addEarlyCSEOrGVNPass(PassManagerWrapper &PMW) const;
  void addStraightLineScalarOptimizationPasses(PassManagerWrapper &PMW) const;
};

class SGPRRegisterRegAlloc : public RegisterRegAllocBase<SGPRRegisterRegAlloc> {
public:
  SGPRRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

class VGPRRegisterRegAlloc : public RegisterRegAllocBase<VGPRRegisterRegAlloc> {
public:
  VGPRRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

class WWMRegisterRegAlloc : public RegisterRegAllocBase<WWMRegisterRegAlloc> {
public:
  WWMRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

static bool onlyAllocateSGPRs(const TargetRegisterInfo &TRI,
                              const MachineRegisterInfo &MRI,
                              const Register Reg) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return static_cast<const SIRegisterInfo &>(TRI).isSGPRClass(RC);
}

static bool onlyAllocateVGPRs(const TargetRegisterInfo &TRI,
                              const MachineRegisterInfo &MRI,
                              const Register Reg) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return !static_cast<const SIRegisterInfo &>(TRI).isSGPRClass(RC);
}

static bool onlyAllocateWWMRegs(const TargetRegisterInfo &TRI,
                                const MachineRegisterInfo &MRI,
                                const Register Reg) {
  const SIMachineFunctionInfo *MFI =
      MRI.getMF().getInfo<SIMachineFunctionInfo>();
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return !static_cast<const SIRegisterInfo &>(TRI).isSGPRClass(RC) &&
         MFI->checkFlag(Reg, AMDGPU::VirtRegFlag::WWM_REG);
}

/// -{sgpr|wwm|vgpr}-regalloc=... command line option.
static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }

/// A dummy default pass factory indicates whether the register allocator is
/// overridden on the command line.
static llvm::once_flag InitializeDefaultSGPRRegisterAllocatorFlag;
static llvm::once_flag InitializeDefaultVGPRRegisterAllocatorFlag;
static llvm::once_flag InitializeDefaultWWMRegisterAllocatorFlag;

static SGPRRegisterRegAlloc
    defaultSGPRRegAlloc("default",
                        "pick SGPR register allocator based on -O option",
                        useDefaultRegisterAllocator);

// The AMDGPU translation unit owns the "sgpr-regalloc" / "vgpr-regalloc" /
// "wwm-regalloc" and their "-npm" siblings. We look up the already-registered
// options via the CL argument map instead of redeclaring them here.
inline cl::Option *SGPRRegAllocOpt() {
  return llvm::cl::getRegisteredOptions().lookup("sgpr-regalloc");
}
inline cl::Option *VGPRRegAllocOpt() {
  return llvm::cl::getRegisteredOptions().lookup("vgpr-regalloc");
}
inline cl::Option *WWMRegAllocOpt() {
  return llvm::cl::getRegisteredOptions().lookup("wwm-regalloc");
}

inline cl::opt<RegAllocType, false, RegAllocTypeParser> &SGPRRegAllocNPM() {
  return lookupRegisteredOpt<
      cl::opt<RegAllocType, false, RegAllocTypeParser>>("sgpr-regalloc-npm");
}
inline cl::opt<RegAllocType, false, RegAllocTypeParser> &VGPRRegAllocNPM() {
  return lookupRegisteredOpt<
      cl::opt<RegAllocType, false, RegAllocTypeParser>>("vgpr-regalloc-npm");
}
inline cl::opt<RegAllocType, false, RegAllocTypeParser> &WWMRegAllocNPM() {
  return lookupRegisteredOpt<
      cl::opt<RegAllocType, false, RegAllocTypeParser>>("wwm-regalloc-npm");
}

/// Check if the given RegAllocType is supported for AMDGPU NPM register
/// allocation. Only Fast and Greedy are supported; Basic and PBQP are not.
static Error checkRegAllocSupported(RegAllocType RAType, StringRef RegName) {
  if (RAType == RegAllocType::Basic || RAType == RegAllocType::PBQP) {
    return make_error<StringError>(
        Twine("unsupported register allocator '") +
            (RAType == RegAllocType::Basic ? "basic" : "pbqp") + "' for " +
            RegName + " registers",
        inconvertibleErrorCode());
  }
  return Error::success();
}

Error AMDGPUCodeGenPassBuilder::validateRegAllocOptions() const {
  // 1. Generic --regalloc-npm is not supported for AMDGPU.
  if (Opt.RegAlloc != RegAllocType::Unset) {
    return make_error<StringError>(
        "-regalloc-npm not supported for amdgcn. Use -sgpr-regalloc-npm, "
        "-vgpr-regalloc-npm, and -wwm-regalloc-npm",
        inconvertibleErrorCode());
  }

  // 2. Legacy PM regalloc options are not compatible with NPM.
  cl::Option *SGPR = SGPRRegAllocOpt();
  cl::Option *VGPR = VGPRRegAllocOpt();
  cl::Option *WWM = WWMRegAllocOpt();
  if ((SGPR && SGPR->getNumOccurrences() > 0) ||
      (VGPR && VGPR->getNumOccurrences() > 0) ||
      (WWM && WWM->getNumOccurrences() > 0)) {
    return make_error<StringError>(
        "-sgpr-regalloc, -vgpr-regalloc, and -wwm-regalloc are legacy PM "
        "options. Use -sgpr-regalloc-npm, -vgpr-regalloc-npm, and "
        "-wwm-regalloc-npm with the new pass manager",
        inconvertibleErrorCode());
  }

  // 3. Only Fast and Greedy allocators are supported for AMDGPU.
  if (auto Err = checkRegAllocSupported(SGPRRegAllocNPM(), "SGPR"))
    return Err;
  if (auto Err = checkRegAllocSupported(WWMRegAllocNPM(), "WWM"))
    return Err;
  if (auto Err = checkRegAllocSupported(VGPRRegAllocNPM(), "VGPR"))
    return Err;

  return Error::success();
}

static FunctionPass *createBasicSGPRRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateSGPRs);
}

static FunctionPass *createGreedySGPRRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateSGPRs);
}

static FunctionPass *createFastSGPRRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateSGPRs, false);
}

static FunctionPass *createBasicVGPRRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateVGPRs);
}

static FunctionPass *createGreedyVGPRRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateVGPRs);
}

static FunctionPass *createFastVGPRRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateVGPRs, true);
}

static FunctionPass *createBasicWWMRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateWWMRegs);
}

static FunctionPass *createGreedyWWMRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateWWMRegs);
}

static FunctionPass *createFastWWMRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateWWMRegs, false);
}

static SGPRRegisterRegAlloc basicRegAllocSGPR("basic",
                                              "basic register allocator",
                                              createBasicSGPRRegisterAllocator);
static SGPRRegisterRegAlloc
    greedyRegAllocSGPR("greedy", "greedy register allocator",
                       createGreedySGPRRegisterAllocator);

static SGPRRegisterRegAlloc fastRegAllocSGPR("fast", "fast register allocator",
                                             createFastSGPRRegisterAllocator);

static VGPRRegisterRegAlloc basicRegAllocVGPR("basic",
                                              "basic register allocator",
                                              createBasicVGPRRegisterAllocator);
static VGPRRegisterRegAlloc
    greedyRegAllocVGPR("greedy", "greedy register allocator",
                       createGreedyVGPRRegisterAllocator);

static VGPRRegisterRegAlloc fastRegAllocVGPR("fast", "fast register allocator",
                                             createFastVGPRRegisterAllocator);

static WWMRegisterRegAlloc basicRegAllocWWMReg("basic",
                                               "basic register allocator",
                                               createBasicWWMRegisterAllocator);
static WWMRegisterRegAlloc
    greedyRegAllocWWMReg("greedy", "greedy register allocator",
                         createGreedyWWMRegisterAllocator);
static WWMRegisterRegAlloc fastRegAllocWWMReg("fast", "fast register allocator",
                                              createFastWWMRegisterAllocator);

static bool isLTOPreLink(ThinOrFullLTOPhase Phase) {
  return Phase == ThinOrFullLTOPhase::FullLTOPreLink ||
         Phase == ThinOrFullLTOPhase::ThinLTOPreLink;
}

//===----------------------------------------------------------------------===//
// AMDGPUCodeGenPassBuilder — implementations (cloned).
//===----------------------------------------------------------------------===//

AMDGPUCodeGenPassBuilder::AMDGPUCodeGenPassBuilder(
    GCNTargetMachine &TM, const CGPassBuilderOption &Opts,
    PassInstrumentationCallbacks *PIC)
    : CodeGenPassBuilder(TM, Opts, PIC) {
  Opt.MISchedPostRA = true;
  Opt.RequiresCodeGenSCCOrder = true;
  // Exceptions and StackMaps are not supported, so these passes will never do
  // anything.
  // Garbage collection is not supported.
  disablePass<StackMapLivenessPass, FuncletLayoutPass, PatchableFunctionPass,
              ShadowStackGCLoweringPass, GCLoweringPass>();
}

//===----------------------------------------------------------------------===//
// Luthier buildPipeline: PPM-level end-to-end instrumentation pipeline.
//===----------------------------------------------------------------------===//

Error AMDGPUCodeGenPassBuilder::buildPipeline(PrototypePassManager &PPM) const {

  // Register the Luthier PEI hook up front, so it fires when the base
  // addMachinePasses eventually reaches PrologEpilogInserterPass on the
  // machine-side wrapper below.
  insertPass<PrologEpilogInserterPass>(InjectedPayloadPEIPass());
  insertPass<SIPreAllocateWWMRegsPass>(SVAPhysVGPRPinPass());

  // ---- Stage 1: ISEL half of the AMDGPU codegen pipeline on the IModule. ---
  //
  // Mirrors the front portion of CodeGenPassBuilder::buildPipeline (the
  // module-analysis require passes, addISelPasses, and addCoreISelPasses),
  // but writes into a scratch ModulePassManager which is then wrapped as a
  // single Prototype-level pass that runs over the instrumentation module.
  {
    ModulePassManager ISelMPM;
    PassManagerWrapper PMW = luthier::makePassManagerWrapper(ISelMPM);

    addModulePass(RequireAnalysisPass<MachineModuleAnalysis, Module>(), PMW,
                  /*Force=*/true);
    addModulePass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>(), PMW,
                  /*Force=*/true);
    addModulePass(RequireAnalysisPass<CollectorMetadataAnalysis, Module>(), PMW,
                  /*Force=*/true);
    addModulePass(RequireAnalysisPass<RuntimeLibraryAnalysis, Module>(), PMW,
                  /*Force=*/true);
    addModulePass(RequireAnalysisPass<LibcallLoweringModuleAnalysis, Module>(),
                  PMW,
                  /*Force=*/true);

    addISelPasses(PMW);
    flushFPMsToMPM(PMW);

    if (auto Err = addCoreISelPasses(PMW))
      return std::move(Err);
    flushFPMsToMPM(PMW);

    PPM.addPass(createRunOnInstrumentationModuleAdaptor(std::move(ISelMPM)));
  }

  // ---- Stage 2: MIR-intrinsic lowering at Prototype level. -----------------
  //
  // Intrinsic lowering runs at the Prototype level because it may inspect
  // both the target and instrumentation modules together and cross-reference
  // MachineFunctions.
  PPM.addPass(IntrinsicMIRLoweringPass());

  PPM.addPass(llvm::RequireAnalysisPass<IPPredicatedLivenessAnalysis,
                                      Prototype, PrototypeAnalysisManager>());
  /// TODO: Add other required analysis here
  PPM.addPass(InjectedPayloadPreserveLiveRegsPass());

  // Stage 3 runs SVAPhysVGPRPinPass and InjectedPayloadPEIPass, which are
  // MachineFunction passes and so can only read Prototype-level analyses out of
  // the cache. Materialize what they need here rather than earlier: every
  // Prototype pass above reports PreservedAnalyses::none(), which drops any
  // Prototype-level result computed before it.
  PPM.addPass(llvm::RequireAnalysisPass<IPPredCFGAnalysis, Prototype,
                                        PrototypeAnalysisManager>());
  PPM.addPass(llvm::RequireAnalysisPass<IPPredicatedLivenessAnalysis,
                                        Prototype, PrototypeAnalysisManager>());
  PPM.addPass(llvm::RequireAnalysisPass<SVStorageAndLoadLocationsAnalysis,
                                        Prototype, PrototypeAnalysisManager>());
  PPM.addPass(llvm::RequireAnalysisPass<StateValueArraySpecsAnalysis, Prototype,
                                        PrototypeAnalysisManager>());

  // ---- Stage 3: machine-passes half of the AMDGPU codegen pipeline. --------
  //
  // Mirrors the back portion of CodeGenPassBuilder::buildPipeline (from
  // addMachinePasses through the MachineVerifier), again wrapped as a
  // single Prototype-level pass over the instrumentation module. The
  // InjectedPayloadPEIPass registered at the top of this function is spliced
  // in immediately after PrologEpilogInserterPass by the AfterCallbacks
  // hook installed via CodeGenPassBuilder::insertPass.
  //
  // No asm-printer / PrintMIR / final MachineFunction free is done here —
  // asm printing in Luthier is scheduled by a later PPM-level pass that
  // patches the instrumentation logic into the target module first.
  {
    ModulePassManager MachineMPM;
    PassManagerWrapper PMW = luthier::makePassManagerWrapper(MachineMPM);

    if (auto Err = addMachinePasses(PMW))
      return std::move(Err);

    if (!Opt.DisableVerify)
      addMachineFunctionPass(MachineVerifierPass(), PMW);

    flushFPMsToMPM(PMW);

    PPM.addPass(createRunOnInstrumentationModuleAdaptor(std::move(MachineMPM)));
  }

  return Error::success();
}

void AMDGPUCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) const {
  if (RemoveIncompatibleFunctions() && TM.getTargetTriple().isAMDGCN()) {
    flushFPMsToMPM(PMW);
    addModulePass(AMDGPURemoveIncompatibleFunctionsPass(TM), PMW);
  }

  flushFPMsToMPM(PMW);

  if (TM.getTargetTriple().isAMDGCN())
    addModulePass(AMDGPUPrintfRuntimeBindingPass(), PMW);

  if (LowerCtorDtor())
    addModulePass(AMDGPUCtorDtorLoweringPass(), PMW);

  if (isPassEnabled(EnableImageIntrinsicOptimizer()))
    addFunctionPass(AMDGPUImageIntrinsicOptimizerPass(TM), PMW);

  if (EnableUniformIntrinsicCombine())
    addFunctionPass(AMDGPUUniformIntrinsicCombinePass(), PMW);
  // This can be disabled by passing ::Disable here or on the command line
  // with --expand-variadics-override=disable.
  flushFPMsToMPM(PMW);
  addModulePass(ExpandVariadicsPass(ExpandVariadicsMode::Lowering), PMW);

  addModulePass(AMDGPUAlwaysInlinePass(), PMW);
  addModulePass(AlwaysInlinerPass(), PMW);

  addModulePass(AMDGPUExportKernelRuntimeHandlesPass(), PMW);

  if (EnableLowerExecSync())
    addModulePass(AMDGPULowerExecSyncPass(), PMW);

  if (EnableSwLowerLDS())
    addModulePass(AMDGPUSwLowerLDSPass(TM), PMW);

  // Runs before PromoteAlloca so the latter can account for function uses
  if (EnableLowerModuleLDS())
    addModulePass(AMDGPULowerModuleLDSPass(TM), PMW);

  // Run atomic optimizer before Atomic Expand
  if (TM.getOptLevel() >= CodeGenOptLevel::Less &&
      (AMDGPUAtomicOptimizerStrategy() != ScanOptions::None))
    addFunctionPass(
        AMDGPUAtomicOptimizerPass(TM, AMDGPUAtomicOptimizerStrategy()), PMW);

  addFunctionPass(AtomicExpandPass(TM), PMW);

  if (TM.getOptLevel() > CodeGenOptLevel::None) {
    addFunctionPass(AMDGPUPromoteAllocaPass(TM), PMW);
    if (isPassEnabled(EnableScalarIRPasses()))
      addStraightLineScalarOptimizationPasses(PMW);

    // TODO: Handle EnableAMDGPUAliasAnalysis

    // TODO: May want to move later or split into an early and late one.
    addFunctionPass(AMDGPUCodeGenPreparePass(TM), PMW);

    // Try to hoist loop invariant parts of divisions AMDGPUCodeGenPrepare may
    // have expanded.
    if (TM.getOptLevel() > CodeGenOptLevel::Less) {
      addFunctionPass(createFunctionToLoopPassAdaptor(LICMPass(LICMOptions()),
                                                      /*UseMemorySSA=*/true),
                      PMW);
    }
  }

  Base::addIRPasses(PMW);

  // EarlyCSE is not always strong enough to clean up what LSR produces. For
  // example, GVN can combine
  //
  //   %0 = add %a, %b
  //   %1 = add %b, %a
  //
  // and
  //
  //   %0 = shl nsw %a, 2
  //   %1 = shl %a, 2
  //
  // but EarlyCSE can do neither of them.
  if (isPassEnabled(EnableScalarIRPasses()))
    addEarlyCSEOrGVNPass(PMW);
}

void AMDGPUCodeGenPassBuilder::addCodeGenPrepare(
    PassManagerWrapper &PMW) const {
  if (TM.getOptLevel() > CodeGenOptLevel::None) {
    flushFPMsToMPM(PMW);
    addModulePass(AMDGPUPreloadKernelArgumentsPass(TM), PMW);
  }

  if (EnableLowerKernelArguments())
    addFunctionPass(AMDGPULowerKernelArgumentsPass(TM), PMW);

  Base::addCodeGenPrepare(PMW);

  if (isPassEnabled(EnableLoadStoreVectorizer()))
    addFunctionPass(LoadStoreVectorizerPass(), PMW);

  // This lowering has been placed after codegenprepare to take advantage of
  // address mode matching (which is why it isn't put with the LDS lowerings).
  // It could be placed anywhere before uniformity annotations (an analysis
  // that it changes by splitting up fat pointers into their components)
  // but has been put before switch lowering and CFG flattening so that those
  // passes can run on the more optimized control flow this pass creates in
  // many cases.
  flushFPMsToMPM(PMW);
  addModulePass(AMDGPULowerBufferFatPointersPass(TM), PMW);
  flushFPMsToMPM(PMW);
  requireCGSCCOrder(PMW);

  addModulePass(AMDGPULowerIntrinsicsPass(TM), PMW);

  // LowerSwitch pass may introduce unreachable blocks that can cause unexpected
  // behavior for subsequent passes. Placing it here seems better that these
  // blocks would get cleaned up by UnreachableBlockElim inserted next in the
  // pass flow.
  addFunctionPass(LowerSwitchPass(), PMW);
}

void AMDGPUCodeGenPassBuilder::addPreISel(PassManagerWrapper &PMW) const {

  if (TM.getOptLevel() > CodeGenOptLevel::None) {
    addFunctionPass(FlattenCFGPass(), PMW);
    addFunctionPass(SinkingPass(), PMW);
    addFunctionPass(AMDGPULateCodeGenPreparePass(TM), PMW);
  }

  // Merge divergent exit nodes. StructurizeCFG won't recognize the multi-exit
  // regions formed by them.

  addFunctionPass(AMDGPUUnifyDivergentExitNodesPass(), PMW);
  addFunctionPass(FixIrreduciblePass(), PMW);
  addFunctionPass(UnifyLoopExitsPass(), PMW);
  addFunctionPass(StructurizeCFGPass(/*SkipUniformRegions=*/false), PMW);

  addFunctionPass(AMDGPUAnnotateUniformValuesPass(), PMW);

  addFunctionPass(SIAnnotateControlFlowPass(TM), PMW);

  // TODO: Move this right after structurizeCFG to avoid extra divergence
  // analysis. This depends on stopping SIAnnotateControlFlow from making
  // control flow modifications.
  addFunctionPass(AMDGPURewriteUndefForPHIPass(), PMW);

  if (!getCGPassBuilderOption().EnableGlobalISelOption ||
      !isGlobalISelAbortEnabled() || !NewRegBankSelect())
    addFunctionPass(LCSSAPass(), PMW);

  if (TM.getOptLevel() > CodeGenOptLevel::Less) {
    flushFPMsToMPM(PMW);
    addModulePass(AMDGPUPerfHintAnalysisPass(TM), PMW);
  }

  // FIXME: Why isn't this queried as required from AMDGPUISelDAGToDAG, and why
  // isn't this in addInstSelector?
  addFunctionPass(RequireAnalysisPass<UniformityInfoAnalysis, Function>(), PMW,
                  /*Force=*/true);
}

void AMDGPUCodeGenPassBuilder::addILPOpts(PassManagerWrapper &PMW) const {
  if (EnableEarlyIfConversion())
    addMachineFunctionPass(EarlyIfConverterPass(), PMW);

  Base::addILPOpts(PMW);
}

Error AMDGPUCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(AMDGPUISelDAGToDAGPass(TM), PMW);
  addMachineFunctionPass(SIFixSGPRCopiesPass(), PMW);
  addMachineFunctionPass(SILowerI1CopiesPass(), PMW);
  return Error::success();
}

void AMDGPUCodeGenPassBuilder::addPreRewrite(PassManagerWrapper &PMW) const {
  if (EnableRegReassign()) {
    addMachineFunctionPass(GCNNSAReassignPass(), PMW);
  }

  addMachineFunctionPass(AMDGPURewriteAGPRCopyMFMAPass(), PMW);
}

void AMDGPUCodeGenPassBuilder::addMachineSSAOptimization(
    PassManagerWrapper &PMW) const {
  Base::addMachineSSAOptimization(PMW);

  addMachineFunctionPass(SIFoldOperandsPass(), PMW);
  if (EnableDPPCombine()) {
    addMachineFunctionPass(GCNDPPCombinePass(), PMW);
  }
  addMachineFunctionPass(SILoadStoreOptimizerPass(), PMW);
  if (isPassEnabled(EnableSDWAPeephole())) {
    addMachineFunctionPass(SIPeepholeSDWAPass(), PMW);
    addMachineFunctionPass(EarlyMachineLICMPass(), PMW);
    addMachineFunctionPass(MachineCSEPass(), PMW);
    addMachineFunctionPass(SIFoldOperandsPass(), PMW);
  }
  addMachineFunctionPass(DeadMachineInstructionElimPass(), PMW);
  addMachineFunctionPass(SIShrinkInstructionsPass(), PMW);
}

Error AMDGPUCodeGenPassBuilder::addFastRegAlloc(PassManagerWrapper &PMW) const {
  insertPass<PHIEliminationPass>(SILowerControlFlowPass());

  insertPass<TwoAddressInstructionPass>(SIWholeQuadModePass());

  return Base::addFastRegAlloc(PMW);
}

Error AMDGPUCodeGenPassBuilder::addRegAssignmentFast(
    PassManagerWrapper &PMW) const {
  if (auto Err = validateRegAllocOptions())
    return Err;

  addMachineFunctionPass(GCNPreRALongBranchRegPass(), PMW);

  // SGPR allocation - default to fast at -O0.
  if (SGPRRegAllocNPM() == RegAllocType::Greedy)
    addMachineFunctionPass(RAGreedyPass({onlyAllocateSGPRs, "sgpr"}), PMW);
  else
    addMachineFunctionPass(RegAllocFastPass({onlyAllocateSGPRs, "sgpr", false}),
                           PMW);

  // Equivalent of PEI for SGPRs.
  addMachineFunctionPass(SILowerSGPRSpillsPass(), PMW);

  // To Allocate wwm registers used in whole quad mode operations (for shaders).
  addMachineFunctionPass(SIPreAllocateWWMRegsPass(), PMW);

  // WWM allocation - default to fast at -O0.
  if (WWMRegAllocNPM() == RegAllocType::Greedy)
    addMachineFunctionPass(RAGreedyPass({onlyAllocateWWMRegs, "wwm"}), PMW);
  else
    addMachineFunctionPass(
        RegAllocFastPass({onlyAllocateWWMRegs, "wwm", false}), PMW);

  addMachineFunctionPass(SILowerWWMCopiesPass(), PMW);
  addMachineFunctionPass(AMDGPUReserveWWMRegsPass(), PMW);

  // VGPR allocation - default to fast at -O0.
  if (VGPRRegAllocNPM() == RegAllocType::Greedy)
    addMachineFunctionPass(RAGreedyPass({onlyAllocateVGPRs, "vgpr"}), PMW);
  else
    addMachineFunctionPass(RegAllocFastPass({onlyAllocateVGPRs, "vgpr"}), PMW);

  return Error::success();
}

Error AMDGPUCodeGenPassBuilder::addOptimizedRegAlloc(
    PassManagerWrapper &PMW) const {
  if (EnableDCEInRA())
    insertPass<DetectDeadLanesPass>(DeadMachineInstructionElimPass());

  // FIXME: when an instruction has a Killed operand, and the instruction is
  // inside a bundle, seems only the BUNDLE instruction appears as the Kills of
  // the register in LiveVariables, this would trigger a failure in verifier,
  // we should fix it and enable the verifier.
  if (OptVGPRLiveRange())
    insertPass<RequireAnalysisPass<LiveVariablesAnalysis, MachineFunction>>(
        SIOptimizeVGPRLiveRangePass());

  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.
  insertPass<PHIEliminationPass>(SILowerControlFlowPass());

  if (EnableRewritePartialRegUses())
    insertPass<RenameIndependentSubregsPass>(GCNRewritePartialRegUsesPass());

  if (isPassEnabled(EnablePreRAOptimizations()))
    insertPass<MachineSchedulerPass>(GCNPreRAOptimizationsPass());

  // Allow the scheduler to run before SIWholeQuadMode inserts exec manipulation
  // instructions that cause scheduling barriers.
  insertPass<MachineSchedulerPass>(SIWholeQuadModePass());

  if (OptExecMaskPreRA())
    insertPass<MachineSchedulerPass>(SIOptimizeExecMaskingPreRAPass());

  // This is not an essential optimization and it has a noticeable impact on
  // compilation time, so we only enable it from O2.
  if (TM.getOptLevel() > CodeGenOptLevel::Less)
    insertPass<MachineSchedulerPass>(SIFormMemoryClausesPass());

  return Base::addOptimizedRegAlloc(PMW);
}

void AMDGPUCodeGenPassBuilder::addPreRegAlloc(PassManagerWrapper &PMW) const {
  if (getOptLevel() != CodeGenOptLevel::None)
    addMachineFunctionPass(AMDGPUPrepareAGPRAllocPass(), PMW);
}

Error AMDGPUCodeGenPassBuilder::addRegAssignmentOptimized(
    PassManagerWrapper &PMW) const {
  if (auto Err = validateRegAllocOptions())
    return Err;

  addMachineFunctionPass(GCNPreRALongBranchRegPass(), PMW);

  // SGPR allocation - default to greedy at -O1 and above.
  if (SGPRRegAllocNPM() == RegAllocType::Fast)
    addMachineFunctionPass(RegAllocFastPass({onlyAllocateSGPRs, "sgpr", false}),
                           PMW);
  else
    addMachineFunctionPass(RAGreedyPass({onlyAllocateSGPRs, "sgpr"}), PMW);

  // Commit allocated register changes. This is mostly necessary because too
  // many things rely on the use lists of the physical registers, such as the
  // verifier. This is only necessary with allocators which use LiveIntervals,
  // since FastRegAlloc does the replacements itself.
  addMachineFunctionPass(VirtRegRewriterPass(false), PMW);

  // At this point, the sgpr-regalloc has been done and it is good to have the
  // stack slot coloring to try to optimize the SGPR spill stack indices before
  // attempting the custom SGPR spill lowering.
  addMachineFunctionPass(StackSlotColoringPass(), PMW);

  // Equivalent of PEI for SGPRs.
  addMachineFunctionPass(SILowerSGPRSpillsPass(), PMW);

  // To Allocate wwm registers used in whole quad mode operations (for shaders).
  addMachineFunctionPass(SIPreAllocateWWMRegsPass(), PMW);

  // WWM allocation - default to greedy at -O1 and above.
  if (WWMRegAllocNPM() == RegAllocType::Fast)
    addMachineFunctionPass(
        RegAllocFastPass({onlyAllocateWWMRegs, "wwm", false}), PMW);
  else
    addMachineFunctionPass(RAGreedyPass({onlyAllocateWWMRegs, "wwm"}), PMW);
  addMachineFunctionPass(SILowerWWMCopiesPass(), PMW);
  addMachineFunctionPass(VirtRegRewriterPass(false), PMW);
  addMachineFunctionPass(AMDGPUReserveWWMRegsPass(), PMW);

  // VGPR allocation - default to greedy at -O1 and above.
  if (VGPRRegAllocNPM() == RegAllocType::Fast)
    addMachineFunctionPass(RegAllocFastPass({onlyAllocateVGPRs, "vgpr"}), PMW);
  else
    addMachineFunctionPass(RAGreedyPass({onlyAllocateVGPRs, "vgpr"}), PMW);

  addPreRewrite(PMW);
  addMachineFunctionPass(VirtRegRewriterPass(true), PMW);

  addMachineFunctionPass(AMDGPUMarkLastScratchLoadPass(), PMW);
  return Error::success();
}

void AMDGPUCodeGenPassBuilder::addPostRegAlloc(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(SIFixVGPRCopiesPass(), PMW);
  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addMachineFunctionPass(SIOptimizeExecMaskingPass(), PMW);
  Base::addPostRegAlloc(PMW);
}

void AMDGPUCodeGenPassBuilder::addPreSched2(PassManagerWrapper &PMW) const {
  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addMachineFunctionPass(SIShrinkInstructionsPass(), PMW);
  addMachineFunctionPass(SIPostRABundlerPass(), PMW);
}

void AMDGPUCodeGenPassBuilder::addPostBBSections(
    PassManagerWrapper &PMW) const {
  // We run this later to avoid passes like livedebugvalues and BBSections
  // having to deal with the apparent multi-entry functions we may generate.
  addMachineFunctionPass(AMDGPUPreloadKernArgPrologPass(), PMW);
}

void AMDGPUCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) const {
  if (isPassEnabled(EnableVOPD(), CodeGenOptLevel::Less)) {
    addMachineFunctionPass(GCNCreateVOPDPass(), PMW);
  }

  addMachineFunctionPass(SIMemoryLegalizerPass(), PMW);
  addMachineFunctionPass(SIInsertWaitcntsPass(), PMW);

  addMachineFunctionPass(SIModeRegisterPass(), PMW);

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addMachineFunctionPass(SIInsertHardClausesPass(), PMW);

  addMachineFunctionPass(SILateBranchLoweringPass(), PMW);

  if (isPassEnabled(EnableSetWavePriority(), CodeGenOptLevel::Less))
    addMachineFunctionPass(AMDGPUSetWavePriorityPass(), PMW);

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addMachineFunctionPass(SIPreEmitPeepholePass(), PMW);

  // The hazard recognizer that runs as part of the post-ra scheduler does not
  // guarantee to be able handle all hazards correctly. This is because if there
  // are multiple scheduling regions in a basic block, the regions are scheduled
  // bottom up, so when we begin to schedule a region we don't know what
  // instructions were emitted directly before it.
  //
  // Here we add a stand-alone hazard recognizer pass which can handle all
  // cases.
  addMachineFunctionPass(PostRAHazardRecognizerPass(), PMW);
  addMachineFunctionPass(AMDGPUWaitSGPRHazardsPass(), PMW);
  addMachineFunctionPass(AMDGPULowerVGPREncodingPass(), PMW);

  if (isPassEnabled(EnableInsertDelayAlu(), CodeGenOptLevel::Less)) {
    addMachineFunctionPass(AMDGPUInsertDelayAluPass(), PMW);
  }

  addMachineFunctionPass(BranchRelaxationPass(), PMW);
}

bool AMDGPUCodeGenPassBuilder::isPassEnabled(const cl::opt<bool> &Opt,
                                             CodeGenOptLevel Level) const {
  if (Opt.getNumOccurrences())
    return Opt;
  if (TM.getOptLevel() < Level)
    return false;
  return Opt;
}

void AMDGPUCodeGenPassBuilder::addEarlyCSEOrGVNPass(
    PassManagerWrapper &PMW) const {
  if (TM.getOptLevel() == CodeGenOptLevel::Aggressive)
    addFunctionPass(GVNPass(), PMW);
  else
    addFunctionPass(EarlyCSEPass(), PMW);
}

void AMDGPUCodeGenPassBuilder::addStraightLineScalarOptimizationPasses(
    PassManagerWrapper &PMW) const {
  if (isPassEnabled(EnableLoopPrefetch(), CodeGenOptLevel::Aggressive))
    addFunctionPass(LoopDataPrefetchPass(), PMW);

  addFunctionPass(SeparateConstOffsetFromGEPPass(), PMW);

  // ReassociateGEPs exposes more opportunities for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addFunctionPass(StraightLineStrengthReducePass(), PMW);

  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse.
  addEarlyCSEOrGVNPass(PMW);

  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addFunctionPass(NaryReassociatePass(), PMW);

  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addFunctionPass(EarlyCSEPass(), PMW);
}

} // namespace

//===----------------------------------------------------------------------===//
// InstrumentationPassBuilder — construction / destruction.
//===----------------------------------------------------------------------===//

InstrumentationPassBuilder::InstrumentationPassBuilder(
    llvm::TargetMachine &TM, llvm::PipelineTuningOptions PTO,
    std::optional<llvm::PGOOptions> PGOOpt, PassInstrumentationCallbacks *PIC)
    : TM(TM), PIC(PIC),
      PB(std::make_unique<PassBuilder>(&TM, PTO, PGOOpt, PIC)) {
  // Module passes — parsed via -passes=<...>.
  PB->registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager &MPM,
         llvm::ArrayRef<PassBuilder::PipelineElement>) {
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME) {                                                          \
    MPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"
        return false;
      });

  // Machine-function passes.
  PB->registerPipelineParsingCallback(
      [](StringRef Name, llvm::MachineFunctionPassManager &MFPM,
         llvm::ArrayRef<PassBuilder::PipelineElement>) {
#define MACHINE_FUNCTION_PASS(NAME, CREATE_PASS)                               \
  if (Name == NAME) {                                                          \
    MFPM.addPass(CREATE_PASS);                                                 \
    return true;                                                               \
  }
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"
        return false;
      });
}

InstrumentationPassBuilder::~InstrumentationPassBuilder() = default;

//===----------------------------------------------------------------------===//
// Cross-level proxy registration.
//===----------------------------------------------------------------------===//

void InstrumentationPassBuilder::crossRegisterProxies(
    PrototypeAnalysisManager &PAM, const ModuleAnalysisManagers &Target,
    const ModuleAnalysisManagers &Instrumentation) {
  // PassManager<Prototype>::run requires PassInstrumentationAnalysis on the
  // Prototype analysis manager. It must NOT be the PIC held by the wrapped
  // llvm::PassBuilder: that one carries StandardInstrumentations' callbacks,
  // which cannot name a Prototype IR unit and abort on the first pass. See
  // PrototypePIC's declaration for the full rationale. The adaptors instrument
  // the module passes they wrap using the module-level PassInstrumentation
  // instead (see runModulePass in Prototype.cpp), so wrapped passes keep the
  // real callbacks.
  PAM.registerPass(
      [this] { return llvm::PassInstrumentationAnalysis(&PrototypePIC); });

  // Route Prototype-level passes into `--print-before-all` / `--print-after-all`
  // / `--print-{before,after}=<pass>`. LLVM's StandardInstrumentations can't
  // name a Prototype IR unit (see PrototypePIC's declaration), so its
  // PrintIRInstrumentation never registers callbacks against PrototypePIC.
  auto DumpPrototype = [&Target, &Instrumentation](llvm::StringRef Header,
                                                   llvm::StringRef PassID,
                                                   llvm::Any IR) {
    const auto *const *PPtr = llvm::any_cast<const Prototype *>(&IR);
    if (!PPtr)
      return;
    const Prototype &P = **PPtr;
    llvm::dbgs() << Header << " " << PassID << " on prototype '"
                 << P.getName() << "' ***\n";
    P.print(llvm::dbgs(), Target.FAM, Instrumentation.FAM);
  };
  PrototypePIC.registerBeforeNonSkippedPassCallback(
      [DumpPrototype](llvm::StringRef PassID, llvm::Any IR) {
        if (llvm::shouldPrintBeforePass(PassID))
          DumpPrototype("*** IR Dump Before", PassID, IR);
      });
  PrototypePIC.registerAfterPassCallback(
      [DumpPrototype](llvm::StringRef PassID, llvm::Any IR,
                      const llvm::PreservedAnalyses &) {
        if (llvm::shouldPrintAfterPass(PassID))
          DumpPrototype("*** IR Dump After", PassID, IR);
      });

  // One inner proxy per IR level per module, so that a Prototype-level pass can
  // name exactly the managers it disturbed.
  PAM.registerPass(
      [&] { return TargetModuleAnalysisManagerPrototypeProxy(Target.MAM); });
  PAM.registerPass(
      [&] { return TargetFunctionAnalysisManagerPrototypeProxy(Target.FAM); });
  PAM.registerPass([&] {
    return TargetMachineFunctionAnalysisManagerPrototypeProxy(Target.MFAM);
  });
  PAM.registerPass([&] {
    return IModuleAnalysisManagerPrototypeProxy(Instrumentation.MAM);
  });
  PAM.registerPass([&] {
    return IModuleFunctionAnalysisManagerPrototypeProxy(Instrumentation.FAM);
  });
  PAM.registerPass([&] {
    return IModuleMachineFunctionAnalysisManagerPrototypeProxy(
        Instrumentation.MFAM);
  });

  // Each module's inner levels are wired up separately. That is what keeps the
  // wholesale InnerAM->clear() in LLVM's per-module proxies (PassManager.cpp,
  // FunctionAnalysisManagerModuleProxy::Result::invalidate) confined to the
  // managers of the module whose pass triggered it: a nested
  // llvm::ModulePassManager re-invalidates after every pass it runs, so a pass
  // over the instrumentation module would otherwise wipe the target module's
  // cached MachineFunctionAnalysis results — and the MachineFunctions they own.
  for (const ModuleAnalysisManagers &AMs : {Target, Instrumentation}) {
    AMs.MAM.registerPass(
        [&] { return PrototypeAnalysisManagerModuleProxy(PAM); });
    AMs.FAM.registerPass(
        [&] { return PrototypeAnalysisManagerFunctionProxy(PAM); });
    AMs.MFAM.registerPass(
        [&] { return PrototypeAnalysisManagerMachineFunctionProxy(PAM); });

    PB->crossRegisterProxies(AMs.LAM, AMs.FAM, AMs.CGAM, AMs.MAM, &AMs.MFAM);
  }
}

void InstrumentationPassBuilder::registerAnalyses(
    const ModuleAnalysisManagers &AMs) {
  registerModuleAnalyses(AMs.MAM);
  registerCGSCCAnalyses(AMs.CGAM);
  registerFunctionAnalyses(AMs.FAM);
  registerLoopAnalyses(AMs.LAM);
  registerMachineFunctionAnalyses(AMs.MFAM);
}

void InstrumentationPassBuilder::registerPrototypeAnalyses(
    PrototypeAnalysisManager &PAM) {
#define PROTOTYPE_ANALYSIS(NAME, CREATE_PASS)                                  \
  PAM.registerPass([&] { return CREATE_PASS; });
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"
}

void InstrumentationPassBuilder::registerModuleAnalyses(
    llvm::ModuleAnalysisManager &MAM) {
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  MAM.registerPass([&] { return CREATE_PASS; });
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"

  PB->registerModuleAnalyses(MAM);
}

void InstrumentationPassBuilder::registerCGSCCAnalyses(
    llvm::CGSCCAnalysisManager &CGAM) {
  PB->registerCGSCCAnalyses(CGAM);
}

void InstrumentationPassBuilder::registerFunctionAnalyses(
    llvm::FunctionAnalysisManager &FAM) {
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  FAM.registerPass([&] { return CREATE_PASS; });
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"

  // Must forward to the wrapped builder, exactly like registerModuleAnalyses /
  // registerCGSCCAnalyses / registerLoopAnalyses do. Without this none of the
  // stock LLVM function analyses — MachineFunctionAnalysis among them — are
  // registered, and the first pass to query one trips "Analysis passes must be
  // registered prior to being queried!".
  PB->registerFunctionAnalyses(FAM);
}

void InstrumentationPassBuilder::registerLoopAnalyses(
    llvm::LoopAnalysisManager &LAM) {
  PB->registerLoopAnalyses(LAM);
}

void InstrumentationPassBuilder::registerMachineFunctionAnalyses(
    llvm::MachineFunctionAnalysisManager &MFAM) {
#define MACHINE_FUNCTION_ANALYSIS(NAME, CREATE_PASS)                           \
  MFAM.registerPass([&] { return CREATE_PASS; });
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"

  // See registerFunctionAnalyses: the stock LLVM machine-function analyses have
  // to be registered too.
  PB->registerMachineFunctionAnalyses(MFAM);
}

//===----------------------------------------------------------------------===//
// parsePipeline — target(...) / instrumentation(...) grammar.
//===----------------------------------------------------------------------===//

Error InstrumentationPassBuilder::parsePipeline(PrototypePassManager &PPM,
                                                StringRef PipelineText) {
  StringRef Remaining = PipelineText.trim();

  while (!Remaining.empty()) {
    bool IsTarget = Remaining.consume_front("target(");
    bool IsInstrumentation =
        !IsTarget && Remaining.consume_front("instrumentation(");

    if (!IsTarget && !IsInstrumentation) {
      // A bare name at the top level denotes a Prototype-level pass: it runs
      // over the whole prototype rather than over one of its two modules, so
      // there is no target(...) / instrumentation(...) wrapper to put it in.
      StringRef Name = Remaining.take_until([](char C) { return C == ','; });
      Remaining = Remaining.drop_front(Name.size()).ltrim();
      if (Remaining.consume_front(","))
        Remaining = Remaining.ltrim();
      Name = Name.trim();

      bool Found = false;
#define PROTOTYPE_PASS(NAME, CREATE_PASS)                                      \
  if (!Found && Name == NAME) {                                                \
    PPM.addPass(CREATE_PASS);                                                  \
    Found = true;                                                              \
  }
#include "luthier/ToolCodeGen/LuthierPassRegistry.def"

      if (Found)
        continue;

      // Give plugins a shot at the bare name before giving up. They are handed
      // the name as the "inner text" with no enclosing block, which the
      // IsTarget=false / IsInstrumentation=false pair below distinguishes from
      // a real instrumentation(...) block.
      for (auto &Cb : ParseCallbacks) {
        if (Cb(Name, PPM, /*IsTarget=*/false)) {
          Found = true;
          break;
        }
      }
      if (Found)
        continue;

      return LUTHIER_MAKE_GENERIC_ERROR(
          ("unknown Prototype-level pass name '" + Name +
           "' at top level of -passes (expected a Prototype pass, "
           "'target(...)', or 'instrumentation(...)')")
              .str());
    }

    size_t Depth = 1;
    size_t Pos = 0;
    while (Pos < Remaining.size() && Depth > 0) {
      if (Remaining[Pos] == '(')
        Depth++;
      else if (Remaining[Pos] == ')')
        Depth--;
      if (Depth > 0)
        Pos++;
    }

    if (Depth != 0) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          "unmatched parentheses in Prototype pass pipeline");
    }

    StringRef InnerText = Remaining.substr(0, Pos);
    Remaining = Remaining.substr(Pos + 1).ltrim();
    if (Remaining.consume_front(","))
      Remaining = Remaining.ltrim();

    bool Handled = false;
    for (auto &Cb : ParseCallbacks) {
      if (Cb(InnerText, PPM, IsTarget)) {
        Handled = true;
        break;
      }
    }
    if (Handled)
      continue;

    ModulePassManager InnerMPM;
    if (auto Err = PB->parsePassPipeline(InnerMPM, InnerText))
      return Err;

    if (IsTarget)
      PPM.addPass(createRunOnTargetModuleAdaptor(std::move(InnerMPM)));
    else
      PPM.addPass(createRunOnInstrumentationModuleAdaptor(std::move(InnerMPM)));
  }

  return Error::success();
}

//===----------------------------------------------------------------------===//
// Top-level pipeline builder.
//===----------------------------------------------------------------------===//

Error InstrumentationPassBuilder::buildInstrumentationPipeline(
    PrototypePassManager &PPM, PreInstrumentationCallback InstCallback,
    PatchPCUsagesPass::TargetAddressHostResolverFn PatchPCUsagesHostCallback,
    llvm::OptimizationLevel Level, llvm::CodeGenFileType FileType,
    llvm::CGPassBuilderOption &CGPBO, llvm::raw_pwrite_stream *Out,
    llvm::PassInstrumentationCallbacks *PIC) {
  /// Invoke Pre-code discovery callbacks
  for (auto &CB : PreCodeDiscoveryCallBacks) {
    CB(PPM, Level);
  }
  /// Add the code discovery pass
  PPM.addPass(CodeDiscoveryPass());

  /// Ivoke pre-instrumentation callbacks
  for (auto &CB: PreInstrumentationCallbacks) {
    CB(PPM, Level);
  }

  /// Add the instrumentation passes
  InstCallback(PPM, Level);

  /// Run Patch-PC-Usages runs immediately after the tool's payloads are
  /// created
  {
    llvm::Error PatcherErr = llvm::Error::success();
    PPM.addPass(PatchPCUsagesPass(PatchPCUsagesHostCallback,
                                         PatcherErr));
    if (PatcherErr)
      return PatcherErr;
  }

  /// Invoke pre-IR optimization callbacks
  for (auto &CB : PreInstrumentationOptimizationCallbacks) {
    CB(PPM, Level);
  }
  addInstrumentationModulePass(PPM, PB->buildPerModuleDefaultPipeline(Level));

  for (auto &CB : PreInstrumentationISelCallbacks) {
    CB(PPM, Level);
  }

  addInstrumentationModulePass(
      PPM, llvm::createModuleToFunctionPassAdaptor(
               llvm::RequireAnalysisPass<InjectedPayloadSideEffectsAnalysis,
                                         llvm::Function>()));

  addInstrumentationModulePass(PPM, ProcessIntrinsicsAtIRLevelPass());

  for (auto &CB: PreInstrumentationCodeGenPassesCallbacks) {
    CB(PPM, Level);
  }

  AMDGPUCodeGenPassBuilder CGPB(static_cast<llvm::GCNTargetMachine &>(this->TM),
                                CGPBO, PIC);

  if (auto Err = CGPB.buildPipeline(PPM)) {
    return Err;
  }

  // Final target-module patch step.
  PPM.addPass(TargetModulePatcherPass());
  if (Out)
    addTargetModulePass(PPM, NewPMAsmPrinter(FileType, *Out, true));

  return Error::success();
}

} // namespace luthier
