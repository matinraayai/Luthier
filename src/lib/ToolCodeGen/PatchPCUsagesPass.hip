//===-- PatchPCUsagesPass.hip --------------------------------*- C++ -*-===//
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
///
/// \file
/// Implements both the host and device-side logic of
/// \c luthier::PatchPCUsagesPass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/PatchPCUsagesPass.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h"
#include "luthier/Intrinsic/Intrinsics.h"
#include <cstdint>
#include <hip/hip_runtime.h>

#ifndef __HIP_DEVICE_COMPILE__
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/EntryPoint.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessPass.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include "luthier/ToolCodeGen/PrototypeCallGraph.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include "luthier/ToolCodeGen/TraceFunctionTranslationAnalysis.h"
#include <SIInstrInfo.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#endif // __HIP_DEVICE_COMPILE__

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-patch-pc-usages"

//===----------------------------------------------------------------------===//
// Device side
//===----------------------------------------------------------------------===//

#if defined(__HIPCC__)


__attribute__((device, used))
luthier::PatchPCUsagesPass::EntryPointTraceFunctionEntry *
    luthier::PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMap;

__attribute__((device, used))
std::uint32_t luthier::PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMapSize;

__attribute__((device, used))
std::uint32_t
    luthier::PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMapMaxSize;

__attribute__((device, used))
std::uint32_t luthier::PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMapLock;

__attribute__((device, used, constant))
const luthier::PatchPCUsagesPass::EntryPointTraceFunctionEntry *
    luthier::PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMapSeed =
        nullptr;

__attribute__((device, used, constant))
const std::uint32_t
    luthier::PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMapSeedSize =
        0;

__attribute__((device, used, constructor))
void luthier::PatchPCUsagesPass::Parser::initEntryPointToTraceFunctionAddrMap() {
  using P = luthier::PatchPCUsagesPass::Parser;
  using TE = luthier::PatchPCUsagesPass::EntryPointTraceFunctionEntry;
  // Size the allocation to the codegen-time seed count + 5, so the host
  // callback can append 5 more entries before it has to reallocate.
  const std::uint32_t SeedN = P::EntryPointToTraceFunctionAddrMapSeedSize;

  // Two entry modes: a first-time init that allocates a fresh runtime
  // table sized to the seed + 5-entry slack, and an extend-into-place mode
  // that keeps whatever entries a prior constructor already installed and
  // appends this code object's seed to them. The pointer being non-null
  // (paired with a non-zero \c MaxSize ) is the "already initialised" gate:
  // it lets a later-loaded code object contribute its own trace-fn entries
  // without a "discover the initial entry point" step at load time — the
  // table just grows.
  if (P::EntryPointToTraceFunctionAddrMap != nullptr &&
      P::EntryPointToTraceFunctionAddrMapMaxSize > 0u) {
    const std::uint32_t OldSize = P::EntryPointToTraceFunctionAddrMapSize;
    const std::uint32_t Needed = OldSize + SeedN;
    if (Needed > P::EntryPointToTraceFunctionAddrMapMaxSize) {
      // Grow to \c Needed + 5-entry slack so subsequent extends /
      // host-callback appends have headroom.
      const std::uint32_t NewCapacity = Needed + 5u;
      TE *NewBuf = static_cast<TE *>(malloc(sizeof(TE) * NewCapacity));
      for (std::uint32_t I = 0; I < OldSize; ++I)
        NewBuf[I] = P::EntryPointToTraceFunctionAddrMap[I];
      free(P::EntryPointToTraceFunctionAddrMap);
      P::EntryPointToTraceFunctionAddrMap = NewBuf;
      P::EntryPointToTraceFunctionAddrMapMaxSize = NewCapacity;
    }
    for (std::uint32_t I = 0; I < SeedN; ++I)
      P::EntryPointToTraceFunctionAddrMap[OldSize + I] =
          P::EntryPointToTraceFunctionAddrMapSeed[I];
    P::EntryPointToTraceFunctionAddrMapSize = Needed;
    return;
  }

  // First-time init: size the allocation to the codegen-time seed count
  // + 5, so the host callback can append 5 more entries before it has to
  // reallocate.
  const std::uint32_t Capacity = SeedN + 5u;
  P::EntryPointToTraceFunctionAddrMap =
      static_cast<TE *>(malloc(sizeof(TE) * Capacity));
  // Copy the constant-AS seed entries into the freshly-allocated runtime
  // table. Ordering matches the seed exactly.
  for (std::uint32_t I = 0; I < SeedN; ++I)
    P::EntryPointToTraceFunctionAddrMap[I] =
        P::EntryPointToTraceFunctionAddrMapSeed[I];
  P::EntryPointToTraceFunctionAddrMapSize = SeedN;
  P::EntryPointToTraceFunctionAddrMapMaxSize = Capacity;
  P::EntryPointToTraceFunctionAddrMapLock = 0;
}

__attribute__((device, used, destructor))
void luthier::PatchPCUsagesPass::Parser::finiEntryPointToTraceFunctionAddrMap() {
  using P = luthier::PatchPCUsagesPass::Parser;
  if (P::EntryPointToTraceFunctionAddrMap != nullptr) {
    free(P::EntryPointToTraceFunctionAddrMap);
    P::EntryPointToTraceFunctionAddrMap = nullptr;
  }
  P::EntryPointToTraceFunctionAddrMapSize = 0;
  P::EntryPointToTraceFunctionAddrMapMaxSize = 0;
}

// ROCm device libraries' host-function-call hostcall entrypoint.
using __luthier_ockl_long2 =
    long __attribute__((ext_vector_type(2)));

extern "C" __attribute__((device)) __luthier_ockl_long2
__ockl_call_host_function(unsigned long fptr, unsigned long arg0,
                          unsigned long arg1, unsigned long arg2,
                          unsigned long arg3, unsigned long arg4,
                          unsigned long arg5, unsigned long arg6);

__attribute__((device, used)) void
luthier::PatchPCUsagesPass::Parser::patchRegSingle(
    llvm::MCRegister Reg, std::uint64_t FnHandleAddr) {
  luthier::writeReg<std::uint64_t>(Reg, FnHandleAddr);
}

namespace {

/// Shared fallback body used by every explicit instantiation of
/// \c Parser::patchRegMulti once its compile-time candidate list has been
/// exhausted. Acquires \c EntryPointToTraceFunctionAddrMapLock, walks the runtime table,
/// hands unresolved traces to the host callback (which appends under the
/// lock), then releases and installs the resolved handle. Always inlined
/// so the codegen-time-match hot path stays cheap and the lock's atomic
/// operands remain compile-time constants (scalar-atomic-eligible in the
/// AMDGPU backend).
__attribute__((always_inline, device)) inline void
patchRegMultiFallback(llvm::MCRegister Reg, std::uint64_t Cur,
                      std::uint64_t HostFnPtr) {
  using PCP = luthier::PatchPCUsagesPass;

  // Acquire the wave-uniform spinlock — but only through hardware lane 0.
  if (__lane_id() == 0u) {
    while (atomicCAS(&PCP::Parser::EntryPointToTraceFunctionAddrMapLock, 0u,
                     1u) != 0u) {
    }
  }

  const PCP::EntryPointTraceFunctionEntry *BR = PCP::Parser::EntryPointToTraceFunctionAddrMap;
  const std::uint32_t BRN = PCP::Parser::EntryPointToTraceFunctionAddrMapSize;
  for (std::uint32_t I = 0; I < BRN; ++I) {
    if (BR[I].TraceAddr == Cur) {
      luthier::writeReg<std::uint64_t>(Reg, BR[I].FnHandleAddr);
      if (__lane_id() == 0u) {
        atomicExch(&PCP::Parser::EntryPointToTraceFunctionAddrMapLock, 0u);
      }
      return;
    }
  }

  // Miss: hand off to the host, still holding the lock so the callback can
  // safely append to (or reallocate) the table under exclusive access.
  const unsigned long DispatchPacketAddr =
      reinterpret_cast<unsigned long>(__builtin_amdgcn_dispatch_ptr());
  __luthier_ockl_long2 Result = __ockl_call_host_function(
      HostFnPtr, static_cast<unsigned long>(Cur), DispatchPacketAddr,
      reinterpret_cast<unsigned long>(&PCP::Parser::EntryPointToTraceFunctionAddrMap),
      reinterpret_cast<unsigned long>(&PCP::Parser::EntryPointToTraceFunctionAddrMapSize),
      reinterpret_cast<unsigned long>(&PCP::Parser::EntryPointToTraceFunctionAddrMapMaxSize),
      0, 0);
  if (__lane_id() == 0u) {
    atomicExch(&PCP::Parser::EntryPointToTraceFunctionAddrMapLock, 0u);
  }
  luthier::writeReg<std::uint64_t>(Reg, static_cast<std::uint64_t>(Result[0]));
}

} // namespace

//===----------------------------------------------------------------------===//
// Variadic-template multi-target hook definition + explicit instantiations
// for pack sizes 1..5.
//===----------------------------------------------------------------------===//

template <typename... Pairs>
__attribute__((device, used)) void
luthier::PatchPCUsagesPass::Parser::patchRegMulti(
    llvm::MCRegister Reg, std::uint64_t HostFnPtr, Pairs... Ps) {
  const std::uint64_t Cur = luthier::readReg<std::uint64_t>(Reg);
  // Fold expression over the parameter pack: for every pair, check
  // \c TraceAddr against \c Cur; on a match, write \c FnHandleAddr and
  // fold-return \c true so the \c || short-circuits the remaining pairs.
  // The whole check collapses to a straight-line sequence of compares +
  // conditional writes — no stack-materialized array, no loop counter.
  const bool Matched =
      ((Ps.TraceAddr == Cur &&
        (luthier::writeReg<std::uint64_t>(Reg, Ps.FnHandleAddr), true)) ||
       ...);
  if (!Matched)
    patchRegMultiFallback(Reg, Cur, HostFnPtr);
}

using __luthier_relocatable_pc_TE =
    luthier::PatchPCUsagesPass::EntryPointTraceFunctionEntry;

// Force emission of each specialization.
template __attribute__((device, used)) void
luthier::PatchPCUsagesPass::Parser::patchRegMulti<
    __luthier_relocatable_pc_TE>(llvm::MCRegister, std::uint64_t,
                                  __luthier_relocatable_pc_TE);
template __attribute__((device, used)) void
luthier::PatchPCUsagesPass::Parser::patchRegMulti<
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE>(
    llvm::MCRegister, std::uint64_t, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE);
template __attribute__((device, used)) void
luthier::PatchPCUsagesPass::Parser::patchRegMulti<
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE>(
    llvm::MCRegister, std::uint64_t, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE);
template __attribute__((device, used)) void
luthier::PatchPCUsagesPass::Parser::patchRegMulti<
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE>(
    llvm::MCRegister, std::uint64_t, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE);
template __attribute__((device, used)) void
luthier::PatchPCUsagesPass::Parser::patchRegMulti<
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE>(
    llvm::MCRegister, std::uint64_t, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE,
    __luthier_relocatable_pc_TE, __luthier_relocatable_pc_TE);

#endif // __HIPCC__

//===----------------------------------------------------------------------===//
// Host side — LLVM CodeGen driver logic. Only compiled during the HIP host
// compile pass (the device pass has no include path for the AMDGPU backend
// headers).
//===----------------------------------------------------------------------===//

#ifndef __HIP_DEVICE_COMPILE__

namespace luthier {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {


/// Read the trace-instruction address recorded on an IR Instruction via
/// `MD_pcsections`
std::optional<std::uint64_t> getIRTraceAddr(const llvm::Instruction *I) {
  auto *MD = llvm::dyn_cast_or_null<TargetMachineInstrMDNode>(
      I->getMetadata(llvm::LLVMContext::MD_pcsections));
  if (!MD)
    return std::nullopt;
  return MD->getTraceInstrAddress();
}

/// Reads the trace address off a MachineInstr's `TargetMachineInstrMDNode`.
std::optional<std::uint64_t> getMITraceAddr(const llvm::MachineInstr &MI) {
  if (auto *MD = TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI))
    return MD->getTraceInstrAddress();
  return std::nullopt;
}

/// Callgraph-derived resolution for a single trace site. If the site is
/// direct (single, fully-determined target), the payload should be the
/// `patchRegSingle` shape; otherwise the callee list becomes a codegen-time
/// constant table.
struct SiteResolution {
  llvm::SmallVector<llvm::Function *, 4> Targets;
  bool IsIncomplete{true};
};

/// Build a `trace-addr → SiteResolution` map by walking every CallInst the
/// PrototypeCallGraph produced targets for and pulling each CallInst's
/// trace address out of its `MD_pcsections` metadata.
llvm::DenseMap<std::uint64_t, SiteResolution>
buildTraceAddrIndex(const PrototypeCallGraph &CG) {
  llvm::DenseMap<std::uint64_t, SiteResolution> Out;
  for (const auto &Entry : CG.call_targets()) {
    llvm::CallInst *CI = Entry.first;
    auto Addr = getIRTraceAddr(CI);
    if (!Addr)
      continue;
    SiteResolution R;
    R.Targets.assign(Entry.second.begin(), Entry.second.end());
    R.IsIncomplete = CG.containsIncompleteCallSite(CI);
    Out[*Addr] = std::move(R);
  }
  return Out;
}

/// Find a dead SGPR pair at `MI` for use as a scavenged jump-target register
/// for `S_CALL_B64` / `S_ADD_PC_I64` rewrites.
llvm::MCRegister scavengeSReg64PairAt(const llvm::MachineInstr &MI,
                                      const IPPredicatedCFG &IPCFG,
                                      const IPPredicatedLiveness &IPLiveness) {
  const llvm::MachineFunction &MF = *MI.getMF();
  const auto &ST = MF.getSubtarget();
  const auto &TRI = *ST.getRegisterInfo();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  const llvm::MachineBasicBlock &MBB = *MI.getParent();

  llvm::LivePhysRegs Live(TRI);

  // Live-outs of MBB = ⋃ live-ins(succ PMBB), summing Active + Inactive
  // partitions from IPPredicatedLiveness for every successor of the
  // enclosing PMBB.
  if (!MBB.empty()) {
    const PredicatedMachineBasicBlock &PMBB =
        const_cast<IPPredicatedCFG &>(IPCFG).getPredMBB(MBB.front());
    for (const PredicatedMachineBasicBlock &SuccPMBB : PMBB.successors()) {
      if (const llvm::LivePhysRegs *ActiveLI =
              IPLiveness.getPMBBActiveLiveIns(SuccPMBB)) {
        for (llvm::MCPhysReg R : *ActiveLI)
          Live.addReg(R);
      }
      if (const llvm::LivePhysRegs *InactiveLI =
              IPLiveness.getPMBBInactiveLiveIns(SuccPMBB)) {
        for (llvm::MCPhysReg R : *InactiveLI)
          Live.addReg(R);
      }
    }
  }

  for (auto It = MBB.rbegin(); It != MBB.rend() && &*It != &MI; ++It)
    Live.stepBackward(*It);

  for (llvm::MCPhysReg Reg : llvm::AMDGPU::SReg_64RegClass) {
    if (Live.available(MRI, Reg))
      return llvm::MCRegister(Reg);
  }
  return llvm::MCRegister();
}

/// Emit `S_MOV_B32 dst.sub0, imm_lo; S_MOV_B32 dst.sub1, imm_hi` before
/// `InsertBefore` in `MBB`.
void emitMaterialize64BitImm(llvm::MachineBasicBlock &MBB,
                             llvm::MachineInstr &InsertBefore,
                             llvm::MCRegister DstPair, std::uint64_t Imm,
                             const llvm::TargetInstrInfo &TII,
                             const llvm::TargetRegisterInfo &TRI) {
  const llvm::DebugLoc DL = InsertBefore.getDebugLoc();
  llvm::MCRegister Sub0 = TRI.getSubReg(DstPair, llvm::AMDGPU::sub0);
  llvm::MCRegister Sub1 = TRI.getSubReg(DstPair, llvm::AMDGPU::sub1);
  const std::uint32_t Lo = static_cast<std::uint32_t>(Imm & 0xFFFFFFFFu);
  const std::uint32_t Hi = static_cast<std::uint32_t>(Imm >> 32);
  llvm::MachineFunction &MF = *InsertBefore.getMF();
  // Propagate the source MI's PC-sections MDNode (\c TargetMachineInstrMDNode)
  // onto every replacement MI so downstream analyses that key off the trace
  // metadata (\c PrototypeCallGraph, \c TargetModuleBranchRelaxation, …) still
  // see the same trace address / relaxation flag / retranslation bit on the
  // rewritten code.
  llvm::MDNode *TraceMD = InsertBefore.getPCSections();
  auto NewLo =
      llvm::BuildMI(MBB, InsertBefore, DL, TII.get(llvm::AMDGPU::S_MOV_B32),
                    Sub0)
          .addImm(static_cast<std::int64_t>(Lo));
  auto NewHi =
      llvm::BuildMI(MBB, InsertBefore, DL, TII.get(llvm::AMDGPU::S_MOV_B32),
                    Sub1)
          .addImm(static_cast<std::int64_t>(Hi));
  if (TraceMD) {
    NewLo->setPCSections(MF, TraceMD);
    NewHi->setPCSections(MF, TraceMD);
  }
}

//===----------------------------------------------------------------------===//
// EntryPointToTraceFunctionAddrMap global emission
//===----------------------------------------------------------------------===//

/// Link this pass' module into the IModule
llvm::Error linkParserModuleIntoIModule(
    PatchPCUsagesPass::Parser &OffloadParser, llvm::Module &IModule,
    llvm::ModuleAnalysisManager &IMAM) {
  llvm::LLVMContext &Ctx = IModule.getContext();
  const llvm::TargetMachine &TM =
      IMAM.getResult<llvm::MachineModuleAnalysis>(IModule).getMMI().getTarget();
  llvm::Triple TT = TM.getTargetTriple();
  std::string CPU(TM.getTargetCPU());
  llvm::SubtargetFeatures Features(TM.getTargetFeatureString());

  auto SubOrErr = OffloadParser.parseModule(TT, CPU, Features, Ctx);
  LUTHIER_RETURN_ON_ERROR(SubOrErr.takeError());
  std::unique_ptr<llvm::Module> Sub = std::move(*SubOrErr);
  Sub->setTargetTriple(IModule.getTargetTriple());
  Sub->setDataLayout(IModule.getDataLayout());

  llvm::Linker L(IModule);
  if (L.linkInModule(std::move(Sub), llvm::Linker::Flags::LinkOnlyNeeded))
    return LUTHIER_MAKE_GENERIC_ERROR(
        "PatchPCUsagesPass: failed to link the pass's device-side "
        "payload module into the prototype's instrumentation module.");
  return llvm::Error::success();
}

/// Remove every entry in \p CtorArrayGV (\c @llvm.global_ctors or
/// \c @llvm.global_dtors) that references \p Target, then eliminate the
/// array if the removal leaves it empty. Called when the pass has to strip
/// the HIP-native ctor/dtor for the device-function-entry case.
///
/// The array's shape is \c \[N x { i32, void()*, i8* }\]; slot 1 of each
/// struct is the ctor/dtor function pointer.
void removeFromCtorDtorArray(llvm::Module &M, llvm::StringRef ArrayName,
                             llvm::Function &Target) {
  llvm::GlobalVariable *GV = M.getNamedGlobal(ArrayName);
  if (!GV || !GV->hasInitializer())
    return;
  auto *Init = llvm::dyn_cast<llvm::ConstantArray>(GV->getInitializer());
  if (!Init)
    return;

  llvm::SmallVector<llvm::Constant *, 4> Keep;
  for (unsigned I = 0, E = Init->getNumOperands(); I != E; ++I) {
    auto *Entry = llvm::dyn_cast<llvm::ConstantStruct>(Init->getOperand(I));
    if (!Entry) {
      Keep.push_back(Init->getOperand(I));
      continue;
    }
    llvm::Constant *FnSlot = Entry->getOperand(1)->stripPointerCasts();
    if (FnSlot == &Target)
      continue; // drop
    Keep.push_back(Entry);
  }

  llvm::Type *EltTy = Init->getType()->getElementType();
  if (Keep.empty()) {
    // Nothing left — retire the array entirely.
    GV->eraseFromParent();
    return;
  }
  auto *NewATy = llvm::ArrayType::get(EltTy, Keep.size());
  auto *NewInit = llvm::ConstantArray::get(NewATy, Keep);
  auto *NewGV = new llvm::GlobalVariable(
      M, NewATy, /*isConstant=*/false, GV->getLinkage(), NewInit,
      "" /*rename below*/, GV, GV->getThreadLocalMode(),
      GV->getAddressSpace());
  NewGV->takeName(GV);
  NewGV->copyAttributesFrom(GV);
  GV->eraseFromParent();
}

/// Small helper: look up a shadow-handle → device symbol name in
/// \p OffloadParser and return the matching \c GlobalVariable in
/// \p IModule (or \c nullptr if it doesn't exist).
template <typename T>
llvm::Expected<llvm::GlobalVariable *>
findGVByShadowHandle(PatchPCUsagesPass::Parser &OffloadParser,
                     llvm::Module &IModule, T *Handle) {
  auto NameOrErr = OffloadParser.lookupHandleName(Handle);
  LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
  return IModule.getNamedGlobal(*NameOrErr);
}

/// Populate \c EntryPointToTraceFunctionAddrMapSeed and
/// \c EntryPointToTraceFunctionAddrMapSeedSize on the IModule from the
/// set of already-discovered trace Functions in the target module.
/// Emits a fresh constant \c [N x EntryPointTraceFunctionEntry] global at address
/// space 4 holding each Function's \c (TraceAddr, FnHandleAddr) pair and
/// rewrites the seed pointer's initializer to reference it.
///
/// Called only when the initial entry point is a kernel — for device-fn
/// entry points, the enclosing kernel's module owns both the runtime
/// table and the seed.
llvm::Error emitEntryPointSeed(
    Prototype &P, PatchPCUsagesPass::Parser &OffloadParser) {
  llvm::Module &IModule = P.getInstrumentationModule();
  llvm::Module &TM = P.getTargetModule();
  llvm::LLVMContext &Ctx = IModule.getContext();

  auto SeedPtrGVOrErr = findGVByShadowHandle(
      OffloadParser, IModule,
      &PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMapSeed);
  LUTHIER_RETURN_ON_ERROR(SeedPtrGVOrErr.takeError());
  llvm::GlobalVariable *SeedPtrGV = *SeedPtrGVOrErr;
  auto SeedSizeGVOrErr = findGVByShadowHandle(
      OffloadParser, IModule,
      &PatchPCUsagesPass::Parser::
          EntryPointToTraceFunctionAddrMapSeedSize);
  LUTHIER_RETURN_ON_ERROR(SeedSizeGVOrErr.takeError());
  llvm::GlobalVariable *SeedSizeGV = *SeedSizeGVOrErr;

  if (!SeedPtrGV || !SeedSizeGV)
    return llvm::Error::success();

  llvm::Type *I32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *I64Ty = llvm::Type::getInt64Ty(Ctx);
  llvm::StructType *EntryTy = llvm::StructType::get(Ctx, {I64Ty, I64Ty});

  // Walk the target module for every Function whose entry instruction
  // carries a trace address — those are the codegen-time-discovered
  // entry points.
  //
  // The seed's HandleC needs to end up as the address of the trace function's
  // definition in the linked object. Naming a target-module Function directly
  // in this IModule-owned initializer would embed a cross-module Function*
  // reference: LazyCallGraphAnalysis walks GlobalVariable initializers when
  // it builds the CG for a module (LazyCallGraph.cpp:192-206), so that Node
  // would land in IModule's CG with getParent() pointing at TargetModule. The
  // CGSCC pipeline then trips its assert that FAMProxy is cached on the SCC
  // function's parent module (CGSCCPassManager.cpp:683) — no MAM covering
  // TargetModule is in scope when the O3 pipeline runs on IModule.
  //
  // Instead, synthesize an extern-linkage declaration in IModule for each
  // target-module Function and reference the *declaration* here. The Node
  // stays inside IModule, and \c TargetModulePatcherPass later redirects the
  // declaration to the target-module definition via its existing
  // "same-name declaration ↔ definition" merge (see the loop at
  // TargetModulePatcherPass.cpp:2708-2740), leaving the seed initializer
  // pointing at the real function after module merge.
  llvm::SmallVector<llvm::Constant *, 8> Elts;
  for (llvm::Function &F : TM) {
    if (F.isDeclaration() || F.empty())
      continue;
    auto TraceAddr = getIRTraceAddr(&F.getEntryBlock().front());
    if (!TraceAddr)
      continue;
    llvm::Constant *TraceC = llvm::ConstantInt::get(I64Ty, *TraceAddr);
    llvm::FunctionCallee IDecl = IModule.getOrInsertFunction(
        F.getName(), F.getFunctionType());
    // getOrInsertFunction returns a bitcast-wrapped Constant when the module
    // already has a same-named GV of a different type; both branches wrap a
    // Constant, so ptrtoint is always valid.
    llvm::Constant *HandleCallee =
        llvm::cast<llvm::Constant>(IDecl.getCallee());
    // Same DSO-local rationale as \c getOrInsertIModulePlaceholder: this
    // extern will be merged with the target-module definition of the
    // same name by \c TargetModulePatcherPass, so the symbol is in the
    // same DSO at link time and \c amdgpu-rel32 direct addressing is
    // both valid and needed (GOT-relative on a now-local symbol
    // triggers MC's "expected relocatable expression").
    if (auto *DeclFn = llvm::dyn_cast<llvm::Function>(HandleCallee))
      DeclFn->setDSOLocal(true);
    llvm::Constant *HandleC =
        llvm::ConstantExpr::getPtrToInt(HandleCallee, I64Ty);
    Elts.push_back(llvm::ConstantStruct::get(EntryTy, {TraceC, HandleC}));
  }

  const std::uint32_t N = static_cast<std::uint32_t>(Elts.size());

  // Emit the seed array at AS(4). Even when N == 0 we
  // still leave the seed pointer at nullptr and let the ctor observe
  // SeedSize == 0 — no need to reserve an empty array.
  if (N > 0) {
    llvm::ArrayType *ATy = llvm::ArrayType::get(EntryTy, N);
    llvm::Constant *Init = llvm::ConstantArray::get(ATy, Elts);
    auto *SeedArrGV = new llvm::GlobalVariable(
        IModule, ATy, /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage, Init,
        "__luthier.relocatable_pc.seed_array", /*InsertBefore=*/nullptr,
        llvm::GlobalValue::NotThreadLocal, /*AddressSpace=*/4);
    // Point the constant-AS seed pointer at the new array. The pointer
    // GV's value type is opaque `ptr`; a plain assignment is enough (no
    // bitcast).
    SeedPtrGV->setInitializer(SeedArrGV);
  }
  SeedSizeGV->setInitializer(llvm::ConstantInt::get(I32Ty, N));
  return llvm::Error::success();
}

/// Emit the codegen-time seed when the initial entry point is a kernel
/// AND the callgraph still has indirect edges the runtime resolver has to
/// serve. When the initial entry point is a device function, OR when it's
/// a kernel whose callgraph is fully recovered and consists only of
/// direct edges, strip \c initEntryPointToTraceFunctionAddrMap,
/// \c finiEntryPointToTraceFunctionAddrMap, and both seed globals from
/// the IModule (plus their \c @llvm.global_ctors / \c @llvm.global_dtors
/// entries) — no wave ever hits the fallback path in either case, and the
/// runtime table + spinlock become pure dead weight.
llvm::Error handleEntryPointToTraceFunctionAddrMapCtorDtor(
    Prototype &P, const EntryPoint &EP, const PrototypeCallGraph &CG,
    PatchPCUsagesPass::Parser &OffloadParser) {
  const bool AllSingleTargetAndFullyRecovered =
      CG.isFullyRecovered() &&
      llvm::all_of(CG.call_targets(), [](const auto &Entry) {
        return Entry.second.size() == 1;
      });

  if (EP.isKernel() && !AllSingleTargetAndFullyRecovered)
    return emitEntryPointSeed(P, OffloadParser);

  llvm::Module &IModule = P.getInstrumentationModule();

  // Collect the four Parser globals we're about to drop so we can strip
  // them from `@llvm.used` / `@llvm.compiler.used`
  llvm::SmallPtrSet<llvm::Constant *, 8> ToUnpin;
  auto CollectByShadowHandleFn = [&](auto *Handle) -> llvm::Error {
    auto NameOrErr = OffloadParser.lookupHandleName(Handle);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    if (llvm::Function *F = IModule.getFunction(*NameOrErr))
      ToUnpin.insert(F);
    return llvm::Error::success();
  };
  auto CollectByShadowHandleGV = [&](auto *Handle) -> llvm::Error {
    auto GVOrErr = findGVByShadowHandle(OffloadParser, IModule, Handle);
    LUTHIER_RETURN_ON_ERROR(GVOrErr.takeError());
    if (llvm::GlobalVariable *GV = *GVOrErr)
      ToUnpin.insert(GV);
    return llvm::Error::success();
  };
  LUTHIER_RETURN_ON_ERROR(CollectByShadowHandleFn(
      &PatchPCUsagesPass::Parser::initEntryPointToTraceFunctionAddrMap));
  LUTHIER_RETURN_ON_ERROR(CollectByShadowHandleFn(
      &PatchPCUsagesPass::Parser::finiEntryPointToTraceFunctionAddrMap));
  LUTHIER_RETURN_ON_ERROR(CollectByShadowHandleGV(
      &PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMapSeed));
  LUTHIER_RETURN_ON_ERROR(CollectByShadowHandleGV(
      &PatchPCUsagesPass::Parser::
          EntryPointToTraceFunctionAddrMapSeedSize));
  llvm::removeFromUsedLists(IModule, [&](llvm::Constant *C) {
    return C && ToUnpin.contains(C->stripPointerCasts());
  });

  auto EraseByShadowHandle = [&](auto *Handle, llvm::StringRef ArrayName)
      -> llvm::Error {
    auto NameOrErr = OffloadParser.lookupHandleName(Handle);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    if (llvm::Function *F = IModule.getFunction(*NameOrErr)) {
      removeFromCtorDtorArray(IModule, ArrayName, *F);
      // `replaceAllUsesWith` with `nullptr` isn't valid on Functions
      // (their pointer type is fixed), but at this point the only
      // remaining uses are the `@llvm.global_ctors`/`@llvm.global_dtors`
      // struct entries we just stripped, plus the `@llvm.used` /
      // `@llvm.compiler.used` slots the sweep above dropped. Force-erase
      // the body: it references malloc/free bodies from ROCm device
      // libs, and leaving it in the IModule keeps those bodies alive
      // through DCE and blows up downstream passes.
      F->deleteBody();
      F->eraseFromParent();
    }
    return llvm::Error::success();
  };

  LUTHIER_RETURN_ON_ERROR(EraseByShadowHandle(
      &PatchPCUsagesPass::Parser::initEntryPointToTraceFunctionAddrMap,
      "llvm.global_ctors"));
  LUTHIER_RETURN_ON_ERROR(EraseByShadowHandle(
      &PatchPCUsagesPass::Parser::finiEntryPointToTraceFunctionAddrMap,
      "llvm.global_dtors"));

  auto DropGVByShadowHandle = [&](auto *Handle) -> llvm::Error {
    auto GVOrErr = findGVByShadowHandle(OffloadParser, IModule, Handle);
    LUTHIER_RETURN_ON_ERROR(GVOrErr.takeError());
    if (llvm::GlobalVariable *GV = *GVOrErr;
        GV && GV->use_empty())
      GV->eraseFromParent();
    return llvm::Error::success();
  };
  LUTHIER_RETURN_ON_ERROR(DropGVByShadowHandle(
      &PatchPCUsagesPass::Parser::EntryPointToTraceFunctionAddrMapSeed));
  LUTHIER_RETURN_ON_ERROR(DropGVByShadowHandle(
      &PatchPCUsagesPass::Parser::
          EntryPointToTraceFunctionAddrMapSeedSize));
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Codegen-time candidate-table emission for `patchRegMulti`
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Payload argument marshaling
//===----------------------------------------------------------------------===//

/// Max number of compile-time candidate targets a single
/// \c PatchRegMultiHook<N> specialization supports; extra candidates fall
/// through to the runtime \c EntryPointToTraceFunctionAddrMap + hostcall path.
constexpr unsigned MaxMultiHookN = 5;

/// Return an extern-linkage IModule declaration standing in for the
/// target-module Function \p Target. Used when a payload's IR needs the
/// address of a target-module Function: naming \p Target directly in
/// IModule IR embeds a cross-module \c Function* whose \c getParent() no
/// longer matches the module its use lives in, which \c LazyCallGraphAnalysis
/// on IModule then materializes as an SCC-of-target-function Node and the
/// CGSCC pipeline chokes on at \c CGSCCPassManager.cpp:683.
///
/// The declaration keeps IModule's own IR self-contained. \c TargetModulePatcherPass
/// merges the two back together at the "definitions merge into TargetModule"
/// step (\c TargetModulePatcherPass.cpp:2716-2727), replacing every use of
/// this placeholder with \p Target and erasing the now-dead declaration.
static llvm::Constant *
getOrInsertIModulePlaceholder(llvm::Module &IModule, llvm::Function &Target) {
  llvm::FunctionCallee C =
      IModule.getOrInsertFunction(Target.getName(), Target.getFunctionType());
  auto *Handle = llvm::cast<llvm::Constant>(C.getCallee());
  // Mark the extern DSO-local. \c TargetModulePatcherPass will merge this
  // declaration into the target-module definition of the same name, so
  // by final link time the symbol is guaranteed to live in the same DSO
  // as any use we're planting here. Without this, ISel treats every
  // \c ptrtoint of the handle as an external reference and lowers it as
  // \c SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo/hi) followed
  // by a GOT load — a GOT-relative relocation that MC rejects at
  // assembly time (\c "expected relocatable expression") once the symbol
  // has become module-local after the merge. Setting \c dso_local
  // switches ISel to \c amdgpu-rel32 direct addressing, no GOT slot, and
  // the sequence stays valid across the merge with only the operand's
  // \c GlobalValue* remapped (which \c moveIModuleIntoTarget already
  // does for payload MFs).
  if (auto *F = llvm::dyn_cast<llvm::Function>(Handle))
    F->setDSOLocal(true);
  return Handle;
}

/// Build the `std::vector<PayloadArg>` for `patchRegSingle(Reg, FnAddr)`.
llvm::SmallVector<PayloadArg, 2>
buildSingleArgs(llvm::Module &IModule, llvm::MCRegister Reg,
                llvm::Function *Target) {
  llvm::LLVMContext &Ctx = IModule.getContext();
  llvm::Type *I32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *I64Ty = llvm::Type::getInt64Ty(Ctx);
  llvm::SmallVector<PayloadArg, 2> Args;
  Args.push_back(llvm::ConstantInt::get(I32Ty, Reg.id()));
  Args.push_back(llvm::ConstantExpr::getPtrToInt(
      getOrInsertIModulePlaceholder(IModule, *Target), I64Ty));
  return Args;
}

/// Build the args vector for a call to
/// \c Parser::patchRegMulti<EntryPointTraceFunctionEntry, ..., EntryPointTraceFunctionEntry>. Signature:
///   \c void(MCRegister, uint64_t HostFnPtr, EntryPointTraceFunctionEntry, ..., EntryPointTraceFunctionEntry).
///
/// HIP-Clang lowers each \c EntryPointTraceFunctionEntry by-value parameter on
/// the AMDGPU device ABI in one of two ways depending on how many previous
/// register-passed slots have been consumed:
///   1. \b Coerce: the struct's two \c i64 fields become two consecutive \c i64
///      scalar parameters (\c %Ps.coerce0, \c %Ps.coerce1). This is used until
///      the CC's coerced-scalar register budget is exhausted.
///   2. \b Byref: the remaining struct(s) become a single
///      \c ptr addrspace(5) byref parameter, and the callee reads the fields
///      through that pointer. The caller (this builder) allocates a stack slot,
///      stores the pair, and passes the pointer.
/// The exact split (e.g. first 3 coerced + rest byref for N=4/5) is decided by
/// Clang's AMDGPU ABI and can shift between compiler versions, so we walk
/// \c Hook 's actual \c FunctionType parameter list and dispatch per param.
///
/// A \c nullptr \c Targets[I] entry (the zero-real-candidates stand-in) becomes
/// the \c i64 sentinel \c 0 for the \c FnHandleAddr field — matching the
/// device-side check `TraceAddr == Cur`, which cannot match a legitimate
/// non-zero trace address.
///
/// Byref allocas are inserted through \p Builder, so this must be called from
/// inside the injected payload's build callback (the lambda passed to
/// \c Prototype::createInjectedPayload).
llvm::SmallVector<llvm::Value *, 8>
buildMultiHookArgs(llvm::Module &IModule, llvm::Function &Hook,
                   llvm::MCRegister Reg,
                   llvm::ArrayRef<llvm::Function *> Targets,
                   llvm::ArrayRef<std::uint64_t> TraceAddrs,
                   std::uint64_t HostFnPtr,
                   llvm::IRBuilderBase &Builder) {
  assert(Targets.size() == TraceAddrs.size() &&
         "targets and trace addresses must be same length");
  llvm::LLVMContext &Ctx = IModule.getContext();
  llvm::Type *I32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *I64Ty = llvm::Type::getInt64Ty(Ctx);
  llvm::FunctionType *FTy = Hook.getFunctionType();

  llvm::SmallVector<llvm::Value *, 8> Args;
  Args.push_back(llvm::ConstantInt::get(I32Ty, Reg.id()));
  Args.push_back(llvm::ConstantInt::get(I64Ty, HostFnPtr));

  // Params 0..1 are Reg + HostFnPtr; the rest are the EntryPointTraceFunctionEntry pack.
  unsigned ParamIdx = 2;
  const unsigned NumParams = FTy->getNumParams();
  for (unsigned I = 0, E = Targets.size(); I != E; ++I) {
    llvm::Constant *TraceC = llvm::ConstantInt::get(I64Ty, TraceAddrs[I]);
    // Same placeholder trick as buildSingleArgs — see the note there.
    llvm::Constant *HandleC =
        Targets[I]
            ? llvm::ConstantExpr::getPtrToInt(
                  getOrInsertIModulePlaceholder(IModule, *Targets[I]), I64Ty)
            : llvm::ConstantInt::get(I64Ty, 0);

    assert(ParamIdx < NumParams && "ran off the end of the hook's params");
    llvm::Type *NextTy = FTy->getParamType(ParamIdx);

    if (NextTy->isIntegerTy(64) && ParamIdx + 1 < NumParams &&
        FTy->getParamType(ParamIdx + 1)->isIntegerTy(64)) {
      // Coerced: two consecutive i64 scalars carry TraceAddr / FnHandleAddr.
      Args.push_back(TraceC);
      Args.push_back(HandleC);
      ParamIdx += 2;
    } else if (NextTy->isPointerTy()) {
      // Byref: alloca a { i64, i64 } in the pointer's address space, store the
      // pair into it, and pass the alloca. Inlining the hook will fold the
      // load/store pair back into direct scalar uses.
      llvm::StructType *PairTy = llvm::StructType::get(I64Ty, I64Ty);
      unsigned AS =
          llvm::cast<llvm::PointerType>(NextTy)->getAddressSpace();
      llvm::AllocaInst *Slot =
          Builder.CreateAlloca(PairTy, AS, /*ArraySize=*/nullptr,
                                "luthier.te.byref");
      Slot->setAlignment(llvm::Align(8));
      Builder.CreateAlignedStore(
          llvm::ConstantStruct::get(PairTy, {TraceC, HandleC}), Slot,
          llvm::Align(8));
      Args.push_back(Slot);
      ParamIdx += 1;
    } else if (auto *PairTy = llvm::dyn_cast<llvm::StructType>(NextTy)) {
      // Legacy direct-struct path: some future Clang / another target may keep
      // the aggregate as a StructType param. Emit a ConstantStruct as before.
      Args.push_back(llvm::ConstantStruct::get(PairTy, {TraceC, HandleC}));
      ParamIdx += 1;
    } else {
      llvm_unreachable(
          "unexpected hook parameter lowering for EntryPointTraceFunctionEntry");
    }
  }
  return Args;
}

} // namespace


LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(
    luthier::PatchPCUsagesPass::Parser)

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

PatchPCUsagesPass::PatchPCUsagesPass(TargetAddressHostResolverFn TargetAddressHostResolver,
                                                   llvm::Error &Err)
    : TargetAddressHostResolver(TargetAddressHostResolver),
      OffloadParser(std::make_unique<Parser>(Err)) {}

namespace {

/// Container for the pass's per-run state. Kept local so the pass itself is
/// stateless outside its members.
struct RunContext {
  Prototype &P;
  PrototypeAnalysisManager &PAM;
  llvm::FunctionAnalysisManager &IFAM;
  llvm::Module &IModule;
  llvm::LLVMContext &Ctx;
  const PrototypeCallGraph &CG;
  // IPPredCFG + IPPredicatedLiveness reached directly by the scavenger:
  // no MBB-livein seeding step, live-outs are read straight off the
  // successor PMBBs.
  const IPPredicatedCFG &IPCFG;
  const IPPredicatedLiveness &IPLiveness;
  llvm::DenseMap<std::uint64_t, SiteResolution> ByTraceAddr;
  PatchPCUsagesPass::TargetAddressHostResolverFn TargetAddressHostResolver;
  std::uint64_t HostFnPtr;
  unsigned SiteIdx{0};

  // Resolve the shadow host handles the pass exposes once per run.
  llvm::Function *PatchRegSingleFn{nullptr};
  // One resolved specialization of \c PatchRegMultiHook<N>::hook per
  // candidate count; index (N-1). Any entry left null renders that
  // specialization unavailable — the fallback path below skips it.
  llvm::Function *PatchRegMultiHookByN[MaxMultiHookN] = {nullptr};

  // MBBs touched by any per-opcode rewrite this run. After the walk,
  // every entry here is passed to \c TranslationState::markDirty so the
  // pinned \c TraceFunctionTranslationAnalysis result stays in sync with
  // the mutated MIR (see the header comment on that analysis for the
  // dirty-marking contract). Direct-initialized to a zero-reserved
  // \c DenseMap so aggregate initialization of \c RunContext doesn't
  // have to name the field (the explicit ctor blocks copy-init on
  // omitted members).
  llvm::DenseMap<llvm::MachineFunction *,
                 llvm::SmallPtrSet<llvm::MachineBasicBlock *, 8>>
      DirtyMBBsByMF{0};
};

/// Record \p MBB as touched by a per-opcode rewrite so we can hand it to
/// \c TranslationState::markDirty at the end of \c run.
void markMBBDirty(RunContext &RC, llvm::MachineBasicBlock &MBB) {
  RC.DirtyMBBsByMF[MBB.getParent()].insert(&MBB);
}

/// Small local helper: `siteResolution` on a callgraph miss returns an
/// empty, "incomplete" record, matching the semantics of a site the
/// analysis did not touch at all.
SiteResolution
resolveSiteAtAddr(const RunContext &RC, std::optional<std::uint64_t> Addr) {
  if (!Addr)
    return SiteResolution{};
  auto It = RC.ByTraceAddr.find(*Addr);
  if (It == RC.ByTraceAddr.end())
    return SiteResolution{};
  return It->second;
}

/// Inject the appropriate hook for \p Reg at \p MI: either
/// \c patchRegSingle (fully-resolved single target) or an explicit
/// specialization of \c PatchRegMultiHook<N>::hook for N ∈ [1..5]. Extra
/// codegen-time candidates beyond \c MaxMultiHookN are dropped from the
/// compile-time check; the specialization's own runtime EntryPointToTraceFunctionAddrMap +
/// hostcall path still resolves those addresses if they arise.
llvm::Error injectRegPatchPayload(RunContext &RC, llvm::MachineInstr &MI,
                                  llvm::MCRegister Reg,
                                  const SiteResolution &SR) {
  // Fast path: exactly one target and the site is fully resolved. This maps
  // to the direct write in `patchRegSingle`, no fallback needed.
  if (SR.Targets.size() == 1 && !SR.IsIncomplete) {
    auto Args = buildSingleArgs(RC.IModule, Reg, SR.Targets.front());
    return RC.P
        .createInjectedPayload(*RC.PatchRegSingleFn, MI, RC.IFAM, Args)
        .takeError();
  }

  // Multi / incomplete: collect at most \c MaxMultiHookN codegen-time
  // candidates. Recover each candidate's trace address by reading the entry
  // instruction's \c MD_pcsections; candidates with no trace address are
  // skipped since we couldn't match them at runtime anyway (the fallback
  // EntryPointToTraceFunctionAddrMap + hostcall still handles them).
  llvm::SmallVector<llvm::Function *, MaxMultiHookN> Targets;
  llvm::SmallVector<std::uint64_t, MaxMultiHookN> TraceAddrs;
  for (llvm::Function *F : SR.Targets) {
    if (Targets.size() == MaxMultiHookN)
      break;
    if (!F || F->empty())
      continue;
    auto Addr = getIRTraceAddr(&F->getEntryBlock().front());
    if (!Addr)
      continue;
    Targets.push_back(F);
    TraceAddrs.push_back(*Addr);
  }

  // Zero-candidate case: fall back to the smallest specialization (N=1)
  // with a bogus (0,0) pair so the compile-time check is guaranteed to
  // miss and control drops straight into the EntryPointToTraceFunctionAddrMap + hostcall
  // fallback. Keeps the dispatch table dense.
  if (Targets.empty()) {
    Targets.push_back(nullptr);
    TraceAddrs.push_back(0);
  }

  const unsigned N = static_cast<unsigned>(Targets.size());
  assert(N >= 1 && N <= MaxMultiHookN &&
         "PatchRegMultiHook candidate count out of the supported range");
  llvm::Function *Hook = RC.PatchRegMultiHookByN[N - 1];
  if (!Hook)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "PatchPCUsagesPass: PatchRegMultiHook<{0}>::hook shadow "
        "handle was not resolved — the pass's device fat binary is missing "
        "this specialization.",
        N));

  // The zero-candidate stand-in for N=1 sends `nullptr` as the Function*;
  // the args builder maps that to a `0`-valued `FnHandleAddr` field — a
  // guaranteed mismatch against every legitimate runtime trace address.
  //
  // Drive the payload build directly (rather than through the ArrayRef<Value*>
  // overload) so \c buildMultiHookArgs can insert allocas for byref-lowered
  // EntryPointTraceFunctionEntry parameters through the payload's builder.
  return RC.P
      .createInjectedPayload(
          MI, RC.IFAM,
          [&](llvm::IRBuilderBase &Builder) -> llvm::Error {
            llvm::SmallVector<llvm::Value *, 8> HookArgs = buildMultiHookArgs(
                RC.IModule, *Hook, Reg, Targets, TraceAddrs, RC.HostFnPtr,
                Builder);
            llvm::CallInst *HookCall = Builder.CreateCall(Hook, HookArgs);
            llvm::InlineFunctionInfo IFI;
            llvm::InlineResult IR = llvm::InlineFunction(*HookCall, IFI);
            if (!IR.isSuccess())
              return LUTHIER_MAKE_GENERIC_ERROR(
                  "Failed to force-inline hook '" + Hook->getName().str() +
                  "' into payload: " + IR.getFailureReason());
            return llvm::Error::success();
          })
      .takeError();
}

/// Rewrite a single S_GETPC_B64 into a pair of S_MOV_B32 immediates
/// materializing `TraceAddr + 4` into its destination pair.
void rewriteGetPC(llvm::MachineInstr &MI, std::uint64_t TraceAddr) {
  llvm::MachineBasicBlock &MBB = *MI.getParent();
  const llvm::MachineFunction &MF = *MI.getMF();
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  const llvm::MCRegister DstPair = MI.getOperand(0).getReg();
  emitMaterialize64BitImm(MBB, MI, DstPair, TraceAddr + 4, TII, TRI);
  MI.eraseFromParent();
}

/// S_SETPC_B64: inject a payload that writes the resolved handle into
/// operand-0's register at runtime; leave the jump in place.
llvm::Error rewriteSetPC(RunContext &RC, llvm::MachineInstr &MI) {
  llvm::MCRegister TargetReg = MI.getOperand(0).getReg().asMCReg();
  auto SR = resolveSiteAtAddr(RC, getMITraceAddr(MI));
  return injectRegPatchPayload(RC, MI, TargetReg, SR);
}

/// Drop the synthetic \c S_ENDPGM continuation \c CodeDiscoveryPass placed
/// after \p CallMBB (which held the original call/branch MI). Once the
/// call MI has been lowered to \c S_SETPC_B64_return the continuation is
/// unreachable: the return is a barrier terminator, control never falls
/// through to it. Leaving the stale \c CallMBB → \c EndpgmMBB successor
/// edge would tell downstream liveness / IP-predicated-CFG consumers that
/// this MBB has an intra-procedural successor when it really has none —
/// so remove the edge here, and delete the dead MBB when nothing else
/// references it.
///
/// A synthetic-endpgm MBB is identified by its shape rather than a flag,
/// mirroring the "1 MI, opcode \c S_ENDPGM" pattern \c CodeDiscoveryPass
/// emits (see the "Isolated synthetic S_ENDPGM MBB" section).
void dropDeadSyntheticEndpgmContinuation(llvm::MachineBasicBlock &CallMBB) {
  llvm::MachineFunction &MF = *CallMBB.getParent();
  llvm::SmallVector<llvm::MachineBasicBlock *, 2> DeadSuccs;
  for (llvm::MachineBasicBlock *Succ : CallMBB.successors()) {
    if (Succ->size() != 1)
      continue;
    if (Succ->back().getOpcode() != llvm::AMDGPU::S_ENDPGM)
      continue;
    DeadSuccs.push_back(Succ);
  }
  for (llvm::MachineBasicBlock *Succ : DeadSuccs) {
    CallMBB.removeSuccessor(Succ);
    if (Succ->pred_empty()) {
      Succ->eraseFromParent();
      LLVM_DEBUG(llvm::dbgs() << "[PatchPCUsagesPass]     dropped dead "
                                 "synthetic S_ENDPGM MBB\n");
    }
  }
}

/// S_SWAPPC_B64: materialize `TraceAddr + 4` into the RA-dest pair via
/// direct S_MOV_B32s, inject the target-reg payload as above, and lower
/// the opcode from S_SWAPPC_B64 to S_SETPC_B64 so the atomic RA write
/// baked into S_SWAPPC_B64's semantics no longer overwrites our value.
llvm::Error rewriteSwapPC(RunContext &RC, llvm::MachineInstr &MI,
                          std::uint64_t TraceAddr) {
  llvm::MachineBasicBlock &MBB = *MI.getParent();
  const auto &MF = *MI.getMF();
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();

  const llvm::MCRegister DstPair = MI.getOperand(0).getReg();
  const llvm::MCRegister TargetReg = MI.getOperand(1).getReg().asMCReg();

  // 1. Manually pre-write the RA reg with the trace-domain return address.
  emitMaterialize64BitImm(MBB, MI, DstPair, TraceAddr + 4, TII, TRI);

  // 2. Inject the target-reg payload — it patches TargetReg to the handle
  //    of the resolved trace Function before the jump fires.
  auto SR = resolveSiteAtAddr(RC, TraceAddr);
  if (auto Err = injectRegPatchPayload(RC, MI, TargetReg, SR))
    return Err;

  // 3. Convert S_SWAPPC_B64 into a plain S_SETPC_B64: build a fresh MI
  //    that inherits only the callee-reg operand, then drop the original.
  //    Copy the pcsections MDNode (\c TargetMachineInstrMDNode) across so
  //    downstream analyses still see the same trace address / relaxation
  //    flag / retranslation bit on the rewritten jump.
  const llvm::DebugLoc DL = MI.getDebugLoc();
  llvm::MDNode *TraceMD = MI.getPCSections();
  auto NewMI = llvm::BuildMI(MBB, MI, DL, TII.get(llvm::AMDGPU::S_SETPC_B64_return))
                   .addReg(TargetReg);
  if (TraceMD)
    NewMI->setPCSections(*MI.getMF(), TraceMD);
  MI.eraseFromParent();
  dropDeadSyntheticEndpgmContinuation(MBB);
  return llvm::Error::success();
}

/// S_CALL_B64: scavenge an SReg_64 pair to hold the jump target, materialize
/// the trace-domain RA (`TraceAddr + 4`) into operand-0's RA reg, inject the
/// callgraph-driven target-reg payload writing the scavenged pair, then
/// lower to `S_SETPC_B64 <scavenged>`.
llvm::Error rewriteCall(RunContext &RC, llvm::MachineInstr &MI,
                        std::uint64_t TraceAddr) {
  llvm::MachineBasicBlock &MBB = *MI.getParent();
  const auto &MF = *MI.getMF();
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();

  llvm::MCRegister ScavPair =
      scavengeSReg64PairAt(MI, RC.IPCFG, RC.IPLiveness);
  if (!ScavPair)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "PatchPCUsagesPass: could not scavenge SReg_64 for S_CALL_B64 "
        "target in MF '{0}' MBB {1}",
        MF.getName(), MBB.getNumber()));

  // Materialize RA (dst operand of S_CALL_B64).
  const llvm::MCRegister DstPair = MI.getOperand(0).getReg();
  emitMaterialize64BitImm(MBB, MI, DstPair, TraceAddr + 4, TII, TRI);

  // Inject the callee-reg patch payload against the scavenged pair.
  auto SR = resolveSiteAtAddr(RC, TraceAddr);
  if (auto Err = injectRegPatchPayload(RC, MI, ScavPair, SR))
    return Err;

  // Emit `S_SETPC_B64 <scavenged>` and carry the original S_CALL_B64's
  // pcsections MDNode across.
  const llvm::DebugLoc DL = MI.getDebugLoc();
  llvm::MDNode *TraceMD = MI.getPCSections();
  auto NewMI = llvm::BuildMI(MBB, MI, DL, TII.get(llvm::AMDGPU::S_SETPC_B64_return))
                   .addReg(ScavPair);
  if (TraceMD)
    NewMI->setPCSections(*MI.getMF(), TraceMD);
  MI.eraseFromParent();
  dropDeadSyntheticEndpgmContinuation(MBB);
  return llvm::Error::success();
}

/// S_ADD_PC_I64: as S_CALL_B64 but with no return address to materialize.
llvm::Error rewriteAddPC(RunContext &RC, llvm::MachineInstr &MI,
                        std::uint64_t TraceAddr) {
  llvm::MachineBasicBlock &MBB = *MI.getParent();
  const auto &MF = *MI.getMF();
  const auto &TII = *MF.getSubtarget().getInstrInfo();

  llvm::MCRegister ScavPair =
      scavengeSReg64PairAt(MI, RC.IPCFG, RC.IPLiveness);
  if (!ScavPair)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "PatchPCUsagesPass: could not scavenge SReg_64 for "
        "S_ADD_PC_I64 target in MF '{0}' MBB {1}",
        MF.getName(), MBB.getNumber()));

  auto SR = resolveSiteAtAddr(RC, TraceAddr);
  if (auto Err = injectRegPatchPayload(RC, MI, ScavPair, SR))
    return Err;

  // Emit `S_SETPC_B64 <scavenged>` and carry the original S_ADD_PC_I64's
  // pcsections MDNode across.
  const llvm::DebugLoc DL = MI.getDebugLoc();
  llvm::MDNode *TraceMD = MI.getPCSections();
  auto NewMI = llvm::BuildMI(MBB, MI, DL, TII.get(llvm::AMDGPU::S_SETPC_B64_return))
                   .addReg(ScavPair);
  if (TraceMD)
    NewMI->setPCSections(*MI.getMF(), TraceMD);
  MI.eraseFromParent();
  dropDeadSyntheticEndpgmContinuation(MBB);
  return llvm::Error::success();
}

/// Dispatch on `MI`'s opcode. Returns `llvm::Error::success()` for
/// instructions the pass does not touch.
llvm::Error rewriteIfRelevant(RunContext &RC, llvm::MachineInstr &MI) {
  auto TraceAddr = getMITraceAddr(MI);
  if (!TraceAddr)
    return llvm::Error::success();

  // Every opcode handled below mutates the containing MBB in-place (an
  // instruction is inserted, replaced, or erased). Mark the MBB dirty for
  // \c TraceFunctionTranslationAnalysis before dispatching so the record
  // survives if the rewrite has to return an error partway through.
  const unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  case llvm::AMDGPU::S_GETPC_B64:
  case llvm::AMDGPU::S_SETPC_B64:
  case llvm::AMDGPU::S_SETPC_B64_return:
  case llvm::AMDGPU::S_SWAPPC_B64:
  case llvm::AMDGPU::S_CALL_B64:
  case llvm::AMDGPU::S_ADD_PC_I64:
    markMBBDirty(RC, *MI.getParent());
    break;
  default:
    return llvm::Error::success();
  }

  switch (Opcode) {
  case llvm::AMDGPU::S_GETPC_B64:
    rewriteGetPC(MI, *TraceAddr);
    return llvm::Error::success();
  case llvm::AMDGPU::S_SETPC_B64:
    return rewriteSetPC(RC, MI);
  case llvm::AMDGPU::S_SWAPPC_B64:
    return rewriteSwapPC(RC, MI, *TraceAddr);
  case llvm::AMDGPU::S_CALL_B64:
    return rewriteCall(RC, MI, *TraceAddr);
  case llvm::AMDGPU::S_ADD_PC_I64:
    return rewriteAddPC(RC, MI, *TraceAddr);
  default:
    return llvm::Error::success();
  }
}

} // namespace

llvm::PreservedAnalyses
PatchPCUsagesPass::run(Prototype &P, PrototypeAnalysisManager &PAM) {
  llvm::Module &IModule = P.getInstrumentationModule();
  llvm::LLVMContext &Ctx = IModule.getContext();

  assert(OffloadParser && "Offload parser is nullptr");

  auto &TargetMAM =
      PAM.getResult<TargetModuleAnalysisManagerPrototypeProxy>(P).getManager();
  auto &IMAM =
      PAM.getResult<IModuleAnalysisManagerPrototypeProxy>(P).getManager();
  auto &EPRes = TargetMAM.getResult<InitialEntryPointAnalysis>(
      P.getTargetModule());
  const EntryPoint EP = EPRes.getInitialEntryPoint();


  auto EmitAndReturn = [&](llvm::Error Err, bool Mutated) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return Mutated ? llvm::PreservedAnalyses::none()
                   : llvm::PreservedAnalyses::all();
  };

  // Splice the pass's device-side payloads (patchRegSingle, patchRegMulti,
  // and the EntryPointToTraceFunctionAddrMap globals) into the prototype's
  // instrumentation module. Their `llvm::Function*` and `GlobalVariable*`
  // handles are what every subsequent shadow-handle lookup below expects
  // to find in the IModule.
  if (auto Err = linkParserModuleIntoIModule(*OffloadParser, IModule, IMAM))
    return EmitAndReturn(std::move(Err), /*Mutated=*/false);

  // Fetch the callgraph before the ctor/dtor decision — the kernel-entry
  // path can short-circuit the whole runtime resolver setup when every
  // edge is direct and the callgraph is fully recovered.
  const auto &CG = PAM.getResult<PrototypeCallGraphAnalysis>(P);

  if (auto Err = handleEntryPointToTraceFunctionAddrMapCtorDtor(
          P, EP, CG, *OffloadParser))
    return EmitAndReturn(std::move(Err), /*Mutated=*/true);

  // Inter-procedural predicated liveness. The scavenger reads live-outs
  // straight off each MI's PMBB successors — no per-MBB livein-seeding
  // step is needed.
  const IPPredicatedCFG &IPCFG =
      PAM.getResult<IPPredCFGAnalysis>(P).getVecCFG();
  const auto &IPLiveness = PAM.getResult<IPPredicatedLivenessAnalysis>(P);

  RunContext RC{
      P, PAM,
      PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
          .getManager(),
      IModule, Ctx, CG,
      IPCFG, IPLiveness,
      buildTraceAddrIndex(CG),
      TargetAddressHostResolver,
      // Function pointer → integer. `reinterpret_cast<uintptr_t>` on a
      // function pointer is impl-defined-but-portable on every target
      // Luthier supports (POSIX AMD64/aarch64). This raw address is what
      // the hostcall FUNCTION_CALL service will dereference on the host
      // side.
      static_cast<std::uint64_t>(
          reinterpret_cast<std::uintptr_t>(TargetAddressHostResolver)),
      /*SiteIdx=*/0,
  };

  // Resolve the payload shadow handles up front — they're the same
  // llvm::Function*s for every site. The trait's Parser carries the
  // handle→symbol mapping; the corresponding `llvm::Function` lives in the
  // IModule under that symbol name. `EmitAndReturn` above already knows to
  // switch to PA::none() once we've reached this point since
  // `linkParserModuleIntoIModule` mutated the IModule.
  auto ResolveHandle = [&](auto *HostHandle) -> llvm::Expected<llvm::Function *> {
    auto NameOrErr = OffloadParser->lookupHandleName(HostHandle);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    if (llvm::Function *F = IModule.getFunction(*NameOrErr))
      return F;
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "failed to resolve device-side symbol '{0}' in the "
        "instrumentation module after linking the offload bundle.",
        *NameOrErr));
  };

  auto BindHandle = [&](llvm::Function *&Dst, auto *HostHandle) -> llvm::Error {
    auto FnOrErr = ResolveHandle(HostHandle);
    LUTHIER_RETURN_ON_ERROR(FnOrErr.takeError());
    Dst = *FnOrErr;
    return llvm::Error::success();
  };

  using TE = EntryPointTraceFunctionEntry;
  if (auto Err = BindHandle(RC.PatchRegSingleFn, &Parser::patchRegSingle))
    return EmitAndReturn(std::move(Err), /*Mutated=*/true);
  if (auto Err = BindHandle(RC.PatchRegMultiHookByN[0],
                            &Parser::patchRegMulti<TE>))
    return EmitAndReturn(std::move(Err), /*Mutated=*/true);
  if (auto Err = BindHandle(RC.PatchRegMultiHookByN[1],
                            &Parser::patchRegMulti<TE, TE>))
    return EmitAndReturn(std::move(Err), /*Mutated=*/true);
  if (auto Err = BindHandle(RC.PatchRegMultiHookByN[2],
                            &Parser::patchRegMulti<TE, TE, TE>))
    return EmitAndReturn(std::move(Err), /*Mutated=*/true);
  if (auto Err = BindHandle(RC.PatchRegMultiHookByN[3],
                            &Parser::patchRegMulti<TE, TE, TE, TE>))
    return EmitAndReturn(std::move(Err), /*Mutated=*/true);
  if (auto Err = BindHandle(RC.PatchRegMultiHookByN[4],
                            &Parser::patchRegMulti<TE, TE, TE, TE, TE>))
    return EmitAndReturn(std::move(Err), /*Mutated=*/true);

  llvm::Error WalkErr = llvm::Error::success();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    if (WalkErr)
      return;
    for (llvm::MachineBasicBlock &MBB : MF) {
      for (llvm::MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
        if (auto Err = rewriteIfRelevant(RC, MI)) {
          WalkErr = std::move(Err);
          return;
        }
      }
    }
  });
  if (WalkErr)
    return EmitAndReturn(std::move(WalkErr), /*Mutated=*/true);

  // Trigger a re-translation of every mutated target MF: the pinned
  // \c TraceFunctionTranslationAnalysis lives in the target-side
  // MachineFunctionAnalysisManager, and the correct contract for a
  // mutating pass is to mark each touched MBB dirty on its
  // \c TranslationState — the next consumer's \c flush() picks up the
  // metadata and re-lifts the affected blocks. See the header for
  // \c TranslationState's dirty-marking rules.
  llvm::MachineFunctionAnalysisManager &TargetMFAM =
      PAM.getResult<TargetMachineFunctionAnalysisManagerPrototypeProxy>(P)
          .getManager();
  for (auto &[MF, MBBs] : RC.DirtyMBBsByMF) {
    TranslationState &TState =
        TargetMFAM.getResult<TraceFunctionTranslationAnalysis>(*MF);
    for (llvm::MachineBasicBlock *MBB : MBBs)
      TState.markDirty(*MBB);
    if (auto Err = TState.flush())
      return EmitAndReturn(std::move(Err), /*Mutated=*/true);
  }

  // Every injected payload that was going to reference a device-side
  // hook has now been emitted, so the `used` pin HIP-Clang applied to
  // each Parser::patchReg* declaration is now the only thing keeping the
  // unreferenced variants alive. Strip them out of `@llvm.used` and
  // `@llvm.compiler.used` so downstream DCE can drop the dead
  // specializations (and their unused shadow handles).
  llvm::SmallPtrSet<llvm::Constant *, 8> HookFns;
  HookFns.insert(RC.PatchRegSingleFn);
  for (llvm::Function *H : RC.PatchRegMultiHookByN)
    HookFns.insert(H);
  llvm::removeFromUsedLists(IModule, [&](llvm::Constant *C) {
    return C && HookFns.contains(C->stripPointerCasts());
  });

  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::all();
  // PA.preserve<TargetModuleAnalysisManagerPrototypeProxy>();
  // PA.preserve<TargetFunctionAnalysisManagerPrototypeProxy>();
  /// PA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
  /// PA.preserve<llvm::CGSCCAnalysisManagerModuleProxy>();
  /// PA.preserve<TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  /// PA.preserve<IModuleAnalysisManagerPrototypeProxy>();
  // PA.preserve<IModuleFunctionAnalysisManagerPrototypeProxy>();
  // PA.preserve<IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.abandon<InjectedPayloadAndInstPointAnalysis>();
  PA.abandon<PrototypeCallGraphAnalysis>();
  PA.abandon<IPPredicatedLivenessAnalysis>();
  PA.abandon<IPPredCFGAnalysis>();
  return PA;
}

} // namespace luthier

#endif // __HIP_DEVICE_COMPILE__
