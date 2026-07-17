//===-- TraceCallGraph.cpp - Luthier IR call graph analysis ---------------===//
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
/// Implements the \c TraceCallGraphAnalysis InstrumentPrototype analysis.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TraceCallGraph.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InstrumentPrototype.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include <AMDGPU.h>
#include <SIInstrInfo.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

#undef DEBUG_TYPE
#define DEBUG_TYPE "trace-callgraph"

namespace luthier {

bool TraceCallGraph::invalidate(
    InstrumentPrototype &, const llvm::PreservedAnalyses &PA,
    InstrumentPrototypeAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<TraceCallGraphAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<llvm::AllAnalysesOn<InstrumentPrototype>>();
}

llvm::AnalysisKey TraceCallGraphAnalysis::Key;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

using ValConstMap = llvm::DenseMap<llvm::Value *, llvm::Constant *>;

/// Return the trace-instruction address stored in the \c MD_pcsections
/// metadata of \p I, if present.
static std::optional<uint64_t> getTraceAddr(const llvm::Instruction *I) {
  auto *MD = llvm::dyn_cast_or_null<TargetMachineInstrMDNode>(
      I->getMetadata(llvm::LLVMContext::MD_pcsections));
  if (!MD)
    return std::nullopt;
  return MD->getTraceInstrAddress();
}

/// Strip value-preserving pointer reinterpretations (inttoptr / bitcast /
/// addrspacecast) to expose the underlying address expression.
static llvm::Value *stripPtrCasts(llvm::Value *P) {
  while (auto *PI = llvm::dyn_cast<llvm::Instruction>(P)) {
    if (!llvm::isa<llvm::IntToPtrInst, llvm::BitCastInst,
                   llvm::AddrSpaceCastInst>(PI))
      break;
    P = PI->getOperand(0);
  }
  return P;
}

/// \return \c true if \p SI provably writes exactly the bytes \p LI reads back
/// (same address and same byte width), so the loaded value equals the stored
/// value (bit-reinterpreted to the loaded type).  AA proves the address match
/// for GEP-style addressing; for the inttoptr-based scratch/global pointers the
/// translator emits (which AA treats opaquely) we additionally accept
/// syntactically identical address expressions modulo pointer casts.  That
/// shortcut is only sound within one address space: the same numeric offset
/// names different storage in different address spaces, so require matching
/// address spaces (AA never reports must-alias across them).
static bool storeMatchesLoad(llvm::StoreInst *SI, llvm::LoadInst *LI,
                             llvm::AAResults &AA, const llvm::DataLayout &DL) {
  if (!SI->isSimple())
    return false;
  bool SameAddr =
      AA.isMustAlias(llvm::MemoryLocation::get(LI),
                     llvm::MemoryLocation::get(SI)) ||
      (LI->getPointerAddressSpace() == SI->getPointerAddressSpace() &&
       stripPtrCasts(LI->getPointerOperand()) ==
           stripPtrCasts(SI->getPointerOperand()));
  return SameAddr && DL.getTypeStoreSize(SI->getValueOperand()->getType()) ==
                         DL.getTypeStoreSize(LI->getType());
}

/// Symbolically evaluate \p V as a \c Constant, using \p SubstMap for
/// \c Argument substitution and \p Cache for memoisation.  Returns nullptr
/// when \p V cannot be fully folded.
///
/// \p FAM provides the per-function MemorySSA / alias analysis used to trace
/// loads back to their defining store (see the \c LoadInst case below).
///
/// Special cases handled beyond generic \c ConstantFoldInstOperands:
///  - \c call \c \@llvm.amdgcn.s.getpc() with \c MD_pcsections at addr A
///    → folded to the integer constant \c A+4.
///  - \c PHINode → nullptr (ambiguous without knowing the taken edge).
///  - \c LoadInst → value of the unique clobbering store, when MemorySSA finds
///    a single \c StoreInst at the same address whose stored value has the same
///    byte width (bit-reinterpreted to the loaded type). This is address-space
///    agnostic; in practice it recovers callee addresses spilled to scratch
///    (private memory) and reloaded under register pressure, since that is the
///    only spill path on AMDGPU that keeps the value scalar/foldable.
static llvm::Constant *tryEvalConst(llvm::Value *V, const ValConstMap &SubstMap,
                                    ValConstMap &Cache,
                                    const llvm::DataLayout &DL,
                                    llvm::FunctionAnalysisManager &FAM) {
  if (auto It = Cache.find(V); It != Cache.end())
    return It->second;

  auto cache = [&](llvm::Constant *C) -> llvm::Constant * {
    Cache[V] = C;
    return C;
  };

  if (auto *C = llvm::dyn_cast<llvm::Constant>(V))
    return cache(C);

  if (auto It = SubstMap.find(V); It != SubstMap.end())
    return cache(It->second);

  auto *I = llvm::dyn_cast<llvm::Instruction>(V);
  if (!I || llvm::isa<llvm::PHINode>(I))
    return cache(nullptr);

  if (auto *CI = llvm::dyn_cast<llvm::CallInst>(I)) {
    const llvm::Intrinsic::ID IID = CI->getIntrinsicID();

    // amdgcn_s_getpc() with pcsections → addr + 4
    if (IID == llvm::Intrinsic::amdgcn_s_getpc) {
      if (auto Addr = getTraceAddr(CI))
        return cache(llvm::ConstantInt::get(CI->getType(), *Addr + 4));
      return cache(nullptr);
    }

    // ssa.copy is a plain SSA copy — the translator emits it for register
    // moves (e.g. v_mov). Forward through it.
    if (IID == llvm::Intrinsic::ssa_copy)
      return cache(
          tryEvalConst(CI->getArgOperand(0), SubstMap, Cache, DL, FAM));

    // TODO: cross-lane VGPR reads (readfirstlane / readlane / writelane) are
    // NOT traced. Resolving them properly needs IR-translator changes
    // (per-lane register-value tracking across basic blocks, recording a lane
    // value when tracing through VGPRs), which would also enable tracing vector
    // loads from global memory.
  }

  // Load → trace through the unique clobbering store via MemorySSA, in any
  // address space. This recovers callee addresses spilled to memory (commonly
  // scratch/private under register pressure) and reloaded. A MemoryPhi /
  // liveOnEntry clobber is ambiguous.
  if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(I)) {
    if (!LI->isSimple())
      return cache(nullptr);
    llvm::Function &F = *LI->getFunction();
    llvm::MemorySSA &MSSA = FAM.getResult<llvm::MemorySSAAnalysis>(F).getMSSA();
    llvm::AAResults &AA = FAM.getResult<llvm::AAManager>(F);
    auto *Def = llvm::dyn_cast<llvm::MemoryDef>(
        MSSA.getWalker()->getClobberingMemoryAccess(LI));
    if (!Def || MSSA.isLiveOnEntryDef(Def))
      return cache(nullptr);
    auto *SI = llvm::dyn_cast_or_null<llvm::StoreInst>(Def->getMemoryInst());
    if (!SI || !storeMatchesLoad(SI, LI, AA, DL))
      return cache(nullptr);
    // Break potential memory cycles before recursing into the stored value,
    // then bit-reinterpret it to the loaded type (a spilled i64 pointer is
    // reloaded as e.g. <2 x i32>).
    Cache[V] = nullptr;
    llvm::Constant *SC =
        tryEvalConst(SI->getValueOperand(), SubstMap, Cache, DL, FAM);
    llvm::Constant *R = nullptr;
    if (SC)
      R = SC->getType() == LI->getType()
              ? SC
              : llvm::ConstantFoldCastOperand(llvm::Instruction::BitCast, SC,
                                              LI->getType(), DL);
    Cache[V] = R;
    return R;
  }

  // General case: fold all operands then try ConstantFoldInstOperands.
  llvm::SmallVector<llvm::Constant *> Ops;
  Ops.reserve(I->getNumOperands());
  for (llvm::Value *Op : I->operands()) {
    llvm::Constant *C = tryEvalConst(Op, SubstMap, Cache, DL, FAM);
    if (!C)
      return cache(nullptr);
    Ops.push_back(C);
  }
  return cache(llvm::ConstantFoldInstOperands(I, Ops, DL));
}

/// Set-valued companion to \c tryEvalConst for the call-site target operand.
/// Appends to \p Out (deduplicated) every constant \p V may take, enumerating
/// intra-procedural fan-out that \c tryEvalConst (single-valued) cannot:
///   - \c PHINode / \c SelectInst → union over incoming values / both arms;
///   - \c LoadInst with a \c MemoryPhi clobber → union over the values of the
///     same-address stores reaching the load (a callee pointer stored on
///     divergent paths and reloaded — "memory fan-out");
///   - \c CastInst → each element mapped through the cast.
/// Anything else contributes nothing (conservative).  No cartesian products are
/// formed: a value reconstructed from two independently-varying operands (e.g.
/// the lo/hi halves of a pointer phi'd as separate 32-bit registers) is left
/// unresolved rather than expanded into spurious mixed targets.  \p Active is
/// the current DFS path, used to break cycles (loop-carried phis).
static void evalConstTargets(llvm::Value *V, const ValConstMap &SubstMap,
                             const llvm::DataLayout &DL,
                             llvm::FunctionAnalysisManager &FAM,
                             llvm::SmallPtrSetImpl<llvm::Value *> &Active,
                             llvm::SmallVectorImpl<llvm::Constant *> &Out) {
  auto add = [&](llvm::Constant *C) {
    if (C && !llvm::is_contained(Out, C))
      Out.push_back(C);
  };

  // Fast path: anything that folds to a single constant (manifest constants,
  // getpc chains, single-store spills, fully-constant subtrees).
  {
    ValConstMap Cache;
    if (llvm::Constant *C = tryEvalConst(V, SubstMap, Cache, DL, FAM))
      return add(C);
  }

  if (!Active.insert(V).second)
    return; // cycle on the current path
  llvm::scope_exit Pop([&] { Active.erase(V); });

  auto *I = llvm::dyn_cast<llvm::Instruction>(V);
  if (!I)
    return;

  if (auto *P = llvm::dyn_cast<llvm::PHINode>(I)) {
    for (llvm::Value *In : P->incoming_values())
      evalConstTargets(In, SubstMap, DL, FAM, Active, Out);
    return;
  }
  if (auto *Sel = llvm::dyn_cast<llvm::SelectInst>(I)) {
    evalConstTargets(Sel->getTrueValue(), SubstMap, DL, FAM, Active, Out);
    evalConstTargets(Sel->getFalseValue(), SubstMap, DL, FAM, Active, Out);
    return;
  }
  if (auto *Cast = llvm::dyn_cast<llvm::CastInst>(I)) {
    llvm::SmallVector<llvm::Constant *> Ops;
    evalConstTargets(Cast->getOperand(0), SubstMap, DL, FAM, Active, Ops);
    for (llvm::Constant *C : Ops)
      add(llvm::ConstantFoldCastOperand(Cast->getOpcode(), C, Cast->getType(),
                                        DL));
    return;
  }
  if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(I); LI && LI->isSimple()) {
    // tryEvalConst already handled a single clobbering store; reaching here
    // means the clobber is a MemoryPhi (divergent stores). Union each one.
    llvm::Function &F = *LI->getFunction();
    llvm::MemorySSA &MSSA = FAM.getResult<llvm::MemorySSAAnalysis>(F).getMSSA();
    llvm::AAResults &AA = FAM.getResult<llvm::AAManager>(F);
    auto *MPhi = llvm::dyn_cast<llvm::MemoryPhi>(
        MSSA.getWalker()->getClobberingMemoryAccess(LI));
    if (!MPhi)
      return;
    llvm::MemoryLocation Loc = llvm::MemoryLocation::get(LI);
    auto toLoadTy = [&](llvm::Constant *C) -> llvm::Constant * {
      if (!C || C->getType() == LI->getType())
        return C;
      return llvm::ConstantFoldCastOperand(llvm::Instruction::BitCast, C,
                                           LI->getType(), DL);
    };
    for (unsigned Idx = 0, E = MPhi->getNumIncomingValues(); Idx < E; ++Idx) {
      auto *Def = llvm::dyn_cast<llvm::MemoryDef>(
          MSSA.getWalker()->getClobberingMemoryAccess(
              MPhi->getIncomingValue(Idx), Loc));
      if (!Def || MSSA.isLiveOnEntryDef(Def))
        continue;
      auto *SI = llvm::dyn_cast_or_null<llvm::StoreInst>(Def->getMemoryInst());
      if (!SI || !storeMatchesLoad(SI, LI, AA, DL))
        continue;
      llvm::SmallVector<llvm::Constant *> SVals;
      evalConstTargets(SI->getValueOperand(), SubstMap, DL, FAM, Active, SVals);
      for (llvm::Constant *C : SVals)
        add(toLoadTy(C));
    }
    return;
  }
}

/// Extract a uint64_t address from a folded constant (ConstantInt or
/// ConstantExpr wrapping an inttoptr).  Returns 0 on failure.
static uint64_t extractAddr(llvm::Constant *C) {
  if (!C)
    return 0;
  if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(C)) {
    if (CE->getOpcode() == llvm::Instruction::IntToPtr)
      if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(CE->getOperand(0)))
        return CI->getZExtValue();
    if (CE->getOpcode() == llvm::Instruction::PtrToInt)
      if (auto *Inner = llvm::dyn_cast<llvm::Constant>(CE->getOperand(0)))
        return extractAddr(Inner);
  }
  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(C))
    return CI->getZExtValue();
  return 0;
}

/// Extract a target-module \c Function* handle from \p C, when it is (or wraps)
/// a direct pointer to a function. Returns \c nullptr otherwise. Handles the
/// common inttoptr/ptrtoint/bitcast wrappers a payload might emit around a
/// function-symbol constant.
static llvm::Function *extractFunctionHandle(llvm::Constant *C) {
  if (!C)
    return nullptr;
  if (auto *F = llvm::dyn_cast<llvm::Function>(C))
    return F;
  if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::PtrToInt:
    case llvm::Instruction::BitCast:
    case llvm::Instruction::AddrSpaceCast:
      if (auto *Inner = llvm::dyn_cast<llvm::Constant>(CE->getOperand(0)))
        return extractFunctionHandle(Inner);
      break;
    default:
      break;
    }
  }
  return nullptr;
}

/// Return the physical register that the machine call \p MI uses as the
/// destination of its indirect branch/call (i.e. the value read as the callee
/// address). Returns an empty \c MCRegister for opcodes not recognized as
/// register-mediated indirect calls.
static llvm::MCRegister getIndirectCallTargetReg(const llvm::MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case llvm::AMDGPU::S_SWAPPC_B64:
    // Operand 0 is the return-address def; operand 1 is the callee register.
    if (MI.getNumOperands() >= 2 && MI.getOperand(1).isReg())
      return MI.getOperand(1).getReg();
    return {};
  case llvm::AMDGPU::S_SETPC_B64:
  case llvm::AMDGPU::S_SETPC_B64_return:
    // Single register operand: the branch target.
    if (MI.getNumOperands() >= 1 && MI.getOperand(0).isReg())
      return MI.getOperand(0).getReg();
    return {};
  default:
    return {};
  }
}

// ---------------------------------------------------------------------------
// Payload map + payload-side resolution helpers
// ---------------------------------------------------------------------------

namespace {

/// Maps each target-module \c MachineInstr (identified via its pcsections
/// \c MDNode) to the ordered list of injected-payload functions attached to
/// it. Built from the IModule + the target module's cached MFs so
/// \c TraceCallGraphAnalysis is self-contained (independent of
/// \c InjectedPayloadAndInstPointAnalysis, which cannot be queried at
/// \c CodeDiscoveryPass time).
using AppMIToPayloadsMap =
    llvm::DenseMap<llvm::MachineInstr *,
                   llvm::SmallVector<llvm::Function *, 2>>;

} // namespace

/// Build the AppMI → payloads map by scanning \p IModule for functions
/// tagged as injected payloads (\c luthier.function.injected_payload) and
/// matching each payload's \c luthier.target_instr_point metadata (which is
/// the pcsections \c MDNode of the target MI it attaches to) against the
/// pcsections nodes of the target module's cached MIs.
static AppMIToPayloadsMap
buildAppMIToPayloadsMap(llvm::Module &TargetModule,
                        llvm::FunctionAnalysisManager &TargetFAM,
                        llvm::Module &IModule) {
  llvm::DenseMap<const llvm::MDNode *, llvm::MachineInstr *> PCSToMI;
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;
    auto *MFRes = TargetFAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);
    if (!MFRes)
      continue;
    for (llvm::MachineBasicBlock &MBB : MFRes->getMF()) {
      for (llvm::MachineInstr &MI : MBB) {
        if (llvm::MDNode *PCS = MI.getPCSections())
          PCSToMI.insert({PCS, &MI});
      }
    }
  }

  AppMIToPayloadsMap Out;
  for (llvm::Function &F : IModule) {
    if (!F.hasFnAttribute(InjectedPayloadAttribute))
      continue;
    llvm::MDNode *MD = F.getMetadata(TargetInstrPointAttr);
    if (!MD)
      continue;
    auto It = PCSToMI.find(MD);
    if (It != PCSToMI.end())
      Out[It->second].push_back(&F);
  }
  return Out;
}

// Forward declaration; defined below \c runTrace.
static bool resolveViaPayloads(
    llvm::Module &TargetModule, llvm::Module &IModule,
    const llvm::DataLayout &IDL, llvm::FunctionAnalysisManager &IFAM,
    const AppMIToPayloadsMap &AppMIToPayloads,
    llvm::DenseMap<uint64_t, llvm::Function *> &AddrToFunc,
    llvm::DenseMap<
        llvm::Function *,
        llvm::SmallVector<std::pair<llvm::CallInst *, llvm::Function *>>>
        &KnownCallers,
    llvm::LLVMContext &Ctx, TraceCallGraph &Out);

// ---------------------------------------------------------------------------
// Full IR-level call-target trace (target module + injected payloads)
// ---------------------------------------------------------------------------

/// Core call-target trace. Populates \p Out with resolved call targets and
/// discovered addresses. Payload-side writes attached to target instrumentation
/// points are always consulted; when \p AppMIToPayloads is empty (e.g. at
/// CodeDiscoveryPass time before any payload has been injected) the payload
/// step is a natural no-op.
static void runTrace(llvm::Module &TargetModule,
                     llvm::FunctionAnalysisManager &TargetFAM,
                     llvm::Module &IModule,
                     llvm::FunctionAnalysisManager &IFAM,
                     const AppMIToPayloadsMap &AppMIToPayloads,
                     TraceCallGraph &Out) {
  const llvm::DataLayout &DL = TargetModule.getDataLayout();
  const llvm::DataLayout &IDL = IModule.getDataLayout();
  llvm::LLVMContext &Ctx = TargetModule.getContext();

  // Build addr → Function* map from functions that have entry-point
  // annotations (device functions only; kernels are excluded because they
  // are never called by other IR functions).
  llvm::DenseMap<uint64_t, llvm::Function *> AddrToFunc;
  for (llvm::Function &F : TargetModule) {
    auto EP = getFunctionEntryPoint(F);
    if (!EP || F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    AddrToFunc[EP->getRawAddress()] = &F;
  }

  // Known-callers map: Function* → list of (call_site, caller_function).
  // Seeded with direct IR calls; extended as indirect calls get resolved.
  using CallerInfo = std::pair<llvm::CallInst *, llvm::Function *>;
  llvm::DenseMap<llvm::Function *, llvm::SmallVector<CallerInfo>> KnownCallers;

  for (llvm::Function &F : TargetModule) {
    for (auto &I : llvm::instructions(F)) {
      auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
      if (!CI)
        continue;
      auto *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;
      KnownCallers[Callee].emplace_back(CI, &F);
      // Record direct call edges in the global call graph
      if (Callee->isIntrinsic() || Callee->isDeclaration())
        continue;
      Out.CallTargets[CI].push_back(Callee);
      if (auto EP = getFunctionEntryPoint(*Callee))
        Out.DiscoveredCallTargetAddresses.insert(EP->getRawAddress());
    }
  }

  // Iterative resolution loop.
  // Each pass may resolve indirect calls, adding new entries to KnownCallers
  // and enabling the next pass to resolve callee-side indirect calls whose
  // target depends on arguments supplied by newly resolved callers.
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (llvm::Function &F : TargetModule) {
      if (F.isDeclaration())
        continue;
      for (auto &I : llvm::instructions(F)) {
        auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
        if (!CI || CI->getCalledFunction() ||
            llvm::isa<llvm::InlineAsm>(CI->getCalledOperand()))
          continue;

        auto tryResolve = [&](const ValConstMap &SubstMap) {
          llvm::SmallVector<llvm::Constant *> Cs;
          llvm::SmallPtrSet<llvm::Value *, 16> Active;
          evalConstTargets(CI->getCalledOperand(), SubstMap, DL, TargetFAM,
                           Active, Cs);
          for (llvm::Constant *C : Cs) {
            // Prefer a direct Function-handle resolution (Constant wraps a
            // Function*): the payload path may emit ptrtoint(@fn) which is
            // reachable in the target module directly without an address.
            if (llvm::Function *TgtFn = extractFunctionHandle(C)) {
              if (TgtFn->getParent() != &TargetModule)
                continue; // filtered by the payload-side error path
              auto &Targets = Out.CallTargets[CI];
              if (llvm::is_contained(Targets, TgtFn))
                continue;
              LLVM_DEBUG(luthier::dbgs()
                         << "[TraceCallGraph] Resolved call in "
                         << F.getName() << " → " << TgtFn->getName()
                         << " (via function handle)\n");
              Targets.push_back(TgtFn);
              KnownCallers[TgtFn].emplace_back(CI, &F);
              Changed = true;
              continue;
            }
            uint64_t Addr = extractAddr(C);
            if (!Addr)
              continue;
            // Always record the raw address so CodeDiscoveryPass can enqueue
            // it as a new entry point even before the callee stub exists.
            Out.DiscoveredCallTargetAddresses.insert(Addr);
            auto It = AddrToFunc.find(Addr);
            if (It == AddrToFunc.end())
              continue;
            llvm::Function *Target = It->second;
            auto &Targets = Out.CallTargets[CI];
            if (llvm::is_contained(Targets, Target))
              continue;
            LLVM_DEBUG(luthier::dbgs()
                       << "[TraceCallGraph] Resolved call in " << F.getName()
                       << " → " << Target->getName() << "\n");
            Targets.push_back(Target);
            KnownCallers[Target].emplace_back(CI, &F);
            Changed = true;
          }
        };

        // Phase 1: evaluate callee with no substitution (manifest constants).
        {
          ValConstMap EmptyMap;
          tryResolve(EmptyMap);
        }

        // Phase 2: inter-procedural — try each known call site of F as a
        // source of argument constants.
        if (auto CallerIt = KnownCallers.find(&F);
            CallerIt != KnownCallers.end()) {
          for (auto &[SiteCI, CallerF] : CallerIt->second) {
            ValConstMap SubstMap;
            ValConstMap SiteCache;
            for (unsigned Idx = 0;
                 Idx < SiteCI->arg_size() && Idx < F.arg_size(); ++Idx) {
              ValConstMap EmptyMap;
              if (llvm::Constant *ArgC = tryEvalConst(
                      SiteCI->getArgOperand(Idx), EmptyMap, SiteCache, DL,
                      TargetFAM))
                SubstMap[F.getArg(Idx)] = ArgC;
            }
            if (!SubstMap.empty())
              tryResolve(SubstMap);
          }
        }
      }
    }

    // Payload-side resolution: consult injected payloads for callee writes
    // that resolve remaining unresolved sites. If it makes progress the
    // fixed-point loop above needs another pass so the newly-resolved sites
    // can feed inter-procedural argument propagation.
    if (!Changed &&
        resolveViaPayloads(TargetModule, IModule, IDL, IFAM,
                           AppMIToPayloads, AddrToFunc, KnownCallers, Ctx, Out))
      Changed = true;
  }

  // Mark incomplete call sites — any indirect call that was not fully
  // resolved (i.e. not present in CallTargets at all, or where at least one
  // call-site of its containing function failed to provide a constant for the
  // callee operand).
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;
    for (auto &I : llvm::instructions(F)) {
      auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
      if (!CI || CI->getCalledFunction() ||
          llvm::isa<llvm::InlineAsm>(CI->getCalledOperand()))
        continue;
      if (!Out.CallTargets.contains(CI)) {
        LLVM_DEBUG(luthier::dbgs() << "[TraceCallGraph] Unresolved call in "
                                   << F.getName() << "\n");
        Out.IncompleteCallSites.insert(CI);
        Out.FullyRecovered = false;
      }
    }
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[TraceCallGraph] Resolved " << Out.CallTargets.size()
             << " call sites; " << Out.IncompleteCallSites.size()
             << " incomplete; fully_recovered=" << Out.FullyRecovered << "\n");
}

// ---------------------------------------------------------------------------
// Payload-side resolution
// ---------------------------------------------------------------------------

/// Union all target-side physical registers written by \p Payload via
/// \c luthier::writeReg calls, together with the folded value(s) each write
/// produces. Callers filter by the physreg the app MI reads as its callee.
namespace {
struct PayloadWrite {
  llvm::MCRegister Dest;
  llvm::SmallVector<llvm::Constant *, 2> Values;
};
} // namespace

static void collectPayloadWrites(
    llvm::Function &Payload, const llvm::DataLayout &IDL,
    llvm::FunctionAnalysisManager &IFAM,
    llvm::SmallVectorImpl<PayloadWrite> &Out) {
  for (llvm::Instruction &I : llvm::instructions(Payload)) {
    auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
    if (!CI)
      continue;
    llvm::Function *Callee = CI->getCalledFunction();
    if (!Callee || !Callee->hasFnAttribute(IntrinsicAttribute))
      continue;
    llvm::StringRef Name =
        Callee->getFnAttribute(IntrinsicAttribute).getValueAsString();
    if (Name != "luthier::writeReg")
      continue;
    if (CI->arg_size() < 2)
      continue;
    auto *DestC = llvm::dyn_cast<llvm::ConstantInt>(CI->getArgOperand(0));
    if (!DestC)
      continue;
    llvm::MCRegister Dest(DestC->getZExtValue());

    PayloadWrite W;
    W.Dest = Dest;
    // Fold the write's value operand in the IModule's per-function
    // MemorySSA/AA context (its FAM). Payload bodies are small and mostly
    // straight-line, so the single-constant fast path in tryEvalConst /
    // evalConstTargets is usually enough.
    llvm::SmallPtrSet<llvm::Value *, 16> Active;
    ValConstMap Empty;
    evalConstTargets(CI->getArgOperand(1), Empty, IDL, IFAM, Active, W.Values);
    if (!W.Values.empty())
      Out.push_back(std::move(W));
  }
}

/// Build a map from a target-module MI's trace address to the MI itself,
/// used to go from a target IR \c CallInst's \c MD_pcsections back to the
/// \c MachineInstr it was lifted from.
static llvm::DenseMap<uint64_t, llvm::MachineInstr *>
buildTraceAddrToMIMap(const AppMIToPayloadsMap &AppMIToPayloads) {
  llvm::DenseMap<uint64_t, llvm::MachineInstr *> Out;
  for (const auto &[MI, _] : AppMIToPayloads) {
    auto *MD = TargetMachineInstrMDNode::getInstrMDNodeIfExists(*MI);
    if (!MD)
      continue;
    if (auto Addr = MD->getTraceInstrAddress())
      Out[*Addr] = MI;
  }
  return Out;
}

/// For each still-unresolved indirect call site, locate the corresponding
/// \c MachineInstr, walk every payload attached to it, and treat any
/// \c writeReg targeting the call MI's callee register as a candidate call
/// target. Returns \c true if it made any progress.
///
/// Called unconditionally from \c runTrace; when \p AppMIToPayloads is empty
/// (no payloads injected yet) this is a natural no-op.
static bool resolveViaPayloads(
    llvm::Module &TargetModule, llvm::Module &IModule,
    const llvm::DataLayout &IDL, llvm::FunctionAnalysisManager &IFAM,
    const AppMIToPayloadsMap &AppMIToPayloads,
    llvm::DenseMap<uint64_t, llvm::Function *> &AddrToFunc,
    llvm::DenseMap<
        llvm::Function *,
        llvm::SmallVector<std::pair<llvm::CallInst *, llvm::Function *>>>
        &KnownCallers,
    llvm::LLVMContext &Ctx, TraceCallGraph &Out) {
  if (AppMIToPayloads.empty())
    return false;

  llvm::DenseMap<uint64_t, llvm::MachineInstr *> TraceAddrToMI =
      buildTraceAddrToMIMap(AppMIToPayloads);

  bool Changed = false;
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;
    for (auto &I : llvm::instructions(F)) {
      auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
      if (!CI || CI->getCalledFunction() ||
          llvm::isa<llvm::InlineAsm>(CI->getCalledOperand()))
        continue;
      if (Out.CallTargets.contains(CI))
        continue; // already resolved (may not be complete but has candidates)

      auto TAddr = getTraceAddr(CI);
      if (!TAddr)
        continue;
      auto MIIt = TraceAddrToMI.find(*TAddr);
      if (MIIt == TraceAddrToMI.end())
        continue;
      llvm::MachineInstr *AppMI = MIIt->second;
      auto PayloadsIt = AppMIToPayloads.find(AppMI);
      if (PayloadsIt == AppMIToPayloads.end())
        continue;

      llvm::MCRegister CalleeReg = getIndirectCallTargetReg(*AppMI);
      if (!CalleeReg)
        continue;
      const llvm::TargetRegisterInfo *TRI =
          AppMI->getMF()->getSubtarget().getRegisterInfo();

      for (llvm::Function *Payload : PayloadsIt->second) {
        llvm::SmallVector<PayloadWrite, 4> Writes;
        collectPayloadWrites(*Payload, IDL, IFAM, Writes);
        for (PayloadWrite &W : Writes) {
          if (!TRI || !TRI->regsOverlap(W.Dest, CalleeReg))
            continue;
          for (llvm::Constant *C : W.Values) {
            llvm::Function *TgtFn = extractFunctionHandle(C);
            if (!TgtFn) {
              if (uint64_t Addr = extractAddr(C)) {
                Out.DiscoveredCallTargetAddresses.insert(Addr);
                TgtFn = AddrToFunc.lookup(Addr);
              }
            }
            if (!TgtFn)
              continue;
            if (TgtFn->getParent() != &TargetModule) {
              // The payload wrote a handle pointing outside the target
              // module — most likely into the IModule itself, which is
              // never a legal call target for the target application.
              Ctx.emitError(
                  llvm::formatv("[TraceCallGraph] Injected payload '{0}' "
                                "attached to instrumentation point in "
                                "target function '{1}' writes register "
                                "'{2}' with a handle to non-target-module "
                                "function '{3}'; instrumentation module "
                                "functions cannot be called from the "
                                "target module.",
                                Payload->getName(), F.getName(),
                                TRI->getName(CalleeReg), TgtFn->getName())
                      .str());
              continue;
            }
            auto &Targets = Out.CallTargets[CI];
            if (llvm::is_contained(Targets, TgtFn))
              continue;
            LLVM_DEBUG(luthier::dbgs()
                       << "[TraceCallGraph] Resolved call in " << F.getName()
                       << " → " << TgtFn->getName() << " via payload '"
                       << Payload->getName() << "'\n");
            Targets.push_back(TgtFn);
            KnownCallers[TgtFn].emplace_back(CI, &F);
            Changed = true;
          }
        }
      }
    }
  }
  return Changed;
}

// ---------------------------------------------------------------------------
// Analysis implementation
// ---------------------------------------------------------------------------

TraceCallGraph
TraceCallGraphAnalysis::run(InstrumentPrototype &IP,
                            InstrumentPrototypeAnalysisManager &IPAM) {
  TraceCallGraph Out;

  llvm::Module &TargetModule = IP.getTargetModule();
  llvm::Module &IModule = IP.getInstrumentationModule();

  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<ModuleAnalysisManagerInstrumentPrototypeProxy>(IP)
          .getManager();

  llvm::FunctionAnalysisManager &TargetFAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();
  llvm::FunctionAnalysisManager &IFAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(IModule)
          .getManager();

  // The AppMI → payloads map is built inline from the IModule's payload
  // functions and the target module's cached MFs. Payload-side resolution
  // always runs; an empty map (no injected payloads yet) makes it a no-op.
  AppMIToPayloadsMap AppMIToPayloads =
      buildAppMIToPayloadsMap(TargetModule, TargetFAM, IModule);

  runTrace(TargetModule, TargetFAM, IModule, IFAM, AppMIToPayloads, Out);
  return Out;
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

void TraceCallGraph::print(llvm::raw_ostream &OS) const {
  OS << "TraceCallGraph (fully_recovered=" << (FullyRecovered ? "yes" : "no")
     << "):\n";

  // CallTargets / IncompleteCallSites are hashed containers with no stable
  // iteration order; format each entry into a string and sort before printing
  // so the output is deterministic across runs.
  auto printSorted = [&OS](llvm::SmallVectorImpl<std::string> &Lines) {
    llvm::sort(Lines);
    for (const std::string &Line : Lines)
      OS << Line << "\n";
  };

  OS << "  Resolved call sites (" << CallTargets.size() << "):\n";
  llvm::SmallVector<std::string> ResolvedLines;
  for (const auto &[CI, Targets] : CallTargets) {
    std::string Line;
    llvm::raw_string_ostream LS(Line);
    LS << "    " << CI->getFunction()->getName() << " -> [";
    llvm::interleaveComma(Targets, LS,
                          [&](llvm::Function *T) { LS << T->getName(); });
    LS << "]";
    ResolvedLines.push_back(std::move(Line));
  }
  printSorted(ResolvedLines);

  OS << "  Incomplete call sites (" << IncompleteCallSites.size() << "):\n";
  llvm::SmallVector<std::string> IncompleteLines;
  for (llvm::CallInst *CI : IncompleteCallSites) {
    std::string Line;
    llvm::raw_string_ostream LS(Line);
    LS << "    " << CI->getFunction()->getName();
    IncompleteLines.push_back(std::move(Line));
  }
  printSorted(IncompleteLines);

  OS << "  Discovered call target addresses ("
     << DiscoveredCallTargetAddresses.size() << "):\n";
  llvm::SmallVector<uint64_t> Sorted(DiscoveredCallTargetAddresses.begin(),
                                     DiscoveredCallTargetAddresses.end());
  llvm::sort(Sorted);
  for (uint64_t Addr : Sorted)
    OS << "    " << llvm::format("0x%" PRIx64 "\n", Addr);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void TraceCallGraph::dump() const { print(luthier::dbgs()); }
#endif

// ---------------------------------------------------------------------------
// Printer pass
// ---------------------------------------------------------------------------

llvm::PreservedAnalyses
TraceCallGraphPrinter::run(InstrumentPrototype &IP,
                           InstrumentPrototypeAnalysisManager &IPAM) {
  IPAM.getResult<TraceCallGraphAnalysis>(IP).print(OS);
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
