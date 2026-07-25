//===-- ForwardISAStateToCalleesPass.cpp ----------------------------------===//
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
/// \file ForwardISAStateToCalleesPass.cpp
/// Implements the \c ForwardISAStateToCalleesPass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/ForwardISAStateToCalleesPass.h"
#include "AMDGPUTargetMachine.h"
#include "SIRegisterInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/AttributeMask.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/SSAUpdater.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-forward-isa-state-to-callees"

namespace luthier {

namespace {

//===----------------------------------------------------------------------===//
// Common types
//===----------------------------------------------------------------------===//

/// Per-Function accumulated ISA-state needs. Phys-regs are stored at 32-bit
/// channel granularity (after TRI decomposition). Sorted vectors are derived
/// from these sets when building signatures.
struct RequiredISAState {
  llvm::SmallDenseSet<ScalarValueArgument, 4> SVA;
  llvm::DenseSet<llvm::MCRegister> ReadPhysRegs;
  llvm::DenseSet<llvm::MCRegister> WrittenPhysRegs;

  bool empty() const {
    return SVA.empty() && ReadPhysRegs.empty() && WrittenPhysRegs.empty();
  }
};

/// Lookup table: placeholder key → cached effects record decoded from the
/// named-MD entry's 4th operand. Placeholders whose intrinsic has no
/// callee-visible ISA-state effects map to an entry with all three vectors
/// empty.
using PlaceholderEffectsMap =
    llvm::DenseMap<llvm::StringRef, IntrinsicISAStateEffects>;

PlaceholderEffectsMap buildPlaceholderEffectsMap(const llvm::Module &IModule) {
  PlaceholderEffectsMap Out;
  const llvm::NamedMDNode *NamedMD =
      IModule.getNamedMetadata(LuthierIntrinsicNamedMDName);
  if (!NamedMD)
    return Out;
  for (const llvm::MDNode *Entry : NamedMD->operands()) {
    // Need key (operand 0) and effects (operand 3). Older 3-operand entries
    // (no effects) implicitly have no effects.
    if (!Entry || Entry->getNumOperands() < 3)
      continue;
    auto *KeyMD = llvm::dyn_cast<llvm::MDString>(Entry->getOperand(0));
    if (!KeyMD)
      continue;
    const llvm::MDNode *EffNode =
        Entry->getNumOperands() >= 4
            ? llvm::dyn_cast<llvm::MDNode>(Entry->getOperand(3))
            : nullptr;
    Out.try_emplace(KeyMD->getString(),
                    decodeIntrinsicISAStateEffects(EffNode));
  }
  return Out;
}

/// Identify a Luthier inline-asm placeholder call. Returns the placeholder
/// key (the asm template string) if \p CB is one; empty StringRef otherwise.
llvm::StringRef getPlaceholderKeyOfCall(const llvm::CallBase &CB) {
  auto *IA = llvm::dyn_cast<llvm::InlineAsm>(CB.getCalledOperand());
  if (!IA)
    return {};
  llvm::StringRef AsmStr = IA->getAsmString();
  if (!AsmStr.starts_with(LuthierIntrinsicPlaceholderKeyPrefix))
    return {};
  return AsmStr;
}

/// Decompose \p Reg into its 32-bit channels using TRI. For a 32-bit register
/// returns {Reg}. For wider registers, returns the 32-bit sub-registers in
/// channel order. For sub-32-bit registers, returns the 32-bit super register
/// (so writes/reads aggregate at channel granularity, matching the MIR
/// processors' behaviour).
llvm::SmallVector<llvm::MCRegister, 4>
decomposeRegToChannels(llvm::MCRegister Reg, const llvm::SIRegisterInfo &TRI) {
  llvm::SmallVector<llvm::MCRegister, 4> Out;
  const llvm::TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
  unsigned Size = RC ? TRI.getRegSizeInBits(*RC) : 32u;
  if (Size <= 32) {
    if (Size < 32)
      Out.push_back(TRI.get32BitRegister(Reg));
    else
      Out.push_back(Reg);
    return Out;
  }
  size_t NumChannels = Size / 32;
  for (size_t I = 0; I < NumChannels; ++I) {
    unsigned SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(I);
    Out.push_back(TRI.getSubReg(Reg, SubIdx));
  }
  return Out;
}

/// Returns true if \p F is a body-bearing Function that participates in the
/// IModule call graph (i.e. not a declaration, not an inline-asm pseudo).
bool isParticipatingFunction(const llvm::Function &F) {
  return !F.isDeclaration() && !F.isIntrinsic();
}

//===----------------------------------------------------------------------===//
// Stage 2a — Needs analysis
//===----------------------------------------------------------------------===//

/// Per-Function placeholder effects, derived from this Function's body alone
/// (no transitive call-graph propagation yet).
llvm::Error collectLocalEffects(llvm::Function &F,
                                const PlaceholderEffectsMap &EffectsByKey,
                                const llvm::SIRegisterInfo &TRI,
                                RequiredISAState &Out) {
  for (llvm::Instruction &I : llvm::instructions(F)) {
    auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
    if (!CB)
      continue;
    llvm::StringRef Key = getPlaceholderKeyOfCall(*CB);
    if (Key.empty())
      continue;
    auto It = EffectsByKey.find(Key);
    if (It == EffectsByKey.end())
      continue;
    const IntrinsicISAStateEffects &Eff = It->second;
    for (ScalarValueArgument SA : Eff.ReadSVAs)
      Out.SVA.insert(SA);
    for (llvm::MCRegister R : Eff.ReadPhysRegs)
      for (llvm::MCRegister C : decomposeRegToChannels(R, TRI))
        Out.ReadPhysRegs.insert(C);
    for (llvm::MCRegister R : Eff.WrittenPhysRegs)
      for (llvm::MCRegister C : decomposeRegToChannels(R, TRI))
        Out.WrittenPhysRegs.insert(C);
  }
  return llvm::Error::success();
}

/// Resolve the callee of \p CB to a concrete Function, allowing one level of
/// pointer-cast stripping. Returns nullptr for truly opaque indirect calls
/// (those need to be diagnosed by the caller).
llvm::Function *resolveCallee(const llvm::CallBase &CB) {
  if (llvm::Function *Direct = CB.getCalledFunction())
    return Direct;
  if (llvm::Value *Op = CB.getCalledOperand())
    return llvm::dyn_cast<llvm::Function>(Op->stripPointerCasts());
  return nullptr;
}

/// Build the per-Function transitive Needs map. Also collects:
///   - Functions reachable from any payload (used downstream to limit
///     signature rewrites and to drive diagnostics).
///   - Opaque-indirect-call sites for diagnostics.
struct NeedsAnalysisResult {
  llvm::DenseMap<llvm::Function *, RequiredISAState> Needs;
  llvm::SmallPtrSet<llvm::Function *, 16> PayloadReachable;
};

llvm::Expected<NeedsAnalysisResult>
runNeedsAnalysis(llvm::Module &IModule,
                 const PlaceholderEffectsMap &EffectsByKey,
                 const llvm::SIRegisterInfo &TRI) {
  NeedsAnalysisResult R;

  // Seed: local effects per Function.
  for (llvm::Function &F : IModule.functions()) {
    if (!isParticipatingFunction(F))
      continue;
    RequiredISAState Local;
    LUTHIER_RETURN_ON_ERROR(collectLocalEffects(F, EffectsByKey, TRI, Local));
    if (!Local.empty() || true) {
      // Always insert so callers can find the entry.
      R.Needs.try_emplace(&F, std::move(Local));
    }
  }

  // Build IR call graph (caller -> callees), report opaque indirect calls.
  llvm::DenseMap<llvm::Function *, llvm::SmallPtrSet<llvm::Function *, 4>>
      Callees;
  for (llvm::Function &F : IModule.functions()) {
    if (!isParticipatingFunction(F))
      continue;
    auto &Out = Callees[&F];
    for (llvm::Instruction &I : llvm::instructions(F)) {
      auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
      if (!CB)
        continue;
      // Skip Luthier placeholders — they don't represent IR-level callees.
      if (!getPlaceholderKeyOfCall(*CB).empty())
        continue;
      llvm::Function *Callee = resolveCallee(*CB);
      if (!Callee) {
        // Truly opaque indirect call: only fail if the surrounding chain is
        // payload-reachable (cheaper diagnostic). We accumulate now and check
        // payload-reachability after the propagation step.
        Callee = nullptr; // explicit
        continue;
      }
      Out.insert(Callee);
    }
  }

  // Determine payload-reachable Functions.
  llvm::SmallVector<llvm::Function *, 16> Worklist;
  for (llvm::Function &F : IModule.functions()) {
    if (isParticipatingFunction(F) &&
        F.hasFnAttribute(InjectedPayloadAttribute) &&
        R.PayloadReachable.insert(&F).second)
      Worklist.push_back(&F);
  }
  while (!Worklist.empty()) {
    llvm::Function *F = Worklist.pop_back_val();
    auto It = Callees.find(F);
    if (It == Callees.end())
      continue;
    for (llvm::Function *C : It->second)
      if (R.PayloadReachable.insert(C).second)
        Worklist.push_back(C);
  }

  // Diagnose opaque indirect calls + bodyless transitively-needed callees in
  // payload-reachable scope.
  bool HardError = false;
  for (llvm::Function *F : R.PayloadReachable) {
    for (llvm::Instruction &I : llvm::instructions(*F)) {
      auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
      if (!CB)
        continue;
      if (!getPlaceholderKeyOfCall(*CB).empty())
        continue;
      if (CB->getCalledFunction())
        continue;
      llvm::Value *Op = CB->getCalledOperand();
      if (!Op)
        continue;
      if (llvm::isa<llvm::Function>(Op->stripPointerCasts()))
        continue;
      IModule.getContext().emitError(
          CB, "luthier: opaque indirect call in injected-payload-reachable "
              "function — ISA-state forwarding cannot be planned through "
              "this edge.");
      HardError = true;
    }
  }
  if (HardError)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "ForwardISAStateToCalleesPass: opaque indirect call(s) blocked "
        "forwarding analysis.");

  // Transitive propagation: Needs[F] ∪= Needs[callee] for every direct edge.
  // Worklist-driven until fixed point.
  llvm::DenseMap<llvm::Function *, llvm::SmallPtrSet<llvm::Function *, 4>>
      Callers;
  for (auto &[F, CS] : Callees)
    for (llvm::Function *C : CS)
      Callers[C].insert(F);

  llvm::SmallVector<llvm::Function *, 32> WL;
  for (auto &[F, _] : R.Needs)
    WL.push_back(F);
  while (!WL.empty()) {
    llvm::Function *F = WL.pop_back_val();
    const auto &NF = R.Needs[F];
    auto CallerIt = Callers.find(F);
    if (CallerIt == Callers.end())
      continue;
    for (llvm::Function *P : CallerIt->second) {
      auto &NP = R.Needs[P];
      bool Changed = false;
      for (auto X : NF.SVA)
        Changed |= NP.SVA.insert(X).second;
      for (auto X : NF.ReadPhysRegs)
        Changed |= NP.ReadPhysRegs.insert(X).second;
      for (auto X : NF.WrittenPhysRegs)
        Changed |= NP.WrittenPhysRegs.insert(X).second;
      if (Changed)
        WL.push_back(P);
    }
  }

  // Error if a bodyless Function is transitively needed by a payload.
  for (llvm::Function *F : R.PayloadReachable) {
    auto NIt = R.Needs.find(F);
    if (NIt == R.Needs.end() || NIt->second.empty())
      continue;
    if (F->isDeclaration() && !F->isIntrinsic()) {
      IModule.getContext().emitError(llvm::formatv(
          "luthier: external-declaration callee '{0}' is transitively "
          "needed by an injected payload but has no body to rewrite.",
          F->getName()));
      HardError = true;
    }
  }
  if (HardError)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "ForwardISAStateToCalleesPass: bodyless transitively-needed "
        "callee(s) blocked forwarding analysis.");

  return R;
}

//===----------------------------------------------------------------------===//
// Stage 2b — signature rewrite
//===----------------------------------------------------------------------===//

/// Per-Function rewrite plan. Param layout after the rewrite is:
///   <orig params> ++ <one i32 per SVALaneSlots entry, in order>
///                 ++ <one i32 per InputChannelSlots entry, in order>
/// Return layout: <orig return> if WriteChannelSlots is empty; else
/// <StructType{ orig return (if non-void), one i32 per WriteChannelSlots }>.
struct RewritePlan {
  llvm::SmallVector<std::pair<ScalarValueArgument, uint8_t>, 4> SVALaneSlots;
  llvm::SmallVector<llvm::MCRegister, 4> InputChannelSlots;
  llvm::SmallVector<llvm::MCRegister, 4> WriteChannelSlots;
};

/// True if \p F is a candidate for signature rewriting.
bool isRewriteCandidate(const llvm::Function &F) {
  if (!isParticipatingFunction(F))
    return false;
  if (F.hasFnAttribute(InjectedPayloadAttribute))
    return false;
  if (F.hasAddressTaken())
    return false;
  return true;
}

RewritePlan buildRewritePlan(const RequiredISAState &Needs) {
  RewritePlan P;
  // SVA: sorted by enum value, expanded per-lane.
  llvm::SmallVector<ScalarValueArgument, 4> SortedSVA(Needs.SVA.begin(),
                                                      Needs.SVA.end());
  llvm::sort(SortedSVA);
  for (ScalarValueArgument SA : SortedSVA) {
    unsigned Lanes = StateValueArraySpecs::getArgumentLaneSize(SA);
    for (uint8_t L = 0; L < Lanes; ++L)
      P.SVALaneSlots.emplace_back(SA, L);
  }
  // Input channels: union of read + write channels, sorted by id.
  llvm::DenseSet<llvm::MCRegister> InputSet;
  for (auto R : Needs.ReadPhysRegs)
    InputSet.insert(R);
  for (auto R : Needs.WrittenPhysRegs)
    InputSet.insert(R);
  P.InputChannelSlots.assign(InputSet.begin(), InputSet.end());
  llvm::sort(P.InputChannelSlots, [](llvm::MCRegister A, llvm::MCRegister B) {
    return A.id() < B.id();
  });
  // Write channels: sorted by id.
  P.WriteChannelSlots.assign(Needs.WrittenPhysRegs.begin(),
                             Needs.WrittenPhysRegs.end());
  llvm::sort(P.WriteChannelSlots, [](llvm::MCRegister A, llvm::MCRegister B) {
    return A.id() < B.id();
  });
  return P;
}

/// Compute the new return type for a Function given its rewrite plan and
/// original return type.
llvm::Type *computeNewReturnType(llvm::Type *OrigRet, const RewritePlan &P,
                                 llvm::LLVMContext &Ctx) {
  if (P.WriteChannelSlots.empty())
    return OrigRet;
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  llvm::SmallVector<llvm::Type *, 4> Members;
  if (!OrigRet->isVoidTy())
    Members.push_back(OrigRet);
  for (size_t I = 0; I < P.WriteChannelSlots.size(); ++I)
    Members.push_back(I32);
  return llvm::StructType::get(Ctx, Members);
}

/// Assemble \p Lanes (each i32, in lane-0-first order) into a value of type
/// \p TargetTy. Returns the assembled IR Value, inserted via \p B.
llvm::Value *assembleLanesToType(llvm::IRBuilderBase &B,
                                 llvm::ArrayRef<llvm::Value *> Lanes,
                                 llvm::Type *TargetTy) {
  unsigned Width = Lanes.size() * 32;
  llvm::Type *WideInt =
      llvm::IntegerType::get(B.getContext(), std::max(Width, 32u));
  llvm::Value *Acc = llvm::ConstantInt::get(WideInt, 0);
  for (size_t I = 0; I < Lanes.size(); ++I) {
    llvm::Value *Z = B.CreateZExt(Lanes[I], WideInt);
    if (I != 0)
      Z = B.CreateShl(Z, llvm::ConstantInt::get(WideInt, I * 32));
    Acc = B.CreateOr(Acc, Z);
  }
  // Cast to TargetTy if the types differ.
  if (Acc->getType() == TargetTy)
    return Acc;
  if (TargetTy->isPointerTy())
    return B.CreateIntToPtr(Acc, TargetTy);
  if (TargetTy->isIntegerTy())
    return B.CreateZExtOrTrunc(Acc, TargetTy);
  return B.CreateBitCast(Acc, TargetTy);
}

/// Disassemble \p V (typed `Ty`) into \p NumLanes i32 values, lane-0-first.
void disassembleValueToLanes(llvm::IRBuilderBase &B, llvm::Value *V,
                             unsigned NumLanes,
                             llvm::SmallVectorImpl<llvm::Value *> &Out) {
  llvm::Type *WideInt =
      llvm::IntegerType::get(B.getContext(), std::max(NumLanes * 32u, 32u));
  llvm::Value *Wide = V;
  if (V->getType()->isPointerTy())
    Wide = B.CreatePtrToInt(V, WideInt);
  else if (V->getType()->isIntegerTy())
    Wide = B.CreateZExtOrTrunc(V, WideInt);
  else
    Wide = B.CreateBitCast(V, WideInt);
  llvm::Type *I32 = llvm::Type::getInt32Ty(B.getContext());
  for (unsigned I = 0; I < NumLanes; ++I) {
    llvm::Value *Shifted =
        I == 0 ? Wide
               : B.CreateLShr(Wide, llvm::ConstantInt::get(WideInt, I * 32));
    Out.push_back(B.CreateTrunc(Shifted, I32));
  }
}

/// Information accumulated while rewriting a single Function body. Owned
/// per-Function.
struct BodyRewriteState {
  /// SA → ordered list of i32 lane params (NewFn arg pointers).
  llvm::DenseMap<ScalarValueArgument, llvm::SmallVector<llvm::Argument *, 4>>
      SVALaneParams;
  /// Input channel → its i32 lane param.
  llvm::DenseMap<llvm::MCRegister, llvm::Argument *> InputChannelParams;
};

/// Rewrite placeholders inside \p NewFn (a freshly-cloned callee body) and
/// build the per-channel SSA-Updater state so return blocks can assemble
/// the struct return.
llvm::Error rewriteCalleeBody(llvm::Function &NewFn, const RewritePlan &Plan,
                              const PlaceholderEffectsMap &EffectsByKey,
                              const llvm::SIRegisterInfo &TRI,
                              const BodyRewriteState &BRS) {
  llvm::LLVMContext &Ctx = NewFn.getContext();

  // Per-channel ordered list of writes encountered in program order. Final
  // value per (BB, channel) is the last entry.
  using WritesPerBB = llvm::DenseMap<llvm::BasicBlock *, llvm::Value *>;
  llvm::DenseMap<llvm::MCRegister, WritesPerBB> WritesByChannel;

  // Collect placeholders to process; we mutate while walking, so snapshot.
  llvm::SmallVector<llvm::CallInst *, 16> Placeholders_v;
  for (llvm::Instruction &I : llvm::instructions(NewFn))
    if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
      if (!getPlaceholderKeyOfCall(*CI).empty())
        Placeholders_v.push_back(CI);

  for (llvm::CallInst *CI : Placeholders_v) {
    llvm::StringRef Key = getPlaceholderKeyOfCall(*CI);
    auto It = EffectsByKey.find(Key);
    if (It == EffectsByKey.end())
      continue;
    const IntrinsicISAStateEffects &Eff = It->second;
    if (Eff.ReadSVAs.empty() && Eff.ReadPhysRegs.empty() &&
        Eff.WrittenPhysRegs.empty())
      continue;

    llvm::IRBuilder<> B(CI);

    if (!Eff.ReadSVAs.empty()) {
      // Single-SVA placeholder: assemble lane params, RAUW, erase.
      ScalarValueArgument SA = Eff.ReadSVAs.front();
      auto LIt = BRS.SVALaneParams.find(SA);
      if (LIt == BRS.SVALaneParams.end())
        return LUTHIER_MAKE_GENERIC_ERROR(
            "Internal: readSVA placeholder references SA not in plan.");
      llvm::SmallVector<llvm::Value *, 4> Lanes;
      for (llvm::Argument *A : LIt->second)
        Lanes.push_back(A);
      llvm::Value *Assembled = assembleLanesToType(B, Lanes, CI->getType());
      CI->replaceAllUsesWith(Assembled);
      CI->eraseFromParent();
      continue;
    }

    if (!Eff.ReadPhysRegs.empty()) {
      llvm::MCRegister R = Eff.ReadPhysRegs.front();
      auto Channels = decomposeRegToChannels(R, TRI);
      llvm::SmallVector<llvm::Value *, 4> Lanes;
      for (llvm::MCRegister C : Channels) {
        auto CIt = BRS.InputChannelParams.find(C);
        if (CIt == BRS.InputChannelParams.end())
          return LUTHIER_MAKE_GENERIC_ERROR(
              "Internal: readReg placeholder references channel not in plan.");
        Lanes.push_back(CIt->second);
      }
      llvm::Value *Assembled = assembleLanesToType(B, Lanes, CI->getType());
      CI->replaceAllUsesWith(Assembled);
      CI->eraseFromParent();
      continue;
    }

    if (!Eff.WrittenPhysRegs.empty()) {
      llvm::MCRegister R = Eff.WrittenPhysRegs.front();
      auto Channels = decomposeRegToChannels(R, TRI);
      // writeReg placeholder: arg layout produced by writeRegIRProcessor is
      // [value-to-write] (plus an inline-asm-return slot that we ignore).
      if (CI->arg_size() == 0)
        return LUTHIER_MAKE_GENERIC_ERROR(
            "Internal: writeReg placeholder has no input operand.");
      llvm::Value *Input = CI->getArgOperand(0);
      llvm::SmallVector<llvm::Value *, 4> Lanes;
      disassembleValueToLanes(B, Input, Channels.size(), Lanes);
      for (size_t I = 0; I < Channels.size(); ++I) {
        WritesByChannel[Channels[I]][CI->getParent()] = Lanes[I];
      }
      // The placeholder has a (dead) inline-asm return; RAUW with undef to
      // be safe in case downstream uses leaked through.
      if (!CI->getType()->isVoidTy())
        CI->replaceAllUsesWith(llvm::UndefValue::get(CI->getType()));
      CI->eraseFromParent();
      continue;
    }
  }

  // If there are no write channels, leave returns alone.
  if (Plan.WriteChannelSlots.empty())
    return llvm::Error::success();

  // Per-channel SSAUpdater: seed input parameter as the entry-block initial
  // value, then add per-BB final-write availability.
  llvm::DenseMap<llvm::MCRegister, std::unique_ptr<llvm::SSAUpdater>> Updaters;
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);

  llvm::BasicBlock *EntryBB = &NewFn.getEntryBlock();
  for (llvm::MCRegister C : Plan.WriteChannelSlots) {
    auto U = std::make_unique<llvm::SSAUpdater>();
    U->Initialize(I32, ("luthier.fwd.chan." + llvm::Twine(C.id())).str());
    // Seed entry with input param. If there's a write in EntryBB, override.
    auto IPIt = BRS.InputChannelParams.find(C);
    if (IPIt == BRS.InputChannelParams.end())
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Internal: write-only channel missing input param (plan should "
          "have added it for preserve semantics).");
    U->AddAvailableValue(EntryBB, IPIt->second);
    auto WIt = WritesByChannel.find(C);
    if (WIt != WritesByChannel.end()) {
      for (auto &[BB, V] : WIt->second) {
        // Last write in each BB wins; if BB == EntryBB, the write overrides
        // the input-param seed.
        U->AddAvailableValue(BB, V);
      }
    }
    Updaters[C] = std::move(U);
  }

  // Rewrite every ReturnInst to produce the struct.
  llvm::SmallVector<llvm::ReturnInst *, 4> Returns;
  for (llvm::BasicBlock &BB : NewFn)
    if (auto *RI = llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator()))
      Returns.push_back(RI);

  llvm::Type *NewRetTy = NewFn.getReturnType();
  for (llvm::ReturnInst *RI : Returns) {
    llvm::IRBuilder<> B(RI);
    llvm::Value *Aggregate = llvm::UndefValue::get(NewRetTy);
    unsigned NextSlot = 0;
    llvm::Type *OrigRetTy = RI->getReturnValue()
                                ? RI->getReturnValue()->getType()
                                : llvm::Type::getVoidTy(Ctx);
    if (!OrigRetTy->isVoidTy()) {
      Aggregate =
          B.CreateInsertValue(Aggregate, RI->getReturnValue(), {NextSlot++});
    }
    for (llvm::MCRegister C : Plan.WriteChannelSlots) {
      llvm::Value *V = Updaters[C]->GetValueAtEndOfBlock(RI->getParent());
      Aggregate = B.CreateInsertValue(Aggregate, V, {NextSlot++});
    }
    B.CreateRet(Aggregate);
    RI->eraseFromParent();
  }

  return llvm::Error::success();
}

/// Clone \p OldFn into a fresh Function with the rewrite plan applied, and
/// rewrite its body. Returns the new Function (with the old name; the old
/// Function is RAUW'd and erased by the caller). Populates \p PlanByOldFn so
/// stage 2c can rewrite call sites.
llvm::Expected<llvm::Function *>
rewriteCallee(llvm::Function &OldFn, const RewritePlan &Plan,
              const PlaceholderEffectsMap &EffectsByKey,
              const llvm::SIRegisterInfo &TRI) {
  llvm::LLVMContext &Ctx = OldFn.getContext();
  llvm::FunctionType *OldFT = OldFn.getFunctionType();

  llvm::SmallVector<llvm::Type *, 8> NewParams(OldFT->params().begin(),
                                               OldFT->params().end());
  size_t OrigParamCount = NewParams.size();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  for (size_t I = 0; I < Plan.SVALaneSlots.size(); ++I)
    NewParams.push_back(I32);
  for (size_t I = 0; I < Plan.InputChannelSlots.size(); ++I)
    NewParams.push_back(I32);

  llvm::Type *NewRetTy =
      computeNewReturnType(OldFT->getReturnType(), Plan, Ctx);
  llvm::FunctionType *NewFT =
      llvm::FunctionType::get(NewRetTy, NewParams, OldFT->isVarArg());

  llvm::Function *NewFn =
      llvm::Function::Create(NewFT, OldFn.getLinkage(), OldFn.getAddressSpace(),
                             "", OldFn.getParent());
  NewFn->takeName(&OldFn);
  NewFn->copyAttributesFrom(&OldFn);
  // Strip any return-type attributes that no longer apply to the new return.
  llvm::AttributeMask RetMask;
  for (llvm::Attribute A : NewFn->getAttributes().getRetAttrs())
    RetMask.addAttribute(A);
  NewFn->removeRetAttrs(RetMask);

  // Set arg names for readability.
  llvm::ValueToValueMapTy VMap;
  auto NewArgIt = NewFn->arg_begin();
  for (auto &OldArg : OldFn.args()) {
    llvm::StringRef OldName = OldArg.getName();
    if (!OldName.empty())
      NewArgIt->setName(OldName);
    VMap[&OldArg] = &*NewArgIt;
    ++NewArgIt;
  }
  BodyRewriteState BRS;
  for (auto &Slot : Plan.SVALaneSlots) {
    std::string Name =
        ("luthier.fwd.sva." + llvm::Twine(static_cast<unsigned>(Slot.first)) +
         "." + llvm::Twine(static_cast<unsigned>(Slot.second)))
            .str();
    NewArgIt->setName(llvm::StringRef(Name));
    BRS.SVALaneParams[Slot.first].push_back(&*NewArgIt);
    ++NewArgIt;
  }
  for (llvm::MCRegister C : Plan.InputChannelSlots) {
    std::string Name = ("luthier.fwd.chan." + llvm::Twine(C.id())).str();
    NewArgIt->setName(llvm::StringRef(Name));
    BRS.InputChannelParams[C] = &*NewArgIt;
    ++NewArgIt;
  }
  (void)OrigParamCount;

  llvm::SmallVector<llvm::ReturnInst *, 4> Returns;
  llvm::CloneFunctionInto(NewFn, &OldFn, VMap,
                          llvm::CloneFunctionChangeType::LocalChangesOnly,
                          Returns);

  // Now rewrite placeholders + build return-struct.
  if (auto Err = rewriteCalleeBody(*NewFn, Plan, EffectsByKey, TRI, BRS)) {
    NewFn->eraseFromParent();
    return std::move(Err);
  }

  return NewFn;
}

//===----------------------------------------------------------------------===//
// Stage 2c — call-site rewrite
//===----------------------------------------------------------------------===//

/// Per-caller cache: for each (SA, lane) the i32 lane Value to pass; for each
/// channel the i32 Value to pass. Populated lazily by emitting fresh
/// readSVA/readReg inline-asm placeholders in the caller and stashing the
/// extracted lane/channel.
struct CallerForwardCache {
  llvm::DenseMap<ScalarValueArgument, llvm::SmallVector<llvm::Value *, 4>>
      SVALaneValues;
  llvm::DenseMap<llvm::MCRegister, llvm::Value *> ChannelValues;
};

/// Build a new placeholder call mirroring the shape
/// `ProcessIntrinsicsAtIRLevelPass` would have produced for the requested
/// intrinsic, registering it in the module's named-MD if not already present.
/// Returns the call's value.
///
/// Only supports the read-shape (no args, single inline-asm output) and the
/// writeReg-shape (one i32 arg, no output). Suffices for callee forwarding.
struct PlaceholderFactory {
  llvm::Module &M;
  llvm::LLVMContext &Ctx;
  unsigned NextId;
  // Map: (intrinsic name, aux-signature-string) -> placeholder key. Allows
  // re-use across calls in the same factory lifetime.
  llvm::StringMap<std::string> KeyByContent;

  explicit PlaceholderFactory(llvm::Module &M)
      : M(M), Ctx(M.getContext()),
        NextId(static_cast<unsigned>(
            M.getOrInsertNamedMetadata(LuthierIntrinsicNamedMDName)
                ->getNumOperands())) {}

  llvm::CallInst *emitReadPlaceholder(llvm::IRBuilderBase &B,
                                      llvm::StringRef IntrinsicName,
                                      llvm::StringRef Constraint,
                                      llvm::Type *RetTy,
                                      int32_t AuxConstantI32) {
    std::string ContentKey =
        (IntrinsicName + "#" + llvm::Twine(AuxConstantI32) + "#" + Constraint)
            .str();
    std::string Key;
    auto It = KeyByContent.find(ContentKey);
    if (It != KeyByContent.end()) {
      Key = It->second;
    } else {
      Key = (llvm::Twine(LuthierIntrinsicPlaceholderKeyPrefix) +
             llvm::Twine(NextId++))
                .str();
      KeyByContent[ContentKey] = Key;
      llvm::Metadata *AuxOps[1] = {llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), AuxConstantI32))};
      llvm::MDNode *AuxNode = llvm::MDNode::get(Ctx, AuxOps);
      llvm::NamedMDNode *NMD =
          M.getOrInsertNamedMetadata(LuthierIntrinsicNamedMDName);
      NMD->addOperand(llvm::MDNode::get(
          Ctx, {llvm::MDString::get(Ctx, Key),
                llvm::MDString::get(Ctx, IntrinsicName), AuxNode}));
    }
    auto *FT = llvm::FunctionType::get(RetTy, {}, false);
    auto *IA = llvm::InlineAsm::get(FT, Key, ("=" + Constraint).str(),
                                    /*hasSideEffects=*/true);
    return B.CreateCall(IA, {});
  }

  llvm::CallInst *emitWriteRegPlaceholder(llvm::IRBuilderBase &B,
                                          llvm::MCRegister Channel,
                                          llvm::Value *I32Value) {
    // Re-use any prior writeReg placeholder for this exact channel + value
    // type. The inline-asm has type (i32) -> i32 (def slot is unused).
    llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
    std::string ContentKey =
        ("luthier::writeReg#" + llvm::Twine(Channel.id()) + "#sv").str();
    std::string Key;
    auto It = KeyByContent.find(ContentKey);
    if (It != KeyByContent.end()) {
      Key = It->second;
    } else {
      Key = (llvm::Twine(LuthierIntrinsicPlaceholderKeyPrefix) +
             llvm::Twine(NextId++))
                .str();
      KeyByContent[ContentKey] = Key;
      llvm::Metadata *AuxOps[1] = {llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(I32, Channel.id()))};
      llvm::MDNode *AuxNode = llvm::MDNode::get(Ctx, AuxOps);
      llvm::NamedMDNode *NMD =
          M.getOrInsertNamedMetadata(LuthierIntrinsicNamedMDName);
      NMD->addOperand(llvm::MDNode::get(
          Ctx, {llvm::MDString::get(Ctx, Key),
                llvm::MDString::get(Ctx, "luthier::writeReg"), AuxNode}));
    }
    auto *FT = llvm::FunctionType::get(I32, {I32}, false);
    auto *IA = llvm::InlineAsm::get(FT, Key, "=s,s",
                                    /*hasSideEffects=*/true);
    return B.CreateCall(IA, {I32Value});
  }
};

/// Ensure that the caller-side cache for \p Caller has values for every
/// (SA, lane) and channel needed by \p Plan; emit fresh placeholders at the
/// top of the caller's entry block if missing.
void primeCallerCache(llvm::Function &Caller, const RewritePlan &Plan,
                      const llvm::SIRegisterInfo &TRI, CallerForwardCache &C,
                      PlaceholderFactory &PF) {
  llvm::IRBuilder<> B(&*Caller.getEntryBlock().getFirstInsertionPt());

  // For each unique SA used in Plan: emit a single readSVA placeholder of the
  // SA's natural width, then trunc+lshr into per-lane i32s. Cache lanes.
  llvm::SmallDenseSet<ScalarValueArgument, 4> SeenSVA;
  for (auto &Slot : Plan.SVALaneSlots) {
    if (!SeenSVA.insert(Slot.first).second)
      continue;
    auto &Lanes = C.SVALaneValues[Slot.first];
    if (!Lanes.empty())
      continue;
    unsigned NumLanes = StateValueArraySpecs::getArgumentLaneSize(Slot.first);
    llvm::Type *Wide =
        llvm::IntegerType::get(B.getContext(), std::max(NumLanes * 32u, 32u));
    llvm::CallInst *Read = PF.emitReadPlaceholder(
        B, "luthier::readSVA", "s", Wide, static_cast<int32_t>(Slot.first));
    disassembleValueToLanes(B, Read, NumLanes, Lanes);
  }

  // For each input channel: emit a readReg placeholder per channel.
  for (llvm::MCRegister Channel : Plan.InputChannelSlots) {
    if (C.ChannelValues.count(Channel))
      continue;
    llvm::CallInst *Read = PF.emitReadPlaceholder(
        B, "luthier::readReg", "s", llvm::Type::getInt32Ty(B.getContext()),
        static_cast<int32_t>(Channel.id()));
    C.ChannelValues[Channel] = Read;
  }
}

/// Rewrite a single call site that targets \p OldCallee → \p NewCallee. The
/// new call's extra args come from \p CC (which must already be primed for
/// the plan).
void rewriteCallSite(llvm::CallBase *CB, llvm::Function *OldCallee,
                     llvm::Function *NewCallee, const RewritePlan &Plan,
                     const llvm::SIRegisterInfo &TRI, CallerForwardCache &CC,
                     PlaceholderFactory &PF) {
  llvm::IRBuilder<> B(CB);
  llvm::SmallVector<llvm::Value *, 8> Args(CB->args());
  for (auto &Slot : Plan.SVALaneSlots)
    Args.push_back(CC.SVALaneValues[Slot.first][Slot.second]);
  for (llvm::MCRegister C : Plan.InputChannelSlots)
    Args.push_back(CC.ChannelValues[C]);

  llvm::CallInst *NewCall =
      B.CreateCall(NewCallee->getFunctionType(), NewCallee, Args);
  NewCall->setCallingConv(NewCallee->getCallingConv());
  NewCall->setDebugLoc(CB->getDebugLoc());
  // Copy metadata that's meaningful at the call site.
  llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 4> MDs;
  CB->getAllMetadata(MDs);
  for (auto &[Kind, MD] : MDs)
    NewCall->setMetadata(Kind, MD);

  // Materialize result(s).
  llvm::Type *OrigRetTy = OldCallee->getReturnType();
  llvm::Value *NewOrigRet = nullptr;
  unsigned NextSlot = 0;
  if (!Plan.WriteChannelSlots.empty()) {
    if (!OrigRetTy->isVoidTy()) {
      NewOrigRet = B.CreateExtractValue(NewCall, {NextSlot++});
    }
    for (llvm::MCRegister Channel : Plan.WriteChannelSlots) {
      llvm::Value *Extracted = B.CreateExtractValue(NewCall, {NextSlot++});
      PF.emitWriteRegPlaceholder(B, Channel, Extracted);
    }
  } else {
    NewOrigRet = NewCall;
  }

  if (!OrigRetTy->isVoidTy() && NewOrigRet != nullptr)
    CB->replaceAllUsesWith(NewOrigRet);
  CB->eraseFromParent();
}

} // namespace

llvm::PreservedAnalyses
ForwardISAStateToCalleesPass::run(llvm::Module &IModule,
                                  llvm::ModuleAnalysisManager &MAM) {
  LLVM_DEBUG(luthier::dbgs() << "=== ForwardISAStateToCalleesPass: module '"
                          << IModule.getName() << "' ===\n");

  PlaceholderEffectsMap EffectsByKey = buildPlaceholderEffectsMap(IModule);
  if (EffectsByKey.empty()) {
    LLVM_DEBUG(luthier::dbgs() << "  No placeholders; nothing to do.\n");
    return llvm::PreservedAnalyses::all();
  }
  const auto &TM =
      MAM.getResult<llvm::MachineModuleAnalysis>(IModule).getMMI().getTarget();

  const auto *Subtarget = TM.getSubtargetImpl(*IModule.functions().begin());
  // Fall back to a Function-specific subtarget if the first Function is a
  // declaration (no body). In practice the IModule always contains at least
  // one body-bearing Function by the time this pass runs.
  if (!Subtarget) {
    IModule.getContext().emitError(
        "ForwardISAStateToCalleesPass: could not resolve subtarget.");
    return llvm::PreservedAnalyses::all();
  }
  const auto *TRI =
      static_cast<const llvm::SIRegisterInfo *>(Subtarget->getRegisterInfo());

  auto NeedsOrErr = runNeedsAnalysis(IModule, EffectsByKey, *TRI);
  if (auto Err = NeedsOrErr.takeError()) {
    IModule.getContext().emitError(llvm::toString(std::move(Err)));
    return llvm::PreservedAnalyses::all();
  }

  LLVM_DEBUG({
    for (auto &[F, N] : NeedsOrErr->Needs) {
      if (N.empty())
        continue;
      luthier::dbgs() << "  Needs[" << F->getName() << "]: " << N.SVA.size()
                   << " SVA, " << N.ReadPhysRegs.size() << " readReg, "
                   << N.WrittenPhysRegs.size() << " writeReg\n";
    }
  });

  // Stage 2b: build a rewrite plan and clone every non-payload, non-hook
  // payload-reachable Function with non-empty Needs.
  llvm::DenseMap<llvm::Function *, llvm::Function *> OldToNew;
  llvm::DenseMap<llvm::Function *, RewritePlan> PlanByOldFn;
  for (auto *OldFn : NeedsOrErr->PayloadReachable) {
    if (!isRewriteCandidate(*OldFn))
      continue;
    auto NIt = NeedsOrErr->Needs.find(OldFn);
    if (NIt == NeedsOrErr->Needs.end() || NIt->second.empty())
      continue;
    RewritePlan Plan = buildRewritePlan(NIt->second);
    auto NewOrErr = rewriteCallee(*OldFn, Plan, EffectsByKey, *TRI);
    if (auto Err = NewOrErr.takeError()) {
      IModule.getContext().emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
    OldToNew[OldFn] = *NewOrErr;
    PlanByOldFn[OldFn] = std::move(Plan);
  }

  if (OldToNew.empty())
    return llvm::PreservedAnalyses::all();

  // Stage 2c: rewrite all call sites that reference a rewritten Function.
  // Walk the module's current Functions (still-present plus the new clones —
  // payload entries are in the former set, the clones in the latter).
  PlaceholderFactory PF(IModule);
  llvm::DenseMap<llvm::Function *, CallerForwardCache> CallerCaches;

  // The set of Functions whose call sites we need to inspect: everything
  // payload-reachable except the old Functions we're about to delete.
  llvm::SmallVector<llvm::Function *, 16> Callers;
  for (llvm::Function *F : NeedsOrErr->PayloadReachable) {
    if (OldToNew.count(F))
      continue; // its body will be deleted; skip
    Callers.push_back(F);
  }
  // Also add the NEW clones — they may call other rewritten callees.
  for (auto &[Old, New] : OldToNew)
    Callers.push_back(New);

  for (llvm::Function *Caller : Callers) {
    // Snapshot calls because we'll mutate.
    llvm::SmallVector<llvm::CallBase *, 8> Calls;
    for (llvm::Instruction &I : llvm::instructions(*Caller)) {
      auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
      if (!CB)
        continue;
      llvm::Function *Direct = CB->getCalledFunction();
      if (!Direct) {
        if (llvm::Value *Op = CB->getCalledOperand())
          Direct = llvm::dyn_cast<llvm::Function>(Op->stripPointerCasts());
      }
      if (!Direct)
        continue;
      if (OldToNew.count(Direct))
        Calls.push_back(CB);
    }
    if (Calls.empty())
      continue;

    auto &Cache = CallerCaches[Caller];
    for (llvm::CallBase *CB : Calls) {
      llvm::Function *OldCallee = CB->getCalledFunction();
      if (!OldCallee) {
        OldCallee = llvm::dyn_cast<llvm::Function>(
            CB->getCalledOperand()->stripPointerCasts());
      }
      const RewritePlan &Plan = PlanByOldFn[OldCallee];
      primeCallerCache(*Caller, Plan, *TRI, Cache, PF);
      rewriteCallSite(CB, OldCallee, OldToNew[OldCallee], Plan, *TRI, Cache,
                      PF);
    }
  }

  // Drop old Functions.
  for (auto &[Old, New] : OldToNew) {
    Old->replaceAllUsesWith(llvm::UndefValue::get(Old->getType()));
    Old->eraseFromParent();
  }
  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
