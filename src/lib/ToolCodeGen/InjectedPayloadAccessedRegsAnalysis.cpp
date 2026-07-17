//===-- InjectedPayloadAccessedRegsAnalysis.cpp ---------------------------===//
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
/// Implements the function-level \c InjectedPayloadAccessedRegsAnalysis.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InjectedPayloadAccessedRegsAnalysis.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorsAnalysis.h"
#include <AMDGPUTargetMachine.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace luthier {

llvm::AnalysisKey InjectedPayloadAccessedRegsAnalysis::Key;

bool InjectedPayloadAccessedRegs::invalidate(
    llvm::Function &, const llvm::PreservedAnalyses &PA,
    llvm::FunctionAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<InjectedPayloadAccessedRegsAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<llvm::AllAnalysesOn<llvm::Function>>();
}

namespace {

/// If \p CI's callee is a Luthier inline-asm placeholder (emitted by
/// \c ProcessIntrinsicsAtIRLevelPass ), returns its opaque template-string
/// key; otherwise returns an empty \c StringRef .
llvm::StringRef getPlaceholderKey(const llvm::CallInst &CI) {
  auto *IA = llvm::dyn_cast<llvm::InlineAsm>(CI.getCalledOperand());
  if (!IA)
    return {};
  llvm::StringRef AsmStr = IA->getAsmString();
  if (!AsmStr.starts_with(LuthierIntrinsicPlaceholderKeyPrefix))
    return {};
  return AsmStr;
}

using PlaceholderEffectsMap =
    llvm::DenseMap<llvm::StringRef, IntrinsicISAStateEffects>;

/// Build a map from placeholder key to decoded
/// \c IntrinsicISAStateEffects by scanning the module's
/// \c !luthier.intrinsic.placeholders named metadata. Returns an empty map
/// when the named node is absent (i.e. \c ProcessIntrinsicsAtIRLevelPass
/// has not run yet).
PlaceholderEffectsMap buildPlaceholderEffectsMap(const llvm::Module &M) {
  PlaceholderEffectsMap Out;
  const llvm::NamedMDNode *NamedMD =
      M.getNamedMetadata(LuthierIntrinsicNamedMDName);
  if (!NamedMD)
    return Out;
  for (const llvm::MDNode *Entry : NamedMD->operands()) {
    if (!Entry || Entry->getNumOperands() < 4)
      continue;
    auto *KeyMD = llvm::dyn_cast<llvm::MDString>(Entry->getOperand(0));
    if (!KeyMD)
      continue;
    const auto *EffNode = llvm::dyn_cast<llvm::MDNode>(Entry->getOperand(3));
    Out.try_emplace(KeyMD->getString(),
                    decodeIntrinsicISAStateEffects(EffNode));
  }
  return Out;
}

} // namespace

InjectedPayloadAccessedRegsAnalysis::Result
InjectedPayloadAccessedRegsAnalysis::run(llvm::Function &F,
                                         llvm::FunctionAnalysisManager &FAM) {
  Result Out;
  if (!F.hasFnAttribute(InjectedPayloadAttribute))
    return Out;

  llvm::Module &IModule = *F.getParent();
  llvm::LLVMContext &Ctx = F.getContext();

  auto &MAMProxy = FAM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);

  // Post-lowering path is driven by module-level metadata; populated lazily
  // and left empty when the named MD is absent (pre-lowering).
  PlaceholderEffectsMap Placeholders = buildPlaceholderEffectsMap(IModule);

  // TM and the intrinsic-processor registry are only required for the
  // pre-lowering path — resolve on first use so purely post-lowering runs
  // don't fail if either happens to be uncached.
  const llvm::GCNTargetMachine *TM = nullptr;
  const IntrinsicsProcessorsAnalysis::Result *Processors = nullptr;
  bool ProcessorsLookupFailed = false;
  bool TMLookupFailed = false;

  auto getProcessors = [&]() -> const IntrinsicsProcessorsAnalysis::Result * {
    if (Processors || ProcessorsLookupFailed)
      return Processors;
    Processors =
        MAMProxy.getCachedResult<IntrinsicsProcessorsAnalysis>(IModule);
    if (!Processors) {
      Ctx.emitError("InjectedPayloadAccessedRegsAnalysis: "
                    "IntrinsicsProcessorsAnalysis was not cached in the "
                    "module analysis manager.");
      ProcessorsLookupFailed = true;
    }
    return Processors;
  };

  auto getTM = [&]() -> const llvm::GCNTargetMachine * {
    if (TM || TMLookupFailed)
      return TM;
    auto *MMA = MAMProxy.getCachedResult<llvm::MachineModuleAnalysis>(IModule);
    if (!MMA) {
      Ctx.emitError(
          "InjectedPayloadAccessedRegsAnalysis: "
          "MachineModuleAnalysis is required but not cached in the module "
          "analysis manager.");
      TMLookupFailed = true;
      return nullptr;
    }
    TM =
        &static_cast<const llvm::GCNTargetMachine &>(MMA->getMMI().getTarget());
    return TM;
  };

  auto unionEffects = [&](const IntrinsicISAStateEffects &Eff) {
    for (llvm::MCRegister R : Eff.ReadPhysRegs)
      Out.Reads.insert(R);
    for (llvm::MCRegister R : Eff.WrittenPhysRegs)
      Out.Writes.insert(R);
  };

  for (llvm::Instruction &I : llvm::instructions(F)) {
    auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
    if (!CI)
      continue;

    // Post-lowering path: the call is a Luthier inline-asm placeholder;
    // read effects out of the module's placeholder named-MD side channel.
    if (llvm::StringRef Key = getPlaceholderKey(*CI); !Key.empty()) {
      auto It = Placeholders.find(Key);
      if (It != Placeholders.end())
        unionEffects(It->second);
      continue;
    }

    // Pre-lowering path: the call targets a Function decl attributed as a
    // Luthier intrinsic; invoke the IR processor to obtain its effects.
    llvm::Function *Callee = CI->getCalledFunction();
    if (!Callee || !Callee->hasFnAttribute(IntrinsicAttribute))
      continue;
    llvm::StringRef IntrinsicName =
        Callee->getFnAttribute(IntrinsicAttribute).getValueAsString();

    const auto *P = getProcessors();
    if (!P)
      continue;
    std::optional<IntrinsicProcessor> Processor =
        P->getProcessorIfRegistered(IntrinsicName);
    if (!Processor.has_value()) {
      Ctx.emitError(
          CI, llvm::formatv("Intrinsic {0} is not registered", IntrinsicName)
                  .str());
      continue;
    }
    const llvm::GCNTargetMachine *TargetM = getTM();
    if (!TargetM)
      continue;

    llvm::Expected<IntrinsicIRLoweringInfo> InfoOrErr =
        Processor->IRProcessor(*Callee, *CI, *TargetM);
    if (auto Err = InfoOrErr.takeError()) {
      Ctx.emitError(CI, llvm::toString(std::move(Err)));
      continue;
    }
    unionEffects(InfoOrErr->getEffects());
  }

  return Out;
}

llvm::PreservedAnalyses InjectedPayloadAccessedRegsPrinterPass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
  const auto &Result = FAM.getResult<InjectedPayloadAccessedRegsAnalysis>(F);
  if (Result.reads_empty() && Result.writes_empty())
    return llvm::PreservedAnalyses::all();

  const llvm::TargetRegisterInfo *TRI = nullptr;
  if (auto *MMA =
          FAM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F)
              .getCachedResult<llvm::MachineModuleAnalysis>(*F.getParent())) {
    TRI = static_cast<const llvm::GCNTargetMachine &>(MMA->getMMI().getTarget())
              .getSubtargetImpl(F)
              ->getRegisterInfo();
  }

  auto printRegs =
      [&](const char *Label,
          llvm::iterator_range<InjectedPayloadAccessedRegs::iterator> Regs) {
        llvm::SmallVector<llvm::MCRegister> Sorted(Regs.begin(), Regs.end());
        llvm::sort(Sorted, [](llvm::MCRegister A, llvm::MCRegister B) {
          return A.id() < B.id();
        });
        OS << "    " << Label << ":";
        for (llvm::MCRegister R : Sorted) {
          OS << " ";
          if (TRI)
            OS << TRI->getName(R);
          else
            OS << R.id();
        }
        OS << "\n";
      };

  OS << "Payload " << F.getName() << ":\n";
  printRegs("Reads", Result.reads());
  printRegs("Writes", Result.writes());
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
