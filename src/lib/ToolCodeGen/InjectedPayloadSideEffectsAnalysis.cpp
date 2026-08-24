//===-- InjectedPayloadSideEffectsAnalysis.cpp ----------------------------===//
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
/// Implements the function-level \c InjectedPayloadSideEffectsAnalysis.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
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

llvm::AnalysisKey InjectedPayloadSideEffectsAnalysis::Key;

bool InjectedPayloadSideEffects::invalidate(
    llvm::Function &, const llvm::PreservedAnalyses &PA,
    llvm::FunctionAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<InjectedPayloadSideEffectsAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<llvm::AllAnalysesOn<llvm::Function>>();
}

InjectedPayloadSideEffectsAnalysis::Result
InjectedPayloadSideEffectsAnalysis::run(llvm::Function &F,
                                         llvm::FunctionAnalysisManager &FAM) {
  Result Out;
  if (!F.hasFnAttribute(InjectedPayloadAttribute))
    return Out;

  for (llvm::Instruction &I : llvm::instructions(F)) {
    auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
    if (!CI)
      continue;

    llvm::StringRef IntrinsicName;
    // The value carrying the phys-reg-enum / SVA-enum argument, in whichever
    // operand position the current call form places it.
    const llvm::Value *EnumArg = nullptr;

    if (auto *IA = llvm::dyn_cast<llvm::InlineAsm>(CI->getCalledOperand())) {
      // Post-IR-lowering form: the call is an inline-asm placeholder whose
      // template string is the intrinsic name.
      IntrinsicName = IA->getAsmString();
      // readReg / readSVA: `"=s,i"(i32 enum)`         → arg 0
      // writeReg:         `"s,i"(T val, i32 enum)`    → arg 1
      if (IntrinsicName == "luthier::readReg" ||
          IntrinsicName == "luthier::readSVA")
        EnumArg = CI->getArgOperand(0);
      else if (IntrinsicName == "luthier::writeReg")
        EnumArg = CI->getArgOperand(1);
      else
        continue;
    } else if (llvm::Function *Callee = CI->getCalledFunction();
               Callee && Callee->hasFnAttribute(IntrinsicAttribute)) {
      // Pre-IR-lowering form: the call targets the intrinsic Function decl.
      // Both readReg/readSVA (Reg) and writeReg (Reg, Val) put the phys-reg
      // enum in operand 0.
      IntrinsicName =
          Callee->getFnAttribute(IntrinsicAttribute).getValueAsString();
      if (IntrinsicName != "luthier::readReg" &&
          IntrinsicName != "luthier::writeReg" &&
          IntrinsicName != "luthier::readSVA")
        continue;
      EnumArg = CI->getArgOperand(0);
    } else {
      continue;
    }

    auto *CI32 = llvm::dyn_cast<llvm::ConstantInt>(EnumArg);
    if (!CI32)
      continue;
    uint64_t v = CI32->getZExtValue();
    if (IntrinsicName == "luthier::readReg")
      Out.Reads.insert(v);
    else if (IntrinsicName == "luthier::writeReg")
      Out.Writes.insert(v);
    else // readSVA
      Out.SVAs.insert(static_cast<ScalarValueArgument>(v));
  }

  // Aggregate implicit arg related attributes
  if (!Out.SVAs.contains(IMPLICIT_ARG_BUFFER)) {
    return Out;
  }

  for (auto &[Val, Attr] : std::initializer_list<
           std::pair<amdgpu::hsamd::ValueKind, llvm::StringRef>>{
           {amdgpu::hsamd::ValueKind::HiddenHostcallBuffer,
            "amdgpu-no-hostcall-ptr"},
           {amdgpu::hsamd::ValueKind::HiddenHeapV1, "amdgpu-no-heap-ptr"},
           {amdgpu::hsamd::ValueKind::HiddenMultiGridSyncArg,
            "amdgpu-no-multigrid-sync-arg"},
           {amdgpu::hsamd::ValueKind::HiddenDefaultQueue,
            "amdgpu-no-default-queue"},
           {amdgpu::hsamd::ValueKind::HiddenCompletionAction,
            "amdgpu-no-completion-action"},
           {amdgpu::hsamd::ValueKind::HiddenQueuePtr, "amdgpu-no-queue-ptr"}})
    if (!F.hasFnAttribute(Attr))
      Out.ImplicitArgs.insert(Val);

  return Out;
}

llvm::PreservedAnalyses InjectedPayloadSideEffectsPrinterPass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
  const auto &Result = FAM.getResult<InjectedPayloadSideEffectsAnalysis>(F);
  if (Result.reads_empty() && Result.writes_empty() && Result.svas_empty() &&
      Result.implicit_args_empty())
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
          llvm::iterator_range<InjectedPayloadSideEffects::iterator> Regs) {
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

  auto printSVAs = [&](llvm::iterator_range<
                       InjectedPayloadSideEffects::sva_iterator> SAs) {
    llvm::SmallVector<ScalarValueArgument> Sorted(SAs.begin(), SAs.end());
    llvm::sort(Sorted);
    OS << "    SVAs:";
    for (ScalarValueArgument SA : Sorted)
      OS << " " << static_cast<unsigned>(SA);
    OS << "\n";
  };

  auto printImplicitArgs =
      [&](llvm::iterator_range<
          InjectedPayloadSideEffects::implicit_arg_iterator> Args) {
        llvm::SmallVector<llvm::StringRef> Sorted(Args.begin(), Args.end());
        llvm::sort(Sorted);
        OS << "    ImplicitArgs:";
        for (llvm::StringRef A : Sorted)
          OS << " " << A;
        OS << "\n";
      };

  OS << "Payload " << F.getName() << ":\n";
  printRegs("Reads", Result.reads());
  printRegs("Writes", Result.writes());
  printSVAs(Result.svas());
  printImplicitArgs(Result.implicit_args());
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
