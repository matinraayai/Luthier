//===-- InstrumentPrototypePassBuilder.cpp -------------------------------===//
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
/// Implements \c InstrumentPrototypePassBuilder::parsePipeline.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InstrumentPrototypePassBuilder.h"
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/ErrorHandling.h>

namespace luthier {

void InstrumentPrototypePassBuilder::crossRegisterProxies(
    llvm::ModuleAnalysisManager &MAM, llvm::FunctionAnalysisManager &FAM,
    llvm::MachineFunctionAnalysisManager &MFAM,
    InstrumentPrototypeAnalysisManager &IPAM) {
  // The adaptors run pass instrumentation at the InstrumentPrototype level, so
  // PassInstrumentationAnalysis must be available on its analysis manager.
  // Register it against the PIC held by the wrapped llvm::PassBuilder so
  // callbacks are shared with the nested Module/Function/MachineFunction
  // managers.
  llvm::PassInstrumentationCallbacks *PIC =
      PB.getPassInstrumentationCallbacks();
  IPAM.registerPass([PIC] { return llvm::PassInstrumentationAnalysis(PIC); });

  IPAM.registerPass(
      [&] { return ModuleAnalysisManagerInstrumentPrototypeProxy(MAM); });
  IPAM.registerPass(
      [&] { return FunctionAnalysisManagerInstrumentPrototypeProxy(FAM); });
  IPAM.registerPass([&] {
    return MachineFunctionAnalysisManagerInstrumentPrototypeProxy(MFAM);
  });

  MAM.registerPass(
      [&] { return InstrumentPrototypeAnalysisManagerModuleProxy(IPAM); });
  FAM.registerPass(
      [&] { return InstrumentPrototypeAnalysisManagerFunctionProxy(IPAM); });
  MFAM.registerPass([&] {
    return InstrumentPrototypeAnalysisManagerMachineFunctionProxy(IPAM);
  });
}

llvm::Error InstrumentPrototypePassBuilder::parsePipeline(
    InstrumentPrototypePassManager &IPPM, llvm::StringRef PipelineText) {
  llvm::StringRef Remaining = PipelineText.trim();

  while (!Remaining.empty()) {
    bool IsTarget = Remaining.consume_front("target(");
    bool IsInstrumentation =
        !IsTarget && Remaining.consume_front("instrumentation(");

    if (!IsTarget && !IsInstrumentation) {
      return llvm::make_error<llvm::StringError>(
          "expected 'target(...)' or 'instrumentation(...)' at top level of "
          "-passes (bare pass names are not allowed)",
          llvm::inconvertibleErrorCode());
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
      return llvm::make_error<llvm::StringError>(
          "unmatched parentheses in InstrumentPrototype pass pipeline",
          llvm::inconvertibleErrorCode());
    }

    llvm::StringRef InnerText = Remaining.substr(0, Pos);
    Remaining = Remaining.substr(Pos + 1).ltrim();
    if (Remaining.consume_front(","))
      Remaining = Remaining.ltrim();

    bool Handled = false;
    for (auto &Cb : ParseCallbacks) {
      if (Cb(InnerText, IPPM, IsTarget)) {
        Handled = true;
        break;
      }
    }
    if (Handled)
      continue;

    llvm::ModulePassManager InnerMPM;
    if (auto Err = PB.parsePassPipeline(InnerMPM, InnerText))
      return Err;

    if (IsTarget)
      IPPM.addPass(createRunOnTargetModuleAdaptor(std::move(InnerMPM)));
    else
      IPPM.addPass(
          createRunOnInstrumentationModuleAdaptor(std::move(InnerMPM)));
  }

  return llvm::Error::success();
}

} // namespace luthier
