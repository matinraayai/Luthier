//===-- InstrumentationPMDriver.h -------------------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// \file InstrumentationPMDriver.h
/// Describes the \c InstrumentationPMDriver which is a target module pass
/// in charge of the high-level instrumentation process in Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_APPLY_INSTRUMENTATION_PASS_H
#define LUTHIER_TOOLING_APPLY_INSTRUMENTATION_PASS_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/Plugins/LuthierPassPlugin.h"
#include <llvm/IR/PassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <utility>

namespace llvm {

class PassRegistry;
}

namespace luthier {

struct InstrumentationPMDriverOptions {
  llvm::cl::OptionCategory InstrumentationPMDriverOptionsCat{
      "Instrumentation PM Driver Options",
      "Options regarding the Instrumentation PM driver"};

  llvm::cl::opt<bool> ForceFlatScratchInstructions{
      "force-scratch-insts", llvm::cl::init(true),
      llvm::cl::desc(
          "For targets that support it , forces use of scratch instructions "
          "instead of buffer instructions to access scratch memory."),
      llvm::cl::NotHidden, llvm::cl::cat(InstrumentationPMDriverOptionsCat)};
};

class InstrumentationPMDriver
    : public llvm::PassInfoMixin<InstrumentationPMDriver> {

  const InstrumentationPMDriverOptions &Options;

  /// Luthier pass plugins registered with the outer driver
  llvm::ArrayRef<PassPlugin> PassPlugins{};

  using IModuleCreationFnType =
      std::function<std::unique_ptr<llvm::Module>(llvm::LLVMContext &)>;

  /// Function used to materialize the instrumentation module
  IModuleCreationFnType IModuleCreatorFn;

  /// Function used to augment the pass builder
  /// Can be used to add additional analysis and add passes to the IR
  /// optimization stage
  std::function<void(llvm::PassBuilder &)> PassBuilderAugmentationCallback{};

  /// Callbacks invoked before adding any other passes to the IR pass manager
  std::function<void(llvm::ModulePassManager &)> PreIROptimizationCallback{};

  /// Callback invoked before performing the Luthier Intrinsic IR Lowering
  std::function<void(llvm::ModulePassManager &)>
      PreIRIntrinsicLoweringCallback{};

  /// Callback invoked after performing the Luthier Intrinsic IR Lowering
  std::function<void(llvm::ModulePassManager &)>
      PostIRIntrinsicLoweringCallback{};

  /// Callback to augment the instrumentation code generation legacy pipeline
  std::function<void(llvm::PassRegistry &, llvm::TargetPassConfig &,
                     llvm::TargetMachine &)>
      AugmentTargetPassConfigCallback;

public:
  explicit InstrumentationPMDriver(
      const InstrumentationPMDriverOptions &Options,
      llvm::ArrayRef<PassPlugin> PassPlugins = {},
      IModuleCreationFnType ModuleCreatorFn =
          [](llvm::LLVMContext &Ctx) {
            return std::make_unique<llvm::Module>("Instrumentation Module",
                                                  Ctx);
          },
      std::function<void(llvm::PassBuilder &)> PassBuilderAugmentationCallback =
          [](llvm::PassBuilder &PB) {},
      std::function<void(llvm::ModulePassManager &)> PreIROptimizationCallback =
          [](llvm::ModulePassManager &) {},
      std::function<void(llvm::ModulePassManager &)>
          PreIRIntrinsicLoweringCallback = [](llvm::ModulePassManager &) {},
      std::function<void(llvm::ModulePassManager &)>
          PostIRIntrinsicLoweringCallback = [](llvm::ModulePassManager &) {},
      std::function<void(llvm::PassRegistry &, llvm::TargetPassConfig &,
                         llvm::TargetMachine &)>
          AugmentTargetPassConfigCallback = [](llvm::PassRegistry &,
                                               llvm::TargetPassConfig &,
                                               llvm::TargetMachine &) {});

  llvm::PreservedAnalyses run(llvm::Module &TargetAppM,
                              llvm::ModuleAnalysisManager &TargetMAM);
};
} // namespace luthier

#endif