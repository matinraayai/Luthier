//===-- OptPlugin.cpp -----------------------------------------------------===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
/// Main file for the Luthier "opt" compiler plugin, which registers Luthier
/// passes and their names with the new pass manager's pass builder when loaded.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/AMDGPURegisterLiveness.h"
#include "luthier/Tooling/InstrumentationPMDriver.h"
#include "luthier/Tooling/IntrinsicMIRLoweringPass.h"
#include "luthier/Tooling/LRCallgraph.h"
#include "luthier/Tooling/MMISlotIndexesAnalysis.h"
#include "luthier/Tooling/MockAMDGPULoader.h"
#include "luthier/Tooling/MockLoadAMDGPUCodeObjects.h"
#include "luthier/Tooling/PhysRegsNotInLiveInsAnalysis.h"
#include "luthier/Tooling/PrePostAmbleEmitter.h"
#include "luthier/Tooling/RunIRPassesOnIModulePass.h"
#include "luthier/Tooling/SVStorageAndLoadLocations.h"
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Plugins/PassPlugin.h>

namespace luthier {

static std::unique_ptr<MockAMDGPULoader> Loader{nullptr};

static MockAMDGPULoaderAnalysisOptions MockLoaderOptions;

static InstrumentationPMDriverOptions InstrumentationPMOptions;

} // namespace luthier

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {

  const auto Callback = [](llvm::PassBuilder &PB) {
    /// Register Luthier module analysis passes
    PB.registerAnalysisRegistrationCallback(
        [](llvm::ModuleAnalysisManager &MAM) {
          MAM.registerPass(
              []() { return luthier::AMDGPURegLivenessAnalysis(); });
          MAM.registerPass([]() { return luthier::LRCallGraphAnalysis(); });
          MAM.registerPass([]() { return luthier::MMISlotIndexesAnalysis(); });
          MAM.registerPass([]() {
            return luthier::LRStateValueStorageAndLoadLocationsAnalysis();
          });
          MAM.registerPass(
              []() { return luthier::FunctionPreambleDescriptorAnalysis(); });
          MAM.registerPass(
              []() { return luthier::MockAMDGPULoaderAnalysis(); });
        });

    PB.registerPipelineParsingCallback(
        [&](llvm::StringRef Name, llvm::ModulePassManager &MPM,
            llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
          if (Name == "luthier-apply-instrumentation") {
            MPM.addPass(luthier::InstrumentationPMDriver(
                luthier::InstrumentationPMOptions));
            return true;
          }
          return false;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "luthier-opt", LLVM_VERSION_STRING, Callback,
          nullptr};
}