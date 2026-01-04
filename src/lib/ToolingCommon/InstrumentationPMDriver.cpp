//===-- InstrumentationPMDriver.cpp ---------------------------------------===//
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
/// \file InstrumentationPMDriver.cpp
/// Implements the \c InstrumentationPMDriver pass.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/InstrumentationPMDriver.h"
#include "luthier/Tooling/CodeLifter.h"
#include "luthier/Tooling/IModuleIRGeneratorPass.h"
#include "luthier/Tooling/InjectedPayloadPEIPass.h"
#include "luthier/Tooling/InstrumentationModule.h"
#include "luthier/Tooling/IntrinsicMIRLoweringPass.h"
#include "luthier/Tooling/IntrinsicProcessorsAnalysis.h"
#include "luthier/Tooling/PatchLiftedRepresentationPass.h"
#include "luthier/Tooling/PhysRegsNotInLiveInsAnalysis.h"
#include "luthier/Tooling/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/Tooling/WrapperAnalysisPasses.h"
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/RuntimeLibcallInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-apply-instrumentation"

namespace luthier {

InstrumentationPMDriver::InstrumentationPMDriver(
    const InstrumentationPMDriverOptions &Options,
    llvm::ArrayRef<PassPlugin> PassPlugins,
    IModuleCreationFnType ModuleCreatorFn,
    std::function<void(llvm::ModulePassManager &)> PreIROptimizationCallback,
    std::function<void(llvm::ModulePassManager &)>
        PreIRIntrinsicLoweringCallback,
    std::function<void(llvm::ModulePassManager &)>
        PostIRIntrinsicLoweringCallback,
    std::function<void(llvm::PassRegistry &, llvm::TargetPassConfig &,
                       llvm::TargetMachine &)>
        AugmentTargetPassConfigCallback)
    : Options(Options), PassPlugins(PassPlugins),
      IModuleCreatorFn(std::move(ModuleCreatorFn)),
      PreIROptimizationCallback(std::move(PreIROptimizationCallback)),
      PreIRIntrinsicLoweringCallback(std::move(PreIRIntrinsicLoweringCallback)),
      PostIRIntrinsicLoweringCallback(
          std::move(PostIRIntrinsicLoweringCallback)),
      AugmentTargetPassConfigCallback(
          std::move(AugmentTargetPassConfigCallback)) {
  llvm::PassRegistry *Registry = llvm::PassRegistry::getPassRegistry();
  initializeIModuleMAMWrapperPass(*Registry);
  initializePhysicalRegAccessVirtualizationPass(*Registry);
  initializeInjectedPayloadPEIPass(*Registry);
  initializeIntrinsicMIRLoweringPass(*Registry);
  for (const auto &Plugin : PassPlugins) {
    Plugin.registerLegacyCodegenPassesCallback(*Registry);
  }
};

llvm::PreservedAnalyses
InstrumentationPMDriver::run(llvm::Module &TargetAppM,
                             llvm::ModuleAnalysisManager &TargetMAM) {
  llvm::LLVMContext &Context = TargetAppM.getContext();

  const auto &TargetAppTM =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetAppM)
          .getMMI()
          .getTarget();
  const llvm::Target &Target = TargetAppTM.getTarget();

  /// For target machines with absolute flat scratch the AMDGPU backend will
  /// default to using buffer instructions to access scratch unless it is
  /// forced to use scratch instructions by setting "enable-flat-scratch" to
  /// true in its features when creating its target machine
  std::string Features;

  auto ForceEnableFS = [&]() {
    llvm::SubtargetFeatures TargetAppFeatures(
        TargetAppTM.getTargetFeatureString());
    llvm::SubtargetFeatures IModuleFeatures;
    for (llvm::StringRef Feature : TargetAppFeatures.getFeatures()) {
      if (llvm::SubtargetFeatures::StripFlag(Feature) !=
          "enable-flat-scratch") {
        IModuleFeatures.AddFeature(Feature,
                                   llvm::SubtargetFeatures::hasFlag(Feature));
      }
    }
    IModuleFeatures.AddFeature("enable-flat-scratch");
    return IModuleFeatures.getString();
  };

  /// TODO: Add CL options to control TM options and the codegen optimization
  /// level for the Instrumentation TM

  std::unique_ptr<llvm::GCNTargetMachine> ITM{
      static_cast<llvm::GCNTargetMachine *>(Target.createTargetMachine(
          TargetAppTM.getTargetTriple(), TargetAppTM.getTargetCPU(),
          Options.ForceFlatScratchInstructions
              ? ForceEnableFS()
              : TargetAppTM.getTargetFeatureString(),
          TargetAppTM.Options, TargetAppTM.getRelocationModel(),
          TargetAppTM.getCodeModel(), TargetAppTM.getOptLevel()))};

  std::unique_ptr<llvm::Module> IModule = IModuleCreatorFn(Context);
  /// Invoke the module creation callbacks in the plugins and link them with
  /// the current instrumentation module
  for (const auto &Plugin : PassPlugins) {
    std::unique_ptr<llvm::Module> PluginIModule =
        Plugin.instrumentationModuleCreationCallback(
            Context, ITM->getTargetTriple(), ITM->getTargetCPU(),
            ITM->getTargetFeatureString());
    if (PluginIModule != nullptr) {
      /// TODO: Add CL parameter to control the linking flag here
      if (llvm::Linker::linkModules(*IModule, std::move(PluginIModule))) {
        LUTHIER_EMIT_ERROR_IN_CONTEXT(
            Context,
            LUTHIER_MAKE_GENERIC_ERROR("Failed to link modules together"));
      }
    }
  }

  auto MMI = std::make_unique<llvm::MachineModuleInfo>(ITM.get());

  llvm::ModulePassManager IMPM;

  llvm::LoopAnalysisManager ILAM;
  llvm::FunctionAnalysisManager IFAM;
  llvm::CGSCCAnalysisManager ICGAM;
  llvm::ModuleAnalysisManager IMAM;
  llvm::MachineFunctionPassManager IMFPM;
  llvm::MachineFunctionAnalysisManager IMFAM;

  llvm::PassInstrumentationCallbacks PIC;
  llvm::StandardInstrumentations SI(IModule->getContext(), true);

  // Create a PM Builder for the IR pipeline
  llvm::PassBuilder PB(ITM.get(), llvm::PipelineTuningOptions(), std::nullopt,
                       &PIC);

  /// Augment the pass builder
  PassBuilderAugmentationCallback(PB);

  for (const auto &Plugin : PassPlugins) {
    Plugin.registerInstrumentationPassBuilderCallback(PB);
  }
  {
    llvm::TimeTraceScope Scope("Instrumentation Module IR Optimization");
    SI.registerCallbacks(PIC, &IMAM);
    // Add the Intrinsic Lowering Info analysis pass
    IMAM.registerPass([&]() { return IntrinsicIRLoweringInfoMapAnalysis(); });
    // Add the Intrinsic processors Map analysis pass
    IMAM.registerPass([&]() { return IntrinsicsProcessorsAnalysis(); });
    // Add the analysis that accumulates the physical registers accessed that
    // are not in live-ins sets
    IMAM.registerPass([&]() { return PhysRegsNotInLiveInsAnalysis(); });
    // Add the Target app's MAM as an analysis pass
    IMAM.registerPass(
        [&]() { return TargetAppModuleAndMAMAnalysis(TargetMAM, TargetAppM); });
    // Add the analysis for holding the instrumentation point to injected
    // payload mapping
    IMAM.registerPass([&]() { return InjectedPayloadAndInstPointAnalysis(); });
    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(IMAM);
    PB.registerCGSCCAnalyses(ICGAM);
    PB.registerFunctionAnalyses(IFAM);
    PB.registerLoopAnalyses(ILAM);
    PB.crossRegisterProxies(ILAM, IFAM, ICGAM, IMAM);
    /// Call the pre pipeline consturction callbacks
    PreIROptimizationCallback(IMPM);
    for (const auto &Plugin : PassPlugins) {
      Plugin.registerPreIROptimizationPasses(IMPM);
    }
    // Add the IR optimization pipeline
    IMPM.addPass(PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3));
    /// Call the pre IR intrinsic lowering callback
    PreIRIntrinsicLoweringCallback(IMPM);
    for (const auto &Plugin : PassPlugins) {
      Plugin.invokePreLuthierIRIntrinsicLoweringPassesCallback(IMPM);
    }
    // Add the Intrinsic Processing IR stage pass
    IMPM.addPass(ProcessIntrinsicsAtIRLevelPass(*ITM));
    PostIRIntrinsicLoweringCallback(IMPM);
    for (const auto &Plugin : PassPlugins) {
      Plugin.invokePostLuthierIRIntrinsicLoweringPassesCallback(IMPM);
    }
    // Run the scheduled IR passes
    IMPM.run(*IModule, IMAM);
  }

  LLVM_DEBUG(

      llvm::dbgs() << "Instrumentation Module after IR optimization:\n";
      IModule->print(llvm::dbgs(), nullptr)

  );

  // Trace for profiling
  llvm::TimeTraceScope Scope("Instrumentation Module MIR CodeGen Optimization");

  // Target library info pass, required by the code gen pipeline
  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(IModule->getTargetTriple()));

  // Instantiate the Legacy PM for running the modified codegen pipeline
  // on the instrumentation module and MMI
  // We allocate this on the heap to have the most control over its lifetime,
  // as if it goes out of scope it will also delete the instrumentation
  // MMI
  auto LegacyIPM = new llvm::legacy::PassManager();
  // Instrumentation module MMI wrapper pass, which will house the final
  // generate instrumented code
  auto *IMMIWP = new llvm::MachineModuleInfoWrapperPass(&TargetAppTM);

  LegacyIPM->add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  auto *TPC = ITM->createPassConfig(*LegacyIPM);

  TPC->setDisableVerify(true);

  LegacyIPM->add(TPC);

  LegacyIPM->add(IMMIWP);

  TPC->addISelPasses();

  LegacyIPM->add(new PhysicalRegAccessVirtualizationPass());
  LegacyIPM->add(new IntrinsicMIRLoweringPass());
  TPC->insertPass(&llvm::PrologEpilogCodeInserterID,
                  new InjectedPayloadPEIPass());
  TPC->addMachinePasses();

  // Add the kernel pre-amble emission pass
  LegacyIPM->add(new PrePostAmbleEmitter());
  // Add the lifted representation patching pass
  LegacyIPM->add(new PatchLiftedRepresentationPass());

  llvm::PassRegistry *Registry = llvm::PassRegistry::getPassRegistry();

  /// Invoke the codegen pipeline augmentation callback
  AugmentTargetPassConfigCallback(*Registry, *TPC, *ITM);

  for (const auto &Plugin : PassPlugins) {
    Plugin.invokeAugmentTargetPassConfigCallback(*Registry, *TPC, *ITM);
  }

  TPC->setInitialized();

  bool Modified = LegacyIPM->run(*IModule);

  delete LegacyIPM;

  return Modified ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all();
}
} // namespace luthier