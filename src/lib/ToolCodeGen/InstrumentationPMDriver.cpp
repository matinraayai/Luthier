//===-- InstrumentationPMDriver.cpp ---------------------------------------===//
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
/// \file InstrumentationPMDriver.cpp
/// Implements the \c InstrumentationPMDriver pass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InstrumentationPMDriver.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/ForwardISAStateToCalleesPass.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessIModulePass.h"
#include "luthier/ToolCodeGen/InjectedPayloadAccessedRegsAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadPreserveLiveRegsPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadPEIPass.h"
#include "luthier/ToolCodeGen/IntrinsicMIRLoweringPass.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorsAnalysis.h"
#include "luthier/ToolCodeGen/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/ToolCodeGen/SVAPhysVGPRPinPass.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/TargetModulePatcherPass.h"
// #include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
// #include "luthier/ToolCodeGen/InjectedPayloadPEIPass.h"
// #include "luthier/ToolCodeGen/IntrinsicMIRLoweringPass.h"
// #include "luthier/ToolCodeGen/IntrinsicProcessorsAnalysis.h"
// #include "luthier/ToolCodeGen/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/WrapperAnalysisPasses.h"
#include "luthier/ToolCodeGenTesting/LuthierFile.h"
#include <AMDGPU.h>
#include <AMDGPUTargetMachine.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/RuntimeLibcallInfo.h>
#include <llvm/CodeGen/MIRParser/MIRParser.h>
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/PassInfo.h>
#include <llvm/PassRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-apply-instrumentation"

namespace luthier {

namespace {

llvm::OptimizationLevel optLevelFromUnsigned(unsigned Level) {
  switch (Level) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 3:
    return llvm::OptimizationLevel::O3;
  default:
    return llvm::OptimizationLevel::O2;
  }
}

struct LoadedIModule {
  std::unique_ptr<llvm::Module> Module;
  std::unique_ptr<llvm::MIRParser> MIRParser; // non-null only for .mir inputs
};

llvm::Expected<LoadedIModule> loadIModuleFromFile(llvm::StringRef Path,
                                                  llvm::LLVMContext &Ctx) {
  if (Path.ends_with(".mir")) {
    auto MBOrErr = llvm::MemoryBuffer::getFile(Path);
    if (!MBOrErr)
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to open imodule file '" +
                                        Path.str() +
                                        "': " + MBOrErr.getError().message());
    auto Parser = llvm::createMIRParser(std::move(*MBOrErr), Ctx);
    auto M = Parser->parseIRModule();
    if (!M)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Failed to parse IR from imodule file '" + Path.str() + "'");
    return LoadedIModule{std::move(M), std::move(Parser)};
  }

  if (Path.ends_with(".ll") || Path.ends_with(".bc")) {
    llvm::SMDiagnostic Err;
    auto M = llvm::parseIRFile(Path, Err, Ctx);
    if (!M)
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to parse imodule file '" +
                                        Path.str() +
                                        "': " + Err.getMessage().str());
    return LoadedIModule{std::move(M), nullptr};
  }

  return LUTHIER_MAKE_GENERIC_ERROR(
      "Unrecognized imodule file extension for '" + Path.str() +
      "': expected .mir, .ll, or .bc");
}

/// Parses a series of codegen pipeline passes specified in \p PipelineStr
/// and adds them directly to the legacy \p PM. For ModulePass-level passes with
/// cross-level analysis dependencies (e.g. IntrinsicMIRLoweringPass
/// depending on per-function MachineDominatorTree), direct PM.add() is
/// required.
///
/// Two special tokens parallel LLVM's TargetPassConfig API:
///   - "isel" → TPC.addISelPasses()
///   - "machine-passes" → TPC.addMachinePasses() (the full mid+late MIR
///     pipeline including RA and stock PEI). Pin Luthier MIR passes to a
///     specific slot by listing them BEFORE or AFTER this token; the driver
///     also inserts InjectedPayloadPEIPass via TPC->insertPass(...) so it
///     fires after PrologEpilogCodeInserterID inside the machine-passes
///     block automatically (gated by --disable-injected-payload-pei).
llvm::Error parseCodeGenPipeline(llvm::StringRef PipelineStr,
                                 llvm::legacy::PassManager &PM,
                                 llvm::TargetPassConfig &TPC,
                                 llvm::raw_ostream *MIRPrintStream,
                                 bool &UserInsertedMIRPrint) {
  UserInsertedMIRPrint = false;
  llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
  llvm::SmallVector<llvm::StringRef, 8> PassNames;
  PipelineStr.split(PassNames, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef Name : PassNames) {
    Name = Name.trim();
    if (Name == "isel") {
      TPC.addISelPasses();
      continue;
    }
    if (Name == "machine-passes") {
      // Insert SVAPhysVGPRPinPass right before SIPreAllocateWWMRegs so the
      // WWM greedy regalloc honors our hint pinning the SVA LaneVGPR to
      // the load-plan physreg. The insertPass call must come BEFORE
      // addMachinePasses for the AMDGPU TPC to honor it.
      TPC.insertPass(&llvm::SIPreAllocateWWMRegsLegacyID,
                     &SVAPhysVGPRPinPass::ID);
      TPC.addMachinePasses();
      continue;
    }
    if (Name == "print-mir") {
      // Boundary-level MIR dump. Adds the printer at the current position
      // in the outer pass list (between two `parseCodeGenPipeline` tokens,
      // NOT inside `machine-passes`). Useful for dumping post-mir-lowering
      // MIR before kicking off RA.
      if (!MIRPrintStream)
        return LUTHIER_MAKE_GENERIC_ERROR(
            "print-mir token used but no -imodule-output stream configured");
      PM.add(llvm::createPrintMIRPass(*MIRPrintStream));
      UserInsertedMIRPrint = true;
      continue;
    }
    if (Name.starts_with("print-mir-after=")) {
      // Insert a MachineFunction printer AFTER a specific machine pass
      // inside `addMachinePasses()`. Routed through TPC.insertPass, which
      // stores the request in the TPC's InsertedPasses map; the actual
      // scheduling happens when `machine-passes` triggers
      // `addMachinePasses()`.
      //
      // Uses createMachineFunctionPrinterPass (plain MF::print) rather
      // than createPrintMIRPass (YAML round-trippable). The plain printer
      // is immune to the YAML printer's stale-stack-object-index crash
      // when called mid-pipeline (post-RA / pre-frame-elim, MFI can
      // contain orphan CSI references that MIRPrinter's convertStackObjects
      // chokes on). Output isn't valid MIR for re-parsing but is fine for
      // FileCheck.
      llvm::StringRef PassName = Name.substr(strlen("print-mir-after="));
      const llvm::PassInfo *AfterPI = Registry.getPassInfo(PassName);
      if (!AfterPI)
        return LUTHIER_MAKE_GENERIC_ERROR(
            "print-mir-after=: unknown pass name '" + PassName.str() + "'");
      if (!MIRPrintStream)
        return LUTHIER_MAKE_GENERIC_ERROR(
            "print-mir-after= used but no -imodule-output stream configured");
      llvm::AnalysisID AfterID = AfterPI->getTypeInfo();
      std::string Banner =
          "After " + std::string(AfterPI->getPassName()) + " (Luthier)";
      TPC.insertPass(AfterID,
                     llvm::createMachineFunctionPrinterPass(*MIRPrintStream,
                                                            Banner));
      UserInsertedMIRPrint = true;
      continue;
    }
    const llvm::PassInfo *PI = Registry.getPassInfo(Name);
    if (!PI || !PI->getNormalCtor())
      return LUTHIER_MAKE_GENERIC_ERROR("Unknown or unregistered MIR pass '" +
                                        Name.str() + "'");
    TPC.addMachinePrePasses();
    PM.add(PI->getNormalCtor()());
    std::string Banner = std::string("After ") + std::string(PI->getPassName());
    TPC.addMachinePostPasses(Banner);
  }
  return llvm::Error::success();
}

} // namespace

InstrumentationPMDriver::InstrumentationPMDriver(
    const InstrumentationPMDriverOptions &Options,
    IntrinsicProcessorRegistry &Registry,
    llvm::ArrayRef<PassPlugin> PassPlugins,
    IModuleCreationFnType ModuleCreatorFn,
    std::function<void(llvm::PassBuilder &)> PassBuilderAugmentationCallback,
    std::function<void(llvm::ModulePassManager &)> PreIROptimizationCallback,
    std::function<void(llvm::ModulePassManager &)>
        PreIRIntrinsicLoweringCallback,
    std::function<void(llvm::ModulePassManager &)>
        PostIRIntrinsicLoweringCallback,
    std::function<void(llvm::PassRegistry &, llvm::TargetPassConfig &,
                       llvm::TargetMachine &)>
        AugmentTargetPassConfigCallback)
    : Options(Options), Registry(Registry), PassPlugins(PassPlugins),
      IModuleCreatorFn(std::move(ModuleCreatorFn)),
      PassBuilderAugmentationCallback(
          std::move(PassBuilderAugmentationCallback)),
      PreIROptimizationCallback(std::move(PreIROptimizationCallback)),
      PreIRIntrinsicLoweringCallback(std::move(PreIRIntrinsicLoweringCallback)),
      PostIRIntrinsicLoweringCallback(
          std::move(PostIRIntrinsicLoweringCallback)),
      AugmentTargetPassConfigCallback(
          std::move(AugmentTargetPassConfigCallback)) {
  llvm::PassRegistry *LegacyPassRegistry =
      llvm::PassRegistry::getPassRegistry();
  initializeIModuleMAMWrapperPass(*LegacyPassRegistry);
  initializeIntrinsicMIRLoweringPass(*LegacyPassRegistry);
  initializeIModuleIPPredicatedLivenessAnalysis(*LegacyPassRegistry);
  initializeInjectedPayloadPreserveLiveRegsPass(*LegacyPassRegistry);
  initializeLRStateValueStorageAndLoadLocationsAnalysis(*LegacyPassRegistry);
  initializeSVAPhysVGPRPinPass(*LegacyPassRegistry);
  initializeInjectedPayloadPEIPass(*LegacyPassRegistry);
  initializeTargetModulePatcherPass(*LegacyPassRegistry);

  for (const auto &Plugin : PassPlugins) {
    Plugin.registerLegacyCodegenPassesCallback(*LegacyPassRegistry);
  }
}

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

  /// IModule loading
  std::unique_ptr<llvm::MIRParser> IModuleMIRParser;
  std::unique_ptr<llvm::Module> IModule;

  if (!Options.IModulePath.empty()) {
    llvm::StringRef IModPath = Options.IModulePath.getValue();
    if (IModPath.ends_with(".luthier")) {
      auto ParserOrErr = LuthierFileParser::create(IModPath);
      if (!ParserOrErr) {
        Context.emitError(llvm::toString(ParserOrErr.takeError()));
        return llvm::PreservedAnalyses::all();
      }
      auto IModOrErr = ParserOrErr->loadIModule(Context, TargetAppM);
      if (!IModOrErr) {
        Context.emitError(llvm::toString(IModOrErr.takeError()));
        return llvm::PreservedAnalyses::all();
      }
      IModule = std::move(IModOrErr->first);
      IModuleMIRParser = std::move(IModOrErr->second);
    } else {
      auto LoadedOrErr = loadIModuleFromFile(IModPath, Context);
      if (!LoadedOrErr) {
        Context.emitError(llvm::toString(LoadedOrErr.takeError()));
        return llvm::PreservedAnalyses::all();
      }
      IModule = std::move(LoadedOrErr->Module);
      IModuleMIRParser = std::move(LoadedOrErr->MIRParser);
    }
  } else {
    IModule = IModuleCreatorFn(Context);
  }

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
        Context.emitError(llvm::toString(
            LUTHIER_MAKE_GENERIC_ERROR("Failed to link modules together")));
        return llvm::PreservedAnalyses::all();
      }
    }
  }

  /// The instrumentation-module MMI is created here (rather than at the MIR
  /// stage) so the IR-stage new-PM pipeline can register
  /// \c MachineModuleAnalysis against it. Ownership is later transferred to
  /// the legacy MIR PM.
  auto MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(ITM.get());
  llvm::MachineModuleInfo &MMI = MMIWP->getMMI();

  /// Analysis-manager and PassBuilder setup for the IR pipeline
  llvm::ModulePassManager IMPM;
  llvm::LoopAnalysisManager ILAM;
  llvm::FunctionAnalysisManager IFAM;
  llvm::CGSCCAnalysisManager ICGAM;
  llvm::ModuleAnalysisManager IMAM;
  llvm::MachineFunctionAnalysisManager IMFAM;
  llvm::PassInstrumentationCallbacks PIC;
  llvm::StandardInstrumentations SI(IModule->getContext(),
                                    Options.EnableSIDebugLogging);

  // Create a PM Builder for the IR pipeline
  llvm::PassBuilder PB(ITM.get(), llvm::PipelineTuningOptions(), std::nullopt,
                       &PIC);

  /// Augment the pass builder
  PassBuilderAugmentationCallback(PB);
  for (const auto &Plugin : PassPlugins) {
    Plugin.registerInstrumentationPassBuilderCallback(PB);
  }

  /// Register passes for parsing
  /// The parser is registered here to capture ITM correctly
  PB.registerPipelineParsingCallback(
      [&](llvm::StringRef Name, llvm::ModulePassManager &MPM,
          llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name == "luthier-process-intrinsics-at-ir-level") {
          MPM.addPass(ProcessIntrinsicsAtIRLevelPass(*ITM));
          return true;
        }
        if (Name == "luthier-forward-isa-state-to-callees") {
          MPM.addPass(ForwardISAStateToCalleesPass(*ITM));
          return true;
        }
        return false;
      });

  SI.registerCallbacks(PIC, &IMAM);

  // TODO: re-enable when production-pipeline deps are compiled in:
  IMAM.registerPass([&]() { return InjectedPayloadAndInstPointAnalysis(); });
  IMAM.registerPass([&]() { return IntrinsicsProcessorsAnalysis(Registry); });
  IMAM.registerPass([&]() { return llvm::MachineModuleAnalysis(MMI); });

  IMAM.registerPass(
      [&]() { return TargetAppModuleAndMAMAnalysis(TargetMAM, TargetAppM); });

  IFAM.registerPass(
      [] { return InjectedPayloadAccessedRegsAnalysis(); });

  PB.registerModuleAnalyses(IMAM);
  PB.registerCGSCCAnalyses(ICGAM);
  PB.registerFunctionAnalyses(IFAM);
  PB.registerLoopAnalyses(ILAM);
  PB.crossRegisterProxies(ILAM, IFAM, ICGAM, IMAM);

  auto IRStagePA = llvm::PreservedAnalyses::all();
  {
    llvm::TimeTraceScope Scope("Instrumentation Module IR Optimization");

    bool IsIRPassPipelineNotSpecified =
        Options.IModuleIRPasses.getNumOccurrences() == 0;
    bool IsIRPipelineSpecifiedAndNotEmpty =
        !Options.IModuleIRPasses.getValue().empty();
    bool MustRunIRPipeline =
        IsIRPassPipelineNotSpecified || IsIRPipelineSpecifiedAndNotEmpty;

    if (IsIRPassPipelineNotSpecified) {
      unsigned OptLevelVal =
          Options.IModuleOptLevel.getNumOccurrences() > 0
              ? Options.IModuleOptLevel.getValue()
              : static_cast<unsigned>(TargetAppTM.getOptLevel());
      // TODO: IR pipeline — uncomment when deps are available:
      PreIROptimizationCallback(IMPM);
      for (const auto &Plugin : PassPlugins)
        Plugin.registerPreIROptimizationPasses(IMPM);

      IMPM.addPass(
          PB.buildPerModuleDefaultPipeline(optLevelFromUnsigned(OptLevelVal)));
      PreIRIntrinsicLoweringCallback(IMPM);
      for (const auto &Plugin : PassPlugins)
        Plugin.invokePreLuthierIRIntrinsicLoweringPassesCallback(IMPM);
      // Warm the cache for InjectedPayloadAccessedRegsAnalysis before
      // ProcessIntrinsicsAtIRLevelPass erases the intrinsic-attributed
      // declarations and rewrites each call to an inline-asm placeholder.
      IMPM.addPass(llvm::createModuleToFunctionPassAdaptor(
          llvm::RequireAnalysisPass<InjectedPayloadAccessedRegsAnalysis,
                                    llvm::Function>()));
      IMPM.addPass(ProcessIntrinsicsAtIRLevelPass(*ITM));
      PostIRIntrinsicLoweringCallback(IMPM);
      for (const auto &Plugin : PassPlugins)
        Plugin.invokePostLuthierIRIntrinsicLoweringPassesCallback(IMPM);
    } else if (IsIRPipelineSpecifiedAndNotEmpty) {
      // Run caller-supplied IR pipeline.
      if (auto Err =
              PB.parsePassPipeline(IMPM, Options.IModuleIRPasses.getValue())) {
        Context.emitError(llvm::toString(std::move(Err)));
        return llvm::PreservedAnalyses::all();
      }
    }
    if (MustRunIRPipeline)
      IRStagePA = IMPM.run(*IModule, IMAM);

    LLVM_DEBUG(luthier::dbgs() << "Instrumentation Module after IR pipeline:\n";
               IModule->print(luthier::dbgs(), nullptr));
  }

  /// MIR / codegen pipeline
  bool MIRModified = false;
  auto MIRLegacyPM = std::make_unique<llvm::legacy::PassManager>();

  /// Parse the MIR file in case it was specified in the opts
  if (IModuleMIRParser &&
      IModuleMIRParser->parseMachineFunctions(*IModule, MMI)) {
    Context.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "Failed to parse machine functions from imodule file '" +
        Options.IModulePath + "'")));
    return IRStagePA.areAllPreserved() ? llvm::PreservedAnalyses::all()
                                       : llvm::PreservedAnalyses::none();
  }

  {
    llvm::TimeTraceScope Scope(
        "Instrumentation Module MIR CodeGen Optimization");

    auto *TPC = ITM->createPassConfig(*MIRLegacyPM);
    TPC->setDisableVerify(true);
    MIRLegacyPM->add(TPC);
    MIRLegacyPM->add(MMIWP.release());
    MIRLegacyPM->add(new IModuleMAMWrapperPass(&IMAM));

    bool MIRPassPipelineNotSpecified =
        Options.IModuleMIRPasses.getNumOccurrences() == 0;
    bool MIRPassPipelineSpecifiedAndNotEmpty =
        !Options.IModuleMIRPasses.getValue().empty();
    bool MustRunCodeGen =
        MIRPassPipelineNotSpecified || MIRPassPipelineSpecifiedAndNotEmpty;

    // Set up the optional output file early so parseCodeGenPipeline has
    // a stream to attach a user-requested mid-pipeline `print-mir` to.
    llvm::StringRef OutPath = Options.IModuleOutput;
    bool IsLuthierOutput = OutPath.ends_with(".luthier");
    bool MustDumpMIR = MustRunCodeGen && !OutPath.empty() && !IsLuthierOutput;
    std::unique_ptr<llvm::ToolOutputFile> OutFile;
    if (MustDumpMIR) {
      std::error_code EC;
      OutFile = std::make_unique<llvm::ToolOutputFile>(
          OutPath, EC, llvm::sys::fs::OF_Text);
      if (EC) {
        Context.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
            "Failed to open imodule output file '" + Options.IModuleOutput +
            "': " + EC.message())));
        return IRStagePA.areAllPreserved() ? llvm::PreservedAnalyses::all()
                                           : llvm::PreservedAnalyses::none();
      }
    }
    bool UserInsertedMIRPrint = false;

    if (MIRPassPipelineNotSpecified) {
      // Default IModule codegen pipeline:
      //   isel,
      //   mir-lowering,
      //   injected-payload-accessed-regs,
      //   imodule-ip-pred-liveness,
      //   payload-preserve-live-regs,
      //   lr-sv-storage-load-locs,
      //   machine-passes — runs the full AMDGPU MIR pipeline:
      //     - RA (with SVAPhysVGPRPinPass inserted before
      //       SIPreAllocateWWMRegs so WWM honors the SVA-lane hint)
      //     - stock PrologEpilogInserter
      //     - InjectedPayloadPEIPass inserted via TPC->insertPass
      //       so it fires AFTER stock PEI (it relies on the spill
      //       slots stock PEI materializes) and gets called per-MF
      //       inside the machine-passes block.
      //     - frame finalization, branch folding, block placement.
      //     NOTE: addMachinePasses does NOT call addAsmPrinter
      //     (AsmPrinter is added by addPassesToEmitFile), so the
      //     IModule's MIR ends in physreg form without an emission
      //     step — exactly what we need to hand off to the patcher.
      //   target-module-patcher — runs after the TPC chain finishes.
      //     The IModule's payload + hook MFs are now in physreg form;
      //     inlineInjectedPayload deep-copies them into target MFs
      //     and NewPMAsmPrinter consumes the patched target module.
      //
      // Each non-machine-passes step is bracketed by
      // TPC->addMachinePrePasses() / addMachinePostPasses() the same
      // way parseCodeGenPipeline does when adding registered legacy
      // passes.
      TPC->addISelPasses();

      auto AddModulePass = [&](llvm::Pass *P, llvm::StringRef Name) {
        TPC->addMachinePrePasses();
        MIRLegacyPM->add(P);
        std::string Banner = ("After " + Name).str();
        TPC->addMachinePostPasses(Banner);
      };

      AddModulePass(new IntrinsicMIRLoweringPass(),
                    "Luthier IntrinsicMIRLowering");
      AddModulePass(new IModuleIPPredicatedLivenessAnalysis(),
                    "Luthier IPPredLiveness");
      AddModulePass(new InjectedPayloadPreserveLiveRegsPass(),
                    "Luthier PreserveLiveRegs");
      AddModulePass(new LRStateValueStorageAndLoadLocationsAnalysis(),
                    "Luthier LRStateValueStorage");

      // Schedule SVAPhysVGPRPinPass before SIPreAllocateWWMRegs so
      // the WWM greedy regalloc honors the SVA-lane hint. Schedule
      // InjectedPayloadPEIPass after PrologEpilogCodeInserterID so
      // our payload PEI sees stock-PEI-materialized spill slots.
      // Both must be insertPass-d BEFORE addMachinePasses for the
      // AMDGPU TPC to honor them.
      TPC->insertPass(&llvm::SIPreAllocateWWMRegsLegacyID,
                      &SVAPhysVGPRPinPass::ID);
      if (!Options.DisableInjectedPayloadPEI) {
        TPC->insertPass(&llvm::PrologEpilogCodeInserterID,
                        &InjectedPayloadPEIPass::ID);
      }
      TPC->addMachinePasses();

      AddModulePass(new TargetModulePatcherPass(),
                    "Luthier TargetModulePatcher");
    } else if (MIRPassPipelineSpecifiedAndNotEmpty) {
      llvm::raw_ostream *PrintStream =
          MustDumpMIR ? &OutFile->os() : nullptr;
      llvm::Error ModifiedOrErr = parseCodeGenPipeline(
          Options.IModuleMIRPasses.getValue(), *MIRLegacyPM, *TPC, PrintStream,
          UserInsertedMIRPrint);
      if (ModifiedOrErr) {
        Context.emitError(llvm::toString(std::move(ModifiedOrErr)));
        return IRStagePA;
      }
    }
    if (MustRunCodeGen) {
      TPC->setInitialized();
      // Add the default end-of-pipeline MIR dump only if the user didn't
      // place one mid-pipeline via the `print-mir` token. Skipping the
      // tail dump avoids the MIRPrinter crash when post-frame-elim
      // passes leave MFI in a state the YAML printer can't serialize.
      if (MustDumpMIR && !UserInsertedMIRPrint)
        MIRLegacyPM->add(llvm::createPrintMIRPass(OutFile->os()));
      MIRModified = MIRLegacyPM->run(*IModule);
      if (MustDumpMIR)
        OutFile->keep();

      // Dump the post-patch target module if requested. The patcher
      // mutates target MFs (clones non-payload MFs, strips
      // num-{vgpr,sgpr}, inlines payloads); imodule-output only shows
      // the IModule side, so a separate sink is required for lit tests
      // that need to FileCheck patcher effects.
      llvm::StringRef TgtOutPath = Options.TargetModuleOutput;
      if (!TgtOutPath.empty()) {
        std::error_code TgtEC;
        llvm::ToolOutputFile TgtFile(TgtOutPath, TgtEC, llvm::sys::fs::OF_Text);
        if (TgtEC) {
          Context.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
              "Failed to open target-module-output file '" +
              Options.TargetModuleOutput + "': " + TgtEC.message())));
          return IRStagePA.areAllPreserved() ? llvm::PreservedAnalyses::all()
                                             : llvm::PreservedAnalyses::none();
        }
        // Dump IR (with the stripped attrs visible) + every target-FAM
        // MachineFunction. Use plain MF::print rather than MIRPrinter so
        // stale-stack-object indices don't crash the dumper. Note that
        // target MFs live in the TargetFAM (new-PM), NOT in TargetMMI
        // (legacy MMI) — the legacy MMI is for IModule MFs only.
        TgtFile.os() << "; Luthier target-module-output\n";
        TargetAppM.print(TgtFile.os(), /*AAW=*/nullptr);
        auto &TargetFAM =
            TargetMAM
                .getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetAppM)
                .getManager();
        for (llvm::Function &F : TargetAppM) {
          if (F.isDeclaration())
            continue;
          auto *MFRes =
              TargetFAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);
          if (!MFRes)
            continue;
          TgtFile.os() << "\n# Machine code for function " << F.getName()
                       << ":\n";
          MFRes->getMF().print(TgtFile.os());
        }
        TgtFile.keep();
      }

      if (IsLuthierOutput) {
        if (auto Err = writeLuthierFile(OutPath, TargetAppM, *IModule)) {
          Context.emitError(llvm::toString(std::move(Err)));
          return IRStagePA.areAllPreserved() ? llvm::PreservedAnalyses::all()
                                             : llvm::PreservedAnalyses::none();
        }
      }
    }
  }
  /// Conservatively invalidate all other target module passes even if only the
  // If the user requested an imodule output but opted out of MIR codegen
  // (e.g., --imodule-mir-passes= with a .ll output for IR-only tests),
  // dump the post-IR-pipeline IModule IR here. This preserves the
  // previous-default behavior of emitting the IModule's IR to the
  // output file when no MIR pipeline runs, which IR-level injection
  // tests rely on.
  if (IModule) {
    llvm::StringRef OutPath = Options.IModuleOutput;
    bool IsLuthierOutput = OutPath.ends_with(".luthier");
    bool MIRPassPipelineNotSpecified =
        Options.IModuleMIRPasses.getNumOccurrences() == 0;
    bool MIRPassPipelineSpecifiedAndNotEmpty =
        !Options.IModuleMIRPasses.getValue().empty();
    bool MustRunCodeGen =
        MIRPassPipelineNotSpecified || MIRPassPipelineSpecifiedAndNotEmpty;
    if (!OutPath.empty() && !MustRunCodeGen) {
      if (IsLuthierOutput) {
        // IR-only mode (--imodule-mir-passes=) with a .luthier sink:
        // still need to emit the bundled IModule+target+slot-map dump
        // tests FileCheck against.
        if (auto Err = writeLuthierFile(OutPath, TargetAppM, *IModule)) {
          Context.emitError(llvm::toString(std::move(Err)));
        }
      } else {
        std::error_code EC;
        llvm::ToolOutputFile Out(OutPath, EC, llvm::sys::fs::OF_Text);
        if (!EC) {
          IModule->print(Out.os(), /*AAW=*/nullptr);
          Out.keep();
        }
      }
    }
  }

  // Preserve the new-PM MachineFunctionAnalysis (and the module proxy for
  // it) across our return. If we returned PreservedAnalyses::none() the
  // MachineFunctionAnalysis cache would invalidate; the next consumer
  // (e.g., luthier-asm-printer) would re-run MachineFunctionAnalysis on
  // each Function and produce a *fresh, empty* MachineFunction for any
  // Function whose IR body is empty — which is the case for every
  // target kernel CodeDiscoveryPass lifts (the code lives in the MMI,
  // not in IR BasicBlocks). The result would be an empty MBB list and
  // an AsmPrinter crash inside emitFunctionHeader.
  // Same preservation set as CodeDiscoveryPass uses: explicitly hold on
  // to MachineFunctionAnalysis + its module proxy + the function proxy,
  // because subsequent passes (notably luthier-asm-printer) re-fetch
  // each Function's MachineFunction by name. Empty IR bodies (every
  // target kernel CodeDiscoveryPass lifts has its code in MMI, not
  // IR) would re-materialize as empty MFs if MachineFunctionAnalysis
  // weren't held — AsmPrinter then crashes in emitFunctionHeader on
  // the sentinel MBB iterator.
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<llvm::MachineFunctionAnalysisManagerModuleProxy>();
  PA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
  PA.preserve<llvm::MachineFunctionAnalysis>();
  return PA;
}
} // namespace luthier
