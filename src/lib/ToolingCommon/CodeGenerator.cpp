//===-- CodeGenerator.cpp - Luthier Code Generator ------------------------===//
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
/// \file
/// This file implements Luthier's code generator.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/CodeGenerator.h"
#include "luthier/Comgr/ComgrError.h"
#include "luthier/Common/LuthierError.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include "luthier/Tooling/AMDGPURegisterLiveness.h"
#include "luthier/Tooling/CodeLifter.h"
#include "luthier/Tooling/InjectedPayloadPEIPass.h"
#include "luthier/Tooling/MMISlotIndexesAnalysis.h"
#include "luthier/Tooling/PatchLiftedRepresentationPass.h"
#include "luthier/Tooling/PrePostAmbleEmitter.h"
#include "luthier/Tooling/RunIRPassesOnIModulePass.h"
#include "luthier/Tooling/RunMIRPassesOnIModulePass.h"
#include "luthier/Tooling/ToolExecutableLoader.h"
#include "luthier/Tooling/WrapperAnalysisPasses.h"
#include <AMDGPUResourceUsageAnalysis.h>
#include <AMDGPUTargetMachine.h>
#include <amd_comgr/amd_comgr.h>
#include <llvm/Analysis/CallGraphSCCPass.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/Tooling/InstrumentationPMDriver.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-generator"

namespace luthier {

template <> CodeGenerator *Singleton<CodeGenerator>::Instance{nullptr};

llvm::Error CodeGenerator::printAssembly(
    llvm::Module &Module, llvm::GCNTargetMachine &TM,
    std::unique_ptr<llvm::MachineModuleInfoWrapperPass> &MMIWP,
    llvm::SmallVectorImpl<char> &CompiledObjectFile,
    llvm::CodeGenFileType FileType) {
  llvm::TimeTraceScope Scope("LLVM Assembly Printing");
  // Argument error checking
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      FileType != llvm::CodeGenFileType::Null,
      "Cannot pass file type Null to print assembly."));
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_GENERIC_ERROR_CHECK(MMIWP != nullptr, "MMIWP is nullptr."));

  auto &MMI = MMIWP->getMMI();
  // Create the legacy pass manager with minimal passes to print the
  // assembly file
  llvm::legacy::PassManager PM;
  // Add the target library info pass
  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(Module.getTargetTriple()));
  PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));
  // DummyCGSCCPass must also be added
  PM.add(new llvm::DummyCGSCCPass());
  // TargetPassConfig is expected by the passes involved, so it must be added
  auto *TPC = TM.createPassConfig(PM);
  // Don't run the machine verifier after each pass
  TPC->setDisableVerify(true);
  TPC->setInitialized();
  PM.add(TPC);
  // Add the MMI Wrapper pass, while Releasing the ownership of the unique
  // pointer over MMIWP to avoid freeing MMIWP multiple times
  PM.add(MMIWP.release());
  // Add the resource usage analysis, which is in charge of calculating the
  // kernel descriptor and the metadata fields
  PM.add(new llvm::AMDGPUResourceUsageAnalysisWrapperPass());

  // Finally, add the Assembly printer pass
  llvm::raw_svector_ostream ObjectFileOS(CompiledObjectFile);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !TM.addAsmPrinter(PM, ObjectFileOS, nullptr, FileType, MMI.getContext()),
      "Failed to add the assembly printer pass to the pass manager."));

  // Run the passes on the module to print the assembly
  PM.run(Module);
  return llvm::Error::success();
}

llvm::Error
CodeGenerator::applyInstrumentationTask(const InstrumentationTask &Task,
                                        LiftedRepresentation &LR) {
  // Early exit if no hooks are to be inserted into the LR
  if (Task.getHookInsertionTasks().empty())
    return llvm::Error::success();
  // Acquire the Lifted Representation's lock
  auto Lock = LR.getLock();
  // Each LCO will get its own copy of the instrumented module
  hsa_loaded_code_object_t LCO = LR.getLoadedCodeObject();
  auto Agent = hsa::loadedCodeObjectGetAgent(LoaderApiSnapshot.getTable(), LCO);
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  auto &TM = LR.getTM();
  // Load the bitcode of the instrumentation module into the
  // Lifted Representation's context
  std::unique_ptr<llvm::Module> IModule;
  LUTHIER_RETURN_ON_ERROR(Task.getModule()
                              .readBitcodeIntoContext(LR.getContext(), *Agent)
                              .moveInto(IModule));
  llvm::PassInstrumentationCallbacks PIC;
  llvm::StandardInstrumentations SI(LR.getContext(), true);

  // Create a PM Builder for the IR pipeline
  llvm::PassBuilder PB(&TM);
  llvm::TimeTraceScope Scope("Instrumentation Module IR Optimization");

  // Create a module analysis manager for the target code
  llvm::ModuleAnalysisManager TargetMAM;
  // Create a new Module pass manager, in charge of running the entire
  // pipeline
  llvm::ModulePassManager TargetMPM;

  SI.registerCallbacks(PIC, &TargetMAM);
  // Add the pass instrumentation analysis as it is required by the new PM
  TargetMAM.registerPass([&]() { return llvm::PassInstrumentationAnalysis(); });
  // Add the MMI Analysis pass, pointing to the target app's lifted MMI
  TargetMAM.registerPass(
      [&]() { return llvm::MachineModuleAnalysis(LR.getMMI()); });
  // Add the LR Analysis pass
  TargetMAM.registerPass([&]() { return LiftedRepresentationAnalysis(LR); });
  // Add the LCO Analysis pass
  TargetMAM.registerPass([&]() { return LoadedCodeObjectAnalysis(LCO); });
  // Add the LR Register Liveness pass
  TargetMAM.registerPass([&]() { return AMDGPURegLivenessAnalysis(); });
  // Add the LR Callgraph analysis pass
  TargetMAM.registerPass([&]() { return LRCallGraphAnalysis(); });
  // Add the MMI-wide Slot indexes analysis pass
  TargetMAM.registerPass([&]() { return MMISlotIndexesAnalysis(); });
  // Add the State Value Array storage and load analysis pass
  TargetMAM.registerPass(
      [&]() { return LRStateValueStorageAndLoadLocationsAnalysis(); });
  // Add the Function Preamble Descriptor Analysis pass
  TargetMAM.registerPass(
      [&]() { return FunctionPreambleDescriptorAnalysis(); });
  // Add the instrumentation pass manager driver
  TargetMPM.addPass(InstrumentationPMDriver(InstrumentationPMOptions));

  TargetMPM.run(LR.getModule(), TargetMAM);
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<LiftedRepresentation>> CodeGenerator::instrument(
    const LiftedRepresentation &LR,
    llvm::function_ref<llvm::Error(InstrumentationTask &,
                                   LiftedRepresentation &)>
        Mutator) {
  // Acquire the context lock for thread-safety
  auto Lock = LR.getLock();
  std::unique_ptr<LiftedRepresentation> ClonedLR;
  // Clone the Lifted Representation
  LUTHIER_RETURN_ON_ERROR(
      CodeLifter::instance().cloneRepresentation(LR).moveInto(ClonedLR));
  // Create an instrumentation task to keep track of the hooks called before
  // each MI of the application
  InstrumentationTask IT(*ClonedLR);
  // Run the mutator function on the Lifted Representation and populate the
  // instrumentation task
  LUTHIER_RETURN_ON_ERROR(Mutator(IT, *ClonedLR));
  // Apply the instrumentation task to the Lifted Representation
  LUTHIER_RETURN_ON_ERROR(applyInstrumentationTask(IT, *ClonedLR));

  LLVM_DEBUG(

      llvm::dbgs() << "Final instrumented Lifted Representation Code:\n";
      for (const auto &F : ClonedLR->getModule()) {
        llvm::dbgs() << "Function name in the LLVM Module: " << F.getName()
                     << "\n";
        if (auto MF = ClonedLR->getMMI().getMachineFunction(F)) {
          llvm::dbgs() << "Location of the Machine Function in memory: " << MF
                       << "\n";
          MF->print(llvm::dbgs());
        }
      }

  );

  return std::move(ClonedLR);
}

} // namespace luthier
