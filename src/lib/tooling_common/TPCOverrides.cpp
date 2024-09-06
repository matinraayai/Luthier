//===-- TPCOverrides.hpp --------------------------------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file implements member functions in the target pass config that
/// need to be overridden for Luthier.
//===----------------------------------------------------------------------===//
#include "tooling_common/TPCOverrides.hpp"
#define protected public
#define private public
#include <llvm/CodeGen/TargetPassConfig.h>
#undef protected
#undef private
#include <llvm/CodeGen/BasicBlockSectionsProfileReader.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Target/TargetMachine.h>

namespace llvm {

// Find the FSProfile file name. The internal option takes the precedence
// before getting from TargetMachine.
static std::string getFSProfileFile(const TargetMachine *TM) {
  llvm::StringMap<llvm::cl::Option *> &Map = llvm::cl::getRegisteredOptions();

  const auto &FSProfileFile =
      *reinterpret_cast<llvm::cl::opt<std::string> *>(Map["fs-profile-file"]);
  if (!FSProfileFile.empty())
    return FSProfileFile.getValue();
  const std::optional<PGOOptions> &PGOOpt = TM->getPGOOption();
  if (PGOOpt == std::nullopt || PGOOpt->Action != PGOOptions::SampleUse)
    return {};
  return PGOOpt->ProfileFile;
}

// Find the Profile remapping file name. The internal option takes the
// precedence before getting from TargetMachine.
static std::string getFSRemappingFile(const TargetMachine *TM) {
  llvm::StringMap<llvm::cl::Option *> &Map = llvm::cl::getRegisteredOptions();

  const auto &FSRemappingFile =
      *reinterpret_cast<llvm::cl::opt<std::string> *>(Map["fs-remapping-file"]);

  if (!FSRemappingFile.empty())
    return FSRemappingFile.getValue();
  const std::optional<PGOOptions> &PGOOpt = TM->getPGOOption();
  if (PGOOpt == std::nullopt || PGOOpt->Action != PGOOptions::SampleUse)
    return {};
  return PGOOpt->ProfileRemappingFile;
}

} // namespace llvm

namespace luthier {

void addMachinePassesToTPC(llvm::TargetPassConfig &TPC) {
  llvm::StringMap<llvm::cl::Option *> &Map = llvm::cl::getRegisteredOptions();

  const auto &EnableFSDiscriminator =
      *reinterpret_cast<llvm::cl::opt<bool> *>(Map["enable-fs-discriminator"]);

  const auto &DisableRAFSProfileLoader =
      *reinterpret_cast<llvm::cl::opt<bool> *>(
          Map["disable-ra-fsprofile-loader"]);

  const auto &EnableImplicitNullChecks =
      *reinterpret_cast<llvm::cl::opt<bool> *>(
          Map["enable-implicit-null-checks"]);

  const auto &MISchedPostRA =
      *reinterpret_cast<llvm::cl::opt<bool> *>(Map["misched-postra"]);

  const auto &EnableMachineOutliner =
      *reinterpret_cast<llvm::cl::opt<llvm::RunOutliner> *>(
          Map["enable-machine-outliner"]);

  const auto &GCEmptyBlocks =
      *reinterpret_cast<llvm::cl::opt<bool> *>(Map["gc-empty-basic-blocks"]);

  const auto &EnableMachineFunctionSplitter =
      *reinterpret_cast<llvm::cl::opt<bool> *>(
          Map["enable-split-machine-functions"]);

  const auto &DisableCFIFixup = *reinterpret_cast<llvm::cl::opt<bool> *>(
      Map["disable-cfi-fixup"]);

  /// TODO: This function is a hack; It should be removed once LLVM upstream
  /// finishes migrating to the new pass manager
  TPC.AddingMachinePasses = true;

  // Add passes that optimize machine instructions in SSA form.
  if (TPC.getOptLevel() != llvm::CodeGenOptLevel::None) {
    TPC.addMachineSSAOptimization();
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    TPC.addPass(&llvm::LocalStackSlotAllocationID);
  }

  if (TPC.TM->Options.EnableIPRA)
    TPC.addPass(llvm::createRegUsageInfoPropPass());

  // Run pre-ra passes.
  TPC.addPreRegAlloc();

  // Debugifying the register allocator passes seems to provoke some
  // non-determinism that affects CodeGen and there doesn't seem to be a point
  // where it becomes safe again so stop debugifying here.
  TPC.DebugifyIsSafe = false;

  // Add a FSDiscriminator pass right before RA, so that we could get
  // more precise SampleFDO profile for RA.
  if (EnableFSDiscriminator) {
    TPC.addPass(llvm::createMIRAddFSDiscriminatorsPass(
        llvm::sampleprof::FSDiscriminatorPass::Pass1));
    const std::string ProfileFile = getFSProfileFile(TPC.TM);
    if (!ProfileFile.empty() && !DisableRAFSProfileLoader)
      TPC.addPass(llvm::createMIRProfileLoaderPass(
          ProfileFile, getFSRemappingFile(TPC.TM),
          llvm::sampleprof::FSDiscriminatorPass::Pass1, nullptr));
  }

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (TPC.getOptimizeRegAlloc())
    TPC.addOptimizedRegAlloc();
  else
    TPC.addFastRegAlloc();

  // Run post-ra passes.
  TPC.addPostRegAlloc();

  TPC.addPass(&llvm::RemoveRedundantDebugValuesID);

  TPC.addPass(&llvm::FixupStatepointCallerSavedID);

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (TPC.getOptLevel() != llvm::CodeGenOptLevel::None) {
    TPC.addPass(&llvm::PostRAMachineSinkingID);
    TPC.addPass(&llvm::ShrinkWrapID);
  }

  // Prolog/Epilog inserter needs a TargetMachine to instantiate. But only
  // do so if it hasn't been disabled, substituted, or overridden.
  if (!TPC.isPassSubstitutedOrOverridden(&llvm::PrologEpilogCodeInserterID))
    TPC.addPass(llvm::createPrologEpilogInserterPass());

  /// Add passes that optimize machine instructions after register allocation.
  if (TPC.getOptLevel() != llvm::CodeGenOptLevel::None)
    TPC.addMachineLateOptimization();

  // Expand pseudo instructions before second scheduling pass.
  TPC.addPass(&llvm::ExpandPostRAPseudosID);

  // Run pre-sched2 passes.
  TPC.addPreSched2();

  if (EnableImplicitNullChecks)
    TPC.addPass(&llvm::ImplicitNullChecksID);

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (TPC.getOptLevel() != llvm::CodeGenOptLevel::None &&
      !TPC.TM->targetSchedulesPostRAScheduling()) {
    if (MISchedPostRA)
      TPC.addPass(&llvm::PostMachineSchedulerID);
    else
      TPC.addPass(&llvm::PostRASchedulerID);
  }

  // GC
  TPC.addGCPasses();

  // Basic block placement.
  if (TPC.getOptLevel() != llvm::CodeGenOptLevel::None)
    TPC.addBlockPlacement();

  // Insert before XRay Instrumentation.
  TPC.addPass(&llvm::FEntryInserterID);

  TPC.addPass(&llvm::XRayInstrumentationID);
  TPC.addPass(&llvm::PatchableFunctionID);

  TPC.addPreEmitPass();

  if (TPC.TM->Options.EnableIPRA)
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    TPC.addPass(llvm::createRegUsageInfoCollector());

  // FIXME: Some backends are incompatible with running the verifier after
  // addPreEmitPass.  Maybe only pass "false" here for those targets?
  TPC.addPass(&llvm::FuncletLayoutID);

//  TPC.addPass(&llvm::RemoveLoadsIntoFakeUsesID);
  TPC.addPass(&llvm::StackMapLivenessID);
  TPC.addPass(&llvm::LiveDebugValuesID);
  TPC.addPass(&llvm::MachineSanitizerBinaryMetadataID);

  if (TPC.TM->Options.EnableMachineOutliner &&
      TPC.getOptLevel() != llvm::CodeGenOptLevel::None &&
      EnableMachineOutliner != llvm::RunOutliner::NeverOutline) {
    bool RunOnAllFunctions =
        (EnableMachineOutliner == llvm::RunOutliner::AlwaysOutline);
    bool AddOutliner =
        RunOnAllFunctions || TPC.TM->Options.SupportsDefaultOutlining;
    if (AddOutliner)
      TPC.addPass(llvm::createMachineOutlinerPass(RunOnAllFunctions));
  }

  if (GCEmptyBlocks)
    TPC.addPass(llvm::createGCEmptyBasicBlocksPass());

  if (EnableFSDiscriminator)
    TPC.addPass(llvm::createMIRAddFSDiscriminatorsPass(
        llvm::sampleprof::FSDiscriminatorPass::PassLast));

  bool NeedsBBSections =
      TPC.TM->getBBSectionsType() != llvm::BasicBlockSection::None;
  // Machine function splitter uses the basic block sections feature. Both
  // cannot be enabled at the same time. We do not apply machine function
  // splitter if -basic-block-sections is requested.
  if (!NeedsBBSections && (TPC.TM->Options.EnableMachineFunctionSplitter ||
                           EnableMachineFunctionSplitter)) {
    const std::string ProfileFile = getFSProfileFile(TPC.TM);
    if (!ProfileFile.empty()) {
      if (EnableFSDiscriminator) {
        TPC.addPass(llvm::createMIRProfileLoaderPass(
            ProfileFile, getFSRemappingFile(TPC.TM),
            llvm::sampleprof::FSDiscriminatorPass::PassLast, nullptr));
      } else {
        // Sample profile is given, but FSDiscriminator is not
        // enabled, this may result in performance regression.
        llvm::WithColor::warning()
            << "Using AutoFDO without FSDiscriminator for MFS may regress "
               "performance.\n";
      }
    }
    TPC.addPass(llvm::createMachineFunctionSplitterPass());
  }
  // We run the BasicBlockSections pass if either we need BB sections or BB
  // address map (or both).
  if (NeedsBBSections || TPC.TM->Options.BBAddrMap) {
    if (TPC.TM->getBBSectionsType() == llvm::BasicBlockSection::List) {
      TPC.addPass(llvm::createBasicBlockSectionsProfileReaderWrapperPass(
          TPC.TM->getBBSectionsFuncListBuf()));
      TPC.addPass(llvm::createBasicBlockPathCloningPass());
    }
    TPC.addPass(llvm::createBasicBlockSectionsPass());
  }

  TPC.addPostBBSections();

  if (!DisableCFIFixup && TPC.TM->Options.EnableCFIFixup)
    TPC.addPass(llvm::createCFIFixup());

  TPC.PM->add(llvm::createStackFrameLayoutAnalysisPass());

  // Add passes that directly emit MI after all other MI passes.
  TPC.addPreEmitPass2();

  TPC.AddingMachinePasses = false;
}

} // namespace luthier