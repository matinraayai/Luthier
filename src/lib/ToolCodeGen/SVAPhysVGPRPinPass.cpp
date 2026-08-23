//===-- SVAPhysVGPRPinPass.cpp ----------------------------------*- C++ -*-===//
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
/// Implements SVAPhysVGPRPinPass. For each injected-payload MF, looks up the
/// IP that this payload MF is attached to, asks SVStorageAndLoadLocations for
/// the canonical SVA load destination physreg, and pins the MF's single
/// MFInfo->SGPRSpillVGPRs[] entry to it via MachineRegisterInfo::setSimpleHint.
/// The greedy WWM regalloc honors hints when feasible, and the SVA physreg is
/// feasible by construction (SVStorageAndLoadLocations picked it specifically
/// because nothing else in the target module uses it).
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/SVAPhysVGPRPinPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/ParentPrototypeAnalysis.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"

#include <SIMachineFunctionInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-sva-pin"

namespace luthier {

static void pinInjectedPayloadMF(llvm::MachineFunction &MF,
                                 const InjectedPayloadAndInstPoint &IPIP,
                                 const SVStorageAndLoadLocations &SVLocations) {
  const llvm::Function &F = MF.getFunction();
  if (!F.hasFnAttribute(InjectedPayloadAttribute))
    return;

  auto *MFInfo = MF.getInfo<llvm::SIMachineFunctionInfo>();
  // The LaneVGPR pool: materializeReadlanes pushed exactly one entry
  // (shared across all SA-lane FIs via the framework's monotonic counter).
  // Subsequent SGPR-spills in this MF may have added more entries — those
  // are RA-driven spill targets we DON'T want to pin.
  llvm::ArrayRef<llvm::Register> SpillVGPRs = MFInfo->getSGPRSpillVGPRs();
  if (SpillVGPRs.empty())
    return;

  llvm::LLVMContext &Ctx = F.getContext();

  if (!IPIP.contains(F))
    return;
  const llvm::MachineInstr *AppMI = IPIP.at(F);

  const auto *LoadPlan =
      SVLocations.getStateValueArrayLoadPlanForInstPoint(*AppMI);
  if (!LoadPlan) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "SVAPhysVGPRPinPass: no SVA load plan for IP in {0}", F.getName()))));
    return;
  }
  llvm::MCRegister TargetPhys = LoadPlan->StateValueArrayLoadVGPR;
  if (!TargetPhys) {
    LLVM_DEBUG(luthier::dbgs() << "  no SVA load VGPR for " << F.getName()
                               << "; nothing to pin\n");
    return;
  }

  // The first SpillVGPR is the SVA LaneVGPR (materializeReadlanes
  // allocates it first, before any RA-driven spill can advance the
  // counter). Hint it to the load-plan physreg; the WWM regalloc honors
  // the hint when feasible, which by construction it is.
  llvm::Register LaneVGPR = SpillVGPRs.front();
  llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  MRI.setSimpleHint(LaneVGPR, TargetPhys);

  LLVM_DEBUG(luthier::dbgs()
             << "  pinned LaneVGPR " << llvm::printReg(LaneVGPR) << " to "
             << llvm::printReg(TargetPhys, MF.getSubtarget().getRegisterInfo())
             << " in " << F.getName() << "\n");
}

llvm::PreservedAnalyses
SVAPhysVGPRPinPass::run(llvm::MachineFunction &MF,
                        llvm::MachineFunctionAnalysisManager &MFAM) {
  llvm::Function &F = MF.getFunction();
  if (!F.hasFnAttribute(InjectedPayloadAttribute))
    return llvm::PreservedAnalyses::all();

  llvm::LLVMContext &Ctx = F.getContext();
  llvm::Module &IModule = *F.getParent();

  const auto &MAMProxy =
      MFAM.getResult<llvm::ModuleAnalysisManagerMachineFunctionProxy>(MF);
  const auto &PAMProxy =
      MFAM.getResult<PrototypeAnalysisManagerMachineFunctionProxy>(MF);

  auto *PPA = MAMProxy.getCachedResult<ParentPrototypeAnalysis>(IModule);
  Prototype *P = PPA ? PPA->getPrototype() : nullptr;
  if (!P) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "No parent prototype found for instrumentation module")));
    return llvm::PreservedAnalyses::all();
  }

  const InjectedPayloadAndInstPoint *IPIP =
      PAMProxy.getCachedResult<InjectedPayloadAndInstPointAnalysis>(*P);
  if (!IPIP) {
    Ctx.emitError(llvm::toString(
        LUTHIER_MAKE_GENERIC_ERROR("Injected payload and instrumentation point "
                                   "analysis has not been cached")));
    return llvm::PreservedAnalyses::all();
  }

  const SVStorageAndLoadLocations *SVLocations =
      PAMProxy.getCachedResult<SVStorageAndLoadLocationsAnalysis>(*P);
  if (!SVLocations) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "SV locations analysis has not been cached")));
    return llvm::PreservedAnalyses::all();
  }

  pinInjectedPayloadMF(MF, *IPIP, *SVLocations);

  // Register-hint changes don't invalidate any analyses on the MF's CFG
  // or IR; preserve everything.
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
