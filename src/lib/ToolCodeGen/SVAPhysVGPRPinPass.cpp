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
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"

#include <SIMachineFunctionInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-sva-pin"

namespace luthier {

static bool pinInjectedPayloadMF(llvm::MachineFunction &MF,
                                 const InjectedPayloadAndInstPoint &IPIP,
                                 const SVStorageAndLoadLocations &SVLocations) {
  const llvm::Function &F = MF.getFunction();
  if (!F.hasFnAttribute(InjectedPayloadAttribute))
    return false;

  auto *MFInfo = MF.getInfo<llvm::SIMachineFunctionInfo>();
  // The LaneVGPR pool: materializeReadlanes pushed exactly one entry
  // (shared across all SA-lane FIs via the framework's monotonic counter).
  // Subsequent SGPR-spills in this MF may have added more entries — those
  // are RA-driven spill targets we DON'T want to pin.
  llvm::ArrayRef<llvm::Register> SpillVGPRs = MFInfo->getSGPRSpillVGPRs();
  if (SpillVGPRs.empty())
    return false;

  llvm::LLVMContext &Ctx = F.getContext();

  if (!IPIP.contains(F))
    return false;
  const llvm::MachineInstr *AppMI = IPIP.at(F);

  const auto *LoadPlan =
      SVLocations.getStateValueArrayLoadPlanForInstPoint(*AppMI);
  if (!LoadPlan) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "SVAPhysVGPRPinPass: no SVA load plan for IP in {0}", F.getName()))));
    return false;
  }
  llvm::MCRegister TargetPhys = LoadPlan->StateValueArrayLoadVGPR;
  if (!TargetPhys) {
    LLVM_DEBUG(luthier::dbgs() << "  no SVA load VGPR for " << F.getName()
                               << "; nothing to pin\n");
    return false;
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
  return true;
}

llvm::PreservedAnalyses
SVAPhysVGPRPinPass::run(InstrumentPrototype &IP,
                        InstrumentPrototypeAnalysisManager &IPAM) {
  llvm::Module &IModule = IP.getInstrumentationModule();

  // The IP-side proxy hands out the outer ModuleAnalysisManager; the same
  // MAM caches results for both modules, keyed by the module reference.
  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<ModuleAnalysisManagerInstrumentPrototypeProxy>(IP)
          .getManager();

  llvm::MachineModuleInfo &IMMI =
      MAM.getResult<llvm::MachineModuleAnalysis>(IModule).getMMI();

  const InjectedPayloadAndInstPoint &IPIP =
      MAM.getResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  const SVStorageAndLoadLocations &SVLocations =
      IPAM.getResult<SVStorageAndLoadLocationsAnalysis>(IP);

  bool Changed = false;
  for (llvm::Function &F : IModule) {
    llvm::MachineFunction *MF = IMMI.getMachineFunction(F);
    if (!MF)
      continue;
    Changed |= pinInjectedPayloadMF(*MF, IPIP, SVLocations);
  }

  if (!Changed)
    return llvm::PreservedAnalyses::all();

  // Register-hint changes don't invalidate any analyses on the MF's CFG
  // or IR; preserve everything.
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
