//===-- InjectedPayloadPEIPass.cpp ----------------------------------------===//
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
///
/// \file
/// Implements Luthier's Injected Payload Prologue and Epilogue insertion pass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InjectedPayloadPEIPass.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/ParentPrototypeAnalysis.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include "luthier/ToolCodeGen/StateValueArrayStorage.h"

#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-injected-payload-pei"

namespace luthier {

llvm::PreservedAnalyses
InjectedPayloadPEIPass::run(llvm::MachineFunction &MF,
                            llvm::MachineFunctionAnalysisManager &MFAM) {
  // Skip anything that isn't an injected payload
  llvm::Function &F = MF.getFunction();
  if (!F.hasFnAttribute(InjectedPayloadAttribute)) {
    LLVM_DEBUG(luthier::dbgs()
               << F.getName() << " is not an injected payload; skipping.\n");
    return llvm::PreservedAnalyses::all();
  }

  LLVM_DEBUG(luthier::dbgs()
             << "Running InjectedPayloadPEIPass on " << F.getName() << "\n");

  llvm::LLVMContext &Ctx = F.getContext();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto *TII = ST.getInstrInfo();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  llvm::MachineFrameInfo &MFI = MF.getFrameInfo();
  const auto *SIMFI = MF.getInfo<llvm::SIMachineFunctionInfo>();

  llvm::Module &IModule = *F.getParent();
  const auto &MAMProxy =
      MFAM.getResult<llvm::ModuleAnalysisManagerMachineFunctionProxy>(MF);
  const auto &PAMProxy =
      MFAM.getResult<PrototypeAnalysisManagerMachineFunctionProxy>(MF);

  auto P = [&]() -> Prototype * {
    if (auto *PPA = MAMProxy.getCachedResult<ParentPrototypeAnalysis>(IModule);
        PPA) {
      return PPA->getPrototype();
    }
    return nullptr;
  }();
  if (!P) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("IModule's Prototype was not cached with the "
                      "ParentPrototypeAnalysis"))));
    return llvm::PreservedAnalyses::all();
  }

  const auto *IPIP =
      PAMProxy.getCachedResult<InjectedPayloadAndInstPointAnalysis>(*P);
  if (!IPIP) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "InjectedPayloadAndInstPointAnalysis is required but not cached.")));
    return llvm::PreservedAnalyses::all();
  }
  if (!IPIP->contains(F)) {
    LLVM_DEBUG(luthier::dbgs()
               << F.getName()
               << " has no recorded insertion point; skipping.\n");
    return llvm::PreservedAnalyses::all();
  }
  const llvm::MachineInstr *TargetMI = IPIP->at(F);

  const auto &IPAMProxy =
      MFAM.getResult<PrototypeAnalysisManagerMachineFunctionProxy>(
          MF);

  const SVStorageAndLoadLocations *StateValueLocations =
      IPAMProxy.getCachedResult<SVStorageAndLoadLocationsAnalysis>(*P);
  if (!StateValueLocations) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "SV locations analysis has not been cached")));
    return llvm::PreservedAnalyses::all();
  }

  const InstPointSVALoadPlan *LoadPlan =
      StateValueLocations->getStateValueArrayLoadPlanForInstPoint(*TargetMI);
  if (!LoadPlan) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "No SVA load plan recorded for instrumentation point in {0}",
        F.getName()))));
    return llvm::PreservedAnalyses::all();
  }
  auto &StateValueStorage = LoadPlan->StateValueStorageLocation;

  // Pull the finalized SVA specs from the Prototype-level analysis
  const StateValueArraySpecs *SpecsPtr =
      IPAMProxy.getCachedResult<StateValueArraySpecsAnalysis>(*P);
  if (!SpecsPtr) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "StateValueArraySpecsAnalysis result has not been cached")));
    return llvm::PreservedAnalyses::all();
  }
  const StateValueArraySpecs &Specs = *SpecsPtr;

  llvm::MCRegister SVAVGPR = LoadPlan->StateValueArrayLoadVGPR;

  // Frame regs owned by this payload
  llvm::MCRegister PayloadSPReg = SIMFI->getStackPtrOffsetReg();
  llvm::MCRegister PayloadFPReg = SIMFI->getFrameOffsetReg();
  const bool PayloadSPUsed =
      PayloadSPReg && MRI.isPhysRegUsed(PayloadSPReg);
  const bool PayloadFPUsed =
      PayloadFPReg && MRI.isPhysRegUsed(PayloadFPReg);

  // Wide state regs (PSB and FLAT_SCR) — same save/restore pattern as
  // SP/FP but multi-lane. Their spill lanes on the SVA are gated:
  //   - PSB spill lane exists iff target is non-architected-FS AND
  //     flat scratch has NOT been explicitly enabled (i.e., buffer-scratch
  //     targets).
  //   - FS spill lane exists iff target is non-architected-FS.
  // PSB's function-specific physreg is queried from SIMFI's preloaded
  // arg info rather than hardcoded to SGPR0_SGPR1_SGPR2_SGPR3, so a
  // future ABI relocation stays correct.
  // Both wide regs are subtarget-gated:
  //   * PSB has no preloaded physreg on architected-FS targets (and
  //     \c getPreloadedReg returns \c 0 there), and even on
  //     non-architected-FS targets it is absent when flat-scratch is
  //     explicitly enabled — same conditions the specs uses to gate the
  //     PSB spill lane. Treating PSB as "used" outside that window would
  //     make us try to save an unassigned physreg.
  //   * FLAT_SCR is HW-provided and read-only on architected-FS targets;
  //     the kernel prolog cannot write it to an SVA lane, so we must not
  //     act on it there.
  llvm::MCRegister PayloadPSBReg = SIMFI->getPreloadedReg(
      llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER);
  constexpr llvm::MCRegister PayloadFSReg = llvm::AMDGPU::FLAT_SCR;
  const bool PayloadPSBUsed =
      !ST.hasArchitectedFlatScratch() && !ST.enableFlatScratch() &&
      PayloadPSBReg && MRI.isPhysRegUsed(PayloadPSBReg);
  const bool PayloadFSUsed = !ST.hasArchitectedFlatScratch() &&
                             MRI.isPhysRegUsed(PayloadFSReg);
  const std::optional<uint8_t> PSBSpillLaneOpt =
      Specs.getRsrcBufferSpillLane();
  const std::optional<uint8_t> FSSpillLaneOpt = Specs.getScratchSpillLane();

  // ---- Decide whether this payload actually uses the SVA -----------------
  //
  // Sources of SVA use:
  //   1. The payload reads a lane (or just uses SVA in general)
  //   2. The payload reads/writes to physical registers aliased by the SP
  //   and FP of the payload (e.g. s32, s33).
  //   3. The payload needs frame setup i.e. it spills and has calls.
  bool UsesSVA = false;
  if (SVAVGPR && MRI.isPhysRegUsed(SVAVGPR)) {
    LLVM_DEBUG(luthier::dbgs()
               << "  SVA VGPR " << llvm::printReg(SVAVGPR, ST.getRegisterInfo())
               << " is used\n");
    UsesSVA = true;
  }
  if (!UsesSVA && PayloadSPUsed) {
    LLVM_DEBUG(luthier::dbgs()
               << "  payload SP reg "
               << llvm::printReg(PayloadSPReg, ST.getRegisterInfo())
               << " is used\n");
    UsesSVA = true;
  }
  if (!UsesSVA && PayloadFPUsed) {
    LLVM_DEBUG(luthier::dbgs()
               << "  payload FP reg "
               << llvm::printReg(PayloadFPReg, ST.getRegisterInfo())
               << " is used\n");
    UsesSVA = true;
  }
  if (!UsesSVA && MFI.hasStackObjects()) {
    LLVM_DEBUG(luthier::dbgs() << "  MFI has stack objects\n");
    UsesSVA = true;
  }
  if (!UsesSVA && MFI.hasCalls()) {
    LLVM_DEBUG(luthier::dbgs() << "  MFI has calls\n");
    UsesSVA = true;
  }
  if (!UsesSVA) {
    LLVM_DEBUG(luthier::dbgs()
               << F.getName() << " doesn't use the SVA; skipping PEI.\n");
    return llvm::PreservedAnalyses::all();
  }

  if (!SVAVGPR) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "{0} has no SVA VGPR assigned to it.",
        F.getName()))));
    return llvm::PreservedAnalyses::all();
  }


  // Does the payload need a distinct FP set up? Mirrors predicate is
  // \c SIFrameLowering::hasFPImpl
  const auto *TRI = ST.getRegisterInfo();
  auto payloadNeedsFPSetup = [&]() {
    if (MFI.hasVarSizedObjects() || MFI.hasStackMap() || MFI.hasPatchPoint())
      return true;
    if (MFI.isFrameAddressTaken())
      return true;
    if (TRI->hasStackRealignment(MF))
      return true;
    if (MFI.hasCalls() && !SIMFI->isEntryFunction() && MFI.getStackSize() != 0)
      return true;
    return false;
  };
  const bool NeedsFPSetup =
      PayloadFPReg && PayloadSPReg && PayloadFPReg != PayloadSPReg &&
      payloadNeedsFPSetup();

  bool RequiresAccessToStack = false;
  /// SVA is spilled, therefore the payload needs scratch
  if (StateValueStorage.getStateValueStorageReg() == 0) {
    RequiresAccessToStack = true;
  }
  if (MFI.hasStackObjects() || MFI.hasCalls())
    RequiresAccessToStack = true;

  if (NeedsFPSetup)
    RequiresAccessToStack = true;

  // ---- Emit the prologue ------------------------------------------------
  llvm::MachineBasicBlock &EntryMBB = MF.front();
  auto EntryInsertPt = EntryMBB.SkipPHIsAndLabels(EntryMBB.begin());

  /// Declares the SVA VGPR live-in on \p MBB when nothing in this payload
  /// defines it.
  auto declareSVALiveIn = [&](llvm::MachineBasicBlock &MBB) {
    if (StateValueStorage.requiresLoadAndStoreBeforeUse())
      return;
    if (!MBB.isLiveIn(SVAVGPR))
      MBB.addLiveIn(SVAVGPR);
  };

  declareSVALiveIn(EntryMBB);

  // If the SVS isn't already a free VGPR, load the SVA into the SVA VGPR.
  if (StateValueStorage.requiresLoadAndStoreBeforeUse()) {
    StateValueStorage.emitCodeToLoadSVA(*EntryInsertPt, SVAVGPR);
  }

  // Spill app SP / FP into the SVA lanes the kernel prolog reserved. Only
  // the physregs the payload actually references get spilled.
  auto emitSpillPhysRegToLane = [&](llvm::MachineBasicBlock &MBB,
                                    llvm::MachineBasicBlock::iterator InsertPt,
                                    llvm::MCRegister PhysReg, uint8_t Lane) {
    (void)llvm::BuildMI(MBB, InsertPt, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32), SVAVGPR)
        .addReg(PhysReg, llvm::RegState::Kill)
        .addImm(Lane)
        .addReg(SVAVGPR);
  };
  if (PayloadSPUsed || RequiresAccessToStack) {
    emitSpillPhysRegToLane(EntryMBB, EntryInsertPt, PayloadSPReg,
                           Specs.getStackPointerRegSpillLane());
  }
  if (PayloadFPUsed || NeedsFPSetup) {
    emitSpillPhysRegToLane(EntryMBB, EntryInsertPt, PayloadFPReg,
                           Specs.getFramePointerRegSpillLane());
  }

  // Save/setup PSB and FLAT_SCR.
  auto emitLoadInstrValueFromSALane =
      [&](llvm::MachineBasicBlock &MBB,
          llvm::MachineBasicBlock::iterator InsertPt, llvm::MCRegister BaseReg,
          uint8_t InstrStart, unsigned NumSubLanes) {
        for (unsigned I = 0; I < NumSubLanes; ++I) {
          const llvm::MCRegister Sub =
              (NumSubLanes == 1)
                  ? BaseReg
                  : TRI->getSubReg(
                        BaseReg,
                        llvm::SIRegisterInfo::getSubRegFromChannel(I));
          (void)llvm::BuildMI(MBB, InsertPt, llvm::DebugLoc(),
                              TII->get(llvm::AMDGPU::V_READLANE_B32), Sub)
              .addReg(SVAVGPR)
              .addImm(InstrStart + I);
        }
      };
  auto emitSaveWideRegToSpillLane =
      [&](llvm::MachineBasicBlock &MBB,
          llvm::MachineBasicBlock::iterator InsertPt, llvm::MCRegister BaseReg,
          uint8_t SpillStart, unsigned NumSubLanes) {
        for (unsigned I = 0; I < NumSubLanes; ++I) {
          const llvm::MCRegister Sub =
              (NumSubLanes == 1)
                  ? BaseReg
                  : TRI->getSubReg(
                        BaseReg,
                        llvm::SIRegisterInfo::getSubRegFromChannel(I));
          const bool IsLast = (I + 1 == NumSubLanes);
          (void)llvm::BuildMI(MBB, InsertPt, llvm::DebugLoc(),
                              TII->get(llvm::AMDGPU::V_WRITELANE_B32), SVAVGPR)
              .addReg(Sub, llvm::getKillRegState(IsLast))
              .addImm(SpillStart + I)
              .addReg(SVAVGPR);
        }
      };
  // PSB
  if (PSBSpillLaneOpt && PayloadPSBReg &&
      (PayloadPSBUsed || RequiresAccessToStack)) {
    auto PSBSAIt =
        Specs.findArgumentLane(WAVEFRONT_PRIVATE_SEGMENT_BUFFER);
    if (PSBSAIt == Specs.argument_lane_end()) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "{0}: PSB spill lane is set but SVA has no "
          "WAVEFRONT_PRIVATE_SEGMENT_BUFFER SA lane (instr-side source).",
          F.getName()))));
      return llvm::PreservedAnalyses::all();
    }
    emitSaveWideRegToSpillLane(EntryMBB, EntryInsertPt, PayloadPSBReg,
                               *PSBSpillLaneOpt, /*NumSubLanes=*/4);
    emitLoadInstrValueFromSALane(EntryMBB, EntryInsertPt, PayloadPSBReg,
                                 PSBSAIt->second, /*NumSubLanes=*/4);
  }
  // FLAT_SCR
  if (FSSpillLaneOpt && (PayloadFSUsed || RequiresAccessToStack)) {
    auto FSSAIt = Specs.findArgumentLane(FLAT_SCRATCH);
    if (FSSAIt == Specs.argument_lane_end()) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "{0}: FLAT_SCR spill lane is set but SVA has no FLAT_SCRATCH SA "
          "lane (instr-side source).",
          F.getName()))));
      return llvm::PreservedAnalyses::all();
    }
    emitSaveWideRegToSpillLane(EntryMBB, EntryInsertPt, PayloadFSReg,
                               *FSSpillLaneOpt, /*NumSubLanes=*/2);
    emitLoadInstrValueFromSALane(EntryMBB, EntryInsertPt, PayloadFSReg,
                                 FSSAIt->second, /*NumSubLanes=*/2);
  }

  // If the payload needs stack, read the instrumentation's SP out of the
  // SVA lane the kernel prolog populated, into the payload's SP register.
  // Also setup the frame pointer if needed
  if (RequiresAccessToStack) {
    (void)llvm::BuildMI(EntryMBB, EntryInsertPt, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_READLANE_B32), PayloadSPReg)
        .addReg(SVAVGPR)
        .addImm(Specs.getStackPointerStoreLane());
    if (NeedsFPSetup) {
      (void)llvm::BuildMI(EntryMBB, EntryInsertPt, llvm::DebugLoc(),
                          TII->get(llvm::AMDGPU::S_MOV_B32), PayloadFPReg)
          .addReg(PayloadSPReg);
    }
  }

  // ---- Emit the symmetric epilogue at every return block ----------------
  for (llvm::MachineBasicBlock &MBB : MF) {
    if (!MBB.isReturnBlock())
      continue;
    // A payload with more than one block reaches its returns without ever
    // defining the SVA VGPR, so those blocks need the same live-in as the
    // entry; for a single-block payload this is the entry block and the call
    // is a no-op.
    declareSVALiveIn(MBB);
    auto FirstTerm = MBB.getFirstTerminator();
    // Reverse order: frame-reg restore, then SVS store. Restore only the
    // physregs we spilled on entry.
    auto emitRestorePhysRegFromLane = [&](llvm::MCRegister PhysReg,
                                          uint8_t Lane) {
      (void)llvm::BuildMI(MBB, FirstTerm, llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::V_READLANE_B32), PhysReg)
          .addReg(SVAVGPR)
          .addImm(Lane);
      // Tag the terminator so the live-out is visible to anything that
      // walks operand-level liveness post-PEI.
      if (FirstTerm != MBB.end()) {
        FirstTerm->addOperand(llvm::MachineOperand::CreateReg(
            PhysReg, /*isDef=*/false, /*isImp=*/true));
      }
    };
    // Mirror the prologue's spill conditions: restore whichever of app SP /
    // app FP we saved on entry.
    if (PayloadSPUsed || RequiresAccessToStack) {
      emitRestorePhysRegFromLane(PayloadSPReg,
                                 Specs.getStackPointerRegSpillLane());
    }
    if (PayloadFPUsed || NeedsFPSetup) {
      emitRestorePhysRegFromLane(PayloadFPReg,
                                 Specs.getFramePointerRegSpillLane());
    }
    // Mirror the wide-reg PSB/FS save from the prologue. Reverse of the
    // setup: restore app PSB / FS sub-lanes from their SVA lanes into the
    // physreg. Also mark the terminator with implicit-uses so post-PEI
    // liveness sees the restored physreg live-out.
    auto emitRestoreWideReg = [&](llvm::MCRegister BaseReg, uint8_t StartLane,
                                  unsigned NumSubLanes) {
      for (unsigned I = 0; I < NumSubLanes; ++I) {
        const llvm::MCRegister Sub =
            (NumSubLanes == 1)
                ? BaseReg
                : TRI->getSubReg(
                      BaseReg, llvm::SIRegisterInfo::getSubRegFromChannel(I));
        (void)llvm::BuildMI(MBB, FirstTerm, llvm::DebugLoc(),
                            TII->get(llvm::AMDGPU::V_READLANE_B32), Sub)
            .addReg(SVAVGPR)
            .addImm(StartLane + I);
        if (FirstTerm != MBB.end()) {
          FirstTerm->addOperand(llvm::MachineOperand::CreateReg(
              Sub, /*isDef=*/false, /*isImp=*/true));
        }
      }
    };
    if (PSBSpillLaneOpt && PayloadPSBReg &&
        (PayloadPSBUsed || RequiresAccessToStack)) {
      emitRestoreWideReg(PayloadPSBReg, *PSBSpillLaneOpt,
                         /*NumSubLanes=*/4);
    }
    if (FSSpillLaneOpt && (PayloadFSUsed || RequiresAccessToStack)) {
      emitRestoreWideReg(PayloadFSReg, *FSSpillLaneOpt, /*NumSubLanes=*/2);
    }
    if (StateValueStorage.requiresLoadAndStoreBeforeUse()) {
      // Emit at FirstTerm of THIS return block, not at the entry point
      if (FirstTerm != MBB.end()) {
        StateValueStorage.emitCodeToStoreSVA(*FirstTerm, SVAVGPR);
      }
    }
  }

  LLVM_DEBUG({
    luthier::dbgs() << "After InjectedPayloadPEIPass on " << F.getName()
                    << ":\n";
    MF.print(luthier::dbgs());
  });

  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
