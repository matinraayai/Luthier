//===-- TargetModulePatcherPass.cpp -----------------------------*- C++ -*-===//
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
/// \file TargetModulePatcherPass.cpp
/// Implements the \c TargetModuelPatcherPass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetModulePatcherPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/Cloning.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessIModulePass.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/LuthierBranchRelaxation.h"
#include "luthier/ToolCodeGen/PrePostAmbleEmitter.h"
#include "luthier/ToolCodeGen/SVAFrameLanes.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include "luthier/ToolCodeGen/StateValueArrayStorage.h"
#include "luthier/ToolCodeGen/WrapperAnalysisPasses.h"
#include <AMDGPU.h>
#include <AMDGPUTargetMachine.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalIFunc.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Transforms/Utils/Cloning.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-target-module-patcher"

namespace luthier {

char TargetModulePatcherPass::ID = 0;

LUTHIER_INITIALIZE_LEGACY_PASS_BODY(TargetModulePatcherPass,
                                    "target-module-patcher",
                                    "Luthier Target Module Patcher Pass",
                                    /*CFGOnly=*/false,
                                    /*IsAnalysis=*/false)

void TargetModulePatcherPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<IModuleMAMWrapperPass>();
  AU.addRequired<LRStateValueStorageAndLoadLocationsAnalysis>();
  AU.addRequired<IModuleIPPredicatedLivenessAnalysis>();
  AU.addRequired<llvm::MachineModuleInfoWrapperPass>();
  llvm::ModulePass::getAnalysisUsage(AU);
}

namespace {

/// Emits the per-wave scratch setup at the kernel entry: spills the
/// kernarg-derived PSB.sub0/sub1 and FLAT_SCRATCH_INIT lo/hi into SVA
/// lanes, adds PRIVATE_SEGMENT_WAVE_BYTE_OFFSET to compute the wave's
/// scratch base, stores SGPR32 to the instrumentation-stack-start lane,
/// and reads the spilled kernarg values back into SGPR0/1/FS_LO/HI so
/// the application's prolog still sees them.
///
/// \p UsesDynamicStack and \p PrivateSegmentFixedSize replaced the
/// previous \c amdgpu::hsamd::Kernel::Metadata reference — both are the
/// only fields the function reads. Sourcing them is the caller's job
/// (kernel metadata, per-function attributes, etc.).
llvm::Error emitCodeToSetupScratch(llvm::MachineInstr &EntryInstr,
                                   llvm::MCRegister SVSStorageVGPR,
                                   bool UsesDynamicStack,
                                   unsigned PrivateSegmentFixedSize,
                                   const StateValueArraySpecs &Specs) {
  auto &MF = *EntryInstr.getMF();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  const auto &TRI = *ST.getRegisterInfo();
  auto &MFI = *MF.getInfo<llvm::SIMachineFunctionInfo>();
  // On architected-flat-scratch subtargets (gfx9-CDNA, gfx10.3+, gfx11+,
  // gfx12) FLAT_SCRATCH_INIT is not a preloaded SGPR — the hardware sets
  // up the flat-scratch state into the architectural register directly,
  // so the kernel prolog has no SGPR pair to spill into the SVA. The
  // matching slot lookup returns nullopt, and the corresponding spill/
  // restore step must be skipped. \c InjectedPayloadPEIPass already
  // gates its FrameSpillSlots on this same predicate.
  const bool ArchitectedFS = ST.hasArchitectedFlatScratch();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   emitCodeToSetupScratch MF='"
             << MF.getName()
             << "' SVSVGPR=" << llvm::printReg(SVSStorageVGPR, &TRI)
             << " dynStack=" << UsesDynamicStack << " privSegFixedSize="
             << PrivateSegmentFixedSize << " archFS=" << ArchitectedFS << "\n");
  // Copy SGPR0, SGPR1 (and on non-architected-FS targets,
  // FLAT_SCR_LO / FLAT_SCR_HI) into the state-value register at the
  // lanes the StateValueArraySpecs layout reserves for them. Layout is
  // documented in SVAFrameLanes.h.
  auto SGPR0SpillSlot =
      getKernelPrologFrameSpillLane(llvm::AMDGPU::SGPR0, Specs);
  auto SGPR1SpillSlot =
      getKernelPrologFrameSpillLane(llvm::AMDGPU::SGPR1, Specs);
  std::optional<unsigned> SGPRFlatScrLoSpillSlot;
  std::optional<unsigned> SGPRFlatScrHiSpillSlot;
  if (!ArchitectedFS) {
    SGPRFlatScrLoSpillSlot =
        getKernelPrologFrameSpillLane(llvm::AMDGPU::FLAT_SCR_LO, Specs);
    SGPRFlatScrHiSpillSlot =
        getKernelPrologFrameSpillLane(llvm::AMDGPU::FLAT_SCR_HI, Specs);
  }
  assert(
      SGPR0SpillSlot && SGPR1SpillSlot &&
      (ArchitectedFS || (SGPRFlatScrLoSpillSlot && SGPRFlatScrHiSpillSlot)) &&
      "kernel-prolog SVA lanes must exist for SGPR0/1 (+ FS_LO/HI on "
      "non-architected-FS targets)");

  // The spill/recompute/restore dance around \c PRIVATE_SEGMENT_BUFFER
  // (PSB) and \c FLAT_SCRATCH_INIT is only meaningful on non-
  // architected-FS targets, where (1) the kernel prolog needs the
  // unmodified PSB.sub0/sub1 + FS_LO/HI to compute its own scratch base
  // via PSWO, and (2) the payload's prologue reads those same SGPRs to
  // set up its scratch. On architected-FS subtargets the hardware
  // sets up the per-wave scratch into the architectural register
  // automatically; PSB and FLAT_SCRATCH_INIT are not preloaded SGPRs
  // (so calling \c getPreloadedReg on them returns the null register
  // and any \c getSubReg below would assert), and there is no
  // PSWO-style adjustment to apply.
  if (!ArchitectedFS) {
    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
        .addReg(TRI.getSubReg(
            MFI.getPreloadedReg(
                llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
            llvm::AMDGPU::sub0))
        .addImm(*SGPR0SpillSlot)
        .addReg(SVSStorageVGPR);

    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
        .addReg(TRI.getSubReg(
            MFI.getPreloadedReg(
                llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
            llvm::AMDGPU::sub1))
        .addImm(*SGPR1SpillSlot)
        .addReg(SVSStorageVGPR);

    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
        .addReg(TRI.getSubReg(
            MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
            llvm::AMDGPU::sub0))
        .addImm(*SGPRFlatScrLoSpillSlot)
        .addReg(SVSStorageVGPR);

    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
        .addReg(TRI.getSubReg(
            MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
            llvm::AMDGPU::sub1))
        .addImm(*SGPRFlatScrHiSpillSlot)
        .addReg(SVSStorageVGPR);

    // Add the PSWO to SGPR0/its carry to SGPR1
    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_ADD_U32))
        .addReg(TRI.getSubReg(
                    MFI.getPreloadedReg(
                        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                    llvm::AMDGPU::sub0),
                llvm::RegState::Define)
        .addReg(TRI.getSubReg(
                    MFI.getPreloadedReg(
                        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                    llvm::AMDGPU::sub0),
                llvm::RegState::Kill)
        .addReg(MFI.getPreloadedReg(
            llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET));

    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_ADDC_U32))
        .addReg(TRI.getSubReg(
                    MFI.getPreloadedReg(
                        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                    llvm::AMDGPU::sub1),
                llvm::RegState::Define)
        .addReg(TRI.getSubReg(
                    MFI.getPreloadedReg(
                        llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                    llvm::AMDGPU::sub1),
                llvm::RegState::Kill)
        .addImm(0);
  }
  // Add the PSWO to FS_init_lo/its carry to FS_init_hi. Only needed on
  // non-architected-FS targets — on architected-FS the FS is set up by
  // HW directly and FLAT_SCRATCH_INIT isn't a preloaded SGPR.
  if (!ArchitectedFS) {
    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_ADD_U32))
        .addReg(
            TRI.getSubReg(MFI.getPreloadedReg(
                              llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                          llvm::AMDGPU::sub0),
            llvm::RegState::Define)
        .addReg(
            TRI.getSubReg(MFI.getPreloadedReg(
                              llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                          llvm::AMDGPU::sub0),
            llvm::RegState::Kill)
        .addReg(MFI.getPreloadedReg(
            llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET));
    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_ADDC_U32))
        .addReg(
            TRI.getSubReg(MFI.getPreloadedReg(
                              llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                          llvm::AMDGPU::sub1),
            llvm::RegState::Define)
        .addReg(
            TRI.getSubReg(MFI.getPreloadedReg(
                              llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                          llvm::AMDGPU::sub1),
            llvm::RegState::Kill)
        .addImm(0);
  }

  unsigned int InstrumentationStackStart{0};
  if (UsesDynamicStack)
    llvm_unreachable("Not implemented");
  else {
    InstrumentationStackStart = PrivateSegmentFixedSize;
  }
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                "InstrumentationStackStart="
                             << InstrumentationStackStart << "\n");
  // Set s32 to be the maximum amount of stack requested by the hook
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), llvm::AMDGPU::SGPR32)
      .addImm(InstrumentationStackStart);

  // Store frame registers in their slots. The kernel-prolog's "store" table
  // is the same set of lanes as the spill table (lanes 0-3); the prolog
  // overwrites the original kernarg values with the per-wave-computed ones
  // so the payload prologue reads the per-wave values back via the same
  // lane indices.
  for (const auto &[PhysReg, StoreSlot] :
       getKernelPrologFrameStoreSlots(Specs)) {
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "frame-store "
                               << llvm::printReg(PhysReg, &TRI) << " -> lane "
                               << StoreSlot << "\n");
    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
        .addReg(PhysReg)
        .addImm(StoreSlot)
        .addReg(SVSStorageVGPR);
  }

  // Restore S0, S1 (and FS_init_lo/hi). The restore is symmetric to the
  // spill above — on architected-FS targets we never spilled SGPR0/1
  // (they hold the kernarg-segment ptr, not PSB, and don't get
  // PSWO-adjusted), so there's nothing to restore.
  if (!ArchitectedFS) {
    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_READLANE_B32), llvm::AMDGPU::SGPR0)
        .addReg(SVSStorageVGPR)
        .addImm(*SGPR0SpillSlot);

    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_READLANE_B32), llvm::AMDGPU::SGPR1)
        .addReg(SVSStorageVGPR)
        .addImm(*SGPR1SpillSlot);

    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_READLANE_B32))
        .addReg(
            TRI.getSubReg(MFI.getPreloadedReg(
                              llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                          llvm::AMDGPU::sub0),
            llvm::RegState::Define)
        .addReg(SVSStorageVGPR)
        .addImm(*SGPRFlatScrLoSpillSlot);

    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_READLANE_B32))
        .addReg(
            TRI.getSubReg(MFI.getPreloadedReg(
                              llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                          llvm::AMDGPU::sub1),
            llvm::RegState::Define)
        .addReg(SVSStorageVGPR)
        .addImm(*SGPRFlatScrHiSpillSlot);
  }

  return llvm::Error::success();
}

llvm::Error emitCodeToStoreSGPRKernelArg(llvm::MachineInstr &InsertionPoint,
                                         llvm::MCRegister SrcSGPR,
                                         llvm::MCRegister SVSVGPR,
                                         int SpillSlotStart, int NumSlots,
                                         bool KillAfterUse) {
  const auto &TRI = *InsertionPoint.getMF()->getSubtarget().getRegisterInfo();
  const auto &TII = *InsertionPoint.getMF()->getSubtarget().getInstrInfo();
  size_t Size = TRI.getRegSizeInBits(*TRI.getPhysRegBaseClass(SrcSGPR));
  auto &InsertionPointMBB = *InsertionPoint.getParent();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]     emitCodeToStoreSGPRKernelArg "
                "src="
             << llvm::printReg(SrcSGPR, &TRI)
             << " SVS=" << llvm::printReg(SVSVGPR, &TRI) << " size=" << Size
             << "b slotStart=" << SpillSlotStart << " numSlots=" << NumSlots
             << " kill=" << KillAfterUse << "\n");
  if (Size == 32) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NumSlots == 1, "Mismatch between number of SGPRs in the argument and "
                       "save slot lanes."));
    llvm::BuildMI(InsertionPointMBB, InsertionPoint, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSVGPR)
        .addReg(SrcSGPR, llvm::getKillRegState(KillAfterUse))
        .addImm(SpillSlotStart)
        .addReg(SVSVGPR);
  } else {
    size_t NumChannels = Size / 32;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NumSlots == NumChannels,
        "Mismatch between number of SGPRs in the argument and "
        "save slot lanes."));
    for (int i = 0; i < NumSlots; i++) {
      auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
      llvm::BuildMI(InsertionPointMBB, InsertionPoint, llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSVGPR)
          .addReg(TRI.getSubReg(SrcSGPR, SubIdx),
                  llvm::getKillRegState(KillAfterUse))
          .addImm(SpillSlotStart + i)
          .addReg(SVSVGPR);
    }
  }
  return llvm::Error::success();
}

/// Walk the target MF's per-MBB storage intervals from
/// \c SVStorageAndLoadLocations and emit
/// \c currentSVS.emitCodeToSwitchSVS(MI, nextSVS) at every boundary. This
/// makes the SVA actually migrate between storage schemes across the
/// target's control flow — without this, the load plan exists but the
/// runtime state never matches it.
void emitSVSSwitchesForMF(llvm::MachineFunction &MF,
                          const SVStorageAndLoadLocations &SVLocations,
                          const llvm::SlotIndexes &SI) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   emitSVSSwitchesForMF "
                "MF='"
             << MF.getName() << "' (" << MF.size() << " MBB(s))\n");
  unsigned SwitchesEmitted = 0;
  for (llvm::MachineBasicBlock &MBB : MF) {
    llvm::ArrayRef<StateValueStorageSegment> Segments =
        SVLocations.getStorageIntervals(MBB);
    if (Segments.size() < 2)
      continue;
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                               << llvm::printMBBReference(MBB) << " has "
                               << Segments.size() << " segments\n");
    for (unsigned I = 0, E = Segments.size() - 1; I < E; ++I) {
      const StateValueArrayStorage &Curr = Segments[I].getSVS();
      const StateValueArrayStorage &Next = Segments[I + 1].getSVS();
      if (Curr == Next)
        continue;
      ++SwitchesEmitted;
      // Resolve the boundary slot index → MI inside MBB. The switch
      // must be emitted before the MI at Segments[I+1].begin(); that's
      // where the runtime location of the SVA changes. SlotIndexes maps
      // a slot to its owning MI (or returns null at non-MI slots like
      // block-end), so we use getInstructionFromIndex and fall back to
      // walking forward to the next instruction within MBB.
      llvm::SlotIndex Boundary = Segments[I + 1].begin();
      llvm::MachineInstr *MI = SI.getInstructionFromIndex(Boundary);
      llvm::MachineBasicBlock::iterator InsertPt;
      if (MI && MI->getParent() == &MBB) {
        InsertPt = MI->getIterator();
      } else {
        // No MI is directly anchored at this slot (e.g., the slot
        // corresponds to a block boundary / deleted MI). Fall back to
        // the first terminator so the switch still happens before any
        // control-flow leaves MBB.
        InsertPt = MBB.getFirstTerminator();
        if (InsertPt == MBB.end())
          InsertPt = MBB.end();
      }
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]       "
                                    "emit SVS switch at segment boundary "
                                 << I << " -> " << (I + 1) << "\n");
      Curr.emitCodeToSwitchSVS(InsertPt, Next);
    }
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   emitted " << SwitchesEmitted
             << " SVS switch(es) for MF '" << MF.getName() << "'\n");
}

/// Map a Luthier \c ScalarValueArgument to the AMDGPU
/// \c PreloadedValue that supplies its bits from the HSA kernarg preload.
/// Returns \c std::nullopt for SVA entries that have no kernarg source
/// (e.g., USER_ARG_PTR / IMPLICIT_ARG_OFFSET, which are filled in
/// elsewhere). Used by the initial-entry-kernel setup to find the source
/// SGPR for each requested kernarg spill.
static std::optional<llvm::AMDGPUFunctionArgInfo::PreloadedValue>
preloadedValueForSVA(ScalarValueArgument SA) {
  switch (SA) {
  case WAVEFRONT_PRIVATE_SEGMENT_BUFFER:
    return llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER;
  case KERNEL_ARG_PTR:
    return llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR;
  case DISPATCH_ID:
    return llvm::AMDGPUFunctionArgInfo::DISPATCH_ID;
  case FLAT_SCRATCH:
    return llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT;
  case PRIVATE_SEGMENT_WAVE_BYTE_OFFSET:
    return llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET;
  case DISPATCH_PTR:
    return llvm::AMDGPUFunctionArgInfo::DISPATCH_PTR;
  case QUEUE_PTR:
    return llvm::AMDGPUFunctionArgInfo::QUEUE_PTR;
  case WORK_ITEM_PRIVATE_SEGMENT_SIZE:
    return llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_SIZE;
  case USER_ARG_PTR:
  case IMPLICIT_ARG_OFFSET:
    return std::nullopt;
  }
  return std::nullopt;
}

/// Emit, at every initial-entry kernel's first instruction, the SVA
/// kernarg-preload setup: scratch/stack setup (when requested) plus an
/// \c emitCodeToStoreSGPRKernelArg for each \c ScalarValueArgument in
/// \c KernelInfo.RequestedKernelArguments. The SVA storage register for
/// the entry MBB comes from \c SVLocations.
llvm::Error
emitInitialEntryKernelSetup(llvm::MachineModuleInfo &TargetMMI,
                            llvm::Module &TargetModule,
                            const FunctionPreambleDescriptor &FPD,
                            const SVStorageAndLoadLocations &SVLocations,
                            const StateValueArraySpecs &Specs) {
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] "
                                "emitInitialEntryKernelSetup over "
                             << FPD.Kernels.size() << " kernel(s)\n");
  for (const auto &[KernelMF, KernelInfo] : FPD.Kernels) {
    if (!KernelInfo.usesSVA()) {
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   kernel '"
                                 << (KernelMF ? KernelMF->getName() : "<null>")
                                 << "' does not use SVA, skipping\n");
      continue;
    }
    llvm::MachineFunction *MF = const_cast<llvm::MachineFunction *>(KernelMF);
    if (!MF || MF->empty()) {
      LLVM_DEBUG(luthier::dbgs()
                 << "[TargetModulePatcherPass]   kernel MF null "
                    "or empty, skipping\n");
      continue;
    }
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   kernel '"
                               << MF->getName() << "' uses SVA; setup begin\n");
    llvm::MachineBasicBlock &EntryMBB = MF->front();
    if (EntryMBB.empty())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: kernel '{0}' has an empty entry MBB; "
          "cannot insert SVA setup",
          MF->getName()));
    llvm::MachineInstr &EntryInstr = EntryMBB.front();

    llvm::ArrayRef<StateValueStorageSegment> EntrySegments =
        SVLocations.getStorageIntervals(EntryMBB);
    if (EntrySegments.empty())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: kernel '{0}' has no SVA storage "
          "segment at entry; SV-load-locations analysis is inconsistent",
          MF->getName()));
    llvm::MCRegister SVSStorageReg =
        EntrySegments.front().getSVS().getStateValueStorageReg();
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]     entry SVS storage reg="
               << llvm::printReg(SVSStorageReg,
                                 MF->getSubtarget().getRegisterInfo())
               << "\n");

    if (KernelInfo.RequiresScratchAndStackSetup) {
      const llvm::MachineFrameInfo &MFI = MF->getFrameInfo();
      bool UsesDynamicStack = MFI.hasVarSizedObjects();
      unsigned PrivateSegmentFixedSize =
          static_cast<unsigned>(MFI.getStackSize());
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                    "RequiresScratchAndStackSetup; emitting\n");
      if (auto Err = emitCodeToSetupScratch(EntryInstr, SVSStorageReg,
                                            UsesDynamicStack,
                                            PrivateSegmentFixedSize, Specs))
        return Err;
    }

    const auto &SIMFI = *MF->getInfo<llvm::SIMachineFunctionInfo>();
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]     "
                  "RequestedKernelArguments count="
               << KernelInfo.RequestedKernelArguments.size() << "\n");
    for (ScalarValueArgument SA : KernelInfo.RequestedKernelArguments) {
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]       SVA arg="
                                 << static_cast<unsigned>(SA) << "\n");
      auto LaneIt = Specs.findArgumentLane(SA);
      if (LaneIt == Specs.argument_lane_end())
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "TargetModulePatcherPass: kernel '{0}' requests SVA arg {1} "
            "but the SVA specs do not assign it a lane",
            MF->getName(), static_cast<unsigned>(SA)));
      std::optional<llvm::AMDGPUFunctionArgInfo::PreloadedValue> PV =
          preloadedValueForSVA(SA);
      if (!PV) {
        LLVM_DEBUG(luthier::dbgs()
                   << "[TargetModulePatcherPass]         "
                      "no preloaded value (filled elsewhere)\n");
        continue; // USER_ARG_PTR / IMPLICIT_ARG_OFFSET: filled in elsewhere.
      }
      llvm::MCRegister SrcSGPR = SIMFI.getPreloadedReg(*PV);
      if (!SrcSGPR)
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "TargetModulePatcherPass: kernel '{0}' requests SVA arg {1} "
            "but the source preloaded SGPR is not enabled on the MF",
            MF->getName(), static_cast<unsigned>(SA)));
      int NumSlots =
          static_cast<int>(StateValueArraySpecs::getArgumentLaneSize(SA));
      if (auto Err =
              emitCodeToStoreSGPRKernelArg(EntryInstr, SrcSGPR, SVSStorageReg,
                                           /*SpillSlotStart=*/LaneIt->second,
                                           NumSlots, /*KillAfterUse=*/false))
        return Err;
    }
  }
  (void)TargetMMI;
  (void)TargetModule;
  return llvm::Error::success();
}

/// Clone an IModule \c MachineFunction's frame layout into the target
/// host \c MachineFunction. Frame indices in the cloned payload MIs will
/// continue to be valid because we add equivalent slots to the host MF
/// in the same index order (with the same offset & alignment). This is
/// a verbatim port of the prior \c PatchLiftedRepresentationPass helper.
void patchFrameInfo(const llvm::MachineFunction &InjectedPayloadMF,
                    llvm::MachineFunction &ToBeInstrumentedMF) {
  auto &SrcMFI = InjectedPayloadMF.getFrameInfo();
  if (!SrcMFI.hasStackObjects()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   patchFrameInfo: payload '"
               << InjectedPayloadMF.getName()
               << "' has no stack objects, skipping\n");
    return;
  }
  auto &DstMFI = ToBeInstrumentedMF.getFrameInfo();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   patchFrameInfo: payload '"
             << InjectedPayloadMF.getName()
             << "' srcStack=" << SrcMFI.getStackSize() << " into host '"
             << ToBeInstrumentedMF.getName() << "' dstStack(before)="
             << DstMFI.getStackSize() << " srcObjs=" << SrcMFI.getNumObjects()
             << " srcFixed=" << SrcMFI.getNumFixedObjects() << "\n");
  DstMFI.setStackSize(DstMFI.getStackSize() + SrcMFI.getStackSize());

  auto CopyObjectProperties = [](llvm::MachineFrameInfo &DstMFI,
                                 const llvm::MachineFrameInfo &SrcMFI,
                                 int SrcFI, int DestFI) {
    if (SrcMFI.isStatepointSpillSlotObjectIndex(SrcFI))
      DstMFI.markAsStatepointSpillSlotObjectIndex(DestFI);
    DstMFI.setObjectSSPLayout(DestFI, SrcMFI.getObjectSSPLayout(SrcFI));
    DstMFI.setObjectZExt(DestFI, SrcMFI.isObjectZExt(SrcFI));
    DstMFI.setObjectSExt(DestFI, SrcMFI.isObjectSExt(SrcFI));
  };
  for (int I = 0, E = SrcMFI.getNumObjects() - SrcMFI.getNumFixedObjects();
       I != E; ++I) {
    if (SrcMFI.isDeadObjectIndex(I))
      continue;
    int NewFI;
    if (SrcMFI.isVariableSizedObjectIndex(I)) {
      NewFI = DstMFI.CreateVariableSizedObject(SrcMFI.getObjectAlign(I),
                                               SrcMFI.getObjectAllocation(I));
    } else {
      NewFI = DstMFI.CreateStackObject(
          SrcMFI.getObjectSize(I), SrcMFI.getObjectAlign(I),
          SrcMFI.isSpillSlotObjectIndex(I), SrcMFI.getObjectAllocation(I),
          SrcMFI.getStackID(I));
      DstMFI.setObjectOffset(NewFI, SrcMFI.getObjectOffset(I));
    }
    CopyObjectProperties(DstMFI, SrcMFI, I, NewFI);
  }
  for (int I = -1; I >= (int)-SrcMFI.getNumFixedObjects(); --I) {
    if (SrcMFI.isDeadObjectIndex(I))
      continue;
    int NewFI = DstMFI.CreateFixedObject(
        SrcMFI.getObjectSize(I), SrcMFI.getObjectOffset(I),
        SrcMFI.isImmutableObjectIndex(I), SrcMFI.isAliasedObjectIndex(I));
    CopyObjectProperties(DstMFI, SrcMFI, I, NewFI);
  }
}

/// Inline-clone the body of \p InjectedPayloadMF directly before
/// \p InsertionPointMI in the host MF, splitting the host MBB at the
/// insertion point so that the payload's return blocks branch through
/// to the continuation. \p VMap maps GVs in the IModule to their
/// counterparts in the target module so cross-module operands
/// resolve. Verbatim port of the prior PatchLiftedRepresentationPass
/// helper, scoped down to the inline path (per the new design
/// decision to always inline payloads at their IP).
void inlineInjectedPayload(const llvm::MachineFunction &InjectedPayloadMF,
                           llvm::MachineInstr &InsertionPointMI,
                           llvm::DenseMap<const llvm::MachineBasicBlock *,
                                          llvm::MachineBasicBlock *> &MBBMap,
                           const llvm::ValueToValueMapTy &VMap) {
  auto &InsertionPointMBB = *InsertionPointMI.getParent();
  auto &ToBeInstrumentedMF = *InsertionPointMI.getMF();
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   inlineInjectedPayload payload='"
             << InjectedPayloadMF.getName() << "' (" << InjectedPayloadMF.size()
             << " MBB(s)) into host '" << ToBeInstrumentedMF.getName()
             << "' at " << llvm::printMBBReference(InsertionPointMBB)
             << " before MI opcode=" << InsertionPointMI.getOpcode() << "\n");
  unsigned NumReturnBlocksInHook = 0;
  const llvm::MachineBasicBlock *HookLastReturnMBB = nullptr;
  llvm::MachineBasicBlock *HookLastReturnMBBDest = nullptr;

  for (auto It = InjectedPayloadMF.rbegin(), End = InjectedPayloadMF.rend();
       It != End; ++It) {
    if (It->isReturnBlock() && HookLastReturnMBB == nullptr) {
      HookLastReturnMBB = &*It;
      NumReturnBlocksInHook++;
    }
  }
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     payload return "
                                "block count="
                             << NumReturnBlocksInHook << "\n");
  if (!HookLastReturnMBB) {
    llvm::report_fatal_error(
        "TargetModulePatcherPass: payload MF has no return block");
  }

  if (InjectedPayloadMF.size() > 1) {
    if (InsertionPointMI.getIterator() != InsertionPointMBB.begin())
      HookLastReturnMBBDest =
          InsertionPointMBB.splitAt(*InsertionPointMI.getPrevNode());
    else
      HookLastReturnMBBDest = &InsertionPointMBB;
    if (NumReturnBlocksInHook == 1)
      MBBMap.insert({HookLastReturnMBB, HookLastReturnMBBDest});
  }

  for (const auto &HookMBB : InjectedPayloadMF) {
    if (HookMBB.isEntryBlock()) {
      if (InsertionPointMI.getIterator() != InsertionPointMBB.begin()) {
        MBBMap.insert({&HookMBB, &InsertionPointMBB});
      } else if (InjectedPayloadMF.size() == 1) {
        MBBMap.insert({&InjectedPayloadMF.front(), &InsertionPointMBB});
      } else {
        // Multi-MBB payload, IP at host MBB's first instruction. We
        // create a NewEntryBlock that holds the payload's entry MIs,
        // splice it into the MF's MBB list directly before the host
        // MBB, and reroute every predecessor.
        //
        // `addSuccessor` / `removeSuccessor` only update CFG metadata
        // — the predecessor's ACTUAL terminator `MachineInstr`
        // operands still reference the host MBB. Without rewriting
        // those operands, a predecessor that reaches the host MBB
        // via an explicit branch (e.g. `s_cbranch_execnz` with a
        // target MBB operand) jumps straight past NewEntryBlock and
        // skips the instrumentation; the wave then deadlocks on a
        // later `s_waitcnt` because its half of the wave-mode
        // instrumentation never ran. `ReplaceUsesOfBlockWith` walks
        // the terminator's MBB operands, swaps the reference, AND
        // calls `replaceSuccessor` — so it subsumes the manual
        // `removeSuccessor` call.
        auto *NewEntryBlock = ToBeInstrumentedMF.CreateMachineBasicBlock();
        ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                  NewEntryBlock);
        llvm::SmallVector<llvm::MachineBasicBlock *, 2> PredMBBs;
        for (auto It = InsertionPointMBB.pred_begin();
             It != InsertionPointMBB.pred_end(); ++It) {
          PredMBBs.push_back(*It);
        }
        for (auto *PredMBB : PredMBBs) {
          PredMBB->ReplaceUsesOfBlockWith(&InsertionPointMBB, NewEntryBlock);
        }
        NewEntryBlock->addSuccessor(&InsertionPointMBB);
        MBBMap.insert({&InjectedPayloadMF.front(), NewEntryBlock});
      }
    } else if (HookMBB.isReturnBlock()) {
      if (NumReturnBlocksInHook > 1 || &HookMBB != HookLastReturnMBB) {
        auto *TargetReturnBlock = ToBeInstrumentedMF.CreateMachineBasicBlock();
        ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                  TargetReturnBlock);
        MBBMap.insert({&HookMBB, TargetReturnBlock});
      }
    } else {
      auto *NewHookMBB = ToBeInstrumentedMF.CreateMachineBasicBlock();
      ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                NewHookMBB);
      MBBMap.insert({&HookMBB, NewHookMBB});
    }
  }

  for (auto &MBB : InjectedPayloadMF) {
    auto *DstMBB = MBBMap[&MBB];
    for (auto It = MBB.succ_begin(), IterEnd = MBB.succ_end(); It != IterEnd;
         ++It) {
      auto *DstSuccMBB = MBBMap[*It];
      if (!DstMBB->isSuccessor(DstSuccMBB))
        DstMBB->addSuccessor(DstSuccMBB, MBB.getSuccProbability(It));
    }
    if (MBB.isReturnBlock() && NumReturnBlocksInHook > 1) {
      if (!DstMBB->isSuccessor(HookLastReturnMBBDest))
        DstMBB->addSuccessor(HookLastReturnMBBDest);
    }
  }

  const llvm::TargetSubtargetInfo &STI = ToBeInstrumentedMF.getSubtarget();
  const llvm::TargetInstrInfo *TII = STI.getInstrInfo();
  const llvm::TargetRegisterInfo *TRI = STI.getRegisterInfo();
  auto &SrcMRI = InjectedPayloadMF.getRegInfo();
  auto &DstMRI = ToBeInstrumentedMF.getRegInfo();

  // Allocate fresh virtual registers in the target MF's MRI for every
  // virtual register used by the payload, copying register class /
  // register bank / type / allocation hints. Operand cloning below uses
  // this map to translate payload vreg references into target vreg
  // references. Without this remap, MachineInstr::addOperand walks
  // DstMRI's per-vreg storage tables with payload vreg indices and OOBs
  // when the payload's vreg count exceeds the target MF's.
  llvm::DenseMap<llvm::Register, llvm::Register> SrcToDstVReg;
  for (unsigned I = 0, E = SrcMRI.getNumVirtRegs(); I != E; ++I) {
    llvm::Register SrcReg = llvm::Register::index2VirtReg(I);
    llvm::Register NewReg =
        DstMRI.createIncompleteVirtualRegister(SrcMRI.getVRegName(SrcReg));
    DstMRI.setRegClassOrRegBank(NewReg, SrcMRI.getRegClassOrRegBank(SrcReg));
    llvm::LLT RegTy = SrcMRI.getType(SrcReg);
    if (RegTy.isValid())
      DstMRI.setType(NewReg, RegTy);
    if (const auto *Hints = SrcMRI.getRegAllocationHints(SrcReg))
      for (llvm::Register PrefReg : Hints->second)
        DstMRI.addRegAllocationHint(NewReg, PrefReg);
    SrcToDstVReg[SrcReg] = NewReg;
  }

  llvm::DenseSet<const uint32_t *> ConstRegisterMasks;
  for (const uint32_t *Mask : TRI->getRegMasks())
    ConstRegisterMasks.insert(Mask);

  for (const auto &MBB : InjectedPayloadMF) {
    auto *DstMBB = MBBMap[&MBB];
    llvm::MachineBasicBlock::iterator InsertionPoint;
    // A single-MBB payload's only MBB is BOTH its entry and its (sole)
    // return block. We need the entry-block case (insert at the IP MI)
    // to win — otherwise the return-block branch maps the payload's MIs
    // to DstMBB->begin() and we end up inserting the payload at the
    // start of the host MBB, BEFORE every original MI that was supposed
    // to run prior to the IP.
    if (MBB.isEntryBlock() && InjectedPayloadMF.size() == 1)
      InsertionPoint = InsertionPointMI.getIterator();
    else if (MBB.isReturnBlock() && NumReturnBlocksInHook == 1)
      InsertionPoint = DstMBB->begin();
    else
      InsertionPoint = DstMBB->end();

    for (auto &SrcMI : MBB.instrs()) {
      // Drop the payload's return-edge terminators — the inlined body
      // falls through (or branches) to the host's continuation MBB
      // instead of "returning" to the runtime caller.
      if (MBB.isReturnBlock() && SrcMI.isTerminator())
        break;
      if (SrcMI.isBundle())
        continue;
      const auto &MCID = TII->get(SrcMI.getOpcode());
      auto *DstMI =
          ToBeInstrumentedMF.CreateMachineInstr(MCID, llvm::DebugLoc(),
                                                /*NoImplicit=*/true);
      DstMI->setFlags(SrcMI.getFlags());
      DstMI->setAsmPrinterFlag(SrcMI.getAsmPrinterFlags());
      DstMBB->insert(InsertionPoint, DstMI);
      for (auto &SrcMO : SrcMI.operands()) {
        llvm::MachineOperand DstMO(SrcMO);
        DstMO.clearParent();
        if (DstMO.isReg() && DstMO.getReg().isVirtual()) {
          auto It = SrcToDstVReg.find(DstMO.getReg());
          assert(It != SrcToDstVReg.end() &&
                 "payload references vreg not in source MRI");
          DstMO.setReg(It->second);
        } else if (DstMO.isMBB()) {
          DstMO.setMBB(MBBMap[DstMO.getMBB()]);
        } else if (DstMO.isRegMask()) {
          if (!ConstRegisterMasks.count(DstMO.getRegMask())) {
            uint32_t *DstMask = ToBeInstrumentedMF.allocateRegMask();
            std::memcpy(DstMask, SrcMO.getRegMask(),
                        sizeof(*DstMask) * llvm::MachineOperand::getRegMaskSize(
                                               TRI->getNumRegs()));
            DstMO.setRegMask(DstMask);
          }
        } else if (DstMO.isGlobal()) {
          auto GVEntry = VMap.find(DstMO.getGlobal());
          if (GVEntry == VMap.end()) {
            ToBeInstrumentedMF.getFunction().getContext().emitError(
                llvm::formatv("TargetModulePatcherPass: GV {0} referenced "
                              "by payload MI was not cloned into the "
                              "target module",
                              DstMO.getGlobal()->getName()));
          } else {
            auto *DestGV = llvm::cast<llvm::GlobalValue>(GVEntry->second);
            DstMO.ChangeToGA(DestGV, DstMO.getOffset(), DstMO.getTargetFlags());
          }
        }
        DstMI->addOperand(DstMO);
      }
    }
  }
}

/// Build a VMap that maps each IModule global object (GVs, non-payload
/// non-hook Functions) to a fresh handle created in the target module,
/// and for every IModule function that owns a MachineFunction *and* is
/// neither a payload nor a hook, deep-clone its MIR into the target MMI
/// keyed on the new target-module Function. Returns the VMap so the
/// per-payload inliner can resolve cross-module operands. This subsumes
/// the previous Linker-based approach: we need explicit MF cloning for
/// the non-payload functions, not just IR handles.
llvm::Error cloneIModuleIntoTarget(llvm::Module &IModule,
                                   llvm::Module &TargetModule,
                                   const llvm::MachineModuleInfo &IMMI,
                                   llvm::FunctionAnalysisManager &TargetFAM,
                                   llvm::ValueToValueMapTy &VMap) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] cloneIModuleIntoTarget: "
             << "IModule has "
             << std::distance(IModule.global_begin(), IModule.global_end())
             << " global(s), " << IModule.size() << " function(s)\n");
  unsigned ClonedGVs = 0;
  for (auto &GV : IModule.globals()) {
    ++ClonedGVs;
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   clone GV '"
                               << GV.getName() << "'\n");
    auto *NewGV = new llvm::GlobalVariable(
        TargetModule, GV.getValueType(), GV.isConstant(), GV.getLinkage(),
        /*Initializer=*/nullptr, GV.getName(), /*InsertBefore=*/nullptr,
        GV.getThreadLocalMode(), GV.getType()->getAddressSpace());
    NewGV->copyAttributesFrom(&GV);
    VMap[&GV] = NewGV;
  }

  // Pass 1 — create every (non-payload) Function's IR handle in the
  // target module and stash it in VMap. The MF-clone step in pass 2
  // walks MIs whose `Global` operands may reference any other IModule
  // function (e.g., a helper `__ockl_fprintf_stderr_begin` is called
  // from another helper); resolving those operands through VMap
  // requires every function's handle to already be present, regardless
  // of iteration order. Without this two-pass split, cloning function
  // A's MF when A's body calls function B (and B happens later in
  // IModule iteration order) errors with "Failed to find the
  // corresponding value for B in the cloned value map".
  unsigned ClonedFuncHandles = 0;
  unsigned SkippedPayloads = 0;
  for (auto &F : IModule.functions()) {
    // Hooks and payloads stay in the IModule. Hooks are not cloned at
    // all (they're a tool-author concept that never makes it to the
    // target binary). Payloads are inlined separately at each AppMI
    // via Phase B step 3; cloning their handles here would create a
    // dead duplicate in the target module.
    if (F.hasFnAttribute(InjectedPayloadAttribute)) {
      ++SkippedPayloads;
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   skip payload '"
                                 << F.getName() << "'\n");
      continue;
    }
    ++ClonedFuncHandles;
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   clone Fn handle '" << F.getName()
               << "' (decl=" << F.isDeclaration() << ")\n");

    auto *NewF = llvm::Function::Create(
        llvm::cast<llvm::FunctionType>(F.getValueType()), F.getLinkage(),
        F.getAddressSpace(), F.getName(), &TargetModule);
    NewF->copyAttributesFrom(&F);
    VMap[&F] = NewF;
  }

  // Pass 2 — for each definition with a body, give the target-side
  // Function an unreachable entry block (so its IR is well-formed)
  // and deep-clone the IModule MF into the FAM-managed target MF.
  unsigned ClonedMFs = 0;
  for (auto &F : IModule.functions()) {
    if (F.hasFnAttribute(InjectedPayloadAttribute))
      continue;
    if (F.isDeclaration())
      continue;

    auto *NewF = llvm::cast<llvm::Function>(VMap[&F]);
    auto *BB = llvm::BasicBlock::Create(TargetModule.getContext(), "", NewF);
    new llvm::UnreachableInst(TargetModule.getContext(), BB);

    const auto *SrcMF = IMMI.getMachineFunction(F);
    if (SrcMF == nullptr) {
      LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   no MF for '"
                                 << F.getName() << "', skip MF clone\n");
      continue; // Definition without a lifted MF — rare; skip MF clone.
    }

    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   clone MF '" << F.getName()
               << "' (" << SrcMF->size() << " MBB(s))\n");
    // Ask the target FAM to construct an empty MF for the new
    // Function (MachineFunctionAnalysis::run does this lazily), then
    // populate it via cloneMFInto. The MF now lives in the target
    // FAM cache and is visible to every subsequent
    // FAM.getResult<MachineFunctionAnalysis>(NewF) consumer.
    llvm::MachineFunction &DstMF =
        TargetFAM.getResult<llvm::MachineFunctionAnalysis>(*NewF).getMF();
    if (auto Err = cloneMFInto(*SrcMF, VMap, DstMF))
      return Err;
    ++ClonedMFs;
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] cloneIModuleIntoTarget "
                "done: "
             << ClonedGVs << " GV(s), " << ClonedFuncHandles
             << " Fn handle(s) (" << SkippedPayloads << " payload(s) skipped), "
             << ClonedMFs << " MF clone(s)\n");
  return llvm::Error::success();
}

/// Strip the `amdgpu-num-vgpr` and `amdgpu-num-sgpr` attributes from every
/// function in \p TargetModule. These were set by CodeDiscoveryPass based
/// on the original (pre-instrumentation) code-object's register usage,
/// but are no longer accurate after our instrumentation extends the
/// register footprint. Leaving them present would mislead the LLVM
/// AMDGPU backend's register-pressure heuristics on the next codegen
/// pass over the target.
void stripStaleNumRegsAttrs(llvm::Module &TargetModule) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] stripStaleNumRegsAttrs "
                "over "
             << TargetModule.size() << " function(s)\n");
  for (llvm::Function &F : TargetModule.functions()) {
    F.removeFnAttr("amdgpu-num-vgpr");
    F.removeFnAttr("amdgpu-num-sgpr");
  }
}

/// Walk every non-indirect branch in \p MF, sum byte sizes via
/// \c TII.getInstSizeInBytes to estimate per-MBB layout offsets, and
/// report any branch whose target lies beyond \c s_branch's signed
/// 16-bit-word range (±131,068 bytes). When out-of-range branches are
/// found we currently emit a diagnostic and return their count — the
/// actual relax-to-\c s_setpc_b64 rewrite (with SGPR scavenging from
/// per-MBB live-ins and SVA-lane spill fallback per
/// \c StateValueArraySpecs::findLowestFreeLanes) is the next phase.
///
/// Why we need this even though stock LLVM \c BranchRelaxationPass
/// exists: \c CodeGenerator::printAssemblyFile invokes
/// \c TM.addAsmPrinter directly, skipping the standard post-RA
/// machine-pass chain. So no LLVM pass runs between TargetModulePatcher
/// and AsmPrinter; out-of-range branches would be silently emitted
/// with truncated displacements.
/// One entry per direct branch whose target lies beyond \c s_branch's
/// signed 16-bit-word range. Populated by \c detectOutOfRangeBranches
/// and consumed by the eventual \c s_setpc_b64 rewriter (task #26
/// rewrite phase, not yet implemented).
struct OutOfRangeBranchRecord {
  const llvm::MachineFunction *MF;
  const llvm::MachineInstr *Branch;
  const llvm::MachineBasicBlock *Target;
  int64_t BranchOffset;
  int64_t Delta;
};

unsigned
detectOutOfRangeBranches(const llvm::MachineFunction &MF,
                         llvm::SmallVectorImpl<OutOfRangeBranchRecord> &Out) {
  static constexpr int64_t kSBranchMaxBytes = (1LL << 18) - 4;
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  llvm::DenseMap<const llvm::MachineBasicBlock *, int64_t> MBBOffset;
  int64_t Cursor = 0;
  for (const auto &MBB : MF) {
    MBBOffset[&MBB] = Cursor;
    for (const auto &MI : MBB)
      Cursor += TII.getInstSizeInBytes(MI);
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass]   detectOutOfRangeBranches MF='"
             << MF.getName() << "' totalSize=" << Cursor << " bytes\n");
  unsigned NumOutOfRange = 0;
  Cursor = 0;
  for (const auto &MBB : MF) {
    int64_t MIOffset = Cursor;
    for (const auto &MI : MBB) {
      if (MI.isBranch() && !MI.isIndirectBranch()) {
        if (auto *TargetMBB = TII.getBranchDestBlock(MI)) {
          int64_t TgtOff = MBBOffset[TargetMBB];
          int64_t Delta = TgtOff - (MIOffset + TII.getInstSizeInBytes(MI));
          if (Delta > kSBranchMaxBytes || Delta < -kSBranchMaxBytes) {
            ++NumOutOfRange;
            LLVM_DEBUG(luthier::dbgs()
                       << "[TargetModulePatcherPass]     out-of-range branch "
                          "at 0x"
                       << llvm::Twine::utohexstr(MIOffset) << " -> "
                       << llvm::printMBBReference(*TargetMBB)
                       << " delta=" << Delta << "B\n");
            Out.push_back({&MF, &MI, TargetMBB, MIOffset, Delta});
          }
        }
      }
      MIOffset += TII.getInstSizeInBytes(MI);
    }
    Cursor = MIOffset;
  }
  return NumOutOfRange;
}

} // namespace

// TODO: Restore the kernel descriptor's \c kernarg_preload field for kernels
// that originally used the kernarg-preload feature.
//
// CodeDiscoveryPass captures the original packed field as the two attributes
//   - \c amdgpu.kd.kernarg_preload_length  (bits 0-6, SGPR-dword count)
//   - \c amdgpu.kd.kernarg_preload_offset  (bits 7-15, dword offset)
// on every kernel \c Function whose subtarget has \c hasKernargPreload().
//
// The AMDGPU AsmPrinter (see \c AMDGPUAsmPrinter.cpp ~ line 667) only writes
// LENGTH (derived from the recomputed \c getNumKernargPreloadedSGPRs()) and
// hard-zeroes OFFSET. After instrumentation the recomputed LENGTH may also
// drift from the original if the lifted kernel signature differs from the
// source binary's. Both halves therefore need a post-AsmPrinter patch:
//   1. Locate the kernel descriptor in the emitted relocatable (via the
//      \c <kernel>.kd symbol).
//   2. Read the two \c amdgpu.kd.kernarg_preload_* attributes off the
//      lifted Function.
//   3. Compose \c kernarg_preload = (offset << 7) | length and write to
//      \c KERNARG_PRELOAD_OFFSET (offset 58 in the descriptor).
//
// Skip patching when either attribute is absent (means the subtarget didn't
// support kernarg-preload at lift time and re-emission won't touch the slot).

bool TargetModulePatcherPass::runOnModule(llvm::Module &IModule) {
  LLVM_DEBUG(luthier::dbgs() << "=== " << getPassName() << " ===\n");
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] IModule='" << IModule.getName()
             << "' (" << IModule.size() << " function(s))\n");

  llvm::LLVMContext &Ctx = IModule.getContext();
  llvm::ModuleAnalysisManager &IMAM =
      getAnalysis<IModuleMAMWrapperPass>().getMAM();
  // The legacy MMI we get from MIRLegacyPM holds *IModule* MFs (payloads,
  // hooks, helpers) — NOT target kernels. Target kernels live in the
  // new-PM target FAM cache (populated by CodeDiscoveryPass via
  // MachineFunctionAnalysis). The two MMI universes are completely
  // disjoint, so we must source target MFs via the FAM and IModule MFs
  // via this MMI.
  llvm::MachineModuleInfo &IMMI =
      getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI();

  auto &TargetModAndMAM =
      IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);
  llvm::Module &TargetModule = TargetModAndMAM.getTargetAppModule();
  llvm::ModuleAnalysisManager &TargetMAM = TargetModAndMAM.getTargetAppMAM();
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] TargetModule='"
                             << TargetModule.getName() << "' ("
                             << TargetModule.size() << " function(s))\n");

  // Target-side MF access goes through the new-PM FAM. Each call to
  // FAM.getResult<MachineFunctionAnalysis>(F) returns the MF
  // CodeDiscoveryPass lifted (or constructs an empty one if missing,
  // which only happens for Functions we created post-CodeDiscovery).
  llvm::FunctionAnalysisManager &TargetFAM =
      TargetMAM
          .getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();
  auto getTargetMF = [&](llvm::Function &F) -> llvm::MachineFunction & {
    return TargetFAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();
  };

  const SVStorageAndLoadLocations &SVLocations =
      getAnalysis<LRStateValueStorageAndLoadLocationsAnalysis>().getResult();

  const llvm::TargetMachine &TM = IMMI.getTarget();
  std::unique_ptr<StateValueArraySpecs> SVASpecs =
      StateValueArraySpecs::getSVASpecs(IModule, TM);
  if (!SVASpecs) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "TargetModulePatcherPass: StateValueArraySpecs::getSVASpecs "
        "returned null; expected the IModule to carry SVA-spec "
        "metadata by this stage")));
    return false;
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] SVASpecs resolved\n");

  const FunctionPreambleDescriptor &FPD =
      TargetMAM.getResult<FunctionPreambleDescriptorAnalysis>(TargetModule);

  llvm::MachineFunctionAnalysisManager &TargetMFAM =
      TargetMAM
          .getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(
              TargetModule)
          .getManager();

  // ============= Phase A: SVA Setup & Storage Code Emission ===============
  //
  // For each target MF, walk SVStorageAndLoadLocations'
  // StateValueStorageIntervals and emit the SVS switch code at each
  // interval boundary. The initial-entry-point kernel additionally needs
  // the kernarg-preload setup sequence at its entry; that's deferred to
  // the next iteration alongside the FPD-keyed iteration. For now we
  // emit only the per-MBB switches — which is the part that exercises
  // the cross-MBB storage relocation logic.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase A.1: "
                                "emitSVSSwitchesForMF ===\n");
  bool Changed = false;
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;
    llvm::MachineFunction &MF = getTargetMF(F);
    emitSVSSwitchesForMF(MF, SVLocations,
                         TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(MF));
    Changed = true;
  }

  // Phase A.2: initial-entry-kernel SVA preload setup (scratch + kernarg
  // spills into SVA lanes) at every kernel entry that uses the SVA.
  // emitInitialEntryKernelSetup also needs target-side MF access; it
  // takes a MachineModuleInfo today but operates by looking up MFs for
  // FPD.Kernels. We pass IMMI as a placeholder — the function ignores
  // it where target lookups are needed (a follow-on cleanup will switch
  // it to FAM too once we audit the helper).
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase A.2: "
                                "emitInitialEntryKernelSetup ===\n");
  if (auto Err = emitInitialEntryKernelSetup(IMMI, TargetModule, FPD,
                                             SVLocations, *SVASpecs)) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return Changed;
  }

  // ============= Phase B: Target Patching ===============================
  //
  // Step 1: Clone IModule globals + non-payload non-hook Functions into
  // the target module. The returned VMap is used by the inliner so
  // cross-module operands resolve. Helper MFs are constructed in the
  // target FAM (via MachineFunctionAnalysis::run) and populated with
  // cloneMFInto, so they're visible to subsequent target-side loops
  // and to NewPMAsmPrinter's per-Function MF lookup.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase B.1: "
                                "cloneIModuleIntoTarget ===\n");
  llvm::ValueToValueMapTy VMap;
  if (auto Err = cloneIModuleIntoTarget(IModule, TargetModule, IMMI, TargetFAM,
                                        VMap)) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return Changed;
  }

  // Step 2: Strip stale num-{vgpr,sgpr} attrs — CodeDiscoveryPass set
  // these from the original code object, and they're wrong now that
  // we've inlined payloads + cloned helpers.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase B.2: "
                                "stripStaleNumRegsAttrs ===\n");
  stripStaleNumRegsAttrs(TargetModule);

  // Step 3: Inline every injected payload at its corresponding target
  // MachineInstr. Following the design directive, payload MIs (minus
  // their return terminators) are deep-cloned into the host MF right
  // before the target MI; the payload's return-edge terminators are
  // replaced by fall-through / branch into the host continuation MBB,
  // matching the prior PatchLiftedRepresentationPass inline path.
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] === Phase B.3: inline "
                "injected payloads ===\n");
  const InjectedPayloadAndInstPoint &IPIP =
      IMAM.getResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  // `MachineBasicBlock::splitAt` (called by `inlineInjectedPayload` when
  // the payload entry isn't the first MI of its MBB) needs `TracksLiveness`
  // + populated MBB liveins on the target MF. CodeDiscoveryPass-lifted MFs
  // don't carry liveness info by default. Set this up once per unique
  // target MF before any inlining happens. Phase B.4's per-kernel relaxer
  // recomputes again, but `fullyRecomputeLiveIns` is idempotent (the
  // fixed-point converges in 0 extra iterations when the state is
  // already consistent).
  llvm::DenseSet<llvm::MachineFunction *> TargetMFsTouchedByPayloads;
  for (const auto &[InjectedPayloadFunc, InsertionPointMI] :
       IPIP.payload_mi()) {
    TargetMFsTouchedByPayloads.insert(InsertionPointMI->getMF());
  }
  for (llvm::MachineFunction *MF : TargetMFsTouchedByPayloads) {
    MF->getProperties().setTracksLiveness();
    llvm::SmallVector<llvm::MachineBasicBlock *, 16> AllMBBs;
    for (llvm::MachineBasicBlock &MBB : *MF)
      AllMBBs.push_back(&MBB);
    llvm::fullyRecomputeLiveIns(AllMBBs);
  }

  unsigned PayloadCount = 0;
  for (const auto &[InjectedPayloadFunc, InsertionPointMI] :
       IPIP.payload_mi()) {
    ++PayloadCount;
    llvm::DenseMap<const llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>
        MBBMap;
    const auto *InjectedPayloadMF =
        IMMI.getMachineFunction(*InjectedPayloadFunc);
    if (!InjectedPayloadMF) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "TargetModulePatcherPass: payload function '{0}' has no "
          "MachineFunction in the IModule MMI",
          InjectedPayloadFunc->getName()))));
      return Changed;
    }
    patchFrameInfo(*InjectedPayloadMF,
                   *InsertionPointMI->getParent()->getParent());
    inlineInjectedPayload(*InjectedPayloadMF, *InsertionPointMI, MBBMap, VMap);
  }
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]   Phase B.3 inlined "
                             << PayloadCount << " payload(s)\n");

  // Post-inline branch displacement check. CodeGenerator::printAssemblyFile
  // invokes AsmPrinter directly with no intermediate machine-pass chain, so
  // LLVM's BranchRelaxationPass never runs on the target module — out-of-range
  // branches would otherwise be silently truncated by the encoder. For now we
  // count + diagnose; the actual relaxation (s_setpc_b64 + SGPR scavenging
  // via per-MBB live-ins, falling back to spilling two app SGPRs into the
  // lowest free SVA lanes per StateValueArraySpecs::findLowestFreeLanes) is
  // the next phase.
  // Phase B step 4: invoke Luthier's forked BranchRelaxation on every
  // patched target MF. CodeGenerator::printAssemblyFile takes the MMI
  // straight to AsmPrinter (no addPassesToEmitFile), so far-jumps need
  // explicit relaxation here or they'd silently emit with truncated
  // displacements. We use a sibling-class fork (LuthierBranchRelaxation
  // + LuthierRegScavenger, not subclasses — stock methods are non-
  // virtual) so the scavenger can be told to reserve the SVA storage
  // register and, eventually, route emergency spills to SVA lanes.
  //
  // ReservedForSVA gathers every reg that ever appears in any SVS
  // storage segment for this MF — that's the set the scavenger must
  // not pick when allocating the long-branch PC pair.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase B.4: "
                                "branch relaxation per kernel ===\n");
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;
    // Skip non-kernel target functions. The helpers `cloneIModuleIntoTarget`
    // copies in (utility functions referenced from hooks — atomics, lane
    // ops, ockl printf helpers, etc.) carry terminator shapes
    // (`S_SETPC_B64` returns, indirect calls) that LLVM AMDGPU's
    // `analyzeBranch` doesn't fully understand. Running the long-branch
    // relaxer on them trips `MachineOperand::getMBB()`'s `isMBB()`
    // assert inside `analyzeBranch`. Lit fixtures don't hit this
    // because they only carry payload IModules; the runtime path's
    // embedded bitcode brings in real HIP helpers. Long branches only
    // need correctness on the kernels themselves anyway — they're the
    // ones whose post-instrumentation expansion grows the code far
    // enough to need relaxing.
    if (F.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL) {
      LLVM_DEBUG(luthier::dbgs()
                 << "[TargetModulePatcherPass]   skip non-kernel '"
                 << F.getName() << "' for relaxer\n");
      continue;
    }
    llvm::MachineFunction *MF = &getTargetMF(F);
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   relaxer for kernel '"
               << F.getName() << "' (" << MF->size() << " MBB(s))\n");
    llvm::DenseSet<llvm::MCPhysReg> ReservedForSVA;
    for (const auto &MBB : *MF) {
      for (const auto &Seg : SVLocations.getStorageIntervals(MBB)) {
        llvm::SmallVector<llvm::MCRegister, 4> Regs;
        Seg.getSVS().getAllStorageRegisters(Regs);
        for (llvm::MCRegister R : Regs)
          ReservedForSVA.insert(R.id());
      }
    }
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "ReservedForSVA size="
                               << ReservedForSVA.size() << "\n");

    // Pre-pin a `LongBranchReservedReg` on this MF if any globally
    // dead SReg_64 exists. `SIInstrInfo::insertIndirectBranch` (and
    // our LuthierBranchRelaxation's emitLuthierLongBranch) consult
    // `SIMachineFunctionInfo::getLongBranchReservedReg()` first; when
    // non-zero it bypasses the scavenger entirely. With a pre-pinned
    // reg every trampoline reuses the same pair, no RestoreBB ever
    // gets populated, MBB growth per relaxed branch is minimal, and
    // the relaxer's outer fixed-point loop converges in O(1)
    // iterations regardless of how tight `--amdgpu-s-branch-bits` is
    // (no inserted spill code → no neighboring branches pushed out
    // of range by spill+restore MBBs). Reg selection: any SReg_64 in
    // the allocation order not in ReservedForSVA, not reserved by
    // MRI, and not appearing in any MBB's liveins or MI uses/defs
    // (i.e., globally dead).
    {
      auto &MRI = MF->getRegInfo();
      const auto *TRI = MF->getSubtarget().getRegisterInfo();
      auto *MFI = MF->getInfo<llvm::SIMachineFunctionInfo>();

      // Liveins under the lifted MIR weren't computed by CodeDiscovery
      // — we need them populated before we can read MBB.liveins().
      // LuthierBranchRelaxation will recompute again at its run() start;
      // that's idempotent (fixed-point converges in 0 extra iterations
      // since the state's already consistent).
      MF->getProperties().setTracksLiveness();
      llvm::SmallVector<llvm::MachineBasicBlock *, 16> AllMBBs;
      for (llvm::MachineBasicBlock &MBB : *MF)
        AllMBBs.push_back(&MBB);
      llvm::fullyRecomputeLiveIns(AllMBBs);

      // Build the set of regs that appear anywhere in the MF (liveins
      // post-fullyRecomputeLiveIns + explicit MI operand uses/defs).
      llvm::DenseSet<llvm::MCPhysReg> UsedAnywhere;
      for (const auto &MBB : *MF) {
        for (const auto &LI : MBB.liveins())
          UsedAnywhere.insert(LI.PhysReg);
        for (const auto &MI : MBB) {
          for (const auto &MO : MI.operands()) {
            if (MO.isReg() && MO.getReg().isPhysical())
              UsedAnywhere.insert(MO.getReg().id());
          }
        }
      }

      // Find the first SReg_64 candidate where neither of its 32-bit
      // sub-regs appears in UsedAnywhere and the pair itself isn't
      // reserved or in ReservedForSVA.
      llvm::Register Picked;
      for (llvm::MCRegister Reg :
           llvm::AMDGPU::SReg_64RegClass.getRegisters()) {
        if (!MRI.isAllocatable(Reg) || MRI.isReserved(Reg))
          continue;
        if (ReservedForSVA.contains(Reg.id()))
          continue;
        llvm::MCRegister Sub0 = TRI->getSubReg(Reg, llvm::AMDGPU::sub0);
        llvm::MCRegister Sub1 = TRI->getSubReg(Reg, llvm::AMDGPU::sub1);
        if (!Sub0 || !Sub1)
          continue;
        if (UsedAnywhere.contains(Reg.id()) ||
            UsedAnywhere.contains(Sub0.id()) ||
            UsedAnywhere.contains(Sub1.id()))
          continue;
        Picked = Reg;
        break;
      }
      if (Picked) {
        LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                      "LongBranchReservedReg picked="
                                   << llvm::printReg(Picked, TRI) << "\n");
        MFI->setLongBranchReservedReg(Picked);
      } else {
        LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                      "no globally-dead SReg_64; relaxer must "
                                      "scavenge via SVA-lane spill\n");
      }
    }

    // SVA-lane spill sink: when the long-branch scavenger can't find a
    // globally-free SReg_64, fall through to here. We spill the chosen
    // pair into the two lowest free SVA lanes via V_WRITELANE_B32 at
    // SpillBefore and reload via V_READLANE_B32 at ReloadBefore. The
    // SVA storage VGPR is read from the MF's entry MBB's first segment;
    // this is correct for the common case where SVA storage stays in
    // one VGPR for the whole function. For schemes that switch storage
    // mid-function, the segment covering the actual spill point would
    // be more precise — but the trampoline MBBs created by the relaxer
    // aren't in `SVLocations` (post-analysis MBBs), so entry-segment
    // is the conservative pick that always resolves.
    llvm::MachineFunction *MFPtr = MF;
    const StateValueArraySpecs *SpecsPtr = SVASpecs.get();
    const SVStorageAndLoadLocations *SVLocPtr = &SVLocations;
    auto SpillSink = [MFPtr, SpecsPtr,
                      SVLocPtr](llvm::MachineBasicBlock &SpillMBB,
                                llvm::MachineBasicBlock::iterator SpillBefore,
                                llvm::MachineBasicBlock &ReloadMBB,
                                llvm::MachineBasicBlock::iterator ReloadBefore,
                                llvm::MCRegister Reg,
                                const llvm::TargetRegisterClass &RC) -> bool {
      // Only the SReg_64 case is supported (the only class the
      // long-branch scavenger ever asks for).
      if (&RC != &llvm::AMDGPU::SReg_64RegClass)
        return false;

      // Resolve SVA VGPR from the MF's entry MBB's first segment.
      if (MFPtr->empty())
        return false;
      auto EntrySegs = SVLocPtr->getStorageIntervals(MFPtr->front());
      if (EntrySegs.empty())
        return false;
      llvm::MCRegister SVAVGPR =
          EntrySegs.front().getSVS().getStateValueStorageReg();
      if (!SVAVGPR)
        return false;

      const auto &ST = MFPtr->getSubtarget<llvm::GCNSubtarget>();
      unsigned WaveSize = ST.getWavefrontSize();
      auto FreeLanes = SpecsPtr->findLowestFreeLanes(2, WaveSize);
      if (FreeLanes.size() < 2)
        return false;

      const auto *TII = ST.getInstrInfo();
      const auto *TRI = ST.getRegisterInfo();
      llvm::MCRegister Sub0 = TRI->getSubReg(Reg, llvm::AMDGPU::sub0);
      llvm::MCRegister Sub1 = TRI->getSubReg(Reg, llvm::AMDGPU::sub1);
      if (!Sub0 || !Sub1)
        return false;

      // SVA VGPR is implicitly live everywhere the SVA is in
      // scope (set up by the kernel-entry preload) but isn't in
      // any MBB's live-in set by default. Declare it on SpillMBB
      // explicitly — the relaxer copies BranchBB's liveins from
      // its predecessor's successors and doesn't otherwise see
      // the SVA reg, so MachineVerifier would complain about the
      // V_WRITELANE's tied-def use. ReloadMBB (RestoreBB) gets
      // its liveins recomputed by the relaxer via
      // computeAndAddLiveIns immediately after we return; that
      // path requires liveins to be empty as a precondition AND
      // walks V_READLANE backward (so SVAVGPR ends up in the
      // computed set anyway). Skip the pre-emit addLiveIn for
      // RestoreBB.
      if (!SpillMBB.isLiveIn(SVAVGPR))
        SpillMBB.addLiveIn(SVAVGPR);

      // Spill: write each sub-reg into its reserved SVA lane right
      // before the long-branch arithmetic clobbers them. Note that
      // V_WRITELANE_B32 has a tied-def operand for the destination
      // VGPR so the read-modify-write of unrelated lanes is encoded
      // explicitly.
      llvm::DebugLoc DL;
      llvm::BuildMI(SpillMBB, SpillBefore, DL,
                    TII->get(llvm::AMDGPU::V_WRITELANE_B32), SVAVGPR)
          .addReg(Sub0)
          .addImm(FreeLanes[0])
          .addReg(SVAVGPR);
      llvm::BuildMI(SpillMBB, SpillBefore, DL,
                    TII->get(llvm::AMDGPU::V_WRITELANE_B32), SVAVGPR)
          .addReg(Sub1)
          .addImm(FreeLanes[1])
          .addReg(SVAVGPR);

      // Reload: pull each lane back into the sub-reg in the
      // RestoreBB that runs after the long jump lands.
      llvm::BuildMI(ReloadMBB, ReloadBefore, DL,
                    TII->get(llvm::AMDGPU::V_READLANE_B32), Sub0)
          .addReg(SVAVGPR)
          .addImm(FreeLanes[0]);
      llvm::BuildMI(ReloadMBB, ReloadBefore, DL,
                    TII->get(llvm::AMDGPU::V_READLANE_B32), Sub1)
          .addReg(SVAVGPR)
          .addImm(FreeLanes[1]);
      return true;
    };

    LuthierBranchRelaxation BR(std::move(ReservedForSVA), std::move(SpillSink));
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     running "
                                  "LuthierBranchRelaxation on '"
                               << F.getName() << "'\n");
    BR.run(*MF);
    LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass]     "
                                  "LuthierBranchRelaxation done for '"
                               << F.getName() << "'\n");
  }

  // Sanity-check: re-run the detector and hard-error if anything
  // remains out of range. Either the stock relaxer didn't run (e.g.,
  // AMDGPU-specific corner case) or our offset accounting disagrees
  // with the asm printer's.
  LLVM_DEBUG(luthier::dbgs() << "[TargetModulePatcherPass] === Phase B.5: "
                                "post-relax out-of-range sanity check ===\n");
  llvm::SmallVector<OutOfRangeBranchRecord, 4> OutOfRange;
  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;
    // Same kernel-only filter as the relaxer loop — `analyzeBranch`
    // on non-kernel cloned helpers crashes on `isMBB()`.
    if (F.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    detectOutOfRangeBranches(getTargetMF(F), OutOfRange);
  }
  if (!OutOfRange.empty()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TargetModulePatcherPass]   " << OutOfRange.size()
               << " branch(es) still over-range; failing\n");
    std::string Detail;
    llvm::raw_string_ostream OS(Detail);
    OS << "TargetModulePatcherPass: " << OutOfRange.size()
       << " branch(es) remain over-range after BranchRelaxationPass; "
       << "branches:";
    for (const auto &R : OutOfRange) {
      OS << "\n  - " << R.MF->getName() << " offset 0x";
      OS.write_hex(static_cast<uint64_t>(R.BranchOffset));
      OS << " → " << R.Target->getName() << " (delta " << R.Delta << " bytes)";
    }
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(Detail)));
    return Changed;
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TargetModulePatcherPass] runOnModule complete; "
                "target module is patched and verified\n");
  return true;
}

} // namespace luthier
