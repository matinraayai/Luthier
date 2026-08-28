//===-- IntrinsicMIRLoweringPass.cpp --------------------------------------===//
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
/// Implements the Intrinsic MIR Lowering Pass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/IntrinsicMIRLoweringPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/Intrinsic/ReadReg.h"
#include "luthier/Intrinsic/ReadSVA.h"
#include "luthier/Intrinsic/WriteReg.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorsAnalysis.h"
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineSSAUpdater.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetFrameLowering.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier {

namespace {

/// Decode the trailing register-operand args of an inline-asm MachineInstr
/// into (Flag, MachineOperand) pairs. AsmString operands and other non-flag
/// metadata operands are ignored.
llvm::SmallVector<
    std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
getInlineAsmArgs(const llvm::MachineInstr &MI) {
  llvm::SmallVector<
      std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
      Out;
  for (unsigned I = llvm::InlineAsm::MIOp_FirstOperand,
                NumOps = MI.getNumOperands();
       I < NumOps; ++I) {
    const llvm::MachineOperand &MO = MI.getOperand(I);
    if (!MO.isImm())
      continue;
    const llvm::InlineAsm::Flag F(MO.getImm());
    Out.emplace_back(F, &MI.getOperand(I + 1));
    I += F.getNumOperandRegisters();
  }
  return Out;
}

/// Returns the SGPR register class for \p NumLanes 32-bit sub-registers.
const llvm::TargetRegisterClass *getSGPRRegClassForLanes(unsigned NumLanes) {
  switch (NumLanes) {
  case 1:
    return &llvm::AMDGPU::SGPR_32RegClass;
  case 2:
    return &llvm::AMDGPU::SGPR_64RegClass;
  case 4:
    return &llvm::AMDGPU::SGPR_128RegClass;
  default:
    return nullptr;
  }
}

} // namespace

bool IntrinsicMIRLoweringPass::processMachineFunction(
    llvm::MachineFunction &MF, bool IsInjectedPayload,
    const IntrinsicsProcessorsAnalysis::Result &IntrinsicsProcessors,
    const StateValueArraySpecs &SVASpecs, llvm::MCRegister SVAVGPR) {
  llvm::LLVMContext &Ctx = MF.getFunction().getContext();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const llvm::TargetInstrInfo *TII = ST.getInstrInfo();
  const llvm::TargetRegisterInfo *TRI = ST.getRegisterInfo();
  llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  auto *SIMFI = MF.getInfo<llvm::SIMachineFunctionInfo>();

  /// Per-MF cache: SA -> wide vreg holding the SA's value. Returned vreg may
  /// be a single SGPR_32 (1-lane SAs) or a REG_SEQUENCE'd wide SGPR (multi
  /// lane). One cache entry per SA covers every intrinsic call site in this
  /// MF.
  llvm::DenseMap<ScalarValueArgument, llvm::Register> SAResultCache;

  /// Fixed insertion point for SVA-related instructions in the entry
  /// block. Initially points to the first non-PHI/label/debug MI (often
  /// a \c luthier::readSVA \c INLINEASM). After the SVA VGPR placeholder
  /// IMPLICIT_DEF gets emitted below, this iterator is re-anchored to
  /// \c std::next(SVAImplDef) so subsequent BuildMI insertions land in a
  /// stable position — the intrinsic-lowering loop erases INLINEASM MIs
  /// as it processes them, and holding an iterator to an erased MI
  /// crashes the next BuildMI (e.g. on multi-readSVA payloads).
  llvm::MachineBasicBlock::iterator SVAInsertPt =
      MF.front().SkipPHIsLabelsAndDebug(MF.front().begin());

  /// Per phys-reg-channel SSAUpdater. The updater is created lazily on the
  /// first read/write of that channel by an intrinsic.
  ///
  /// Block iteration order is plain MF order. Single-pass query into
  /// MachineSSAUpdater is unsafe in the presence of back edges: a loop
  /// header is processed before its back-edge predecessor, so any
  /// GetValueInMiddleOfBlock query at the header would synthesize an
  /// IMPLICIT_DEF for the not-yet-registered back-edge def. To fix this we
  /// run a two-phase scheme:
  ///   * Phase 1 (per-MI lowering, current loop): every phys-reg read in a
  ///     non-entry MBB and every return-block restore emits an
  ///     IMPLICIT_DEF placeholder and records a PendingResolution. Writes
  ///     are registered with AddAvailableValue as soon as they're lowered.
  ///   * Phase 2 (after the MBB loop): for every PendingResolution, query
  ///     GetValueInMiddleOfBlock / GetValueAtEndOfBlock on the now-fully-
  ///     populated SSAUpdater, MRI.replaceRegWith the placeholder, and
  ///     erase the IMPLICIT_DEF.
  llvm::DenseMap<llvm::MCRegister, std::unique_ptr<llvm::MachineSSAUpdater>>
      PhysRegValueSSAUpdaters;

  /// Per SVA-lane SSAUpdater for frame-reg writes that must reach the SVA
  /// VGPR. Populated as each writeReg on SP/FP/PSB/FLAT_SCR is lowered,
  /// then queried once per return block after the MBB walk finishes so
  /// the V_WRITELANE_B32 chain lands at the payload's exit rather than
  /// modifying the SVA VGPR mid-body.
  llvm::DenseMap<uint8_t, std::unique_ptr<llvm::MachineSSAUpdater>>
      FrameSVAWriteUpdaters;

  /// Records a placeholder vreg that needs to be replaced in Phase 2 once
  /// the SSAUpdater for \c Channel has all its AvailableValues registered.
  struct PendingPhysRegResolution {
    llvm::MachineBasicBlock *MBB;
    llvm::MCRegister Channel;
    llvm::Register Placeholder;
    enum Kind { Read, ReturnRestore } K;
  };
  llvm::SmallVector<PendingPhysRegResolution, 8> PendingResolutions;

  bool Changed = false;

  // Reserve a virtual VGPR_32 placeholder for the SVA VGPR that the
  // register allocator resolves to \c SVAVGPR via a simple register
  // hint plus the WWM_REG virtual-reg flag. Only set up when there is
  // a valid \c SVAVGPR from the payload's LoadPlan; MFs without one
  // (e.g. non-injected-payload utility functions, or payloads whose
  // target module doesn't yet carry a PATCHPOINT binding) run through
  // the non-SVA \c readReg / \c writeReg SSA-updater path only. The
  // frame-lane read/write and readSVA branches below explicitly error
  // out when they need the placeholder but \c SVAVGPR is invalid.
  llvm::Register SVAVGPRPlaceholder;
  llvm::MachineInstr *SVAImplDef = nullptr;
  auto svaInsertPt = [&]() -> llvm::MachineBasicBlock::iterator {
    // Recompute freshly each time — the initial \c SVAInsertPt often
    // pointed at a \c luthier::readSVA \c INLINEASM the intrinsic
    // lowering loop later erases, which invalidates any captured
    // iterator. Anchoring on \c SVAImplDef (never erased) and using
    // \c std::next keeps SVA-related MIs right after the placeholder
    // def, in dominator order, with a stable in-MBB anchor.
    return std::next(SVAImplDef->getIterator());
  };
  if (SVAVGPR) {
    SVAVGPRPlaceholder =
        MRI.createVirtualRegister(&llvm::AMDGPU::VGPR_32RegClass);
    MRI.setSimpleHint(SVAVGPRPlaceholder, SVAVGPR);
    SIMFI->setFlag(SVAVGPRPlaceholder, llvm::AMDGPU::VirtRegFlag::WWM_REG);
    llvm::MDNode *MarkerNode = llvm::MDNode::get(
        Ctx, {llvm::MDString::get(Ctx, "luthier.sva_vgpr_placeholder")});
    SVAImplDef =
        llvm::BuildMI(MF.front(), SVAInsertPt, llvm::MIMetadata(),
                      TII->get(llvm::AMDGPU::IMPLICIT_DEF), SVAVGPRPlaceholder)
            .getInstr();
    SVAImplDef->setPCSections(MF, MarkerNode);
  }

  // Reserve the SVA lane region on the WWM LaneVGPR in lane order.
  if (SVAVGPR) {
    llvm::MachineFrameInfo &MFI = MF.getFrameInfo();
    llvm::SmallVector<uint8_t, 32> ReservedLanes;
    ReservedLanes.push_back(SVASpecs.getStackPointerRegSpillLane());
    ReservedLanes.push_back(SVASpecs.getFramePointerRegSpillLane());
    ReservedLanes.push_back(SVASpecs.getStackPointerStoreLane());
    if (auto PSBLane = SVASpecs.getRsrcBufferSpillLane())
      for (uint8_t I = 0; I < 4; ++I)
        ReservedLanes.push_back(static_cast<uint8_t>(*PSBLane + I));
    if (auto FSLane = SVASpecs.getScratchSpillLane())
      for (uint8_t I = 0; I < 2; ++I)
        ReservedLanes.push_back(static_cast<uint8_t>(*FSLane + I));
    for (auto It = SVASpecs.argument_lane_begin();
         It != SVASpecs.argument_lane_end(); ++It) {
      const unsigned NumLanes =
          StateValueArraySpecs::getArgumentLaneSize(It->first);
      for (unsigned I = 0; I < NumLanes; ++I)
        ReservedLanes.push_back(static_cast<uint8_t>(It->second + I));
    }
    llvm::sort(ReservedLanes);
    for (uint8_t Lane : ReservedLanes) {
      (void)Lane;
      int FI = MFI.CreateStackObject(/*Size=*/4, llvm::Align(4),
                                     /*isSpillSlot=*/true);
      MFI.setStackID(FI, llvm::TargetStackID::SGPRSpill);
      if (!SIMFI->allocateSGPRSpillToVGPRLane(
              MF, FI, /*SpillToPhysVGPRLane=*/false)) {
        Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "Failed to reserve SVA lane {0} on the WWM LaneVGPR in MF {1}",
            Lane, MF.getName()))));
        return Changed;
      }
    }
  }

  /// Returns the SVA lane for a frame reg read/written by an injected
  /// payload, matching \c InjectedPayloadPEIPass 's convention:
  ///
  ///   SP  (from \c SIMFI->getStackPtrOffsetReg() )     → lane 0
  ///   FP  (from \c SIMFI->getFrameOffsetReg() )        → lane 1
  ///   PSB sub-lanes (SGPR0..3 of the preloaded
  ///        \c PRIVATE_SEGMENT_BUFFER — 4 lanes)        → \c
  ///        getRsrcBufferSpillLane() ..+3
  ///   FLAT_SCR_LO / FLAT_SCR_HI                        → \c
  ///   getScratchSpillLane() and +1
  ///
  /// PSB and FLAT_SCR routing only fires when the SVA layout has the
  /// corresponding lane assignment (i.e., non-architected-FS targets;
  /// PSB additionally requires flat-scratch not to be explicitly
  /// enabled) . Returns \c std::nullopt if \p PhysReg is not one of the frame
  /// regs, or if this isn't an injected payload.
  auto getFrameSVALaneForPhysReg =
      [&](llvm::MCRegister PhysReg) -> std::optional<uint8_t> {
    if (!IsInjectedPayload)
      return std::nullopt;
    if (PhysReg == SIMFI->getStackPtrOffsetReg().asMCReg())
      return SVASpecs.getStackPointerRegSpillLane();
    if (PhysReg == SIMFI->getFrameOffsetReg().asMCReg())
      return SVASpecs.getFramePointerRegSpillLane();
    // PSB / FLAT_SCR routing is only meaningful when the target does NOT
    // have architected flat scratch
    if (!ST.hasArchitectedFlatScratch()) {
      // PSB sub-regs (4 x 32-bit sub-lanes of PRIVATE_SEGMENT_BUFFER).
      if (auto PSBLane = SVASpecs.getRsrcBufferSpillLane()) {
        if (llvm::MCRegister PSBReg = SIMFI->getPreloadedReg(
                llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER)) {
          const auto *SITRI = static_cast<const llvm::SIRegisterInfo *>(TRI);
          for (unsigned I = 0; I < 4; ++I) {
            llvm::MCRegister Sub = SITRI->getSubReg(
                PSBReg, llvm::SIRegisterInfo::getSubRegFromChannel(I));
            if (Sub == PhysReg)
              return static_cast<uint8_t>(*PSBLane + I);
          }
        }
      }
      // FLAT_SCR sub-regs (2 x 32-bit).
      if (auto FSLane = SVASpecs.getScratchSpillLane()) {
        if (PhysReg == llvm::AMDGPU::FLAT_SCR_LO)
          return *FSLane;
        if (PhysReg == llvm::AMDGPU::FLAT_SCR_HI)
          return static_cast<uint8_t>(*FSLane + 1);
      }
    }
    return std::nullopt;
  };

  /// SVA scalar argument accessor: returns a virtual register holding the
  /// value of the requested scalar argument.
  auto SVAScalarArgumentAccessor =
      [&](ScalarValueArgument SA) -> llvm::Register {
    auto CacheIt = SAResultCache.find(SA);
    if (CacheIt != SAResultCache.end())
      return CacheIt->second;

    if (!SVAVGPR) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "luthier::readSVA in MF {0}: no SVA VGPR resolved from the "
          "instrumentation point's load plan; readSVA can only be used "
          "inside an injected payload with a valid LoadPlan.",
          MF.getName()))));
      return llvm::Register();
    }

    unsigned NumLanes = StateValueArraySpecs::getArgumentLaneSize(SA);
    static const unsigned SubRegForLane[] = {
        llvm::AMDGPU::sub0, llvm::AMDGPU::sub1, llvm::AMDGPU::sub2,
        llvm::AMDGPU::sub3};

    auto LaneIt = SVASpecs.findArgumentLane(SA);
    if (LaneIt == SVASpecs.argument_lane_end()) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "luthier::readSVA in MF {0}: SA {1} has no lane assignment in "
          "StateValueArraySpecs; the analysis should have picked it up "
          "from the readSVA use.",
          MF.getName(), static_cast<int>(SA)))));
      return llvm::Register();
    }
    const uint8_t LaneBase = LaneIt->second;

    llvm::SmallVector<llvm::Register, 4> LaneRegs;
    LaneRegs.reserve(NumLanes);
    for (uint8_t Lane = 0; Lane < NumLanes; ++Lane) {
      llvm::Register LaneReg =
          MRI.createVirtualRegister(&llvm::AMDGPU::SReg_32_XM0_XEXECRegClass);
      (void)llvm::BuildMI(MF.front(), svaInsertPt(), llvm::MIMetadata(),
                          TII->get(llvm::AMDGPU::V_READLANE_B32), LaneReg)
          .addReg(SVAVGPRPlaceholder)
          .addImm(LaneBase + Lane);
      LaneRegs.push_back(LaneReg);
    }

    if (NumLanes == 1) {
      SAResultCache[SA] = LaneRegs[0];
      return LaneRegs[0];
    }

    const llvm::TargetRegisterClass *MergedRC =
        getSGPRRegClassForLanes(NumLanes);
    if (!MergedRC) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Unsupported scalar-arg lane count {0} for SA {1}",
                        NumLanes, static_cast<int>(SA)))));
      SAResultCache[SA] = LaneRegs[0];
      return LaneRegs[0];
    }
    llvm::Register MergedReg = MRI.createVirtualRegister(MergedRC);
    auto RSBuilder =
        llvm::BuildMI(MF.front(), svaInsertPt(), llvm::MIMetadata(),
                      TII->get(llvm::AMDGPU::REG_SEQUENCE), MergedReg);
    for (uint8_t Lane = 0; Lane < NumLanes; ++Lane)
      (void)RSBuilder.addReg(LaneRegs[Lane]).addImm(SubRegForLane[Lane]);
    SAResultCache[SA] = MergedReg;
    return MergedReg;
  };

  auto VirtRegBuilder = [&](const llvm::TargetRegisterClass *RC) {
    return MRI.createVirtualRegister(RC);
  };

  /// Initialize an SSAUpdater for a phys-reg root on first access: emits the
  /// entry-block COPY-from-physreg and registers the root vreg with the
  /// updater. The entry block live-in is also added.
  auto initializeSSAUpdaterForRoot = [&](llvm::MCRegister Root)
      -> std::pair<llvm::Register, llvm::MachineSSAUpdater *> {
    auto It = PhysRegValueSSAUpdaters
                  .insert({Root, std::make_unique<llvm::MachineSSAUpdater>(MF)})
                  .first;
    llvm::MachineBasicBlock &EntryBlock = MF.front();
    EntryBlock.addLiveIn(Root);
    const llvm::TargetRegisterClass *RootRegClass =
        TRI->getPhysRegBaseClass(Root);
    if (!RootRegClass) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Physical register {0} doesn't have a reg class",
                        llvm::printReg(Root, TRI)))));
      return {llvm::Register(), nullptr};
    }
    const llvm::TargetRegisterClass *RootCrossCopyRegClass =
        TRI->getCrossCopyRegClass(RootRegClass);
    if (!RootCrossCopyRegClass) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Physical register {0} doesn't have a copy reg class",
                        llvm::printReg(Root, TRI)))));
      return {llvm::Register(), nullptr};
    }
    llvm::Register RootVirtReg =
        MRI.createVirtualRegister(RootCrossCopyRegClass);
    (void)llvm::BuildMI(EntryBlock, EntryBlock.begin(), llvm::MIMetadata(),
                        TII->get(llvm::AMDGPU::IMPLICIT_DEF))
        .addReg(Root, llvm::RegState::Define);
    (void)llvm::BuildMI(EntryBlock, EntryBlock.begin(), llvm::MIMetadata(),
                        TII->get(llvm::AMDGPU::COPY))
        .addReg(RootVirtReg, llvm::RegState::Define)
        .addReg(Root);
    It->getSecond()->Initialize(RootVirtReg);
    // Tell the SSAUpdater that the entry block already has a def for this
    // phys-reg's virtual value. Without this, later
    // GetValueInMiddleOfBlock(MBB) queries on a successor MBB would
    // synthesize an IMPLICIT_DEF "undef" PHI source instead of using the
    // entry COPY, and the return-block restore would COPY the undef back —
    // which would also make InjectedPayloadSideEffectsAnalysis report a
    // spurious Write on a phys-reg the payload only reads.
    It->getSecond()->AddAvailableValue(&EntryBlock, RootVirtReg);
    return {RootVirtReg, It->getSecond().get()};
  };

  /// Materialize the virtual register holding the current value of a 32-bit
  /// physreg \p Channel at the placeholder's program point in \p MBB. Errors
  /// out for non-payload functions — only injected payloads can touch the
  /// target application's ISA-visible state, since their contents are
  /// patched in as-is.
  ///
  /// On AMDGPU, regunit roots are 16-bit (SReg_LO16 / SReg_HI16) — non-
  /// allocatable classes that can't back virtual registers. So the driver
  /// decomposes every read/write at the 32-bit-channel granularity instead;
  /// this helper handles one such channel.
  auto getReadChannelVReg =
      [&](llvm::MCRegister Channel, llvm::MachineBasicBlock *MBB,
          const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder)
      -> llvm::Register {
    if (!IsInjectedPayload) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Function {0} is not an injected payload. Physical "
          "registers can only be accessed inside injected payloads",
          MF.getName()))));
      return llvm::Register();
    }

    auto ChIt = PhysRegValueSSAUpdaters.find(Channel);
    llvm::Register EntryVReg;
    if (ChIt == PhysRegValueSSAUpdaters.end()) {
      auto [V, U] = initializeSSAUpdaterForRoot(Channel);
      if (!U)
        return llvm::Register();
      EntryVReg = V;
    } else {
      // Reconstruct the entry-block COPY's vreg from the existing COPY MI
      // so we can short-circuit the SSAUpdater for entry-block queries.
      llvm::MachineBasicBlock &EntryBlock = MF.front();
      for (llvm::MachineInstr &MI : EntryBlock) {
        if (MI.isCopy() && MI.getOperand(0).isReg() &&
            MI.getOperand(0).getReg().isVirtual() && MI.getOperand(1).isReg() &&
            MI.getOperand(1).getReg().isPhysical() &&
            MI.getOperand(1).getReg() == Channel) {
          EntryVReg = MI.getOperand(0).getReg();
          break;
        }
      }
    }
    // Entry-block shortcut: MachineSSAUpdater::GetValueInMiddleOfBlock
    // ignores defs that live in the queried block, which is wrong for the
    // entry block (whose only def is the COPY-from-physreg we ourselves
    // inserted at .begin()). Return the entry-COPY's vreg directly in
    // that case.
    //
    // For any other MBB, defer the SSAUpdater query to Phase 2 — at this
    // point in the walk we may not yet have processed a back-edge
    // predecessor's overwrite, so GetValueInMiddleOfBlock would
    // synthesize a stale IMPLICIT_DEF PHI source.
    if (MBB == &MF.front() && EntryVReg.isValid())
      return EntryVReg;
    const llvm::TargetRegisterClass *RC =
        TRI->getCrossCopyRegClass(TRI->getPhysRegBaseClass(Channel));
    if (!RC) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Channel {0} has no copy reg class", llvm::printReg(Channel, TRI)))));
      return llvm::Register();
    }
    llvm::Register VReg = MRI.createVirtualRegister(RC);
    // Emit the placeholder right before the consuming intrinsic so dominance
    // is trivially satisfied — the resolved vreg in Phase 2 either already
    // dominates this point (entry COPY / cross-block PHI) or is defined
    // earlier in this MBB.
    (void)MIBuilder(llvm::AMDGPU::IMPLICIT_DEF)
        .addReg(VReg, llvm::RegState::Define);
    PendingResolutions.push_back(
        {MBB, Channel, VReg, PendingPhysRegResolution::Read});
    return VReg;
  };

  /// Record physical registers written by an intrinsic. Each phys-reg gets
  /// per-root COPY-defs that feed the SSAUpdater so subsequent reads of the
  /// same physreg see the new value, and the return-block restore logic
  /// emits a COPY-back of the final value.
  auto recordOverwrittenRegs =
      [&](const llvm::DenseMap<llvm::MCRegister, llvm::Register>
              &ToBeOverwrittenRegs,
          llvm::MachineBasicBlock *MBB,
          const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder) {
        // Same channel-decomposition strategy as PhysRegAccessor: walk
        // 32-bit channels instead of MCRegUnit roots
        for (const auto &[PhysReg, VirtReg] : ToBeOverwrittenRegs) {
          unsigned RegSizeBits = TRI->getRegSizeInBits(PhysReg, MRI);
          unsigned NumChannels = (RegSizeBits + 31) / 32;
          for (unsigned I = 0; I < NumChannels; ++I) {
            llvm::MCRegister Channel = PhysReg;
            unsigned SubIdx = llvm::AMDGPU::NoSubRegister;
            if (NumChannels > 1) {
              SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(I);
              Channel = TRI->getSubReg(PhysReg, SubIdx);
            }
            const llvm::TargetRegisterClass *ChannelRegClass =
                TRI->getPhysRegBaseClass(Channel);
            if (!ChannelRegClass) {
              Ctx.emitError(
                  llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                      "Physical register {0} doesn't have a reg class",
                      llvm::printReg(Channel, TRI)))));
              continue;
            }
            const llvm::TargetRegisterClass *ChannelCrossCopyRegClass =
                TRI->getCrossCopyRegClass(ChannelRegClass);
            if (!ChannelCrossCopyRegClass) {
              Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
                  llvm::formatv("Physical register {0} doesn't have a copy "
                                "reg class",
                                llvm::printReg(Channel, TRI)))));
              continue;
            }
            llvm::Register SubVirtReg =
                MRI.createVirtualRegister(ChannelCrossCopyRegClass);
            // For a single-channel reg, the source operand has no sub-reg
            // index; for multi-channel, slice via the channel's sub-reg.
            if (NumChannels == 1) {
              (void)MIBuilder(llvm::AMDGPU::COPY)
                  .addReg(SubVirtReg, llvm::RegState::Define)
                  .addReg(VirtReg);
            } else {
              (void)MIBuilder(llvm::AMDGPU::COPY)
                  .addReg(SubVirtReg, llvm::RegState::Define)
                  .addReg(VirtReg, llvm::RegState::NoFlags, SubIdx);
            }
            auto ChIt = PhysRegValueSSAUpdaters.find(Channel);
            if (ChIt == PhysRegValueSSAUpdaters.end()) {
              ChIt =
                  PhysRegValueSSAUpdaters
                      .insert({Channel,
                               std::make_unique<llvm::MachineSSAUpdater>(MF)})
                      .first;
              ChIt->getSecond()->Initialize(SubVirtReg);
            }
            ChIt->getSecond()->AddAvailableValue(MBB, SubVirtReg);
          }
        }
      };

  for (llvm::MachineBasicBlock &MBBRef : MF) {
    llvm::MachineBasicBlock *MBB = &MBBRef;
    for (llvm::MachineInstr &MI : llvm::make_early_inc_range(*MBB)) {
      if (!MI.isInlineAsm())
        continue;

      const llvm::MachineOperand &AsmStrOp =
          MI.getOperand(llvm::InlineAsm::MIOp_AsmString);
      const char *AsmStr = AsmStrOp.getSymbolName();
      llvm::StringRef IntrinsicName(AsmStr);

      auto ArgVec = getInlineAsmArgs(MI);

      auto MIBuilder = [&](int Opcode) {
        return llvm::BuildMI(*MBB, MI, llvm::MIMetadata(MI), TII->get(Opcode));
      };

      bool isReadReg = IntrinsicName == "luthier::readReg";
      bool isWriteReg = IntrinsicName == "luthier::writeReg";
      bool isReadSVA = IntrinsicName == "luthier::readSVA";

      if (isReadReg || isWriteReg) {
        // The physical register enum is passed as the immediate operand
        // (ArgVec[1]); ArgVec[0] is the regdef output produced by the IR
        // processor's setReturnValueInfo.
        llvm::DenseMap<llvm::MCRegister, llvm::Register> ReadPhysRegVRegs;
        llvm::MCRegister PhysReg(ArgVec[1].second->getImm());
        unsigned RegSizeBits = TRI->getRegSizeInBits(PhysReg, MRI);
        unsigned NumChannels = std::max(1u, (RegSizeBits + 31) / 32);
        // Frame-reg fast path: readReg / writeReg of SP / FP / PSB /
        // FLAT_SCR sub-channels inside an injected payload address the
        // fixed SVA frame lanes directly via V_READLANE_B32 /
        // V_WRITELANE_B32 against the caller-loaded SVA VGPR. The lane
        // per physreg is set by StateValueArraySpecs, which is a
        // Prototype-level analysis, so every payload sees the same lane
        // for the same physreg.
        std::optional<uint8_t> FrameSVALane = getFrameSVALaneForPhysReg(PhysReg);
        if (FrameSVALane) {
          if (!SVAVGPR) {
            Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
                llvm::formatv("Frame-reg readReg / writeReg in MF {0}: no SVA "
                              "VGPR resolved from the instrumentation point's "
                              "load plan; frame-reg SVA access requires an "
                              "injected payload with a valid LoadPlan.",
                              MF.getName()))));
            return Changed;
          }
          if (isReadReg) {
            // Constrain to SReg_32_XM0_XEXEC (matches what the removed
            // loadRegFromStackSlot path was constraining to, and what
            // readRegMIRProcessor's single-channel COPY expects).
            llvm::Register FrameVReg = MRI.createVirtualRegister(
                &llvm::AMDGPU::SReg_32_XM0_XEXECRegClass);
            llvm::BuildMI(MF.front(), svaInsertPt(), llvm::MIMetadata(),
                          TII->get(llvm::AMDGPU::V_READLANE_B32), FrameVReg)
                .addReg(SVAVGPRPlaceholder)
                .addImm(*FrameSVALane);
            ReadPhysRegVRegs[PhysReg] = FrameVReg;
          }
          // For writes: leave ReadPhysRegVRegs empty for this channel —
          // writeRegMIRProcessor's single-channel path doesn't consult it,
          // and the frame-write side below emits SI_SPILL_S32_SAVE directly.
        } else if (RegSizeBits < 32) {
          // Sub-32 reads are folded into the 32-bit super-register channel
          // when one exists (e.g. SGPR_LO16 → SGPR). Some special 1-bit
          // regs — most notably \c SCC — have no 32-bit super via
          // \c get32BitRegister: they can be read only by materializing
          // their value into a fresh 32-bit vreg via a cross-class \c COPY
          // (LLVM's \c copyPhysReg lowers \c $scc → \c SGPR to
          // \c S_CSELECT_B32 automatically). Route those directly through
          // \c getReadChannelVReg on the sub-32 physreg itself, so
          // \c readRegMIRProcessor's \c SrcRegSize==1 branch finds
          // \c channelVReg(SCC).
          const auto *SITRI = static_cast<const llvm::SIRegisterInfo *>(TRI);
          llvm::MCRegister SuperReg = SITRI->get32BitRegister(PhysReg);
          llvm::MCRegister LookupReg = SuperReg ? SuperReg : PhysReg;
          if (!ReadPhysRegVRegs.count(LookupReg))
            ReadPhysRegVRegs[LookupReg] =
                getReadChannelVReg(LookupReg, MBB, MIBuilder);
        } else {
          for (unsigned I = 0; I < NumChannels; ++I) {
            llvm::MCRegister Channel = PhysReg;
            if (NumChannels > 1) {
              unsigned SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(I);
              Channel = TRI->getSubReg(PhysReg, SubIdx);
            }
            if (!ReadPhysRegVRegs.count(Channel))
              ReadPhysRegVRegs[Channel] =
                  getReadChannelVReg(Channel, MBB, MIBuilder);
          }
        }
        if (isReadReg) {
          if (auto Err = readRegMIRProcessor(
                  MF, ArgVec, MIBuilder, VirtRegBuilder, ReadPhysRegVRegs)) {
            Ctx.emitError(llvm::toString(std::move(Err)));
            return Changed;
          }
        } else {
          // The processor fills WritePhysRegSlots; the driver records each
          // entry with the appropriate SSAUpdater after the processor returns.
          llvm::DenseMap<llvm::MCRegister, llvm::Register> WritePhysRegSlots;
          if (auto Err =
                  writeRegMIRProcessor(MF, ArgVec, MIBuilder, VirtRegBuilder,
                                       ReadPhysRegVRegs, WritePhysRegSlots)) {
            Ctx.emitError(llvm::toString(std::move(Err)));
            return Changed;
          }
          // Frame-reg fast path for writes: deferred to return blocks.
          // Rather than modifying the SVA VGPR mid-body with a
          // V_WRITELANE_B32 here, register the write value with a per-
          // lane SSAUpdater and let the post-MBB-walk emission stage
          // build one V_WRITELANE_B32 chain per return block that folds
          // in every lane the payload wrote.
          if (FrameSVALane) {
            auto WSIt = WritePhysRegSlots.find(PhysReg);
            if (WSIt != WritePhysRegSlots.end()) {
              const uint8_t Lane = *FrameSVALane;
              auto UIt = FrameSVAWriteUpdaters.find(Lane);
              if (UIt == FrameSVAWriteUpdaters.end()) {
                auto Updater =
                    std::make_unique<llvm::MachineSSAUpdater>(MF);
                // Seed with an entry-MBB IMPLICIT_DEF so paths that
                // don't write the lane resolve to a value RA can freely
                // coalesce; the V_WRITELANE at the return block will
                // consume it there.
                llvm::Register InitVReg = MRI.createVirtualRegister(
                    &llvm::AMDGPU::SReg_32RegClass);
                llvm::BuildMI(MF.front(), svaInsertPt(), llvm::MIMetadata(),
                              TII->get(llvm::AMDGPU::IMPLICIT_DEF), InitVReg);
                Updater->Initialize(InitVReg);
                Updater->AddAvailableValue(&MF.front(), InitVReg);
                UIt = FrameSVAWriteUpdaters
                          .insert({Lane, std::move(Updater)})
                          .first;
              }
              UIt->second->AddAvailableValue(MBB, WSIt->second);
              WritePhysRegSlots.erase(WSIt);
            }
          }
          recordOverwrittenRegs(WritePhysRegSlots, MBB, MIBuilder);
        }
      } else if (isReadSVA) {
        // The SA enum is passed as the immediate operand (ArgVec[1]);
        // ArgVec[0] is the regdef output.
        llvm::DenseMap<ScalarValueArgument, llvm::Register> SVAVRegs;
        ScalarValueArgument SA =
            static_cast<ScalarValueArgument>(ArgVec[1].second->getImm());
        SVAVRegs[SA] = SVAScalarArgumentAccessor(SA);
        if (auto Err = readSVAMIRProcessor(MF, ArgVec, MIBuilder, SVAVRegs)) {
          Ctx.emitError(llvm::toString(std::move(Err)));
          return Changed;
        }
      } else {
        std::optional<IntrinsicProcessor> Processor =
            IntrinsicsProcessors.getProcessorIfRegistered(IntrinsicName);
        if (!Processor.has_value()) {
          Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
              llvm::formatv("Intrinsic processor for {0} was not found in the "
                            "intrinsic processors.",
                            IntrinsicName))));
          return Changed;
        }
        if (auto Err = Processor->MIRProcessor(MF, ArgVec, MIBuilder,
                                               VirtRegBuilder)) {
          Ctx.emitError(llvm::toString(std::move(Err)));
          return Changed;
        }
      }
      MI.eraseFromParent();
      Changed = true;
    }

    // Emit physical register restore COPYs at each return block. Done once
    // per block (not per-MI) so a block with multiple intrinsic placeholders
    // emits only one set of restores. Restores go before the first
    // terminator so RA sees them as live-out values, and an implicit-use of
    // the physreg is added to the terminator so RA treats it as live-out of
    // the function.
    if (MBB->isReturnBlock() && !PhysRegValueSSAUpdaters.empty()) {
      auto FirstTerm = MBB->getFirstTerminator();
      for (auto &[PhysReg, SSAUpdater] : PhysRegValueSSAUpdaters) {
        // Emit an IMPLICIT_DEF placeholder whose use is the COPY-to-physreg
        // restore. Phase 2 resolves the placeholder via
        // GetValueAtEndOfBlock once all writes across the function have
        // been registered. Using an immediate query here would be wrong
        // for any phys-reg whose live-in to this return block depends on
        // an MBB that hasn't been processed yet.
        const llvm::TargetRegisterClass *RC =
            TRI->getCrossCopyRegClass(TRI->getPhysRegBaseClass(PhysReg));
        if (!RC) {
          Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
              llvm::formatv("Return-block channel {0} has no copy reg class",
                            llvm::printReg(PhysReg, TRI)))));
          continue;
        }
        llvm::Register Placeholder = MRI.createVirtualRegister(RC);
        (void)llvm::BuildMI(*MBB, FirstTerm, llvm::MIMetadata(),
                            TII->get(llvm::AMDGPU::IMPLICIT_DEF), Placeholder);
        (void)llvm::BuildMI(*MBB, FirstTerm, llvm::MIMetadata(),
                            TII->get(llvm::AMDGPU::COPY))
            .addReg(PhysReg, llvm::RegState::Define)
            .addReg(Placeholder);
        MBB->back().addOperand(llvm::MachineOperand::CreateReg(PhysReg,
                                                               /*isDef=*/false,
                                                               /*isImp=*/true));
        PendingResolutions.push_back({MBB, PhysReg, Placeholder,
                                      PendingPhysRegResolution::ReturnRestore});
      }
    }
  }

  // Phase 2: every AvailableValue across the MF has now been registered
  // with the SSAUpdaters. Resolve each placeholder by querying the
  // updater (which lazily inserts PHIs at this point) and replacing all
  // uses of the placeholder vreg with the resolved one. Erase the
  // IMPLICIT_DEF defining the placeholder so the only def of the
  // resolved vreg is the real one.
  for (const PendingPhysRegResolution &P : PendingResolutions) {
    auto UpdaterIt = PhysRegValueSSAUpdaters.find(P.Channel);
    if (UpdaterIt == PhysRegValueSSAUpdaters.end())
      continue;
    llvm::MachineSSAUpdater &Updater = *UpdaterIt->second;
    llvm::Register Resolved = P.K == PendingPhysRegResolution::ReturnRestore
                                  ? Updater.GetValueAtEndOfBlock(P.MBB)
                                  : Updater.GetValueInMiddleOfBlock(P.MBB);
    if (!Resolved.isValid() || Resolved == P.Placeholder)
      continue;
    llvm::MachineInstr *DefMI = MRI.getUniqueVRegDef(P.Placeholder);
    MRI.replaceRegWith(P.Placeholder, Resolved);
    if (DefMI && DefMI->isImplicitDef())
      DefMI->eraseFromParent();
  }

  // Frame-reg SVA write flushing: chain one V_WRITELANE_B32 per written
  // lane at every return block.
  llvm::Register CurrentSVAVReg = SVAVGPRPlaceholder;
  if (!FrameSVAWriteUpdaters.empty()) {
    llvm::SmallVector<uint8_t, 8> LanesInOrder;
    LanesInOrder.reserve(FrameSVAWriteUpdaters.size());
    for (auto &Entry : FrameSVAWriteUpdaters)
      LanesInOrder.push_back(Entry.first);
    llvm::sort(LanesInOrder);
    for (llvm::MachineBasicBlock &RetMBB : MF) {
      if (!RetMBB.isReturnBlock())
        continue;
      auto FirstTerm = RetMBB.getFirstTerminator();
      llvm::MDNode *SVAVGPRMarker = llvm::MDNode::get(
          Ctx, {llvm::MDString::get(Ctx, "luthier.sva_vgpr_placeholder")});
      for (uint8_t Lane : LanesInOrder) {
        llvm::MachineSSAUpdater &Updater =
            *FrameSVAWriteUpdaters.find(Lane)->second;
        llvm::Register ValueToWrite = Updater.GetValueAtEndOfBlock(&RetMBB);
        llvm::Register NextSVAVReg =
            MRI.createVirtualRegister(&llvm::AMDGPU::VGPR_32RegClass);
        MRI.setSimpleHint(NextSVAVReg, SVAVGPR);
        SIMFI->setFlag(NextSVAVReg, llvm::AMDGPU::VirtRegFlag::WWM_REG);
        auto *WriteMI =
            llvm::BuildMI(RetMBB, FirstTerm, llvm::MIMetadata(),
                          TII->get(llvm::AMDGPU::V_WRITELANE_B32), NextSVAVReg)
                .addReg(ValueToWrite)
                .addImm(Lane)
                .addReg(CurrentSVAVReg)
                .getInstr();
        // Tag each chain-link def with the same pcsections marker as the
        // initial IMPLICIT_DEF placeholder so \c SVAPhysVGPRPinPass picks
        // up every SVA-VGPR virtual register and re-pins it to the
        // LoadPlan physreg — otherwise RA could split the write chain
        // across multiple WWM VGPRs and the caller would only see a
        // partial update.
        WriteMI->setPCSections(MF, SVAVGPRMarker);
        CurrentSVAVReg = NextSVAVReg;
      }
    }
  }

  /// Make the final VReg of the SVA an implicit operand of the return block
  for (llvm::MachineBasicBlock &RetMBB : MF) {
    if (!RetMBB.isReturnBlock())
      continue;
    auto FirstTerm = RetMBB.getFirstTerminator();
    if (FirstTerm != RetMBB.end() && CurrentSVAVReg != SVAVGPRPlaceholder)
      FirstTerm->addOperand(llvm::MachineOperand::CreateReg(
          CurrentSVAVReg, /*isDef=*/false, /*isImp=*/true));
  }

  return Changed;
}

bool IntrinsicMIRLoweringPass::lowerIntrinsics(
    Prototype &IP, PrototypeAnalysisManager &IPAM,
    const StateValueArraySpecs &SVASpecs,
    const InjectedPayloadAndInstPoint &IPIP,
    const SVStorageAndLoadLocations &SVLocations) {
  bool Changed = false;

  llvm::Module &IModule = IP.getInstrumentationModule();

  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<IModuleAnalysisManagerPrototypeProxy>(IP).getManager();

  const auto &IntrinsicsProcessors =
      MAM.getResult<IntrinsicsProcessorsAnalysis>(IModule);

  llvm::FunctionAnalysisManager &IModuleFAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(IModule)
          .getManager();

  for (llvm::Function &F : IModule) {
    if (F.isDeclaration())
      continue;
    bool IsInjectedPayload = F.hasFnAttribute(InjectedPayloadAttribute);
    llvm::MachineFunction *MF = nullptr;
    if (auto *MFRes =
            IModuleFAM.getCachedResult<llvm::MachineFunctionAnalysis>(F))
      MF = &MFRes->getMF();
    else
      continue;
    // Resolve the caller-loaded SVA VGPR for this payload.
    llvm::MCRegister SVAVGPR{};
    if (IsInjectedPayload && IPIP.contains(F)) {
      if (const llvm::MachineInstr *TargetMI = IPIP.at(F)) {
        if (const InstPointSVALoadPlan *LoadPlan =
                SVLocations.getStateValueArrayLoadPlanForInstPoint(*TargetMI))
          SVAVGPR = LoadPlan->StateValueArrayLoadVGPR;
      }
    }
    Changed |= processMachineFunction(*MF, IsInjectedPayload,
                                      IntrinsicsProcessors, SVASpecs, SVAVGPR);
  }

  return Changed;
}

llvm::PreservedAnalyses
IntrinsicMIRLoweringPass::run(Prototype &IP, PrototypeAnalysisManager &IPAM) {
  const StateValueArraySpecs &SVASpecs =
      IPAM.getResult<StateValueArraySpecsAnalysis>(IP);
  const InjectedPayloadAndInstPoint &IPIP =
      IPAM.getResult<InjectedPayloadAndInstPointAnalysis>(IP);
  const SVStorageAndLoadLocations &SVLocations =
      IPAM.getResult<SVStorageAndLoadLocationsAnalysis>(IP);
  bool Changed = lowerIntrinsics(IP, IPAM, SVASpecs, IPIP, SVLocations);

  if (!Changed)
    return llvm::PreservedAnalyses::all();

  // Preserve the outer MAM proxy so the Prototype adaptor doesn't
  // wipe every cached module-level analysis for both modules on the way out —
  // downstream passes still need the cached MachineFunctionAnalysis results
  // for the instrumentation module we just mutated.
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  // This pass rewrites MIR inside the instrumentation module. It does not
  // touch the target module at all, and it handles its own module's MIR in
  // place rather than through an analysis, so every inner analysis-manager
  // proxy stays valid; only Prototype-level analyses derived from the
  // instrumentation module's MIR are dropped.
  PA.preserve<TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  // The SVA specs derived by StateValueArraySpecsAnalysis depend only on
  // the IModule's IR call sites and target-module entry-point metadata,
  // neither of which this pass modifies. Preserve it explicitly.
  PA.preserve<StateValueArraySpecsAnalysis>();
  return PA;
}

} // namespace luthier
