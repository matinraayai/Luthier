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
/// This file implements the Intrinsic MIR Lowering Pass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/IntrinsicMIRLoweringPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorsAnalysis.h"
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include "luthier/ToolCodeGen/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include "luthier/ToolCodeGen/WrapperAnalysisPasses.h"
#include <AMDGPU.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineSSAUpdater.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetFrameLowering.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier {

char IntrinsicMIRLoweringPass::ID = 0;

LUTHIER_INITIALIZE_LEGACY_PASS_BODY(IntrinsicMIRLoweringPass, "mir-lowering",
                                    "Intrinsic MIR lowering pass",
                                    true /* Only looks at CFG */,
                                    false /* Analysis Pass */)

namespace {

/// Decode the trailing register-operand args of an inline-asm MachineInstr
/// into (Flag, Register) pairs. AsmString operands and other non-flag
/// metadata operands are ignored; the iteration skips the register operands
/// described by each flag's \c getNumOperandRegisters().
llvm::SmallVector<std::pair<llvm::InlineAsm::Flag, llvm::Register>>
getInlineAsmArgs(const llvm::MachineInstr &MI) {
  llvm::SmallVector<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Out;
  for (unsigned I = llvm::InlineAsm::MIOp_FirstOperand,
                NumOps = MI.getNumOperands();
       I < NumOps; ++I) {
    const llvm::MachineOperand &MO = MI.getOperand(I);
    if (!MO.isImm())
      continue;
    const llvm::InlineAsm::Flag F(MO.getImm());
    const llvm::Register Reg(MI.getOperand(I + 1).getReg());
    Out.emplace_back(F, Reg);
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

/// Lookup table: placeholder-key (the InlineAsm template string used as
/// operand 0 of an INLINEASM MachineInstr) → (intrinsic-name, aux-MDNode).
/// Aux may be \c nullptr if no aux was forwarded.
struct PlaceholderInfo {
  llvm::StringRef IntrinsicName;
  llvm::MDNode *Aux;
  IntrinsicISAStateEffects Effects;
};
using PlaceholderMap = llvm::DenseMap<llvm::StringRef, PlaceholderInfo>;

/// Walk the module's \c !luthier.intrinsic.placeholders NamedMDNode and
/// build a key → PlaceholderInfo map. Each NamedMD operand is a 3-tuple
/// \c !{key, name, aux} written by \c ProcessIntrinsicsAtIRLevelPass.
PlaceholderMap buildPlaceholderMap(const llvm::Module &IModule) {
  PlaceholderMap Out;
  const llvm::NamedMDNode *NamedMD =
      IModule.getNamedMetadata(LuthierIntrinsicNamedMDName);
  if (!NamedMD)
    return Out;
  for (const llvm::MDNode *Entry : NamedMD->operands()) {
    if (!Entry || Entry->getNumOperands() < 3)
      continue;
    auto *KeyMD = llvm::dyn_cast<llvm::MDString>(Entry->getOperand(0));
    auto *NameMD = llvm::dyn_cast<llvm::MDString>(Entry->getOperand(1));
    if (!KeyMD || !NameMD)
      continue;
    auto *AuxMD = llvm::dyn_cast<llvm::MDNode>(Entry->getOperand(2));
    // An empty aux MDNode means "no aux data"; surface it as nullptr to the
    // processor, matching the prior pcsections behaviour.
    if (AuxMD && AuxMD->getNumOperands() == 0)
      AuxMD = nullptr;
    const llvm::MDNode *EffNode =
        Entry->getNumOperands() >= 4
            ? llvm::dyn_cast<llvm::MDNode>(Entry->getOperand(3))
            : nullptr;
    Out.try_emplace(KeyMD->getString(),
                    PlaceholderInfo{NameMD->getString(), AuxMD,
                                    decodeIntrinsicISAStateEffects(EffNode)});
  }
  return Out;
}

/// Extract the placeholder key from an inline-asm MachineInstr. Returns an
/// empty StringRef if the asm template is not a Luthier placeholder key.
llvm::StringRef getPlaceholderKey(const llvm::MachineInstr &MI) {
  if (MI.getNumOperands() == 0)
    return {};
  const llvm::MachineOperand &Op0 = MI.getOperand(0);
  if (!Op0.isSymbol())
    return {};
  llvm::StringRef AsmStr(Op0.getSymbolName());
  if (!AsmStr.starts_with(LuthierIntrinsicPlaceholderKeyPrefix))
    return {};
  return AsmStr;
}

} // namespace

struct IntrinsicMIRLoweringPass::PlaceholderLookupTable {
  PlaceholderMap Map;
};

bool IntrinsicMIRLoweringPass::processMachineFunction(
    llvm::MachineFunction &MF, bool IsInjectedPayload,
    const IntrinsicsProcessorsAnalysis::Result &IntrinsicsProcessors,
    const PlaceholderLookupTable &Placeholders,
    llvm::SmallDenseSet<ScalarValueArgument> &ScalarArgumentsUsed,
    PerFunctionSVAInfo &MFSVAInfo) {
  llvm::LLVMContext &Ctx = MF.getFunction().getContext();
  const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  llvm::MachineRegisterInfo &MRI = MF.getRegInfo();

  /// Per-MF cache: SA -> wide vreg holding the SA's value. Returned vreg may
  /// be a single SGPR_32 (1-lane SAs) or a REG_SEQUENCE'd wide SGPR (multi
  /// lane). One cache entry per SA covers every intrinsic call site in this
  /// MF.
  llvm::DenseMap<ScalarValueArgument, llvm::Register> SAResultCache;

  /// Fixed insertion point for SVA-related instructions in the entry block.
  /// Always insert before the first terminator so that the order is
  /// deterministic and all inserted instructions dominate the entire
  /// function.
  llvm::MachineBasicBlock::iterator SVAInsertPt =
      MF.front().getFirstTerminator();

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

  /// Returns or lazily creates the per-MF SVA VGPR placeholder, an
  /// IMPLICIT_DEF VGPR_32 in the entry block carrying
  /// !pcsections !{!"luthier.sva_vgpr_placeholder"}. A later (post-RA)
  /// Luthier pass resolves this to the actual SVA VGPR and emits the
  /// SI_SPILL_S32_TO_VGPR / SI_RESTORE_S32_FROM_VGPR pseudos that engage
  /// the WWM machinery — those pseudos can only be safely emitted with
  /// physical-register operands, so the wiring must happen after
  /// register allocation rather than here.
  auto getOrCreateSVAVGPRPlaceholder = [&]() {
    if (MFSVAInfo.SVAVGPRPlaceholder.isValid())
      return MFSVAInfo.SVAVGPRPlaceholder;
    llvm::Register SVAPlaceholder =
        MRI.createVirtualRegister(&llvm::AMDGPU::VGPR_32RegClass);
    llvm::MDNode *MarkerNode = llvm::MDNode::get(
        Ctx, {llvm::MDString::get(Ctx, "luthier.sva_vgpr_placeholder")});
    auto *SVAImplDef =
        llvm::BuildMI(MF.front(), SVAInsertPt, llvm::MIMetadata(),
                      TII->get(llvm::AMDGPU::IMPLICIT_DEF), SVAPlaceholder)
            .getInstr();
    SVAImplDef->setPCSections(MF, MarkerNode);
    MFSVAInfo.SVAVGPRPlaceholder = SVAPlaceholder;
    return MFSVAInfo.SVAVGPRPlaceholder;
  };

  /// SVA scalar argument accessor: returns a virtual register holding the
  /// value of the requested scalar argument.
  ///
  /// On first request for an SA in this MF, materializes per-lane SGPR_32
  /// IMPLICIT_DEF placeholders (recorded in MFSVAInfo.Readlanes for later
  /// replacement by V_READLANE_B32 in phase 2), REG_SEQUENCEs them into a
  /// wide SGPR for multi-lane SAs, and caches the result. Subsequent
  /// requests for the same SA return the cached vreg.
  auto SVAScalarArgumentAccessor =
      [&](ScalarValueArgument SA) -> llvm::Register {
    auto CacheIt = SAResultCache.find(SA);
    if (CacheIt != SAResultCache.end())
      return CacheIt->second;

    ScalarArgumentsUsed.insert(SA);
    (void)getOrCreateSVAVGPRPlaceholder();

    unsigned NumLanes = StateValueArraySpecs::getArgumentLaneSize(SA);
    static const unsigned SubRegForLane[] = {
        llvm::AMDGPU::sub0, llvm::AMDGPU::sub1, llvm::AMDGPU::sub2,
        llvm::AMDGPU::sub3};

    llvm::SmallVector<llvm::Register, 4> LaneRegs;
    LaneRegs.reserve(NumLanes);
    for (uint8_t Lane = 0; Lane < NumLanes; ++Lane) {
      llvm::Register LaneReg =
          MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass);
      llvm::BuildMI(MF.front(), SVAInsertPt, llvm::MIMetadata(),
                    TII->get(llvm::AMDGPU::IMPLICIT_DEF), LaneReg);
      MFSVAInfo.Readlanes.push_back({LaneReg, SA, Lane});
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
        llvm::BuildMI(MF.front(), SVAInsertPt, llvm::MIMetadata(),
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
                        TII->get(llvm::AMDGPU::COPY))
        .addReg(RootVirtReg, llvm::RegState::Define)
        .addReg(Root);
    It->getSecond()->Initialize(RootVirtReg);
    // Tell the SSAUpdater that the entry block already has a def for this
    // phys-reg's virtual value. Without this, later
    // GetValueInMiddleOfBlock(MBB) queries on a successor MBB would
    // synthesize an IMPLICIT_DEF "undef" PHI source instead of using the
    // entry COPY, and the return-block restore would COPY the undef back —
    // which would also make InjectedPayloadAccessedRegsAnalysis report a
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
        // 32-bit channels (always allocatable on AMDGPU) instead of
        // MCRegUnit roots (which are non-allocatable 16-bit slices).
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

      // Recover the intrinsic identity from the placeholder's opaque key.
      // The key is the InlineAsm template string (operand 0 of the
      // INLINEASM MachineInstr), and survives SelectionDAG even though
      // !pcsections does not. The Module's
      // !luthier.intrinsic.placeholders NamedMD maps key -> (name, aux).
      llvm::StringRef Key = getPlaceholderKey(MI);
      if (Key.empty())
        continue;
      auto KeyIt = Placeholders.Map.find(Key);
      if (KeyIt == Placeholders.Map.end())
        continue;
      llvm::StringRef IntrinsicName = KeyIt->second.IntrinsicName;
      llvm::MDNode *IntrinsicPayload = KeyIt->second.Aux;
      const IntrinsicISAStateEffects &Effects = KeyIt->second.Effects;

      auto ArgVec = getInlineAsmArgs(MI);

      auto MIBuilder = [&](int Opcode) {
        return llvm::BuildMI(*MBB, MI, llvm::MIMetadata(MI), TII->get(Opcode));
      };

      std::optional<IntrinsicProcessor> Processor =
          IntrinsicsProcessors.getProcessorIfRegistered(IntrinsicName);
      if (!Processor.has_value()) {
        Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
            llvm::formatv("Intrinsic processor for {0} was not found in the "
                          "intrinsic processors.",
                          IntrinsicName))));
        return Changed;
      }

      // Pre-stage per-SA vregs from this placeholder's declared SVA reads.
      llvm::DenseMap<ScalarValueArgument, llvm::Register> SVAVRegs;
      for (ScalarValueArgument SA : Effects.ReadSVAs)
        SVAVRegs[SA] = SVAScalarArgumentAccessor(SA);

      // Pre-stage per-32-bit-channel vregs from this placeholder's declared
      // phys-reg reads. Wide regs are decomposed channel-wise here so each
      // intrinsic processor only sees 32-bit-channel keys uniformly.
      llvm::DenseMap<llvm::MCRegister, llvm::Register> ReadPhysRegVRegs;
      for (llvm::MCRegister Reg : Effects.ReadPhysRegs) {
        unsigned RegSizeBits = TRI->getRegSizeInBits(Reg, MRI);
        unsigned NumChannels = std::max(1u, (RegSizeBits + 31) / 32);
        if (RegSizeBits < 32) {
          // Sub-32 reads are folded into the 32-bit super-register channel.
          const auto *SITRI = static_cast<const llvm::SIRegisterInfo *>(TRI);
          llvm::MCRegister SuperReg = SITRI->get32BitRegister(Reg);
          if (!ReadPhysRegVRegs.count(SuperReg))
            ReadPhysRegVRegs[SuperReg] =
                getReadChannelVReg(SuperReg, MBB, MIBuilder);
          continue;
        }
        for (unsigned I = 0; I < NumChannels; ++I) {
          llvm::MCRegister Channel = Reg;
          if (NumChannels > 1) {
            unsigned SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(I);
            Channel = TRI->getSubReg(Reg, SubIdx);
          }
          if (!ReadPhysRegVRegs.count(Channel))
            ReadPhysRegVRegs[Channel] =
                getReadChannelVReg(Channel, MBB, MIBuilder);
        }
      }

      // The processor fills WritePhysRegSlots; the driver records each entry
      // with the appropriate SSAUpdater after the processor returns.
      llvm::DenseMap<llvm::MCRegister, llvm::Register> WritePhysRegSlots;
      if (auto Err = Processor->MIRProcessor(
              MF, ArgVec, IntrinsicPayload, MIBuilder, VirtRegBuilder, SVAVRegs,
              ReadPhysRegVRegs, WritePhysRegSlots)) {
        Ctx.emitError(llvm::toString(std::move(Err)));
        return Changed;
      }

      recordOverwrittenRegs(WritePhysRegSlots, MBB, MIBuilder);

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

  return Changed;
}

bool IntrinsicMIRLoweringPass::lowerIntrinsics(
    llvm::Module &IModule,
    llvm::DenseMap<llvm::MachineFunction *, PerFunctionSVAInfo> &SVAInfoByMF,
    std::unique_ptr<StateValueArraySpecs> &SVASpecs) {
  bool Changed = false;

  llvm::MachineModuleInfo &MMI =
      getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI();

  llvm::ModuleAnalysisManager &IMAM =
      getAnalysis<IModuleMAMWrapperPass>().getMAM();

  const auto &IntrinsicsProcessors =
      IMAM.getResult<IntrinsicsProcessorsAnalysis>(IModule);

  auto &TargetModuleAndMAM =
      IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);
  llvm::Module &TargetModule = TargetModuleAndMAM.getTargetAppModule();
  llvm::ModuleAnalysisManager &TargetMAM = TargetModuleAndMAM.getTargetAppMAM();
  bool IsInitialEntryPointKernel =
      TargetMAM.getResult<InitialEntryPointAnalysis>(TargetModule)
          .getInitialEntryPoint()
          .isKernel();

  // Build the placeholder-key -> (name, aux) lookup table once per pass
  // run, by walking the module's !luthier.intrinsic.placeholders NamedMD.
  PlaceholderLookupTable Placeholders{buildPlaceholderMap(IModule)};

  // Set of all scalar arguments used across the entire module.
  llvm::SmallDenseSet<ScalarValueArgument> ScalarArgumentsUsed{};

  // If the initial entry point is not a kernel, then we are instrumenting
  // newly discovered code from an already-running instrumented kernel. In
  // that situation the SVA has already been initialized to save all possible
  // scalar arguments just to be safe.
  if (!IsInitialEntryPointKernel) {
    for (std::underlying_type_t<ScalarValueArgument> I =
             SCALAR_VALUE_ARGUMENT_FIRST;
         I <= SCALAR_VALUE_ARGUMENT_LAST; I++) {
      ScalarArgumentsUsed.insert(static_cast<ScalarValueArgument>(I));
    }
  }

  for (llvm::Function &F : IModule) {
    bool IsInjectedPayload = F.hasFnAttribute(InjectedPayloadAttribute);
    llvm::MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF)
      continue;
    Changed |= processMachineFunction(*MF, IsInjectedPayload,
                                      IntrinsicsProcessors, Placeholders,
                                      ScalarArgumentsUsed, SVAInfoByMF[MF]);
  }

  // Finalize the SVA layout now that we know which SAs were requested.
  SVASpecs = StateValueArraySpecs::setModuleSVASpec(IModule, MMI.getTarget(),
                                                    ScalarArgumentsUsed);
  return Changed;
}

void IntrinsicMIRLoweringPass::materializeReadlanes(
    llvm::DenseMap<llvm::MachineFunction *, PerFunctionSVAInfo> &SVAInfoByMF,
    const StateValueArraySpecs &SVASpecs, bool &Changed) {
  for (auto &[MF, SVAInfo] : SVAInfoByMF) {
    if (SVAInfo.Readlanes.empty())
      continue;

    llvm::LLVMContext &Ctx = MF->getFunction().getContext();
    const llvm::TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
    llvm::MachineRegisterInfo &MRI = MF->getRegInfo();
    llvm::MachineFrameInfo &MFI = MF->getFrameInfo();
    auto *MFInfo = MF->getInfo<llvm::SIMachineFunctionInfo>();

    // The high-level emission contract for SGPR-from-VGPR-lane reads:
    //
    //   1. CreateStackObject + setStackID(SGPRSpill) gives us an FI.
    //   2. MFInfo->allocateSGPRSpillToVGPRLane reserves K lanes on a fresh
    //      framework-owned LaneVGPR (single LaneVGPR shared across all
    //      SA-lane FIs because the monotonic NumVirtualVGPRSpillLanes
    //      counter advances per allocation).
    //   3. TII->loadRegFromStackSlot emits SI_SPILL_S32_RESTORE
    //      %placeholder_sgpr_vreg, FI — the canonical high-level pseudo
    //      that LLVM's RegAlloc path emits and SILowerSGPRSpills knows how
    //      to lower. Vreg operand 0 is fine pre-RA: VirtRegRewriter rewrites
    //      it to a physreg before SILowerSGPRSpills runs.
    //   4. SILowerSGPRSpills eliminates the FI, lowering to
    //      SI_RESTORE_S32_FROM_VGPR $physreg, $lane_vgpr, <lane>, AND emits
    //      IMPLICIT_DEF for LaneVGPR + setFlag(WWM_REG). The downstream
    //      WWM regalloc pool then handles physreg assignment.
    //
    // The placeholder vreg machinery (IMPLICIT_DEF VGPR_32 +
    // replaceRegWith) from the previous design is unneeded — there's no
    // per-MF SVA-VGPR handle to track because the LaneVGPR is fully owned
    // by SILowerSGPRSpills' SpillVGPRs[] list. See
    // [[reference_amdgpu_wwm_pipeline]] for the verified pass sequence.

    // Gather the unique SAs this MF uses, in canonical SVASpecs lane order
    // so the framework's per-MF monotonic lane counter matches our
    // per-lane immediates.
    llvm::SmallDenseSet<ScalarValueArgument> SAsSeen;
    llvm::SmallVector<ScalarValueArgument> SAsUsed;
    for (const PendingSVAReadlane &Entry : SVAInfo.Readlanes) {
      if (SAsSeen.insert(Entry.SA).second)
        SAsUsed.push_back(Entry.SA);
    }
    llvm::sort(SAsUsed, [&](ScalarValueArgument A, ScalarValueArgument B) {
      auto LA = SVASpecs.findArgumentLane(A);
      auto LB = SVASpecs.findArgumentLane(B);
      assert(LA != SVASpecs.argument_lane_end() &&
             LB != SVASpecs.argument_lane_end() &&
             "SA was requested but not found in SVASpecs");
      return LA->second < LB->second;
    });

    // Allocate one SGPRSpill FI per SA + record the per-SA lane base for
    // step (3) below. We don't need to inspect getSGPRSpillToVirtualVGPRLanes
    // because each SA-lane has a 1:1 FI ↔ (LaneVGPR, lane) binding the
    // framework manages; we just hand the FI to loadRegFromStackSlot and
    // pass the within-SA offset as a separate FI per sub-lane.
    struct SALaneSlot {
      int FI;
      uint8_t LaneIndexWithinFI;
    };
    llvm::DenseMap<std::pair<ScalarValueArgument, uint8_t>, SALaneSlot>
        SubLaneToFI;
    bool AllocFailed = false;

    for (ScalarValueArgument SA : SAsUsed) {
      unsigned NumLanes = StateValueArraySpecs::getArgumentLaneSize(SA);
      // Each sub-lane gets its own FI so loadRegFromStackSlot can target
      // a single 32-bit slot. The framework still shares the underlying
      // LaneVGPR across all of them via NumVirtualVGPRSpillLanes.
      for (uint8_t Lane = 0; Lane < NumLanes; ++Lane) {
        int FI = MFI.CreateStackObject(/*Size=*/4, llvm::Align(4),
                                       /*isSpillSlot=*/true);
        MFI.setStackID(FI, llvm::TargetStackID::SGPRSpill);
        if (!MFInfo->allocateSGPRSpillToVGPRLane(
                *MF, FI, /*SpillToPhysVGPRLane=*/false)) {
          Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
              "Failed to allocate SGPR-to-VGPR-lane spill for SVA SA "
              "{0} lane {1} in MF {2} (target may not support "
              "SGPR-to-VGPR spilling, or the SVA exceeds wave width)",
              static_cast<int>(SA), Lane, MF->getName()))));
          AllocFailed = true;
          break;
        }
        SubLaneToFI[{SA, Lane}] = {FI, /*LaneIndexWithinFI=*/0};
      }
      if (AllocFailed)
        break;
    }

    if (AllocFailed)
      continue;

    // Emit SI_SPILL_S32_RESTORE %placeholder, FI for each pending readlane.
    // TII->loadRegFromStackSlot constrains the dest vreg to
    // SReg_32_XM0_XEXEC and stamps TargetStackID::SGPRSpill on the FI
    // (idempotent with what we already did above).
    for (const PendingSVAReadlane &Entry : SVAInfo.Readlanes) {
      auto It = SubLaneToFI.find({Entry.SA, Entry.LaneWithinSA});
      assert(It != SubLaneToFI.end() && "SA sub-lane must have an FI");
      int FI = It->second.FI;

      llvm::MachineInstr *ImplDef = MRI.getUniqueVRegDef(Entry.SGPRPlaceholder);
      assert(ImplDef && ImplDef->isImplicitDef() &&
             "SVA lane placeholder must be defined by an IMPLICIT_DEF");
      llvm::MachineBasicBlock *DefMBB = ImplDef->getParent();
      llvm::MachineBasicBlock::iterator InsertPt = ImplDef->getIterator();

      TII->loadRegFromStackSlot(*DefMBB, InsertPt, Entry.SGPRPlaceholder, FI,
                                &llvm::AMDGPU::SReg_32RegClass,
                                /*VReg=*/{});

      ImplDef->eraseFromParent();
      Changed = true;
    }

    // The per-MF SVA placeholder is no longer load-bearing — the LaneVGPR
    // is fully owned by MFInfo->SpillVGPRs[]. Erase the placeholder
    // IMPLICIT_DEF VGPR_32 and clear the handle. Downstream consumers (PEI)
    // discover the SVA physreg via the SVStorageAndLoadLocations load plan,
    // not via a per-MF vreg.
    if (SVAInfo.SVAVGPRPlaceholder.isValid()) {
      llvm::MachineInstr *PlaceholderDef =
          MRI.getUniqueVRegDef(SVAInfo.SVAVGPRPlaceholder);
      if (PlaceholderDef && PlaceholderDef->isImplicitDef())
        PlaceholderDef->eraseFromParent();
      SVAInfo.SVAVGPRPlaceholder = llvm::Register();
      Changed = true;
    }
  }
}

bool IntrinsicMIRLoweringPass::runOnModule(llvm::Module &IModule) {
  llvm::DenseMap<llvm::MachineFunction *, PerFunctionSVAInfo> SVAInfoByMF;
  std::unique_ptr<StateValueArraySpecs> SVASpecs{nullptr};

  bool Changed = lowerIntrinsics(IModule, SVAInfoByMF, SVASpecs);

  if (SVASpecs)
    materializeReadlanes(SVAInfoByMF, *SVASpecs, Changed);

  return Changed;
}

void IntrinsicMIRLoweringPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<IModuleMAMWrapperPass>();
  AU.addRequired<llvm::MachineModuleInfoWrapperPass>();
  ModulePass::getAnalysisUsage(AU);
};

} // namespace luthier
