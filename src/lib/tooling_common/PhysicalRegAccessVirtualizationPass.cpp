//===-- PhysicalRegAccessVirtualizationPass.cpp ---------------------------===//
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
/// This file implements Luthier's Physical Reg Access Virtualization Pass.
//===----------------------------------------------------------------------===//
#include "tooling_common/PhysicalRegAccessVirtualizationPass.hpp"
#include "luthier/LiftedRepresentation.h"
#include <GCNSubtarget.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/MachineSSAUpdater.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <queue>

namespace luthier {

char PhysicalRegAccessVirtualizationPass::ID = 0;

const static llvm::SmallDenseMap<llvm::MCRegister, unsigned int>
    ValueRegisterSpillSlots{
        {llvm::AMDGPU::SGPR0, 0},      {llvm::AMDGPU::SGPR1, 1},
        {llvm::AMDGPU::SGPR2, 2},      {llvm::AMDGPU::SGPR3, 3},
        {llvm::AMDGPU::SGPR32, 4},     {llvm::AMDGPU::FLAT_SCR_LO, 5},
        {llvm::AMDGPU::FLAT_SCR_HI, 6}};

const static llvm::SmallDenseMap<llvm::MCRegister, unsigned int>
    ValueRegisterInstrumentationSlots{
        {llvm::AMDGPU::SGPR0, 7},       {llvm::AMDGPU::SGPR1, 8},
        {llvm::AMDGPU::SGPR2, 9},       {llvm::AMDGPU::SGPR3, 10},
        {llvm::AMDGPU::SGPR32, 11},     {llvm::AMDGPU::FLAT_SCR_LO, 12},
        {llvm::AMDGPU::FLAT_SCR_HI, 13}};

bool PhysicalRegAccessVirtualizationPass::runOnMachineFunction(
    llvm::MachineFunction &MF) {
  // If the function being processed is not a hook (i.e a device function
  // getting called inside a hook) it cannot access physical registers,
  // so skip it
  if (!PerHookLiveInRegs.contains(&MF.getFunction())) {
    return false;
  }
  llvm::SmallDenseMap<const llvm::MachineBasicBlock *,
                      llvm::SmallDenseSet<llvm::MCRegister, 4>, 4>
      MBBsThatNeedAccessToPhysicalRegs;
  llvm::SmallDenseSet<llvm::MCRegister, 4> AccessedPhysicalRegistersByHook;
  auto *TRI = MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  // Iterate over the MIs in the Hook function, find placeholder inline asms,
  // and get their lowering info
  // Aggregate all the accessed physical registers in the hook in one place
  for (const auto &MBB : MF) {
    for (const auto &MI : MBB) {
      if (MI.isInlineAsm()) {
        auto IntrinsicIdxAsString =
            MI.getOperand(llvm::InlineAsm::MIOp_AsmString).getSymbolName();
        unsigned int IntrinsicIdx = std::stoul(IntrinsicIdxAsString);
        auto &LoweringInfo =
            InlineAsmPlaceHolderToIRLoweringInfoMap[IntrinsicIdx];
        for (const auto &AccessedPhysReg :
             LoweringInfo.second.getAccessedPhysicalRegisters()) {
          if (!MBBsThatNeedAccessToPhysicalRegs.contains(&MBB))
            MBBsThatNeedAccessToPhysicalRegs.insert({&MBB, {}});
          auto &AccessedPhysRegClass =
              *TRI->getPhysRegBaseClass(AccessedPhysReg);
          size_t AccessedRegSize = TRI->getRegSizeInBits(AccessedPhysRegClass);
          if (AccessedRegSize > 32) {
            size_t NumChannels = AccessedRegSize / 32;
            for (int i = 0; i < NumChannels; i++) {
              auto Reg =
                  TRI->getSubReg(AccessedPhysReg,
                                 llvm::SIRegisterInfo::getSubRegFromChannel(i));
              MBBsThatNeedAccessToPhysicalRegs[&MBB].insert(Reg);
              AccessedPhysicalRegistersByHook.insert(Reg);
            }
          } else if (AccessedPhysReg == 16) {
            auto Reg = TRI->get32BitRegister(AccessedPhysReg);
            MBBsThatNeedAccessToPhysicalRegs[&MBB].insert(Reg);
            AccessedPhysicalRegistersByHook.insert(Reg);
          } else {
            MBBsThatNeedAccessToPhysicalRegs[&MBB].insert(AccessedPhysReg);
            AccessedPhysicalRegistersByHook.insert(AccessedPhysReg);
          }
        }
      }
    }
  }
  auto *TII = MF.getSubtarget().getInstrInfo();

  const auto &SVDesc =
      StateValueLocations.getStateValueDescriptorOfHookInsertionPoint(
          *HookFuncToMIMap.at(&MF.getFunction()));

  // For each live-in register that is not used in the hooks, create a copy
  // to its equivalent virtual register in the entry basic block, and
  // a copy back in all the return blocks
  // TODO: Map to the correct location of clobbered registers in case
  // the state value is not in a VGPR
  llvm::DenseMap<llvm::MCRegister, llvm::Register> PhysToVirtRegMap;
  for (const auto &LiveIn : PerHookLiveInRegs[&MF.getFunction()]) {
    if (!AccessedPhysicalRegistersByHook.contains(LiveIn)) {
      auto *PhysRegClass = TRI->getPhysRegBaseClass(LiveIn);
      PhysToVirtRegMap.insert(
          {LiveIn, MRI.createVirtualRegister(PhysRegClass)});
    }
  }
  // Add the physical registers used for state value storage as live
  auto &SVD = StateValueLocations.getStateValueDescriptorOfHookInsertionPoint(
      *HookFuncToMIMap.at(&MF.getFunction()));

  auto &SVS = SVD.StateValueLocation.getSVS();
  if (auto *TwoAGPRSVS = llvm::dyn_cast<TwoAGPRValueStorage>(&SVS)) {
    PhysToVirtRegMap.insert(
        {TwoAGPRSVS->TempAGPR,
         MRI.createVirtualRegister(&llvm::AMDGPU::AGPR_32RegClass)});
  } else if (auto *OneAGPRThreeSGPRSVS =
                 llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(&SVS)) {
    PhysToVirtRegMap.insert(
        {OneAGPRThreeSGPRSVS->FlatScratchSGPRHigh,
         MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass)});
    PhysToVirtRegMap.insert(
        {OneAGPRThreeSGPRSVS->FlatScratchSGPRLow,
         MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass)});
    PhysToVirtRegMap.insert(
        {OneAGPRThreeSGPRSVS->InstrumentationStackPointer,
         MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass)});
  } else if (auto *ThreeSGPRSVS =
                 llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(&SVS)) {
    PhysToVirtRegMap.insert(
        {ThreeSGPRSVS->FlatScratchSGPRHigh,
         MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass)});
    PhysToVirtRegMap.insert(
        {ThreeSGPRSVS->FlatScratchSGPRLow,
         MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass)});
    PhysToVirtRegMap.insert(
        {ThreeSGPRSVS->InstrumentationStackPointer,
         MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass)});
  }

  auto &EntryMBB = *MF.begin();
  for (const auto &[PhysReg, VirtReg] : PhysToVirtRegMap) {
    // If this is a spill slot, it needs to be read from the state value
    // register
    if (ValueRegisterSpillSlots.contains(PhysReg)) {
      if (!EntryMBB.isLiveIn(SVDesc.StateValueVGPR))
        EntryMBB.addLiveIn(SVDesc.StateValueVGPR);
      llvm::BuildMI(EntryMBB, EntryMBB.begin(), llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::V_READLANE_B32), VirtReg)
          .addReg(SVDesc.StateValueVGPR)
          .addImm(ValueRegisterSpillSlots.at(PhysReg));
    } else {
      EntryMBB.addLiveIn(PhysReg);
      llvm::BuildMI(EntryMBB, EntryMBB.begin(), llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::COPY))
          .addReg(VirtReg, llvm::RegState::Define)
          .addReg(PhysReg, llvm::RegState::Kill);
    }
  }

  for (auto &MBB : MF) {
    if (MBB.isReturnBlock()) {
      auto ReturnInst = MBB.getFirstTerminator();
      bool AddedStateValueAsImplicitOp{false};
      for (const auto &[PhysReg, VirtReg] : PhysToVirtRegMap) {
        if (ValueRegisterSpillSlots.contains(PhysReg)) {
          llvm::BuildMI(MBB, ReturnInst, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                        SVDesc.StateValueVGPR)
              .addReg(VirtReg, llvm::RegState::Kill)
              .addImm(ValueRegisterSpillSlots.at(PhysReg))
              .addReg(SVDesc.StateValueVGPR);
          if (!AddedStateValueAsImplicitOp) {
            ReturnInst->addOperand(llvm::MachineOperand::CreateReg(
                ValueRegisterSpillSlots.at(PhysReg), false, true));
            AddedStateValueAsImplicitOp = true;
          }
        } else {
          llvm::BuildMI(MBB, ReturnInst, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::COPY))
              .addReg(PhysReg, llvm::RegState::Define)
              .addReg(VirtReg, llvm::RegState::Kill);
          ReturnInst->addOperand(
              llvm::MachineOperand::CreateReg(PhysReg, false, true));
        }
      }
    }
  }

  llvm::outs() << "Before putting in the accessed physical regs:\n";
  MF.print(llvm::outs());
  llvm::outs() << "Num physical regs accessed by hooks: "
               << AccessedPhysicalRegistersByHook.size() << "\n";
  for (const auto &AccessedReg : AccessedPhysicalRegistersByHook) {
    llvm::outs() << "accessed by hook: " << TRI->getName(AccessedReg) << "\n";
  }

  // For physical registers accessed by Hooks, we have to define the same
  // value and assign it to the values from its predecessor loops, either
  // directly or via the PHI opcode
  llvm::MachineBasicBlock *CurrentMBB{nullptr};
  // Per physical Register SSA Updater.
  // Each SSA updater keeps track of the virtual register that holds
  // the virt reg representing the physical reg in each basic block
  llvm::DenseMap<llvm::MCRegister, std::unique_ptr<llvm::MachineSSAUpdater>>
      PhysRegValueSSAUpdaters;
  std::queue<llvm::MachineBasicBlock *> ToBeVisitedMBBs{{&MF.front()}};
  llvm::DenseSet<llvm::MachineBasicBlock *> VisitedMBBs{};
  while (!ToBeVisitedMBBs.empty()) {
    CurrentMBB = ToBeVisitedMBBs.front();
    llvm::outs() << "Visiting MBB: " << CurrentMBB->getNumber() << "\n";
    if (CurrentMBB->isEntryBlock()) {
      for (const auto AccessedPhysReg : AccessedPhysicalRegistersByHook) {
        auto &SSAUpdater =
            PhysRegValueSSAUpdaters
                .insert(
                    {AccessedPhysReg,
                     std::move(std::make_unique<llvm::MachineSSAUpdater>(MF))})
                .first->getSecond();
        auto VirtReg = MRI.createVirtualRegister(
            TRI->getPhysRegBaseClass(AccessedPhysReg));
        SSAUpdater->Initialize(VirtReg);
        SSAUpdater->AddAvailableValue(CurrentMBB, VirtReg);
        PhysRegLocationPerMBB.insert({{AccessedPhysReg, CurrentMBB}, VirtReg});
        llvm::outs() << "Added " << TRI->getName(AccessedPhysReg) << ","
                     << printReg(VirtReg, TRI) << "\n";
        if (ValueRegisterSpillSlots.contains(AccessedPhysReg)) {
          if (!CurrentMBB->isLiveIn(SVDesc.StateValueVGPR))
            CurrentMBB->addLiveIn(SVDesc.StateValueVGPR);
          llvm::BuildMI(*CurrentMBB, CurrentMBB->begin(), llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_READLANE_B32), VirtReg)
              .addReg(SVDesc.StateValueVGPR)
              .addImm(ValueRegisterSpillSlots.at(AccessedPhysReg));
        } else {
          CurrentMBB->addLiveIn(AccessedPhysReg);
          llvm::BuildMI(*CurrentMBB, CurrentMBB->begin(), llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::COPY))
              .addReg(VirtReg, llvm::RegState::Define)
              .addReg(AccessedPhysReg, llvm::RegState::Kill);
        }
      }
    } else {
      if (MBBsThatNeedAccessToPhysicalRegs.contains(CurrentMBB)) {
        const auto &PhysRegsAccessedByMBB =
            MBBsThatNeedAccessToPhysicalRegs.at(CurrentMBB);
        for (const auto &PhysReg : PhysRegsAccessedByMBB) {
          llvm::Register VirtRegInBlock =
              PhysRegValueSSAUpdaters[PhysReg]->GetValueInMiddleOfBlock(
                  CurrentMBB);
          PhysRegLocationPerMBB.insert({{PhysReg, CurrentMBB}, VirtRegInBlock});
          PhysRegValueSSAUpdaters[PhysReg]->AddAvailableValue(CurrentMBB,
                                                              VirtRegInBlock);
          llvm::outs() << "Added " << TRI->getName(PhysReg) << ","
                       << printReg(VirtRegInBlock, TRI) << "\n";
        }
      }
    }

    if (CurrentMBB->isReturnBlock()) {
      auto ReturnInst = CurrentMBB->getFirstTerminator();
      for (const auto &AccessedPhysReg : AccessedPhysicalRegistersByHook) {
        auto &PhysRegSSAUpdater = PhysRegValueSSAUpdaters[AccessedPhysReg];
        llvm::outs() << "Finishing " << TRI->getName(AccessedPhysReg) << "\n";
        llvm::outs() << "Does it have value available? "
                     << PhysRegSSAUpdater->HasValueForBlock(CurrentMBB) << "\n";
        llvm::Register VirtReg =
            PhysRegSSAUpdater->GetValueAtEndOfBlock(CurrentMBB);
        if (ValueRegisterSpillSlots.contains(AccessedPhysReg)) {
          llvm::BuildMI(*CurrentMBB, ReturnInst, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                        ValueRegisterSpillSlots.at(AccessedPhysReg))
              .addReg(VirtReg, llvm::RegState::Kill)
              .addImm(ValueRegisterSpillSlots.at(AccessedPhysReg));
        } else {
          llvm::BuildMI(*CurrentMBB, ReturnInst, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::COPY))
              .addReg(AccessedPhysReg, llvm::RegState::Define)
              .addReg(VirtReg, llvm::RegState::Kill);
        }
        ReturnInst->addOperand(
            llvm::MachineOperand::CreateReg(AccessedPhysReg, false, true));
      }
    }
    // Add the successor blocks of the current block to the queue
    for (auto It = CurrentMBB->succ_begin(), End = CurrentMBB->succ_end();
         It != End; ++It) {
      if (!VisitedMBBs.contains(*It)) {
        llvm::outs() << "Setting MBB " << (*It)->getNumber()
                     << " to be visited.\n";
        ToBeVisitedMBBs.push(*It);
      }
    }
    // pop the current MBB and add it to the visited list
    VisitedMBBs.insert(CurrentMBB);

    CurrentMBB->print(llvm::outs());
    ToBeVisitedMBBs.pop();
    llvm::outs() << "Queue size: " << ToBeVisitedMBBs.size() << "\n";
  }

  MF.print(llvm::outs());

  return false;
}

void PhysicalRegAccessVirtualizationPass::getAnalysisUsage(
    llvm::AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreservedID(llvm::MachineLoopInfoID);
  llvm::MachineFunctionPass::getAnalysisUsage(AU);
}

static void add32BitLiveRegsToSet(const llvm::LivePhysRegs &LiveRegs,
                                  const llvm::MachineRegisterInfo &MRI,
                                  const llvm::SIRegisterInfo &TRI,
                                  llvm::DenseSet<llvm::MCRegister> &Set) {
  llvm::outs() << "Add 32 bit" << "\n";
  for (const auto &LiveInReg : LiveRegs) {
    llvm::outs() << "checking Reg name: " << TRI.getName(LiveInReg) << "\n";
    auto *PhysRegClass = TRI.getPhysRegBaseClass(LiveInReg);
    if (PhysRegClass == nullptr)
      continue;
    unsigned int RegSize = PhysRegClass->MC->getSizeInBits();
    llvm::outs() << "Size of the reg: " << RegSize << "\n";
    if (RegSize == 32) {
      llvm::outs() << "Reg name: " << TRI.getName(LiveInReg) << "\n";
      Set.insert(LiveInReg);
    }
    if (RegSize == 16) {
      llvm::MCRegister SuperReg32Bit = TRI.get32BitRegister(LiveInReg);
      llvm::outs() << "Reg name: " << TRI.getName(LiveInReg) << "\n";
      llvm::outs() << "Super Reg name: " << TRI.getName(SuperReg32Bit) << "\n";
      if (!LiveRegs.contains(SuperReg32Bit)) {
        llvm::outs() << "16 bit reg: " << TRI.getName(SuperReg32Bit) << "\n";
        Set.insert(SuperReg32Bit);
      }
    }
  }
}

PhysicalRegAccessVirtualizationPass::PhysicalRegAccessVirtualizationPass(
    const LiftedRepresentation &LR,
    const llvm::LivePhysRegs &AccessedPhysicalRegs, const LRCallGraph &CG,
    const LRStateValueLocations &StateValueLocations,
    const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
        &HookFuncToInstPointMI,
    llvm::ArrayRef<std::pair<llvm::Function *, IntrinsicIRLoweringInfo>>
        InlineAsmPlaceHolderToIRLoweringInfoMap,
    const LRRegisterLiveness &RegLiveness)
    : LR(LR), AccessedPhysicalRegs(AccessedPhysicalRegs),
      StateValueLocations(StateValueLocations), RegLiveness(RegLiveness),
      CG(CG), HookFuncToMIMap(HookFuncToInstPointMI),
      InlineAsmPlaceHolderToIRLoweringInfoMap(
          InlineAsmPlaceHolderToIRLoweringInfoMap),
      llvm::MachineFunctionPass(ID) {
  // Figure out the final live regs for each Hook insertion point
  // The final live regs include:
  // 1. The live-in regs calculated during the lifting stage; This is the
  // function-level live-ins
  // 2. If the instrumented function is not a kernel, then the registers live
  // at its call sites are added as well.
  // 3. Registers that the hooks requested read/write access to
  // Note that the final live-in registers must not overlap; In case of
  // overlapping, the two offending registers must be merged together
  for (const auto &[HookFunc, InstPointMI] : HookFuncToInstPointMI) {

    auto *InstPointMBB = InstPointMI->getParent();
    auto *InstPointMF = InstPointMBB->getParent();
    auto &InstPointMRI = InstPointMF->getRegInfo();
    auto &InstPointTRI =
        *InstPointMF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
    auto &HookLiveInRegs =
        PerHookLiveInRegs.insert({HookFunc, {}}).first->getSecond();
    // Get the originally calculated live-in registers at the function level
    // and add them
    auto *MILivePhysRegs =
        RegLiveness.getLiveInPhysRegsOfMachineInstr(*InstPointMI);
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ASSERTION(MILivePhysRegs != nullptr));
    add32BitLiveRegsToSet(*MILivePhysRegs, InstPointMRI, InstPointTRI,
                          HookLiveInRegs);

    // If the MI being instrumented belongs to a device function, then
    // add the Live-in regs for all the call sites that target it
    if (InstPointMF->getFunction().getCallingConv() !=
        llvm::CallingConv::AMDGPU_KERNEL) {
      // If the callgraph is not deterministic, find all call instructions
      // and add their live-ins to hook live regs
      if (CG.hasNonDeterministicCallGraph(
              StateValueLocations.getLCO().asHsaType())) {
        auto LCO = StateValueLocations.getLCO();
        for (const auto &[LRFuncSymbol, LRMF] : LR.functions()) {
          if (LRFuncSymbol->getLoadedCodeObject() == LCO) {
            for (const auto &LRMBB : *LRMF) {
              for (const auto &LRMI : LRMBB) {
                if (LRMI.isCall()) {
                  auto *LRMILivePhysRegs =
                      RegLiveness.getLiveInPhysRegsOfMachineInstr(LRMI);
                  LUTHIER_REPORT_FATAL_ON_ERROR(
                      LUTHIER_ASSERTION(LRMILivePhysRegs != nullptr));
                  add32BitLiveRegsToSet(*LRMILivePhysRegs, InstPointMRI,
                                        InstPointTRI, HookLiveInRegs);
                }
              }
            }
          }
        }
      } else {
        for (const auto &[CallMI, Parent] :
             CG.getCallGraphNode(InstPointMF).CalleeFunctions) {
          auto *CallMILivePhysRegs =
              RegLiveness.getLiveInPhysRegsOfMachineInstr(*CallMI);
          LUTHIER_REPORT_FATAL_ON_ERROR(
              LUTHIER_ASSERTION(CallMILivePhysRegs != nullptr));
          add32BitLiveRegsToSet(*CallMILivePhysRegs, InstPointMRI, InstPointTRI,
                                HookLiveInRegs);
        }
      }
    }
    // Add the physical registers accessed by all the hooks that are not
    // also live-ins
    add32BitLiveRegsToSet(AccessedPhysicalRegs, InstPointMRI, InstPointTRI,
                          HookLiveInRegs);
  }
}

} // namespace luthier