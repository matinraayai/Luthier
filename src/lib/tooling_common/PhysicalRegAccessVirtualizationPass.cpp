//===-- PhysicalRegAccessVirtualizationPass.cpp ---------------------------===//
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
/// This file implements Luthier's Physical Reg Access Virtualization Pass.
//===----------------------------------------------------------------------===//
#include "tooling_common/PhysicalRegAccessVirtualizationPass.hpp"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "tooling_common/PhysRegsNotInLiveInsAnalysis.hpp"
#include "tooling_common/StateValueArraySpecs.hpp"
#include "tooling_common/WrapperAnalysisPasses.hpp"
#include <GCNSubtarget.h>
#include <SIRegisterInfo.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineSSAUpdater.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <queue>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-phys-reg-virtualization"

namespace luthier {

char PhysicalRegAccessVirtualizationPass::ID = 0;

static llvm::RegisterPass<PhysicalRegAccessVirtualizationPass>
    X("phys-virtualization", "Physical Register Access Virtualization Pass",
      true /* Only looks at CFG */, false /* Analysis Pass */);

static void
get32BitSubRegsOfPhysReg(llvm::MCRegister Register,
                         const llvm::SIRegisterInfo &TRI,
                         llvm::SmallVectorImpl<llvm::MCRegister> &SubRegs) {
  LLVM_DEBUG(
      llvm::dbgs() << "Calculating the 32-bit sub-regs of physical register "
                   << llvm::printReg(Register, &TRI) << ".";);

  // Get the register's class
  auto *PhysRegClass = TRI.getPhysRegBaseClass(Register);
  // If no reg class is found for the register, return nothing
  if (PhysRegClass == nullptr) {
    LLVM_DEBUG(llvm::dbgs() << "No reg class was found for register "
                            << llvm::printReg(Register, &TRI)
                            << "; returning nothing.\n";);
    return;
  }

  // Get the register's size
  unsigned int RegSize = PhysRegClass->MC->getSizeInBits();

  LLVM_DEBUG(llvm::dbgs() << "Size of the live register: " << RegSize
                          << ".\n";);

  if (RegSize == 32) {
    LLVM_DEBUG(llvm::dbgs() << "Adding reg " << llvm::printReg(Register, &TRI)
                            << " to the sub regs.\n";);
    SubRegs.push_back(Register);
  } else if (RegSize == 16) {
    // Get the 32 bit super register of the register
    llvm::MCRegister SuperReg32Bit = TRI.get32BitRegister(Register);
    LLVM_DEBUG(llvm::dbgs()
                   << "Adding reg " << llvm::printReg(SuperReg32Bit, &TRI)
                   << " to the sub regs.\n";);
    SubRegs.push_back(SuperReg32Bit);
    // Add 1-bit regs (e.g. SCC)
  } else if (RegSize == 1) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Adding 1-bit register " << llvm::printReg(Register, &TRI)
                   << " to the sub regs.\n";);
  }
  // handle the reg size > 32 case by splitting the register into smaller 32-bit
  // chunks
  else {
    LLVM_DEBUG(llvm::dbgs()
                   << "Register needs to be split into 32-bit sub-regs.\n";);

    size_t NumChannels = RegSize / 32;
    for (int i = 0; i < NumChannels; i++) {
      auto SubReg = TRI.getSubReg(
          Register, llvm::SIRegisterInfo::getSubRegFromChannel(i));

      LLVM_DEBUG(llvm::dbgs()
                     << "Adding sub register " << llvm::printReg(SubReg, &TRI)
                     << " to the sub regs.\n";);
      SubRegs.push_back(SubReg);
    }
  }
}

/// Adds the 32-bit registers clobbered by the \p LiveRegs set into \p Set
/// 16-bit registers in \p LiveRegs will have their 32-bit super-reg inserted,
/// Any register larger than 32-bits will have its 32-bit sub registers
/// inserted into the \p Set
/// Special 1-bit registers (e.g. SCC) are an exception to this rule, and
/// will be added directly to the set
/// \param [in] LiveRegs the set of live physical registers
/// \param [in] TRI the register info of the target
/// \param [out] Set the final set the 32-bit physical registers will be added
/// to
static void
add32BitRegsOfLivePhysRegsToDenseSet(const llvm::LivePhysRegs &LiveRegs,
                                     const llvm::SIRegisterInfo &TRI,
                                     llvm::DenseSet<llvm::MCRegister> &Set) {
  for (const auto &LiveReg : LiveRegs) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Inspecting " << llvm::printReg(LiveReg, &TRI)
                   << " to be added to the set of 32-bit live registers.\n";);

    // Skip the Exec mask, because it is preserved automatically by the function
    // call
    if (TRI.regsOverlap(LiveReg, llvm::AMDGPU::EXEC)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping execute mask register.\n");
      continue;
    }

    // Get the live register's class and its size
    auto *PhysRegClass = TRI.getPhysRegBaseClass(LiveReg);
    if (PhysRegClass == nullptr) {
      LLVM_DEBUG(llvm::dbgs()
                     << "No reg class was found for register "
                     << llvm::printReg(LiveReg, &TRI) << "; Skipping it.\n";);
      continue;
    }

    // TODO: Remove this once it is confirmed we don't need to manually
    // specify this all the time
    //    if (LiveReg == llvm::AMDGPU::EXEC)
    //      continue;
    unsigned int RegSize = PhysRegClass->MC->getSizeInBits();

    LLVM_DEBUG(llvm::dbgs()
                   << "Size of the live register: " << RegSize << "\n";);

    if (RegSize == 32) {
      LLVM_DEBUG(llvm::dbgs() << "Adding reg " << llvm::printReg(LiveReg, &TRI)
                              << "\n";);
      Set.insert(LiveReg);
    }
    if (RegSize == 16) {
      // Get the 32 bit super register of the live register
      llvm::MCRegister SuperReg32Bit = TRI.get32BitRegister(LiveReg);

      LLVM_DEBUG(llvm::dbgs() << "Looking to see if the 32-bit super register "
                              << llvm::printReg(LiveReg, &TRI)
                              << " has already been added to the set.\n";);
      if (!LiveRegs.contains(SuperReg32Bit)) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Adding 32-bit super register "
                       << llvm::printReg(LiveReg, &TRI) << " to the set.\n";);
        Set.insert(SuperReg32Bit);
      }
    }
    // Add 1-bit regs (e.g. SCC)
    else if (RegSize == 1) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Adding 1-bit register "
                     << llvm::printReg(LiveReg, &TRI) << " to the set.\n";);
      Set.insert(LiveReg);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Register is too big to be added to the live "
                                 "set; Skipping.\n";);
    }
  }
}

PhysicalRegAccessVirtualizationPass::PhysicalRegAccessVirtualizationPass()
    : llvm::MachineFunctionPass(ID) {}

bool PhysicalRegAccessVirtualizationPass::runOnMachineFunction(
    llvm::MachineFunction &MF) {
  // Here, we finalize the set of live physical regs before each instrumentation
  // point that needs to be preserved.
  // The final live regs set include:
  // 1. The live-in regs calculated by the LRRegisterLiveness analysis stage;
  // This is the function-level live-ins.
  // 2. If the instrumented function is not a kernel, the set of registers live
  // at its call sites are added in as well. In case the LRCallGraph
  // analysis fails, all registers that are live before all call sites are
  // added.
  // 3. A set of physical registers that the injected payloads requested access
  // to, but (at least in one injected payload) were not in the live-ins set of
  // the instrumentation point at the function-level (item 1). This
  // is to prevent the dead register's value from being used as a spare register
  // in hooks, and therefore preserving its original execution value.
  // For example, an injected payload asks for access to register s1, but it
  // is dead at its instrumentation point. S1 is also dead in the
  // instrumentation point right before the current one. If nothing is done
  // to preserve s1's value in the previous instrumentation point, the next
  // injected payload will read a garbage value.
  // The same case can be argued for writing to s1 if there's an injected
  // payload right after the current instrumentation point, and s1 is dead
  // there as well.
  //
  // Note that LLVM's LivePhysReg set is not used here, and registers are
  // instead stored in a 32-bit wide granularity; For example, if s[4:5] is
  // live only s4 and s5 are saved into the live-in set. This is to prevent
  // overlap between registers so that no additional pseudo instructions
  // would be required to express sub/super register relation and to update
  // sub/super registers when it is written to

  LLVM_DEBUG(
      llvm::dbgs()
          << "Physical Register Access Virtualization Pass's constructor; "
             "Gathering all 32-bit live-in sub registers of each "
             "instrumentation points that need to be preserved.\n";);

  auto &IModule = const_cast<llvm::Module &>(
      *getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI().getModule());

  auto &IMAM = getAnalysis<IModuleMAMWrapperPass>().getMAM();

  auto &TargetMAndModule =
      IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);

  auto &TargetModule = TargetMAndModule.getTargetAppModule();

  auto &TargetMAM = TargetMAndModule.getTargetAppMAM();

  auto &IPIP =
      *IMAM.getCachedResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  auto &RegLiveness = TargetMAM.getResult<LRRegLivenessAnalysis>(TargetModule);

  const auto &CG = TargetMAM.getResult<LRCallGraphAnalysis>(TargetModule);

  const auto &StateValueLocations =
      TargetMAM.getResult<LRStateValueStorageAndLoadLocationsAnalysis>(
          TargetModule);

  auto *SVALoadPlan =
      StateValueLocations.getStateValueArrayLoadPlanForInstPoint(
          *IPIP.at(MF.getFunction()));
  if (!SVALoadPlan) {
    MF.getContext().reportError(
        {},
        llvm::formatv(
            "Could not find the state value load plan for Machine Instr {0}.",
            *IPIP.at(MF.getFunction())));
    return false;
  }

  // If the function being processed is not an injected payload (i.e a device
  // function getting called inside a hook) it cannot access physical registers
  // (it should have been checked by the code generator), so skip it
  if (!MF.getFunction().hasFnAttribute(LUTHIER_HOOK_ATTRIBUTE) &&
      !MF.getFunction().hasFnAttribute(LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE)) {

    // Add the state value array's load VGPR as a live-in for all basic blocks
    // to be preserved throughout the injected payload
    for (auto &MBB : MF) {
      if (!MBB.isLiveIn(SVALoadPlan->StateValueArrayLoadVGPR))
        MBB.addLiveIn(SVALoadPlan->StateValueArrayLoadVGPR);
    }

    for (auto &MBB : MF) {
      if (MBB.isReturnBlock()) {
        // The first return instruction is where we emit the copy instruction
        auto ReturnInst = MBB.getFirstTerminator();
        // Add the state value array load VGPR as an implicit use for the
        // terminator instruction. This is because without a use the register
        // allocator will not preserve the state value VGPR
        ReturnInst->addOperand(llvm::MachineOperand::CreateReg(
            SVALoadPlan->StateValueArrayLoadVGPR, false, true));
      }
    }

    return true;
  }

  const auto &AccessedPhysicalRegs =
      IMAM.getResult<PhysRegsNotInLiveInsAnalysis>(IModule)
          .getPhysRegsNotInLiveIns();

  auto InstPointMI = IPIP.at(MF.getFunction());

  auto *InstPointMBB = InstPointMI->getParent();
  auto *InstPointMF = InstPointMBB->getParent();
  auto &InstPointMRI = InstPointMF->getRegInfo();
  auto &InstPointTRI =
      *InstPointMF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
  // Get the originally calculated live-in registers at the function level
  // and add them to the final set
  auto *MILivePhysRegs =
      RegLiveness.getLiveInPhysRegsOfMachineInstr(*InstPointMI);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      MILivePhysRegs != nullptr,
      "Failed to find the physical live-in regs for machine instruction {0} "
      "inside the lifted representation.",
      *InstPointMI));

  LLVM_DEBUG(
      llvm::dbgs()
          << "Adding the function level live-ins of instrumentation point ";
      InstPointMI->print(llvm::dbgs(), true, false, false, false);
      llvm::dbgs() << ".\n";);

  add32BitRegsOfLivePhysRegsToDenseSet(*MILivePhysRegs, InstPointTRI,
                                       PhysicalLiveInsForInjectedPayload);

  // If the MI being instrumented belongs to a device function, then
  // add the Live-in regs for all the call sites that potentially target it
  if (InstPointMF->getFunction().getCallingConv() !=
      llvm::CallingConv::AMDGPU_KERNEL) {

    LLVM_DEBUG(llvm::dbgs()
                   << "Instrumentation point is inside a non-kernel "
                      "function; Adding its call point live-ins as well.\n";);

    // If the callgraph is not deterministic, find all call instructions
    // and add their live-ins to hook live regs
    if (CG.hasNonDeterministicCallGraph()) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "Lifted representation doesn't have a deterministic call "
                 "graph; Adding the live-ins of all call instructions.\n";);
      auto &TargetMMI =
          TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetModule)
              .getMMI();

      for (const auto &TargetF : TargetModule) {
        if (auto *TargetMF = TargetMMI.getMachineFunction(TargetF)) {
          for (const auto &LRMBB : *TargetMF) {
            for (const auto &LRMI : LRMBB) {
              if (LRMI.isCall()) {
                auto *LRMILivePhysRegs =
                    RegLiveness.getLiveInPhysRegsOfMachineInstr(LRMI);
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    LUTHIER_ERROR_CHECK(LRMILivePhysRegs != nullptr,
                                        "Failed to find the physical live-in "
                                        "regs for machine instruction {0} "
                                        "inside the lifted representation.",
                                        LRMI));
                add32BitRegsOfLivePhysRegsToDenseSet(
                    *LRMILivePhysRegs, InstPointTRI,
                    PhysicalLiveInsForInjectedPayload);
              }
            }
          }
        }
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                     << "Lifted representation has a deterministic call graph; "
                        "Adding the call instruction live-ins.\n";);
      for (const auto &[CallMI, Parent] :
           CG.getCallGraphNode(InstPointMF).CalleeFunctions) {
        auto *CallMILivePhysRegs =
            RegLiveness.getLiveInPhysRegsOfMachineInstr(*CallMI);
        LUTHIER_REPORT_FATAL_ON_ERROR(
            LUTHIER_ERROR_CHECK(CallMILivePhysRegs != nullptr, ""));
        add32BitRegsOfLivePhysRegsToDenseSet(*CallMILivePhysRegs, InstPointTRI,
                                             PhysicalLiveInsForInjectedPayload);
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Adding physical registers accessed by hooks "
                             "that are not in the live-ins.\n");
  // Add the physical registers accessed by all the hooks that are not
  // also live-ins
  add32BitRegsOfLivePhysRegsToDenseSet(AccessedPhysicalRegs, InstPointTRI,
                                       PhysicalLiveInsForInjectedPayload);

  bool Changed{false};
  auto *TRI = MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
  auto &MRI = MF.getRegInfo();

  // A mapping between basic blocks inside the functions that access physical
  // registers, and the set of physical registers they access
  llvm::SmallDenseMap<const llvm::MachineBasicBlock *,
                      llvm::SmallDenseSet<llvm::MCRegister, 4>, 4>
      MBBsToAccessedToPhysicalRegs;
  // The set of all physical registers accessed by the current MF being
  // processed
  llvm::SmallDenseSet<llvm::MCRegister, 4> AllPhysRegsAccessedByAllIntrinsics;

  const auto &InlineAsmPlaceHolderToIRLoweringInfoMap =
      IMAM.getCachedResult<IntrinsicIRLoweringInfoMapAnalysis>(IModule)
          ->getLoweringInfo();

  // Iterate over the MIs in the Hook function, find placeholder inline asms,
  // and get their lowering info
  // Aggregate all the accessed physical registers in the injected payload
  // in one place
  // Also create a mapping between each MBB and the set of physical registers
  // it will access
  for (const auto &MBB : MF) {
    for (const auto &MI : MBB) {
      if (MI.isInlineAsm()) {
        llvm::Expected<unsigned int> IntrinsicIdx =
            getIntrinsicInlineAsmPlaceHolderIdx(MI);
        if (auto Err = IntrinsicIdx.takeError()) {
          MF.getContext().reportError({}, llvm::toString(std::move(Err)));
          return false;
        }
        if (*IntrinsicIdx == -1)
          continue;

        LLVM_DEBUG(llvm::dbgs()
                       << "Virtualizing physical registers for intrinsic ID "
                       << *IntrinsicIdx << "\n";);

        auto &LoweringInfo =
            InlineAsmPlaceHolderToIRLoweringInfoMap[*IntrinsicIdx];
        for (const auto &AccessedPhysReg : LoweringInfo.accessed_phys_regs()) {

          LLVM_DEBUG(llvm::dbgs()
                         << "Intrinsic requested access to physical register "
                         << llvm::printReg(AccessedPhysReg, TRI) << "\n";);

          if (!MBBsToAccessedToPhysicalRegs.contains(&MBB))
            MBBsToAccessedToPhysicalRegs.insert({&MBB, {}});
          llvm::SmallVector<llvm::MCRegister, 4> AccessedPhysicalSubRegs;

          get32BitSubRegsOfPhysReg(AccessedPhysReg, *TRI,
                                   AccessedPhysicalSubRegs);

          LLVM_DEBUG(llvm::dbgs() << "Accessed sub regs by intrinsic "
                                  << *IntrinsicIdx << ":\n");

          for (const auto &AccessedSubReg : AccessedPhysicalSubRegs) {
            LLVM_DEBUG(llvm::dbgs()
                           << "    " << llvm::printReg(AccessedPhysReg, TRI)
                           << "\n";);
            MBBsToAccessedToPhysicalRegs[&MBB].insert(AccessedSubReg);
            AllPhysRegsAccessedByAllIntrinsics.insert(AccessedSubReg);
          };
        }
      }
    }
  }
  // Now that we have all the accessed physical registers in one place,
  // and know which MBBs need access to what physical registers,
  // we start inserting moves to virtual registers, and materializing access
  // to them when
  auto *TII = MF.getSubtarget().getInstrInfo();

  // For each live-in register that is not used in the hooks, create a copy
  // to its equivalent virtual register in the entry basic block, and
  // a copy back in all the return blocks
  // TODO: Map to the correct location of clobbered registers in case
  // the state value is not in a VGPR
  llvm::DenseMap<llvm::MCRegister, llvm::Register>
      PreservedPhysRegToVirtRegStorageMap;
  for (const auto &LiveIn : PhysicalLiveInsForInjectedPayload) {
    if (!AllPhysRegsAccessedByAllIntrinsics.contains(LiveIn) &&
        !stateValueArray::isFrameSpillSlot(LiveIn)) {

      LLVM_DEBUG(llvm::dbgs()
                 << "Live-in register " << llvm::printReg(LiveIn, TRI)
                 << "is not accessed by the intrinsics nor is in the state "
                    "value array spill slot.\n");

      auto *PhysRegClass = TRI->getPhysRegBaseClass(LiveIn);
      // If this is an allocatable register (e.g. SGPR, VGPR, AGPR), then
      // create a new virtual register for it with the same class
      if (TRI->isInAllocatableClass(LiveIn)) {
        llvm::Register CopyVirtualRegister =
            MRI.createVirtualRegister(PhysRegClass);

        LLVM_DEBUG(llvm::dbgs()
                       << "The live-in register " << llvm::printReg(LiveIn, TRI)
                       << " is in an allocatable class, and virtual register "
                       << llvm::printReg(CopyVirtualRegister, TRI)
                       << " is created for its spill.\n";);

        PreservedPhysRegToVirtRegStorageMap.insert(
            {LiveIn, CopyVirtualRegister});
      } else {
        // If this is not an allocatable register, Spill to an SGPR with
        // appropriate size
        auto *CopyClass = TRI->getCrossCopyRegClass(PhysRegClass);
        if (!CopyClass) {
          MF.getContext().reportError(
              {}, llvm::formatv("Failed to get a copy class for physical "
                                "register {0}, in register class {1}.",
                                llvm::printReg(LiveIn, TRI),
                                TRI->getRegClassName(PhysRegClass)));
          return false;
        }

        llvm::Register CopyRegister = MRI.createVirtualRegister(CopyClass);

        LLVM_DEBUG(llvm::dbgs()
                       << "The live-in register " << llvm::printReg(LiveIn, TRI)
                       << " is not an allocatable class. Copy register class "
                       << TRI->getRegClassName(CopyClass)
                       << " was selected and virtual register "
                       << llvm::printReg(CopyRegister, TRI)
                       << " is created for its spill.\n";);

        PreservedPhysRegToVirtRegStorageMap.insert({LiveIn, CopyRegister});
      }
    }
  }

  // Add the state value array's load VGPR as a live-in for all basic blocks
  // to be preserved throughout the injected payload
  for (auto &MBB : MF) {
    if (!MBB.isLiveIn(SVALoadPlan->StateValueArrayLoadVGPR))
      MBB.addLiveIn(SVALoadPlan->StateValueArrayLoadVGPR);
  }

  // We now emit the copy instructions from where the preserved
  // physical registers to their virtual registers they will be stored in
  // Physical registers. A copy from phys reg to virt reg is inserted in the
  // beginning of the entry BB and a copy from virt reg to phys reg is inserted
  // at end of the return blocks
  auto &EntryMBB = *MF.begin();
  // Keep a reference to the first instruction of the entry basic
  auto &EntryInst = *EntryMBB.begin();

  LLVM_DEBUG(llvm::dbgs() << "Emitting copy from physical register to virt "
                             "register storage instructions...\n");

  for (const auto &[PreservedPhysReg, PreservedRegVirtRegStorage] :
       PreservedPhysRegToVirtRegStorageMap) {
    LLVM_DEBUG(llvm::dbgs() << "Handing copy instruction for "
                            << llvm::printReg(PreservedPhysReg, TRI) << "\n";);
    // If this is a spill slot, it needs to be read from the state value
    // register; The prologue/epilogue inserter will then emit a copy
    // from the actual physical register to the appropriate spill lane of the
    // state value array
    if (stateValueArray::isFrameSpillSlot(PreservedPhysReg)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Register " << llvm::printReg(PreservedPhysReg, TRI)
                     << " comes from the state value array spill slot.\n";);
      auto Builder = llvm::BuildMI(EntryMBB, EntryInst, llvm::DebugLoc(),
                                   TII->get(llvm::AMDGPU::V_READLANE_B32),
                                   PreservedRegVirtRegStorage)
                         .addReg(SVALoadPlan->StateValueArrayLoadVGPR)
                         .addImm(stateValueArray::getFrameSpillSlotLaneId(
                             PreservedPhysReg));
      LLVM_DEBUG(llvm::dbgs() << "Adding phys to reg copy instruction "
                              << Builder << "\n";);
    } else {
      // Add the physical register as a live-in to the entry basic block
      LLVM_DEBUG(llvm::dbgs()
                 << "Adding the register to the live-ins of the entry MBB.\n");
      EntryMBB.addLiveIn(PreservedPhysReg);
      auto Builder =
          llvm::BuildMI(EntryMBB, EntryInst, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::COPY))
              .addReg(PreservedRegVirtRegStorage, llvm::RegState::Define)
              .addReg(PreservedPhysReg, llvm::RegState::Kill);
      LLVM_DEBUG(llvm::dbgs() << "Adding phys to reg copy instruction "
                              << Builder << "\n";);
    }
    Changed |= true;
  }

  LLVM_DEBUG(llvm::dbgs()
             << "Entry MBB copies are done.\n"
                "Emitting copy instructions from virtual register storage "
                "to the physical registers they are preserving.\n");

  // We now emit the copy instructions from where the virt registers to their
  // original preserved physical register at the end of each return block
  for (auto &MBB : MF) {
    if (MBB.isReturnBlock()) {
      // The first return instruction is where we emit the copy instruction
      auto ReturnInst = MBB.getFirstTerminator();
      // Add the state value array load VGPR as an implicit use for the
      // terminator instruction. This is because without a use the register
      // allocator will not preserve the state value VGPR
      ReturnInst->addOperand(llvm::MachineOperand::CreateReg(
          SVALoadPlan->StateValueArrayLoadVGPR, false, true));
      for (const auto &[PreservedPhysReg, PhysRegVirtStorage] :
           PreservedPhysRegToVirtRegStorageMap) {
        if (stateValueArray::isFrameSpillSlot(PreservedPhysReg)) {
          // If the preserved reg is in the spill slots, then emit a write lane
          // instruction to put it back in the state value array
          LLVM_DEBUG(llvm::dbgs()
                         << "Register " << llvm::printReg(PreservedPhysReg, TRI)
                         << " comes from the state value array spill slot.\n";);
          auto Builder = llvm::BuildMI(MBB, ReturnInst, llvm::DebugLoc(),
                                       TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                                       SVALoadPlan->StateValueArrayLoadVGPR)
                             .addReg(PhysRegVirtStorage, llvm::RegState::Kill)
                             .addImm(stateValueArray::getFrameSpillSlotLaneId(
                                 PreservedPhysReg))
                             .addReg(SVALoadPlan->StateValueArrayLoadVGPR);
          LLVM_DEBUG(llvm::dbgs() << "Adding reg to phys copy instruction "
                                  << Builder << "\n";);
          // Add the preserved physical reg to the list of implicit operands
          // of the return instruction
          ReturnInst->addOperand(
              llvm::MachineOperand::CreateReg(PreservedPhysReg, false, true));
        } else {
          auto Builder = llvm::BuildMI(MBB, ReturnInst, llvm::DebugLoc(),
                                       TII->get(llvm::AMDGPU::COPY))
                             .addReg(PreservedPhysReg, llvm::RegState::Define)
                             .addReg(PhysRegVirtStorage, llvm::RegState::Kill);
          LLVM_DEBUG(llvm::dbgs() << "Adding reg to phys copy instruction "
                                  << Builder << "\n";);
          // Add the preserved physical reg to the list of implicit operands
          // of the return instruction
          ReturnInst->addOperand(
              llvm::MachineOperand::CreateReg(PreservedPhysReg, false, true));
        }
        Changed |= true;
      }
    }
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Emission of instructions for preserving un-accessed "
                      "live physical registers is complete.\n";

      llvm::dbgs() << "The machine function before ensuring access to physical "
                      "registers requested by intrinsics:\n";

      MF.print(llvm::dbgs()););

  LLVM_DEBUG(
      llvm::dbgs()
          << "Ensuring access to physical registers required by intrinsics.\n";
      llvm::dbgs() << "Number of physical 32-bit registers accessed by this "
                      "injected payload: "
                   << AllPhysRegsAccessedByAllIntrinsics.size() << "\n";);
  // Now we ensure access to physical registers required by intrinsics by
  // creating an SSAUpdater that keeps track of virtual register values that
  // store a single physical register in each basic block
  // We perform a depth-first traversal of the basic blocks:
  // - For entry/return blocks, we insert the same copy instructions we put in
  // for physical registers that only needed to be preserved
  // - If an MBB requires access to a physical register, we create a new virtual
  // value for it and add it to the SSA Updater; After we're done, we
  // materialize instructions that would make copies/phi nodes between all the
  // virtual registers in each basic block, connecting them together, and
  // ensuring the SSA form of the MF is preserved
  // Per physical Register SSA Updater.
  // Each SSA updater keeps track of the virtual register that holds
  // the virt reg representing the physical reg in each basic block that
  // needs it
  llvm::DenseMap<llvm::MCRegister, std::unique_ptr<llvm::MachineSSAUpdater>>
      PhysRegValueSSAUpdaters;
  // Queue that will be populated with the current MBB's successors
  std::queue<llvm::MachineBasicBlock *> ToBeVisitedMBBs{{&MF.front()}};
  // Set of MBBs that have been visited
  llvm::DenseSet<llvm::MachineBasicBlock *> VisitedMBBs{};
  while (!ToBeVisitedMBBs.empty()) {
    // Visit the MBB at the front of the queue
    llvm::MachineBasicBlock *CurrentMBB = ToBeVisitedMBBs.front();

    LLVM_DEBUG(llvm::dbgs()
                   << "Visiting MBB: " << CurrentMBB->getNumber() << "\n";);

    if (CurrentMBB->isEntryBlock()) {

      LLVM_DEBUG(llvm::dbgs() << "MBB being visited is an entry block; "
                                 "Creating SSA Updaters.\n";);

      for (const auto AccessedPhysReg : AllPhysRegsAccessedByAllIntrinsics) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Creating SSA updator for accessed physical register: "
                   << llvm::printReg(AccessedPhysReg, TRI) << ".\n");
        (void)PhysRegValueSSAUpdaters.insert(
            {AccessedPhysReg,
             std::move(std::make_unique<llvm::MachineSSAUpdater>(MF))});
        auto &SSAUpdater = PhysRegValueSSAUpdaters[AccessedPhysReg];

        // Create a virtual register for the physical register being accessed
        auto VirtReg = MRI.createVirtualRegister(
            TRI->getPhysRegBaseClass(AccessedPhysReg));

        LLVM_DEBUG(llvm::dbgs()
                       << "Created virtual register "
                       << llvm::printReg(VirtReg, TRI)
                       << " in entry block for physical register "
                       << llvm::printReg(AccessedPhysReg, TRI) << ".\n";);
        // Initialize and add the virtual register to the ssa updater
        SSAUpdater->Initialize(VirtReg);
        SSAUpdater->AddAvailableValue(CurrentMBB, VirtReg);
        // Add the virtual register as the location of the physical register
        // in the entry MBB
        PhysRegLocationPerMBB.insert({{AccessedPhysReg, CurrentMBB}, VirtReg});
        // If the register has to be loaded from the spill slot in the state
        // value array, create a read lane instruction for it
        if (stateValueArray::isFrameSpillSlot(AccessedPhysReg)) {
          LLVM_DEBUG(llvm::dbgs() << "The accessed physical register is in the "
                                     "state value array spill slot.\n";);
          auto Builder =
              llvm::BuildMI(*CurrentMBB, CurrentMBB->begin(), llvm::DebugLoc(),
                            TII->get(llvm::AMDGPU::V_READLANE_B32), VirtReg)
                  .addReg(SVALoadPlan->StateValueArrayLoadVGPR)
                  .addImm(stateValueArray::getFrameSpillSlotLaneId(AccessedPhysReg));
          LLVM_DEBUG(llvm::dbgs() << "Adding phys to reg copy instruction "
                                  << Builder << "\n";);
        } else {
          // Add the physical register to the live-ins of the current block
          CurrentMBB->addLiveIn(AccessedPhysReg);
          // Create a copy instruction from the physical register to the
          // virtual register
          auto Builder =
              llvm::BuildMI(*CurrentMBB, CurrentMBB->begin(), llvm::DebugLoc(),
                            TII->get(llvm::AMDGPU::COPY))
                  .addReg(VirtReg, llvm::RegState::Define)
                  .addReg(AccessedPhysReg, llvm::RegState::Kill);
          LLVM_DEBUG(llvm::dbgs() << "Adding phys to reg copy instruction "
                                  << Builder << "\n";);
        }
        Changed |= true;
      }
    } else if (MBBsToAccessedToPhysicalRegs.contains(CurrentMBB)) {
      // if the visited block is not an entry block, and it requires access to a
      // physical register, then we need to materialize a new virtual register
      // for each physical register it needs to access
      const auto &PhysRegsAccessedByMBB =
          MBBsToAccessedToPhysicalRegs.at(CurrentMBB);
      for (const auto &PhysReg : PhysRegsAccessedByMBB) {
        llvm::Register VirtRegInCurrentMBB =
            PhysRegValueSSAUpdaters[PhysReg]->GetValueInMiddleOfBlock(
                CurrentMBB);
        PhysRegLocationPerMBB.insert(
            {{PhysReg, CurrentMBB}, VirtRegInCurrentMBB});
        PhysRegValueSSAUpdaters[PhysReg]->AddAvailableValue(
            CurrentMBB, VirtRegInCurrentMBB);

        LLVM_DEBUG(
            llvm::dbgs()
                << "Added " << llvm::printReg(VirtRegInCurrentMBB, TRI)
                << " as the virtual register storing the value of physical "
                   "register "
                << llvm::printReg(PhysReg, TRI) << ".\n";);
        Changed |= true;
      }
    }
    if (CurrentMBB->isReturnBlock()) {
      // If the visited block is (also) a return block, then we need to
      // materialize a virtual register at the end of it for the physical
      // register and copy it back to where it originally was
      LLVM_DEBUG(llvm::dbgs()
                     << "Visited MBB is a return block; Emitting copy "
                        "instructions from virtual to phys register.\n";);

      auto ReturnInst = CurrentMBB->getFirstTerminator();
      for (const auto &AccessedPhysReg : AllPhysRegsAccessedByAllIntrinsics) {
        auto &PhysRegSSAUpdater = PhysRegValueSSAUpdaters[AccessedPhysReg];
        llvm::Register VirtReg =
            PhysRegSSAUpdater->GetValueAtEndOfBlock(CurrentMBB);

        LLVM_DEBUG(llvm::dbgs()
                       << "Materialized virtual register "
                       << llvm::printReg(VirtReg, TRI)
                       << " as the final storage for the physical register "
                       << llvm::printReg(AccessedPhysReg, TRI) << ".\n";);

        if (stateValueArray::isFrameSpillSlot(AccessedPhysReg)) {
          LLVM_DEBUG(llvm::dbgs()
                         << "The physical register has to be stored back "
                            "in the state value array.\n";);
          auto Builder =
              llvm::BuildMI(*CurrentMBB, ReturnInst, llvm::DebugLoc(),
                            TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                            SVALoadPlan->StateValueArrayLoadVGPR)
                  .addReg(VirtReg, llvm::RegState::Kill)
                  .addImm(stateValueArray::getFrameSpillSlotLaneId(AccessedPhysReg))
                  .addReg(SVALoadPlan->StateValueArrayLoadVGPR);
          LLVM_DEBUG(llvm::dbgs()
                         << "Adding virt reg to phys reg copy instruction "
                         << Builder << "\n";);
        } else {
          auto Builder =
              llvm::BuildMI(*CurrentMBB, ReturnInst, llvm::DebugLoc(),
                            TII->get(llvm::AMDGPU::COPY))
                  .addReg(AccessedPhysReg, llvm::RegState::Define)
                  .addReg(VirtReg, llvm::RegState::Kill);
          LLVM_DEBUG(llvm::dbgs()
                         << "Adding virt reg to phys reg copy instruction "
                         << Builder << "\n";);
        }
        // Add the physical register as an implicit operand to the return
        // instruction to ensure presence of a use
        ReturnInst->addOperand(
            llvm::MachineOperand::CreateReg(AccessedPhysReg, false, true));
        Changed |= true;
      }
    }
    // Add the successor blocks of the current block to the queue
    for (auto It = CurrentMBB->succ_begin(), End = CurrentMBB->succ_end();
         It != End; ++It) {
      if (!VisitedMBBs.contains(*It)) {
        LLVM_DEBUG(llvm::dbgs() << "Adding MBB " << (*It)->getNumber()
                                << " to be visited.\n";);
        ToBeVisitedMBBs.push(*It);
      }
    }
    // pop the current MBB and add it to the visited list
    VisitedMBBs.insert(CurrentMBB);

    LLVM_DEBUG(llvm::dbgs()
                   << "MBB " << CurrentMBB->getNumber() << " after visit:\n";
               CurrentMBB->print(llvm::dbgs()););

    ToBeVisitedMBBs.pop();

    LLVM_DEBUG(llvm::dbgs() << "Number of MBBs in the queue to be visited: "
                            << ToBeVisitedMBBs.size() << "\n";);
  }

  LLVM_DEBUG(llvm::dbgs() << "Machine function's final form after virtualizing "
                             "physical registers:\n";
             MF.print(llvm::dbgs()););

  return Changed;
}

void PhysicalRegAccessVirtualizationPass::getAnalysisUsage(
    llvm::AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<IModuleMAMWrapperPass>();
  AU.addPreservedID(llvm::MachineLoopInfoID);
  llvm::MachineFunctionPass::getAnalysisUsage(AU);
}

llvm::Register PhysicalRegAccessVirtualizationPass::getMCRegLocationInMBB(
    llvm::MCRegister PhysReg, const llvm::MachineBasicBlock &MBB) const {
  LLVM_DEBUG(const auto &TRI =
                 MBB.getParent()->getSubtarget().getRegisterInfo();
             llvm::dbgs() << "Requesting access to physical register "
                          << llvm::printReg(PhysReg, TRI) << " in MBB "
                          << MBB.getName() << "\n";);
  auto It = PhysRegLocationPerMBB.find({PhysReg, &MBB});
  if (It == PhysRegLocationPerMBB.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to find the register, returning 0.\n";);
    return {};
  } else {
    LLVM_DEBUG(const auto &TRI =
                   MBB.getParent()->getSubtarget().getRegisterInfo();
               llvm::dbgs()
               << "Found register " << llvm::printReg(It->second, TRI)
               << " for physical register " << llvm::printReg(PhysReg, TRI)
               << " in MBB " << MBB.getFullName() << "\n";);
    return It->second;
  }
}

} // namespace luthier