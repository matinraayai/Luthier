

void PhysicalRegAccessVirtualizationPass::emitHookPrologue(
    llvm::MachineFunction &MF,
    llvm::SmallDenseSet<llvm::MCRegister, 4> &AccessedPhysicalRegisters) const {
  auto &SubTarget = MF.getSubtarget<llvm::GCNSubtarget>();
  auto *TII = SubTarget.getInstrInfo();
  auto *TRI = SubTarget.getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  auto &StateValueIntervals = getAnalysis<luthier::LRStateValueLocations>();
  // Get the Hook's insertion point
  auto *InsertionPoint = HookFuncToMIMap.at(&MF.getFunction());
  // Get the segment value for the hook insertion point
  auto InsertionPointSegment =
      StateValueIntervals.getStateValueDescriptorOfHookInsertionPoint(
          *InsertionPoint);
  const auto &HookLiveInRegs = PerHookLiveInRegs.at(&MF.getFunction());
  // If the state value is in VGPR, only spill s[0:3], s32, and FS pair if they
  // are live or are being accessed by the hook
  llvm::MCRegister ValueRegisterLocation = InsertionPointSegment.StateValueVGPR;
  bool IsStateValueInVGPR = TRI->isVGPR(MRI, ValueRegisterLocation);
  bool IsStateValueInAGPR = TRI->isAGPR(MRI, ValueRegisterLocation);
  llvm::MCRegister AGPRSpillSlot{0};
  bool IsStateValueSpilled = ValueRegisterLocation == 0;
  llvm::MCRegister ScavengedTmpSGPR{0};
  llvm::MCRegister FinalValueRegisterLocation{0};

  if (IsStateValueInAGPR) {
    // Try to scavenge an unused AGPR to spill a live-in, un-accessed VGPR into
    for (auto Reg : llvm::AMDGPU::AGPR_32RegClass) {
      if (MRI.isAllocatable(Reg) && !HookLiveInRegs.contains(Reg) &&
          !AccessedPhysicalRegisters.contains(Reg)) {
        AGPRSpillSlot = Reg;
        break;
      }
    }
  }
  // If the state value is in AGPR, and we weren't able to scavenge an
  // AGPR to spill, or if we have spilled the state value already, then we
  // need to scavenge a free temp SGPR
  if (!IsStateValueInVGPR && ((IsStateValueInAGPR && AGPRSpillSlot == 0) ||
                              (IsStateValueSpilled && !IsStateValueInAGPR))) {
    for (auto Reg : llvm::AMDGPU::SGPR_32RegClass) {
      if (MRI.isAllocatable(Reg) && !HookLiveInRegs.contains(Reg) &&
          !AccessedPhysicalRegisters.contains(Reg))
        ScavengedTmpSGPR = Reg;
      break;
    }
    if (ScavengedTmpSGPR == 0)
      llvm::report_fatal_error("Unable to find a free SGPR to use");
  }

  llvm::MachineBasicBlock::iterator PrologueInsertionPoint = MF.front().begin();
  if (ScavengedTmpSGPR != 0) {
    // If we scavenged a tmp SGPR, then we have to do the following in the
    // prologue:
    // 1. Replace the FLAT_SCRATCH of the app with the one we have saved
    // 2. Spill the app's FLAT_SCRATCH register pair
    // 3. Spill a VGPR, preferably the one not used accessed by the hook
    // 4. Restore the value state into the freed VGPR
    llvm::MCRegister SavedFlatScratch =
        InsertionPointSegment.getInstrumentationStackFlatScratchLocation();
    if (SavedFlatScratch == 0)
      llvm::report_fatal_error("No saved flat scratch was found");
    if (TRI->getRegSizeInBits(*TRI->getPhysRegBaseClass(SavedFlatScratch)) !=
        64)
      llvm::report_fatal_error("Invalid size for the saved flat scratch");
    // step 1.
    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::COPY),
                  ScavengedTmpSGPR)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI, llvm::RegState::Kill);
    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::COPY),
                  llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(SavedFlatScratch, llvm::RegState::Kill, llvm::AMDGPU::sub1);
    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::COPY),
                  ScavengedTmpSGPR)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO, llvm::RegState::Kill);
    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::COPY),
                  llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(SavedFlatScratch, llvm::RegState::Kill, llvm::AMDGPU::sub0);
    // step 2.
    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_SCRATCH_STORE_DWORDX2_IMM))
        .addReg(SavedFlatScratch, llvm::RegState::Kill)
        .addReg(llvm::AMDGPU::FLAT_SCR)
        .addImm(0);
    // step 3.
    for (auto Reg : llvm::AMDGPU::VGPR_32RegClass) {
      if (MRI.isAllocatable(Reg) && !AccessedPhysicalRegisters.contains(Reg))
        FinalValueRegisterLocation = Reg;
      break;
    }
    if (FinalValueRegisterLocation == 0)
      FinalValueRegisterLocation = *AccessedPhysicalRegisters.begin();

    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::S_MOV_B32),
                  ScavengedTmpSGPR)
        .addImm(0);

    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(FinalValueRegisterLocation, llvm::RegState::Kill)
        .addReg(ScavengedTmpSGPR)
        .addImm(8);

    // step 4.
    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                  FinalValueRegisterLocation)
        .addReg(ScavengedTmpSGPR, llvm::RegState::Kill)
        .addImm(12);
  } else if (IsStateValueInAGPR && AGPRSpillSlot != 0) {
    // If we have a free AGPR, spill a VGPR that preferably
    // isn't accessed by the hook into it
    for (auto Reg : llvm::AMDGPU::VGPR_32RegClass) {
      if (MRI.isAllocatable(Reg) && !AccessedPhysicalRegisters.contains(Reg))
        FinalValueRegisterLocation = Reg;
      break;
    }
    if (FinalValueRegisterLocation == 0)
      FinalValueRegisterLocation = *AccessedPhysicalRegisters.begin();
    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::COPY), AGPRSpillSlot)
        .addReg(FinalValueRegisterLocation, llvm::RegState::Kill);
    llvm::BuildMI(*PrologueInsertionPoint->getParent(), PrologueInsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::COPY),
                  FinalValueRegisterLocation)
        .addReg(ValueRegisterLocation, llvm::RegState::Kill);
  } else if (IsStateValueInVGPR) {
    FinalValueRegisterLocation = ValueRegisterLocation;
  } else
    llvm_unreachable("Should not reach here");

  // Spill s[0:3], s32, FS registers (if needed), and load the instrumentation
  // values
  for (auto &[SReg, LaneID] : ValueRegisterSpillSlots) {
    if (SReg == llvm::AMDGPU::FLAT_SCR_LO ||
        SReg == llvm::AMDGPU::FLAT_SCR_HI) {
      // If we spilled the flat scratch regs onto the stack, we won't
      // need to spill it here
      if (!IsStateValueInVGPR || (IsStateValueInAGPR && AGPRSpillSlot == 0)) {
        continue;
      }
    }
    if (HookLiveInRegs.contains(SReg) ||
        AccessedPhysicalRegisters.contains(SReg)) {
      llvm::BuildMI(*InsertionPoint->getParent(), InsertionPoint,
                    llvm::DebugLoc(), TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                    FinalValueRegisterLocation)
          .addReg(SReg)
          .addImm(LaneID)
          .addReg(FinalValueRegisterLocation);
    }
  }

  // Load the PSB, SP, and FS of the instrumentation hook
  for (auto &[SReg, LaneID] : ValueRegisterInstrumentationSlots) {
    if (SReg == llvm::AMDGPU::FLAT_SCR_LO ||
        SReg == llvm::AMDGPU::FLAT_SCR_HI) {
      // If we spilled the flat scratch regs onto the stack, we won't
      // need load it here
      if (!IsStateValueInVGPR || (IsStateValueInAGPR && AGPRSpillSlot == 0)) {
        continue;
      }
    }
    llvm::BuildMI(*InsertionPoint->getParent(), InsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::V_READLANE_B32),
                  SReg)
        .addReg(FinalValueRegisterLocation)
        .addImm(LaneID);
  }
}