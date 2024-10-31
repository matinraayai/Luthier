//===-- Cloning.cpp - IR and MIR Cloning Utilities ------------------------===//
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
/// This file implements utility functions used to clone LLVM MIR constructs.
//===----------------------------------------------------------------------===//
#include "common/Cloning.hpp"
#include "common/Error.hpp"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/PseudoSourceValueManager.h>
#include <llvm/CodeGen/TargetInstrInfo.h>

namespace luthier {

static void cloneFrameInfo(
    llvm::MachineFrameInfo &DstMFI, const llvm::MachineFrameInfo &SrcMFI,
    const llvm::DenseMap<llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>
        &Src2DstMBB) {
  DstMFI.setFrameAddressIsTaken(SrcMFI.isFrameAddressTaken());
  DstMFI.setReturnAddressIsTaken(SrcMFI.isReturnAddressTaken());
  DstMFI.setHasStackMap(SrcMFI.hasStackMap());
  DstMFI.setHasPatchPoint(SrcMFI.hasPatchPoint());
  DstMFI.setUseLocalStackAllocationBlock(
      SrcMFI.getUseLocalStackAllocationBlock());
  DstMFI.setOffsetAdjustment(SrcMFI.getOffsetAdjustment());

  DstMFI.ensureMaxAlignment(SrcMFI.getMaxAlign());
  assert(DstMFI.getMaxAlign() == SrcMFI.getMaxAlign() &&
         "we need to set exact alignment");
  // If the stack size is already estimated, set it in the destination as well
  if (SrcMFI.getStackSize() != 0)
    DstMFI.setStackSize(SrcMFI.getStackSize());
  DstMFI.setAdjustsStack(SrcMFI.adjustsStack());
  DstMFI.setHasCalls(SrcMFI.hasCalls());
  DstMFI.setHasOpaqueSPAdjustment(SrcMFI.hasOpaqueSPAdjustment());
  DstMFI.setHasCopyImplyingStackAdjustment(
      SrcMFI.hasCopyImplyingStackAdjustment());
  DstMFI.setHasVAStart(SrcMFI.hasVAStart());
  DstMFI.setHasMustTailInVarArgFunc(SrcMFI.hasMustTailInVarArgFunc());
  DstMFI.setHasTailCall(SrcMFI.hasTailCall());

  if (SrcMFI.isMaxCallFrameSizeComputed())
    DstMFI.setMaxCallFrameSize(SrcMFI.getMaxCallFrameSize());

  DstMFI.setCVBytesOfCalleeSavedRegisters(
      SrcMFI.getCVBytesOfCalleeSavedRegisters());

  if (llvm::MachineBasicBlock *SavePt = SrcMFI.getSavePoint())
    DstMFI.setSavePoint(Src2DstMBB.find(SavePt)->second);
  if (llvm::MachineBasicBlock *RestorePt = SrcMFI.getRestorePoint())
    DstMFI.setRestorePoint(Src2DstMBB.find(RestorePt)->second);

  auto CopyObjectProperties = [](llvm::MachineFrameInfo &DstMFI,
                                 const llvm::MachineFrameInfo &SrcMFI, int FI) {
    if (SrcMFI.isStatepointSpillSlotObjectIndex(FI))
      DstMFI.markAsStatepointSpillSlotObjectIndex(FI);
    DstMFI.setObjectSSPLayout(FI, SrcMFI.getObjectSSPLayout(FI));
    DstMFI.setObjectZExt(FI, SrcMFI.isObjectZExt(FI));
    DstMFI.setObjectSExt(FI, SrcMFI.isObjectSExt(FI));
  };

  for (int i = 0, e = SrcMFI.getNumObjects() - SrcMFI.getNumFixedObjects();
       i != e; ++i) {
    int NewFI;

    assert(!SrcMFI.isFixedObjectIndex(i));
    if (SrcMFI.isVariableSizedObjectIndex(i)) {
      NewFI = DstMFI.CreateVariableSizedObject(SrcMFI.getObjectAlign(i),
                                               SrcMFI.getObjectAllocation(i));
    } else {
      NewFI = DstMFI.CreateStackObject(
          SrcMFI.getObjectSize(i), SrcMFI.getObjectAlign(i),
          SrcMFI.isSpillSlotObjectIndex(i), SrcMFI.getObjectAllocation(i),
          SrcMFI.getStackID(i));
      DstMFI.setObjectOffset(NewFI, SrcMFI.getObjectOffset(i));
    }

    CopyObjectProperties(DstMFI, SrcMFI, i);

    (void)NewFI;
    assert(i == NewFI && "expected to keep stable frame index numbering");
  }

  // Copy the fixed frame objects backwards to preserve frame index numbers,
  // since CreateFixedObject uses front insertion.
  for (int i = -1; i >= (int)-SrcMFI.getNumFixedObjects(); --i) {
    assert(SrcMFI.isFixedObjectIndex(i));
    int NewFI = DstMFI.CreateFixedObject(
        SrcMFI.getObjectSize(i), SrcMFI.getObjectOffset(i),
        SrcMFI.isImmutableObjectIndex(i), SrcMFI.isAliasedObjectIndex(i));
    CopyObjectProperties(DstMFI, SrcMFI, i);

    (void)NewFI;
    assert(i == NewFI && "expected to keep stable frame index numbering");
  }

  for (unsigned I = 0, E = SrcMFI.getLocalFrameObjectCount(); I < E; ++I) {
    auto LocalObject = SrcMFI.getLocalFrameObjectMap(I);
    DstMFI.mapLocalFrameObject(LocalObject.first, LocalObject.second);
  }

  DstMFI.setCalleeSavedInfo(SrcMFI.getCalleeSavedInfo());

  if (SrcMFI.hasStackProtectorIndex()) {
    DstMFI.setStackProtectorIndex(SrcMFI.getStackProtectorIndex());
  }

  // FIXME: Needs test, missing MIR serialization.
  if (SrcMFI.hasFunctionContextIndex()) {
    DstMFI.setFunctionContextIndex(SrcMFI.getFunctionContextIndex());
  }
}

static llvm::Error cloneMemOperands(llvm::MachineInstr &DstMI,
                                    const llvm::MachineInstr &SrcMI,
                                    const llvm::MachineFunction &SrcMF,
                                    llvm::MachineFunction &DstMF) {
  // The new MachineMemOperands should be owned by the new function's
  // Allocator.
  llvm::PseudoSourceValueManager &PSVMgr = DstMF.getPSVManager();

  // We also need to remap the PseudoSourceValues from the new function's
  // PseudoSourceValueManager.
  llvm::SmallVector<llvm::MachineMemOperand *, 2> NewMMOs;
  for (llvm::MachineMemOperand *OldMMO : SrcMI.memoperands()) {
    llvm::MachinePointerInfo NewPtrInfo(OldMMO->getPointerInfo());
    if (const llvm::PseudoSourceValue *PSV =
            dyn_cast_if_present<const llvm::PseudoSourceValue *>(
                NewPtrInfo.V)) {
      switch (PSV->kind()) {
      case llvm::PseudoSourceValue::Stack:
        NewPtrInfo.V = PSVMgr.getStack();
        break;
      case llvm::PseudoSourceValue::GOT:
        NewPtrInfo.V = PSVMgr.getGOT();
        break;
      case llvm::PseudoSourceValue::JumpTable:
        NewPtrInfo.V = PSVMgr.getJumpTable();
        break;
      case llvm::PseudoSourceValue::ConstantPool:
        NewPtrInfo.V = PSVMgr.getConstantPool();
        break;
      case llvm::PseudoSourceValue::FixedStack:
        NewPtrInfo.V = PSVMgr.getFixedStack(
            cast<llvm::FixedStackPseudoSourceValue>(PSV)->getFrameIndex());
        break;
      case llvm::PseudoSourceValue::GlobalValueCallEntry:
        NewPtrInfo.V = PSVMgr.getGlobalValueCallEntry(
            cast<llvm::GlobalValuePseudoSourceValue>(PSV)->getValue());
        break;
      case llvm::PseudoSourceValue::ExternalSymbolCallEntry:
        NewPtrInfo.V = PSVMgr.getExternalSymbolCallEntry(
            cast<llvm::ExternalSymbolPseudoSourceValue>(PSV)->getSymbol());
        break;
      case llvm::PseudoSourceValue::TargetCustom:
      default:
        // FIXME: We have no generic interface for allocating custom PSVs.
        return LUTHIER_CREATE_ERROR("Cloning TargetCustom PSV not handled");
      }
    }

    llvm::MachineMemOperand *NewMMO = DstMF.getMachineMemOperand(
        NewPtrInfo, OldMMO->getFlags(), OldMMO->getMemoryType(),
        OldMMO->getBaseAlign(), OldMMO->getAAInfo(), OldMMO->getRanges(),
        OldMMO->getSyncScopeID(), OldMMO->getSuccessOrdering(),
        OldMMO->getFailureOrdering());
    NewMMOs.push_back(NewMMO);
  }

  DstMI.setMemRefs(DstMF, NewMMOs);
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<llvm::MachineFunction>> cloneMF(
    const llvm::MachineFunction *SrcMF, const llvm::ValueToValueMapTy &VMap,
    llvm::MachineModuleInfo &DestMMI,
    llvm::DenseMap<llvm::MachineInstr *, llvm::MachineInstr *> *SrcToDstMIMap) {
  // Find the destination function entry in the value map
  auto &SrcF = SrcMF->getFunction();
  auto DestFMapEntry = VMap.find(&SrcF);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DestFMapEntry != VMap.end(),
      "Failed to find the corresponding value for function {0} "
      "in the cloned value map.",
      SrcF.getName()));
  llvm::Function &DestF = *cast<llvm::Function>(DestFMapEntry->second);

  // Construct the destination machine function
  auto DstMF = std::make_unique<llvm::MachineFunction>(
      DestF, SrcMF->getTarget(), SrcMF->getSubtarget(), SrcMF->getContext(),
      SrcMF->getFunctionNumber());
  llvm::DenseMap<llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>
      Src2DstMBB;

  auto *SrcMRI = &SrcMF->getRegInfo();
  auto *DstMRI = &DstMF->getRegInfo();

  // Clone blocks.
  for (const llvm::MachineBasicBlock &SrcMBB : *SrcMF) {
    // We don't assign a BB to the MBB because it is likely to be just
    // a placeholder with an unreachable instruction
    llvm::MachineBasicBlock *DstMBB = DstMF->CreateMachineBasicBlock();
    Src2DstMBB[const_cast<llvm::MachineBasicBlock *>(&SrcMBB)] = DstMBB;

    DstMBB->setCallFrameSize(SrcMBB.getCallFrameSize());

    if (SrcMBB.isIRBlockAddressTaken())
      DstMBB->setAddressTakenIRBlock(SrcMBB.getAddressTakenIRBlock());
    if (SrcMBB.isMachineBlockAddressTaken())
      DstMBB->setMachineBlockAddressTaken();

    if (SrcMBB.hasLabelMustBeEmitted())
      DstMBB->setLabelMustBeEmitted();

    DstMBB->setAlignment(SrcMBB.getAlignment());

    DstMBB->setMaxBytesForAlignment(SrcMBB.getMaxBytesForAlignment());

    DstMBB->setIsEHPad(SrcMBB.isEHPad());
    DstMBB->setIsEHScopeEntry(SrcMBB.isEHScopeEntry());
    DstMBB->setIsEHCatchretTarget(SrcMBB.isEHCatchretTarget());
    DstMBB->setIsEHFuncletEntry(SrcMBB.isEHFuncletEntry());

    DstMBB->setIsCleanupFuncletEntry(SrcMBB.isCleanupFuncletEntry());
    DstMBB->setIsBeginSection(SrcMBB.isBeginSection());
    DstMBB->setIsEndSection(SrcMBB.isEndSection());

    DstMBB->setSectionID(SrcMBB.getSectionID());
    DstMBB->setIsInlineAsmBrIndirectTarget(
        SrcMBB.isInlineAsmBrIndirectTarget());

    if (std::optional<uint64_t> Weight = SrcMBB.getIrrLoopHeaderWeight())
      DstMBB->setIrrLoopHeaderWeight(*Weight);
  }

  const llvm::MachineFrameInfo &SrcMFI = SrcMF->getFrameInfo();
  llvm::MachineFrameInfo &DstMFI = DstMF->getFrameInfo();

  // Copy stack objects and other info
  cloneFrameInfo(DstMFI, SrcMFI, Src2DstMBB);

  // Remap the debug info frame index references.
  DstMF->VariableDbgInfos = SrcMF->VariableDbgInfos;

  // Clone virtual registers
  for (unsigned I = 0, E = SrcMRI->getNumVirtRegs(); I != E; ++I) {
    llvm::Register Reg = llvm::Register::index2VirtReg(I);
    llvm::Register NewReg =
        DstMRI->createIncompleteVirtualRegister(SrcMRI->getVRegName(Reg));
    assert(NewReg == Reg && "expected to preserve virtreg number");

    DstMRI->setRegClassOrRegBank(NewReg, SrcMRI->getRegClassOrRegBank(Reg));

    llvm::LLT RegTy = SrcMRI->getType(Reg);
    if (RegTy.isValid())
      DstMRI->setType(NewReg, RegTy);

    // Copy register allocation hints.
    const auto &Hints = SrcMRI->getRegAllocationHints(Reg);
    for (llvm::Register PrefReg : Hints->second)
      DstMRI->addRegAllocationHint(NewReg, PrefReg);
  }

  const llvm::TargetSubtargetInfo &STI = DstMF->getSubtarget();
  const llvm::TargetInstrInfo *TII = STI.getInstrInfo();
  const llvm::TargetRegisterInfo *TRI = STI.getRegisterInfo();

  // Link blocks.
  for (auto &SrcMBB : *SrcMF) {
    auto *DstMBB = Src2DstMBB[const_cast<llvm::MachineBasicBlock *>(&SrcMBB)];
    DstMF->push_back(DstMBB);

    for (auto It = SrcMBB.succ_begin(), IterEnd = SrcMBB.succ_end();
         It != IterEnd; ++It) {
      auto *SrcSuccMBB = *It;
      auto *DstSuccMBB = Src2DstMBB[SrcSuccMBB];
      DstMBB->addSuccessor(DstSuccMBB, SrcMBB.getSuccProbability(It));
    }

    for (auto &LI : SrcMBB.liveins_dbg())
      DstMBB->addLiveIn(LI);

    // Make sure MRI knows about registers clobbered by unwinder.
    if (DstMBB->isEHPad()) {
      if (auto *RegMask = TRI->getCustomEHPadPreservedMask(*DstMF))
        DstMRI->addPhysRegsUsedFromRegMask(RegMask);
    }
  }

  llvm::DenseSet<const uint32_t *> ConstRegisterMasks;

  // Track predefined/named regmasks which we ignore.
  for (const uint32_t *Mask : TRI->getRegMasks())
    ConstRegisterMasks.insert(Mask);

  // Clone instructions.
  for (auto &SrcMBB : *SrcMF) {
    auto *DstMBB = Src2DstMBB[const_cast<llvm::MachineBasicBlock *>(&SrcMBB)];
    for (auto &SrcMI : SrcMBB) {
      const auto &MCID = TII->get(SrcMI.getOpcode());
      auto *DstMI = DstMF->CreateMachineInstr(MCID, SrcMI.getDebugLoc(),
                                              /*NoImplicit=*/true);
      DstMI->setFlags(SrcMI.getFlags());
      DstMI->setAsmPrinterFlag(SrcMI.getAsmPrinterFlags());

      DstMBB->push_back(DstMI);
      for (auto &SrcMO : SrcMI.operands()) {
        llvm::MachineOperand DstMO(SrcMO);
        DstMO.clearParent();

        // Update MBB.
        if (DstMO.isMBB())
          DstMO.setMBB(Src2DstMBB[DstMO.getMBB()]);
        else if (DstMO.isRegMask()) {
          DstMRI->addPhysRegsUsedFromRegMask(DstMO.getRegMask());

          if (!ConstRegisterMasks.count(DstMO.getRegMask())) {
            uint32_t *DstMask = DstMF->allocateRegMask();
            std::memcpy(DstMask, SrcMO.getRegMask(),
                        sizeof(*DstMask) * llvm::MachineOperand::getRegMaskSize(
                                               TRI->getNumRegs()));
            DstMO.setRegMask(DstMask);
          }
        } else if (DstMO.isGlobal()) {
          auto GVEntry = VMap.find(DstMO.getGlobal());
          LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
              GVEntry != VMap.end(),
              "Failed to find the corresponding value for {0} "
              "in the cloned value map.",
              DstMO.getGlobal()->getName()));
          auto *DestGV = cast<llvm::GlobalValue>(GVEntry->second);
          DstMO.ChangeToGA(DestGV, DstMO.getOffset(), DstMO.getTargetFlags());
        }

        DstMI->addOperand(DstMO);
        if (SrcToDstMIMap != nullptr)
          SrcToDstMIMap->insert(
              {const_cast<llvm::MachineInstr *>(&SrcMI), DstMI});
      }

      LUTHIER_RETURN_ON_ERROR(cloneMemOperands(*DstMI, SrcMI, *SrcMF, *DstMF));
    }
  }

  DstMF->setAlignment(SrcMF->getAlignment());
  DstMF->setExposesReturnsTwice(SrcMF->exposesReturnsTwice());
  DstMF->setHasInlineAsm(SrcMF->hasInlineAsm());
  DstMF->setHasWinCFI(SrcMF->hasWinCFI());

  DstMF->getProperties().reset().set(SrcMF->getProperties());

  if (!SrcMF->getFrameInstructions().empty() ||
      !SrcMF->getLongjmpTargets().empty() ||
      !SrcMF->getCatchretTargets().empty())
    return LUTHIER_CREATE_ERROR(
        "cloning not implemented for machine function property");

  DstMF->setCallsEHReturn(SrcMF->callsEHReturn());
  DstMF->setCallsUnwindInit(SrcMF->callsUnwindInit());
  DstMF->setHasEHCatchret(SrcMF->hasEHCatchret());
  DstMF->setHasEHScopes(SrcMF->hasEHScopes());
  DstMF->setHasEHFunclets(SrcMF->hasEHFunclets());
  DstMF->setIsOutlined(SrcMF->isOutlined());

  if (!SrcMF->getLandingPads().empty() ||
      !SrcMF->getCodeViewAnnotations().empty() ||
      !SrcMF->getTypeInfos().empty() || !SrcMF->getFilterIds().empty() ||
      SrcMF->hasAnyWasmLandingPadIndex() || SrcMF->hasAnyCallSiteLandingPad() ||
      SrcMF->hasAnyCallSiteLabel() || !SrcMF->getCallSitesInfo().empty())
    return LUTHIER_CREATE_ERROR(
        "cloning not implemented for machine function property");

  DstMF->setDebugInstrNumberingCount(SrcMF->DebugInstrNumberingCount);

  if (!DstMF->cloneInfoFrom(*SrcMF, Src2DstMBB))
    return LUTHIER_CREATE_ERROR(
        "target does not implement MachineFunctionInfo cloning");

  DstMRI->freezeReservedRegs();

  //  bool Verified = DstMF->verify(nullptr, "", /*AbortOnError=*/true);
  //  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Verified));
  return DstMF;
}

llvm::Error cloneMMI(
    const llvm::MachineModuleInfo &SrcMMI, const llvm::Module &SrcModule,
    const llvm::ValueToValueMapTy &VMap, llvm::MachineModuleInfo &DestMMI,
    llvm::DenseMap<llvm::MachineInstr *, llvm::MachineInstr *> *SrcToDstMIMap) {
  for (const llvm::Function &SrcF : SrcModule) {
    llvm::MachineFunction *SrcMF = SrcMMI.getMachineFunction(SrcF);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        SrcMF != nullptr,
        "Failed to find the Machine Function associated with "
        "Function {0} inside the MMI during cloning.",
        SrcF.getName()));
    auto DestFMapEntry = VMap.find(&SrcF);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        DestFMapEntry != VMap.end(),
        "Failed to find the corresponding value for function {0} "
        "in the cloned value map.",
        SrcF.getName()));
    llvm::Function &DestF = *cast<llvm::Function>(DestFMapEntry->second);
    auto DestMF = cloneMF(SrcMF, VMap, DestMMI, SrcToDstMIMap);
    LUTHIER_RETURN_ON_ERROR(DestMF.takeError());
    DestMMI.insertFunction(DestF, std::move(*DestMF));
  }
  return llvm::Error::success();
}

} // namespace luthier