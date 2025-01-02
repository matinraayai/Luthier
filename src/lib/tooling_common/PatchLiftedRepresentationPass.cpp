//===-- PatchLiftedRepresentationPass.cpp ---------------------------------===//
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
/// This file implements the Patch lifted representation pass.
//===----------------------------------------------------------------------===//
#include "tooling_common/PatchLiftedRepresentationPass.hpp"
#include "common/Cloning.hpp"
#include "tooling_common/IModuleIRGeneratorPass.hpp"
#include "tooling_common/WrapperAnalysisPasses.hpp"
#include <SIInstrInfo.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <ranges>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-patch-lr"

namespace luthier {

static llvm::cl::opt<bool> OutlineAllInjectedPayloads(
    "luthier-outline-all-injected-payloads",
    llvm::cl::desc("Outline all injected payloads no matter the code size."),
    llvm::cl::init(true));

static void cloneFrameInfo(const llvm::MachineFunction &InjectedPayloadMF,
                           llvm::MachineFunction &ToBeInstrumentedMF) {
  auto &HookMFFrameInfo = InjectedPayloadMF.getFrameInfo();

  if (HookMFFrameInfo.hasStackObjects()) {
    ToBeInstrumentedMF.getFrameInfo().setStackSize(
        ToBeInstrumentedMF.getFrameInfo().getStackSize() +
        HookMFFrameInfo.getStackSize());
  }

  if (HookMFFrameInfo.hasStackObjects()) {
    // Clone the frame objects
    auto &ToBeInstrumentedMFI = ToBeInstrumentedMF.getFrameInfo();

    auto CopyObjectProperties = [](llvm::MachineFrameInfo &DstMFI,
                                   const llvm::MachineFrameInfo &SrcMFI,
                                   int FI) {
      if (SrcMFI.isStatepointSpillSlotObjectIndex(FI))
        DstMFI.markAsStatepointSpillSlotObjectIndex(FI);
      DstMFI.setObjectSSPLayout(FI, SrcMFI.getObjectSSPLayout(FI));
      DstMFI.setObjectZExt(FI, SrcMFI.isObjectZExt(FI));
      DstMFI.setObjectSExt(FI, SrcMFI.isObjectSExt(FI));
    };

    for (int i = 0, e = HookMFFrameInfo.getNumObjects() -
                        HookMFFrameInfo.getNumFixedObjects();
         i != e; ++i) {
      int NewFI;

      assert(!HookMFFrameInfo.isFixedObjectIndex(i));
      if (!HookMFFrameInfo.isDeadObjectIndex(i)) {
        if (HookMFFrameInfo.isVariableSizedObjectIndex(i)) {
          NewFI = ToBeInstrumentedMFI.CreateVariableSizedObject(
              HookMFFrameInfo.getObjectAlign(i),
              HookMFFrameInfo.getObjectAllocation(i));
        } else {
          NewFI = ToBeInstrumentedMFI.CreateStackObject(
              HookMFFrameInfo.getObjectSize(i),
              HookMFFrameInfo.getObjectAlign(i),
              HookMFFrameInfo.isSpillSlotObjectIndex(i),
              HookMFFrameInfo.getObjectAllocation(i),
              HookMFFrameInfo.getStackID(i));
          ToBeInstrumentedMFI.setObjectOffset(
              NewFI, HookMFFrameInfo.getObjectOffset(i));
        }
        CopyObjectProperties(ToBeInstrumentedMFI, HookMFFrameInfo, i);

        (void)NewFI;
        assert(i == NewFI && "expected to keep stable frame index numbering");
      }
    }

    // Copy the fixed frame objects backwards to preserve frame index
    // numbers, since CreateFixedObject uses front insertion.
    for (int i = -1; i >= (int)-HookMFFrameInfo.getNumFixedObjects(); --i) {
      assert(HookMFFrameInfo.isFixedObjectIndex(i));
      if (!HookMFFrameInfo.isDeadObjectIndex(i)) {
        int NewFI = ToBeInstrumentedMFI.CreateFixedObject(
            HookMFFrameInfo.getObjectSize(i),
            HookMFFrameInfo.getObjectOffset(i),
            HookMFFrameInfo.isImmutableObjectIndex(i),
            HookMFFrameInfo.isAliasedObjectIndex(i));
        CopyObjectProperties(ToBeInstrumentedMFI, HookMFFrameInfo, i);

        (void)NewFI;
        assert(i == NewFI && "expected to keep stable frame index numbering");
      }
    }
  }
}

llvm::DenseMap<const llvm::MachineFunction *,
               PatchLiftedRepresentationPass::PatchType>
PatchLiftedRepresentationPass::decidePatchingMethod(
    llvm::Module &TargetAppM, llvm::ModuleAnalysisManager &TargetMAM) {
  // Analysis result output
  llvm::DenseMap<const llvm::MachineFunction *, PatchType> Out;
  // Things we need for this analysis
  auto &IModuleAnalysis =
      *TargetMAM.getCachedResult<IModulePMAnalysis>(TargetAppM);
  auto &TargetMMI =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetAppM).getMMI();
  auto &IMAM = IModuleAnalysis.getMAM();
  const auto &IPIP =
      *IMAM.getCachedResult<InjectedPayloadAndInstPointAnalysis>(IModule);
  // We need to estimate whether we can get away with outlining

  // Mapping between branch instructions and their target MBBs
  llvm::SmallDenseMap<const llvm::MachineInstr *,
                      const llvm::MachineBasicBlock *, 8>
      BranchToTargetMap;
  // Mapping between branch instructions and their offset from the beginning
  // of the function
  llvm::SmallDenseMap<const llvm::MachineInstr *, uint64_t> BranchToOffsetMap;
  // Mapping between the MBBs and their offset from the beginning of the
  // function
  llvm::SmallDenseMap<const llvm::MachineBasicBlock *, uint64_t>
      MBBsToOffsetMap;

  for (const auto &TargetF : TargetAppM) {
    if (auto *TargetMF = TargetMMI.getMachineFunction(TargetF)) {
      const auto &TII = *TargetMF->getSubtarget().getInstrInfo();
      /// Offset from the beginning of the MF; Accumulates anything from
      /// the target app and the instructions we inject to the function
      uint64_t MFBeginningOffset = 0;
      for (const auto &MBB : *TargetMF) {
        uint64_t MBBBeginningOffset = MFBeginningOffset;
        MBBsToOffsetMap.insert({&MBB, MBBBeginningOffset});
        for (const auto &MI : MBB) {
          uint64_t InstSize = TII.getInstSizeInBytes(MI);
          // If this is an instrumentation point, add the estimated size of the
          // code being injected
          if (IPIP.contains(MI)) {
            uint64_t InjectedPayloadSize = IMMI.getMachineFunction(*IPIP.at(MI))
                                               ->estimateFunctionSizeInBytes();
            IModuleFuncSizes.insert(
                {IMMI.getMachineFunction(*IPIP.at(MI)), InjectedPayloadSize});
            MBBBeginningOffset += InjectedPayloadSize;
          }
          if (MI.isBranch() && !MI.isIndirectBranch()) {
            if (auto TargetMBB = TII.getBranchDestBlock(MI)) {
              BranchToTargetMap.insert({&MI, TargetMBB});
              BranchToOffsetMap.insert({&MI, MBBBeginningOffset});
            }
          }
          MBBBeginningOffset += InstSize;
        }
        MFBeginningOffset = MBBBeginningOffset;
      }
      // Now that we have the offset info, we can decide on the way we're going
      // to patch the function
      if (!OutlineAllInjectedPayloads) {
        Out.insert({TargetMF, INLINE});
        for (const auto &[Branch, Target] : BranchToTargetMap) {
          uint64_t BranchOffset = BranchToOffsetMap[Branch];
          uint64_t TargetOffset = MBBsToOffsetMap[Target];
          uint64_t BranchDist = (BranchOffset > TargetOffset)
                                    ? BranchOffset - TargetOffset
                                    : TargetOffset - BranchOffset;
          // If the branch can't make its target with the injected payload code
          // inlined, break out of the loop
          if (BranchDist > (1 << 18)) {
            Out.insert({TargetMF, OUTLINE});
          }
        }
      } else
        Out.insert({TargetMF, OUTLINE});

      // If the injected payload can't be inlined, we have to check what kind
      // of branch we need to do to reach the injected payloads at the
      // end of the function
      if (Out.at(TargetMF) == OUTLINE) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Have to outline MF " << TargetMF->getName() << "\n";
                   llvm::dbgs() << "Size of the MF after patching: "
                                << MFBeginningOffset << ".\n";);
        // Take into account the s_branches we need to insert during outlining
        MFBeginningOffset += IPIP.size() * 4;
        LLVM_DEBUG(llvm::dbgs() << "Size of the MF after outlining: "
                                << MFBeginningOffset << ".\n";);
        if (MFBeginningOffset > (1 << 19)) {
          llvm_unreachable(
              "Have not yet implemented how to deal with big code.");
        }
      }
    }
  }
  return Out;
}

void inlineInjectedPayload(const llvm::MachineFunction &InjectedPayloadMF,
                           llvm::MachineInstr &InsertionPointMI,
                           llvm::DenseMap<const llvm::MachineBasicBlock *,
                                          llvm::MachineBasicBlock *> &MBBMap,
                           const llvm::ValueToValueMapTy &VMap) {
  auto &InsertionPointMBB = *InsertionPointMI.getParent();
  auto &ToBeInstrumentedMF = *InsertionPointMI.getMF();
  // Number of return blocks in the hook
  unsigned int NumReturnBlocksInHook{0};
  // The last return block of the Hook function
  const llvm::MachineBasicBlock *HookLastReturnMBB{nullptr};
  // The target block in the instrumented function which the last return
  // block of the hook will be prepended to if NumReturnBlocksInHook is 1
  // Otherwise, it is the successor of all the return blocks in the hook
  llvm::MachineBasicBlock *HookLastReturnMBBDest{nullptr};

  // Find the last return block of the Hook function + count the number
  // of the return blocks in the hook
  for (const auto &MBB : std::ranges::reverse_view(InjectedPayloadMF)) {
    if (MBB.isReturnBlock() && HookLastReturnMBB == nullptr) {
      HookLastReturnMBB = &MBB;
      NumReturnBlocksInHook++;
    }
  }
  if (HookLastReturnMBB == nullptr)
    llvm_unreachable("No return block found inside the hook");

  if (InjectedPayloadMF.size() > 1) {
    // If there's multiple blocks inside the hook, and the insertion point
    // is not the beginning of the instrumented basic block, then split
    // the insertion point MBB right before the insertion MI, the
    // destination of the last return block of the hook will be the
    // newly-created block
    if (InsertionPointMI.getIterator() != InsertionPointMBB.begin())
      HookLastReturnMBBDest =
          InsertionPointMBB.splitAt(*InsertionPointMI.getPrevNode());
    else
      HookLastReturnMBBDest = &InsertionPointMBB;
    if (NumReturnBlocksInHook == 1)
      MBBMap.insert({HookLastReturnMBB, HookLastReturnMBBDest});
    // If number of return blocks is greater than 1 (very unlikely) we
    // will create a block for it in the next loop
  }
  for (const auto &HookMBB : InjectedPayloadMF) {
    // Special handling of the entry block
    if (HookMBB.isEntryBlock()) {
      // If the Insertion Point is not before the very first instruction,
      // then the Insertion Point MBB will be where the content of the
      // entry block will be appended to
      if (InsertionPointMI.getIterator() != InsertionPointMBB.begin()) {
        MBBMap.insert({&HookMBB, &InsertionPointMBB});
      }
      // If the insertion point is right before the very first instruction
      // of the block, then it should be appended to the return block of
      // the hook instead, unless the hook has only a single basic block
      else if (InjectedPayloadMF.size() == 1) {
        // If there's only a single basic block in the instrumentation
        // function, then the insertion point MBB will be where the hook's
        // first (and last) MBB appends to
        MBBMap.insert({&InjectedPayloadMF.front(), &InsertionPointMBB});
      } else {
        // Otherwise, create a new basic block for the entry block of the
        // hook, and Make all the pred blocks of the Insertion point MBB
        // to point to this newly created block instead
        auto *NewEntryBlock = ToBeInstrumentedMF.CreateMachineBasicBlock();
        ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                  NewEntryBlock);
        // First add the NewEntryBlock as a pred to all
        // InsertionPointMBB's preds
        llvm::SmallVector<llvm::MachineBasicBlock *, 2> PredMBBs;
        for (auto It = InsertionPointMBB.pred_begin();
             It != InsertionPointMBB.pred_end(); ++It) {
          auto PredMBB = *It;
          PredMBB->addSuccessor(NewEntryBlock);
          PredMBBs.push_back(PredMBB);
        }
        // Remove the insertion point mbb from the PredMBB's successor
        // list
        for (auto &PredMBB : PredMBBs) {
          PredMBB->removeSuccessor(&InsertionPointMBB);
        }
        // Add the insertion point MBB as the successor of this block
        NewEntryBlock->addSuccessor(&InsertionPointMBB);
        //
        MBBMap.insert({&InjectedPayloadMF.front(), NewEntryBlock});
      }
    }
    // Special handling for the return blocks
    else if (HookMBB.isReturnBlock()) {
      // If this is not the last return block, or there's more than one
      // return block, then we have to create a new block for it in the
      // target function
      if (NumReturnBlocksInHook > 1 || &HookMBB != HookLastReturnMBB) {
        auto *TargetReturnBlock = ToBeInstrumentedMF.CreateMachineBasicBlock();
        ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                  TargetReturnBlock);
        MBBMap.insert({&HookMBB, TargetReturnBlock});
      }
    } else {
      // Otherwise, we only need to create a new basic block in the
      // instrumented code and just copy its contents over
      auto *NewHookMBB = ToBeInstrumentedMF.CreateMachineBasicBlock();
      ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                NewHookMBB);
      MBBMap.insert({&HookMBB, NewHookMBB});
    }
  }

  // Link blocks
  for (auto &MBB : InjectedPayloadMF) {
    auto *DstMBB = MBBMap[&MBB];
    for (auto It = MBB.succ_begin(), IterEnd = MBB.succ_end(); It != IterEnd;
         ++It) {
      auto *SrcSuccMBB = *It;
      auto *DstSuccMBB = MBBMap[SrcSuccMBB];
      if (!DstMBB->isSuccessor(DstSuccMBB))
        DstMBB->addSuccessor(DstSuccMBB, MBB.getSuccProbability(It));
    }
    if (MBB.isReturnBlock() && NumReturnBlocksInHook > 1) {
      // Add the LastHookMBBDest as a successor to the return block
      // if there's more than one return block in the hook
      if (!DstMBB->isSuccessor(HookLastReturnMBBDest))
        DstMBB->addSuccessor(HookLastReturnMBBDest);
    }
  }
  // Finally, clone the instructions into the new MBBs
  const llvm::TargetSubtargetInfo &STI = ToBeInstrumentedMF.getSubtarget();
  const llvm::TargetInstrInfo *TII = STI.getInstrInfo();
  const llvm::TargetRegisterInfo *TRI = STI.getRegisterInfo();
  auto &TargetMFMRI = ToBeInstrumentedMF.getRegInfo();

  llvm::DenseSet<const uint32_t *> ConstRegisterMasks;

  // Track predefined/named regmasks which we ignore.
  for (const uint32_t *Mask : TRI->getRegMasks())
    ConstRegisterMasks.insert(Mask);
  for (const auto &MBB : InjectedPayloadMF) {
    auto *DstMBB = MBBMap[&MBB];
    llvm::MachineBasicBlock::iterator InsertionPoint;
    if (MBB.isReturnBlock() && NumReturnBlocksInHook == 1) {
      InsertionPoint = DstMBB->begin();
    } else if (MBB.isEntryBlock() && InjectedPayloadMF.size() == 1) {
      InsertionPoint = InsertionPointMI.getIterator();
    } else
      InsertionPoint = DstMBB->end();
    if (MBB.isEntryBlock()) {
      //        auto *DstMI = ToBeInstrumentedMF.CreateMachineInstr(
      //            TII->get(llvm::AMDGPU::S_WAITCNT), llvm::DebugLoc(),
      //            /*NoImplicit=*/true);
      //        DstMBB->insert(InsertionPoint, DstMI);
      //        DstMI->addOperand(llvm::MachineOperand::CreateImm(0));
    }
    for (auto &SrcMI : MBB.instrs()) {
      if (MBB.isReturnBlock() && SrcMI.isTerminator()) {
        break;
      }
      // Don't clone the bundle headers
      if (SrcMI.isBundle())
        continue;
      const auto &MCID = TII->get(SrcMI.getOpcode());
      // TODO: Properly import the debug location
      auto *DstMI =
          ToBeInstrumentedMF.CreateMachineInstr(MCID, llvm::DebugLoc(),
                                                /*NoImplicit=*/true);
      DstMI->setFlags(SrcMI.getFlags());
      DstMI->setAsmPrinterFlag(SrcMI.getAsmPrinterFlags());
      DstMBB->insert(InsertionPoint, DstMI);
      for (auto &SrcMO : SrcMI.operands()) {
        llvm::MachineOperand DstMO(SrcMO);
        DstMO.clearParent();

        // Update MBB.
        if (DstMO.isMBB())
          DstMO.setMBB(MBBMap[DstMO.getMBB()]);
        else if (DstMO.isRegMask()) {
          //          TargetMFMRI.addPhysRegsUsedFromRegMask(DstMO.getRegMask());

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
            ToBeInstrumentedMF.getFunction()
                .getParent()
                ->getContext()
                .emitError(
                    llvm::formatv("Failed to find global variable {0} inside "
                                  "the representation being patched.",
                                  DstMO.getGlobal()));
          }
          auto *DestGV = cast<llvm::GlobalValue>(GVEntry->second);
          DstMO.ChangeToGA(DestGV, DstMO.getOffset(), DstMO.getTargetFlags());
        }

        DstMI->addOperand(DstMO);
      }
    }
    if (MBB.isReturnBlock() ||
        (MBB.isEntryBlock() && InjectedPayloadMF.size() == 1)) {
      //        auto *DstMI = ToBeInstrumentedMF.CreateMachineInstr(
      //            TII->get(llvm::AMDGPU::S_WAITCNT), llvm::DebugLoc(),
      //            /*NoImplicit=*/true);
      //        DstMBB->insert(InsertionPoint, DstMI);
      //        DstMI->addOperand(llvm::MachineOperand::CreateImm(0));
    }
    if (NumReturnBlocksInHook > 1 && MBB.isReturnBlock()) {
      TII->insertUnconditionalBranch(*DstMBB, HookLastReturnMBBDest,
                                     llvm::DebugLoc());
    }
  }
}

void outlineInjectedPayload(const llvm::MachineFunction &InjectedPayloadMF,
                            llvm::MachineInstr &InsertionPointMI,
                            llvm::DenseMap<const llvm::MachineBasicBlock *,
                                           llvm::MachineBasicBlock *> &MBBMap,
                            const llvm::ValueToValueMapTy &VMap) {
  auto &InsertionPointMBB = *InsertionPointMI.getParent();
  auto &ToBeInstrumentedMF = *InsertionPointMI.getMF();
  // The MBB that will jump to the beginning of the injected payload
  llvm::MachineBasicBlock *JumpFromBlock{nullptr};
  // The MBB that the injected payload return blocks will jump to
  llvm::MachineBasicBlock *JumpToBlock{nullptr};

  // Split the insertion point MBB right before the insertion point MI;
  // if the MI is the first instruction in the MBB, then create a new block
  // and insert it before the insertion point MBB
  if (InsertionPointMI == InsertionPointMBB.begin()) {
    JumpFromBlock = ToBeInstrumentedMF.CreateMachineBasicBlock();
    ToBeInstrumentedMF.insert(InsertionPointMBB.getIterator(), JumpFromBlock);
    // All the predecessors of InsertionPointMBB now become the JumpFromBlock's
    // predecessors
    for (auto &Pred :
         llvm::make_early_inc_range(InsertionPointMBB.predecessors())) {
      Pred->addSuccessor(JumpFromBlock);
      if (Pred->isSuccessor(&InsertionPointMBB))
        Pred->removeSuccessor(&InsertionPointMBB);
    }
    // InsertionPointMBB will become the jump to block
    JumpToBlock = &InsertionPointMBB;
  } else {
    JumpToBlock = InsertionPointMBB.splitAt(*InsertionPointMI.getPrevNode());
    JumpFromBlock = &InsertionPointMBB;
  }

  for (const auto &InjectedPayloadMBB : InjectedPayloadMF) {
    // Create MBBs at the end of the function
    auto *NewBlock = ToBeInstrumentedMF.CreateMachineBasicBlock();
    ToBeInstrumentedMF.push_back(NewBlock);
    MBBMap.insert({&InjectedPayloadMBB, NewBlock});
    // If this is the entry block of the injected payload then it is
    // the jump from block's direct successor
    if (InjectedPayloadMBB.isEntryBlock()) {
      JumpFromBlock->addSuccessor(NewBlock);
    }
    // If this is a return block of the injected payload then it is the jump to
    // block's predecessor
    if (InjectedPayloadMBB.isReturnBlock()) {
      NewBlock->addSuccessor(JumpToBlock);
    }
  }

  // Link blocks
  for (auto &InjectedPayloadMBB : InjectedPayloadMF) {
    auto *DstMBB = MBBMap[&InjectedPayloadMBB];
    for (const auto &IPSucc : InjectedPayloadMBB.successors()) {
      auto *DstSuccMBB = MBBMap[IPSucc];
      if (!DstMBB->isSuccessor(DstSuccMBB))
        DstMBB->addSuccessor(DstSuccMBB);
    }
  }
  // Finally, clone the instructions into the new MBBs
  const llvm::TargetSubtargetInfo &STI = ToBeInstrumentedMF.getSubtarget();
  const llvm::TargetInstrInfo *TII = STI.getInstrInfo();
  const llvm::TargetRegisterInfo *TRI = STI.getRegisterInfo();
  auto &TargetMFMRI = ToBeInstrumentedMF.getRegInfo();

  llvm::DenseSet<const uint32_t *> ConstRegisterMasks;

  // Track predefined/named regmasks which we ignore.
  for (const uint32_t *Mask : TRI->getRegMasks())
    ConstRegisterMasks.insert(Mask);
  for (const auto &MBB : InjectedPayloadMF) {
    auto *DstMBB = MBBMap[&MBB];
    llvm::MachineBasicBlock::iterator InsertionPoint = DstMBB->end();
    [&]() {
      for (auto &SrcMI : MBB.instrs()) {
        if (MBB.isReturnBlock() && SrcMI.isTerminator()) {
          break;
        }
        // Don't clone the bundle headers
        if (SrcMI.isBundle())
          continue;
        const auto &MCID = TII->get(SrcMI.getOpcode());
        // TODO: Properly import the debug location
        auto *DstMI =
            ToBeInstrumentedMF.CreateMachineInstr(MCID, llvm::DebugLoc(),
                                                  /*NoImplicit=*/true);
        DstMI->setFlags(SrcMI.getFlags());
        DstMI->setAsmPrinterFlag(SrcMI.getAsmPrinterFlags());
        DstMBB->insert(InsertionPoint, DstMI);
        for (auto &SrcMO : SrcMI.operands()) {
          llvm::MachineOperand DstMO(SrcMO);
          DstMO.clearParent();

          // Update MBB.
          if (DstMO.isMBB())
            DstMO.setMBB(MBBMap[DstMO.getMBB()]);
          else if (DstMO.isRegMask()) {
            TargetMFMRI.addPhysRegsUsedFromRegMask(DstMO.getRegMask());

            if (!ConstRegisterMasks.count(DstMO.getRegMask())) {
              uint32_t *DstMask = ToBeInstrumentedMF.allocateRegMask();
              std::memcpy(
                  DstMask, SrcMO.getRegMask(),
                  sizeof(*DstMask) *
                      llvm::MachineOperand::getRegMaskSize(TRI->getNumRegs()));
              DstMO.setRegMask(DstMask);
            }
          } else if (DstMO.isGlobal()) {
            auto GVEntry = VMap.find(DstMO.getGlobal());
            if (GVEntry == VMap.end()) {
              ToBeInstrumentedMF.getFunction()
                  .getParent()
                  ->getContext()
                  .emitError(
                      llvm::formatv("Failed to find global variable {0} inside "
                                    "the representation being patched.",
                                    DstMO.getGlobal()));
            }
            auto *DestGV = cast<llvm::GlobalValue>(GVEntry->second);
            DstMO.ChangeToGA(DestGV, DstMO.getOffset(), DstMO.getTargetFlags());
          }

          DstMI->addOperand(DstMO);
        }
      }
    }();
    if (MBB.isEntryBlock()) {
      TII->insertUnconditionalBranch(*JumpFromBlock, DstMBB, llvm::DebugLoc());
    }
    if (MBB.isReturnBlock()) {
      TII->insertUnconditionalBranch(*DstMBB, JumpToBlock, llvm::DebugLoc());
    }
  }
}

llvm::PreservedAnalyses
PatchLiftedRepresentationPass::run(llvm::Module &TargetAppM,
                                   llvm::ModuleAnalysisManager &TargetMAM) {
  auto PatchMethods = decidePatchingMethod(TargetAppM, TargetMAM);
  llvm::TimeTraceScope Scope("Lifted Representation Patching");
  auto T1 = std::chrono::high_resolution_clock::now();

  auto &IModuleAnalysis =
      *TargetMAM.getCachedResult<IModulePMAnalysis>(TargetAppM);

  auto &IModule = IModuleAnalysis.getModule();
  auto &IMAM = IModuleAnalysis.getMAM();

  const auto &IPIP =
      *IMAM.getCachedResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  auto &TargetMMI =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetAppM).getMMI();

  // A mapping between Global Variables in the instrumentation module and
  // their corresponding Global Variables in the instrumented code
  llvm::ValueToValueMapTy VMap;
  // Clone the instrumentation module Global Variables into the instrumented
  // code
  for (const auto &GV : IModule.globals()) {
    auto *NewGV = new llvm::GlobalVariable(
        TargetAppM, GV.getValueType(), GV.isConstant(), GV.getLinkage(),
        nullptr, GV.getName(), nullptr, GV.getThreadLocalMode(),
        GV.getType()->getAddressSpace());
    NewGV->copyAttributesFrom(&GV);
    VMap[&GV] = NewGV;
  }

  // Clone only the definition of functions that are not injected payloads
  for (const auto &F : IModule.functions()) {
    if (IMMI.getMachineFunction(F) != nullptr &&
        !F.hasFnAttribute(LUTHIER_HOOK_ATTRIBUTE) &&
        !F.hasFnAttribute(LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE)) {
      auto *NewF = llvm::Function::Create(
          llvm::cast<llvm::FunctionType>(F.getValueType()), F.getLinkage(),
          F.getAddressSpace(), F.getName(), &TargetAppM);
      llvm::BasicBlock *BB =
          llvm::BasicBlock::Create(TargetAppM.getContext(), "", NewF);
      new llvm::UnreachableInst(TargetAppM.getContext(), BB);
      VMap[&F] = NewF;
      auto NewMF = cloneMF(IMMI.getMachineFunction(F), VMap, TargetMMI);
      LUTHIER_REPORT_FATAL_ON_ERROR(NewMF.takeError());
      LLVM_DEBUG(llvm::dbgs() << "Found non-hook function. Patched contents:\n";
                 NewMF->get()->print(llvm::dbgs()););
      TargetMMI.insertFunction(*NewF, std::move(*NewMF));
    }
  }

  for (const auto &[InsertionPointMI, InjectedPayloadFunc] :
       IPIP.mi_payload()) {
    // A mapping between a machine basic block in the instrumentation MMI
    // and its destination in the patched instrumented code
    llvm::DenseMap<const llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>
        MBBMap;

    const auto &InjectedPayloadMF =
        *IMMI.getMachineFunction(*InjectedPayloadFunc);
    auto &InsertionPointMBB = *InsertionPointMI->getParent();
    auto &ToBeInstrumentedMF = *InsertionPointMBB.getParent();

    /// Patch the frame info
    cloneFrameInfo(InjectedPayloadMF, ToBeInstrumentedMF);

    // Clone the MBBs
    if (PatchMethods.at(&ToBeInstrumentedMF) == INLINE) {
      inlineInjectedPayload(InjectedPayloadMF, *InsertionPointMI, MBBMap, VMap);
    } else {
      outlineInjectedPayload(InjectedPayloadMF, *InsertionPointMI, MBBMap,
                             VMap);
    }
  }
  auto T2 = std::chrono::high_resolution_clock::now();
  llvm::outs()
      << "Time to Patch Lifted Representation: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(T2 - T1).count()
      << "ms.\n";
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
