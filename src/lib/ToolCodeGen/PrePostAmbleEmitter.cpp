//===-- PrePostAmbleEmitter.hpp -------------------------------------------===//
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
/// This file implements the Pre and post amble emitter, as well as the
/// \c FunctionPreambleDescriptor and its analysis pass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/PrePostAmbleEmitter.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/SVAFrameLanes.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include <GCNSubtarget.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/SlotIndexes.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-pre-post-amble-emitter"

namespace luthier {

static llvm::MCRegister
getArgReg(const llvm::MachineFunction &MF,
          llvm::AMDGPUFunctionArgInfo::PreloadedValue ArgReg) {
  return MF.getInfo<llvm::SIMachineFunctionInfo>()->getPreloadedReg(ArgReg);
}

#define DEFINE_ENABLE_SGPR_ARGUMENT_WITHOUT_TRI(                               \
    KernArgName, AMDGPUKernArgEnum, MFIEnableFunc)                             \
  static void enable##KernArgName(llvm::SIMachineFunctionInfo &MFI) {          \
    if (!MFI.getPreloadedReg(AMDGPUKernArgEnum)) {                             \
      MFI.MFIEnableFunc();                                                     \
    }                                                                          \
  }

#define DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(KernArgName, AMDGPUKernArgEnum,   \
                                             MFIEnableFunc)                    \
  static void enable##KernArgName(llvm::SIMachineFunctionInfo &MFI,            \
                                  const llvm::SIRegisterInfo &TRI) {           \
    if (!MFI.getPreloadedReg(AMDGPUKernArgEnum)) {                             \
      MFI.MFIEnableFunc(TRI);                                                  \
    }                                                                          \
  }

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(
    PrivateSegmentBuffer, llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER,
    addPrivateSegmentBuffer);

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(
    FlatScratchInit, llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT,
    addFlatScratchInit);

DEFINE_ENABLE_SGPR_ARGUMENT_WITHOUT_TRI(
    PrivateSegmentWaveOffset,
    llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
    addPrivateSegmentWaveByteOffset);

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(
    KernArg, llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR,
    addKernargSegmentPtr)

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(DispatchID,
                                     llvm::AMDGPUFunctionArgInfo::DISPATCH_ID,
                                     addDispatchID)

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(DispatchPtr,
                                     llvm::AMDGPUFunctionArgInfo::DISPATCH_PTR,
                                     addDispatchPtr)

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(QueuePtr,
                                     llvm::AMDGPUFunctionArgInfo::QUEUE_PTR,
                                     addQueuePtr)

static void emitCodeToReturnSGPRArgsToOriginalPlace(
    const llvm::DenseMap<llvm::AMDGPUFunctionArgInfo::PreloadedValue,
                         llvm::MCRegister> &OriginalKernelArguments,
    llvm::MachineInstr &InsertionPoint) {
  auto &MF = *InsertionPoint.getMF();
  auto &MBB = *InsertionPoint.getParent();
  const auto &TRI = *MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
  const auto &TII = *MF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  for (const auto &[KernArg, OriginalArgReg] : OriginalKernelArguments) {
    llvm::MCRegister ModifiedArgReg = getArgReg(MF, KernArg);
    size_t Size =
        TRI.getRegSizeInBits(*TRI.getPhysRegBaseClass(OriginalArgReg));
    if (OriginalArgReg != ModifiedArgReg) {
      if (Size == 32) {
        llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_MOV_B32), OriginalArgReg)
            .addReg(ModifiedArgReg, llvm::RegState::Kill);
      } else if (Size == 64) {
        llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_MOV_B64), OriginalArgReg)
            .addReg(ModifiedArgReg, llvm::RegState::Kill);
      } else {
        size_t NumChannels = Size / 32;
        for (int i = 0; i < NumChannels / 2; i++) {
          auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i, 2);
          llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_MOV_B64))
              .addReg(TRI.getSubReg(OriginalArgReg, SubIdx),
                      llvm::RegState::Define)
              .addReg(TRI.getSubReg(ModifiedArgReg, SubIdx),
                      llvm::RegState::Kill);
        }
      }
    }
  }
}

static void emitCodeToMoveSVA(llvm::ModuleAnalysisManager &TargetMAM,
                              llvm::Module &TargetModule,
                              llvm::MachineFunction *MF,
                              luthier::SVStorageAndLoadLocations &SVLocations) {
  auto &TargetMFAM =
      TargetMAM
          .getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(
              TargetModule)
          .getManager();
  auto &SlotIndexes = TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(*MF);

  // Now we need to emit code that juggles the SVS between different
  // storage schemes
  const auto &TII = *MF->getSubtarget().getInstrInfo();
  for (auto &MBB : *MF) {
    const auto MBBIntervals = SVLocations.getStorageIntervals(MBB);
    for (unsigned int I = 0; I < MBBIntervals.size() - 1; I++) {
      auto &CurMBBInterval = MBBIntervals[I];
      auto &NextMBBInterval = MBBIntervals[I + 1];
      if (CurMBBInterval.getSVS() != NextMBBInterval.getSVS()) {
        auto InsertionMI =
            SlotIndexes.getInstructionFromIndex(NextMBBInterval.begin());
        CurMBBInterval.getSVS().emitCodeToSwitchSVS(*InsertionMI,
                                                    NextMBBInterval.getSVS());
      }
    }
    // Analyze the branch at the end of this block (if exists)
    llvm::MachineBasicBlock *TBB;
    llvm::MachineBasicBlock *FBB;
    llvm::MachineBasicBlock *NewTBB{nullptr};
    llvm::MachineBasicBlock *NewFBB{nullptr};
    llvm::SmallVector<llvm::MachineOperand, 4> Cond;
    bool Fail = TII.analyzeBranch(MBB, TBB, FBB, Cond, false);
    if (Fail)
      continue;
    llvm::SmallVector<
        std::pair<llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>, 4>
        OldToNewSuccessorsList;
    for (llvm::MachineBasicBlock *SuccessorMBB : MBB.successors()) {
      auto &SuccessorIntervalBegin =
          SVLocations.getStorageIntervals(*SuccessorMBB).front();
      // If the successor and the end of this MBB don't have the same
      // storage, we need to emit switch code in between them
      if (SuccessorIntervalBegin.getSVS() != MBBIntervals.back().getSVS()) {
        if (TBB == SuccessorMBB) {
          // Create a new basic block and insert it at the end
          NewTBB = MF->CreateMachineBasicBlock();
          MF->insert(MF->end(), NewTBB);
          // Insert an unconditional branch to the old FBB
          TII.insertUnconditionalBranch(*NewTBB, TBB, llvm::DebugLoc());
          NewTBB->addSuccessor(TBB);
          // Emit the SVS switch code before the branch
          MBBIntervals.back().getSVS().emitCodeToSwitchSVS(
              NewTBB->front(), SuccessorIntervalBegin.getSVS());
          OldToNewSuccessorsList.emplace_back(TBB, NewTBB);
        } else if (FBB == SuccessorMBB) {
          // Create a new basic block and insert it at the end
          NewFBB = MF->CreateMachineBasicBlock();
          MF->insert(MF->end(), NewFBB);
          // Insert an unconditional branch to the old FBB
          TII.insertUnconditionalBranch(*NewFBB, FBB, llvm::DebugLoc());
          NewFBB->addSuccessor(FBB);
          // Emit the SVS switch code before the branch
          MBBIntervals.back().getSVS().emitCodeToSwitchSVS(
              NewFBB->front(), SuccessorIntervalBegin.getSVS());
          OldToNewSuccessorsList.emplace_back(TBB, NewTBB);
        } else {
          // This is a fallthrough block; We insert the SVS code inside
          // an MBB between the two blocks
          llvm::MachineBasicBlock *NewFallthrough =
              MF->CreateMachineBasicBlock();
          MF->insert(MBB.getNextNode()->getIterator(), NewFallthrough);
          NewFallthrough->addSuccessor(SuccessorMBB);
          OldToNewSuccessorsList.emplace_back(SuccessorMBB, NewFallthrough);
        }
      }
    }
    // Insert the new branch if needed
    if ((NewFBB != nullptr && FBB != nullptr) ||
        (NewTBB != nullptr && TBB != nullptr)) {
      auto DebugLoc = MBB.getFirstInstrTerminator()->getDebugLoc();
      TII.removeBranch(MBB);
      TII.insertBranch(MBB, NewTBB, NewFBB, Cond, DebugLoc);
      for (const auto &[OldSuccessor, NewSuccessor] : OldToNewSuccessorsList) {
        MBB.removeSuccessor(OldSuccessor);
        MBB.addSuccessor(NewSuccessor);
      }
    }
  }
}

llvm::AnalysisKey FunctionPreambleDescriptorAnalysis::Key;

FunctionPreambleDescriptorAnalysis::Result
FunctionPreambleDescriptorAnalysis::run(
    Prototype &IP, PrototypeAnalysisManager &IPAM) {
  Result Out;

  llvm::Module &TargetModule = IP.getTargetModule();
  llvm::Module &IModule = IP.getInstrumentationModule();

  // The IP-side proxy hands out the outer ModuleAnalysisManager; the same
  // MAM caches results for both modules, keyed by the module reference.
  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<ModuleAnalysisManagerPrototypeProxy>(IP)
          .getManager();

  llvm::FunctionAnalysisManager &TargetFAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();
  llvm::FunctionAnalysisManager &IFAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(IModule)
          .getManager();

  const InjectedPayloadAndInstPoint &IPIP =
      MAM.getResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  for (llvm::Function &F : TargetModule) {
    if (F.isDeclaration())
      continue;

    llvm::MachineFunction &MF =
        TargetFAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();

    // KernelPreambleSpecs and DeviceFunctionPreambleSpecs each carry their own
    // RequestedKernelArguments field; grab a reference to whichever this MF
    // maps to so the payload walk below is oblivious to the distinction.
    auto &RequestedKernelArguments =
        F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL
            ? Out.Kernels.insert({&MF, {}})
                  .first->second.RequestedKernelArguments
            : Out.DeviceFunctions.insert({&MF, {}})
                  .first->second.RequestedKernelArguments;

    for (llvm::MachineBasicBlock &MBB : MF) {
      for (llvm::MachineInstr &MI : MBB) {
        if (!IPIP.contains(MI))
          continue;
        for (llvm::Function *Payload : IPIP.at(MI)) {
          const InjectedPayloadSideEffects &SE =
              IFAM.getResult<InjectedPayloadSideEffectsAnalysis>(*Payload);
          for (ScalarValueArgument SA : SE.svas())
            RequestedKernelArguments.insert(SA);
        }
      }
    }
  }

  return Out;
}
} // namespace luthier
