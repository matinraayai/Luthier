//===-- IntrinsicMIRLoweringPass.cpp --------------------------------------===//
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
/// This file implements the Intrinsic MIR Lowering Pass.
//===----------------------------------------------------------------------===//
#include "tooling_common/IntrinsicMIRLoweringPass.hpp"
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>
#include <tooling_common/PhysicalRegAccessVirtualizationPass.hpp>

namespace luthier {

char IntrinsicMIRLoweringPass::ID = 0;

bool IntrinsicMIRLoweringPass::runOnMachineFunction(llvm::MachineFunction &MF) {
  bool Changed{false};
  // The set of physical registers used without being defined in the body
  // after intrinsics has been lowered
  llvm::LivePhysRegs InsertedPhysRegs(*MF.getSubtarget().getRegisterInfo());
  for (auto &MBB : MF) {
    const auto &RegAccessVirtualizer =
        getAnalysis<PhysicalRegAccessVirtualizationPass>();
    llvm::DenseMap<llvm::MCRegister, llvm::Register> OverwrittenPhysRegs;

    for (auto &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.isInlineAsm()) {
        // The Asm string is of type symbol
        auto IntrinsicIdxAsString =
            MI.getOperand(llvm::InlineAsm::MIOp_AsmString).getSymbolName();
        // TODO: catch the exception
        unsigned int IntrinsicIdx = std::stoul(IntrinsicIdxAsString);
        llvm::SmallVector<std::pair<llvm::InlineAsm::Flag, llvm::Register>, 4>
            ArgVec;
        for (unsigned I = llvm::InlineAsm::MIOp_FirstOperand,
                      NumOps = MI.getNumOperands();
             I < NumOps; ++I) {
          const llvm::MachineOperand &MO = MI.getOperand(I);
          if (!MO.isImm())
            continue;
          const llvm::InlineAsm::Flag F(MO.getImm());
          const llvm::Register Reg(MI.getOperand(I + 1).getReg());
          ArgVec.emplace_back(F, Reg);
          // Skip to one before the next operand descriptor, if it exists.
          I += F.getNumOperandRegisters();
        }
        auto *TII = MF.getSubtarget().getInstrInfo();
        auto &MRI = MF.getRegInfo();

        auto MIBuilder = [&](int Opcode) {
          auto Builder =
              llvm::BuildMI(MBB, MI, llvm::MIMetadata(MI), TII->get(Opcode));
          return Builder;
        };

        auto VirtRegBuilder = [&](const llvm::TargetRegisterClass *RC) {
          return MRI.createVirtualRegister(RC);
        };

        auto PhysRegAccessor = [&](llvm::MCRegister Reg) {
          if (OverwrittenPhysRegs.contains(Reg))
            return OverwrittenPhysRegs[Reg];
          else
            return RegAccessVirtualizer.getMCRegLocationInMBB(Reg, MBB);
        };

        llvm::DenseMap<llvm::MCRegister, llvm::Register> ToBeOverwrittenRegs;

        auto &IRLoweringInfo = MIRLoweringMap[IntrinsicIdx].second;

        auto IRProcessor =
            IntrinsicsProcessors.find(IRLoweringInfo.getIntrinsicName());
        if (IRProcessor == IntrinsicsProcessors.end())
          MF.getFunction().getContext().emitError(
              "Intrinsic processor was not found in the intrinsic processor "
              "map.");
        if (auto Err = IRProcessor->second.MIRProcessor(
                IRLoweringInfo, ArgVec, MIBuilder, VirtRegBuilder, MF,
                PhysRegAccessor, ToBeOverwrittenRegs)) {
          MF.getFunction().getContext().emitError(
              "Failed to lower the intrinsic; Error message: " +
              toString(std::move(Err)));
        }
        // Remove the dummy inline assembly
        MI.eraseFromParent();
        Changed = true;
        // Take the
        if (!ToBeOverwrittenRegs.empty()) {
          for (const auto &[PhysReg, VirtReg] : ToBeOverwrittenRegs) {
            llvm::outs() << "PhysReg: " << MF.getSubtarget().getRegisterInfo()->getName(PhysReg) << ", virt reg: << " << llvm::printReg(VirtReg, MF.getSubtarget().getRegisterInfo()) << "\n";
            llvm::outs() << "What will be replaced? " << llvm::printReg(PhysRegAccessor(PhysReg), MF.getSubtarget().getRegisterInfo()) << "\n";
            for (auto &Use : MRI.use_operands(PhysRegAccessor(PhysReg))) {
              Use.getParent()->print(llvm::outs());
              Use.setReg(VirtReg);
            }
            OverwrittenPhysRegs.insert_or_assign(PhysReg, VirtReg);
          }
        }
      }
    }
  }
//  // Update the used physical defs without defines to keep the machine verifier
//  // happy
//  llvm::addLiveIns(MF.front(), InsertedPhysRegs);
  return Changed;
}

void IntrinsicMIRLoweringPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<luthier::PhysicalRegAccessVirtualizationPass>();
  AU.addPreserved<luthier::PhysicalRegAccessVirtualizationPass>();
  AU.addPreservedID(llvm::MachineLoopInfoID);
  AU.addPreserved<llvm::SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
};

} // namespace luthier