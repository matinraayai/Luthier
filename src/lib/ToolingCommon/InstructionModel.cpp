#include "luthier/tooling/InstructionModel.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/GenericLuthierError.h>

namespace luthier {

llvm::Expected<std::pair<std::unique_ptr<llvm::Module>,
                         std::unique_ptr<llvm::MachineModuleInfo>>>
getBackSlice(const llvm::MachineOperand &RegOperand) {
  if (!RegOperand.isReg()) {
    return LUTHIER_MAKE_ERROR(GenericLuthierError,
                              "Passed operand is not a register");
  }
  if (!RegOperand.getReg().isPhysical())
    return LUTHIER_MAKE_ERROR(GenericLuthierError,
                              "Passed register operand is not physical");

  const llvm::MachineInstr *OperandInstr = RegOperand.getParent();
  if (!OperandInstr) {
    return LUTHIER_MAKE_ERROR(GenericLuthierError,
                              "Parent of the machine operand is nullptr");
  }

  auto CurrInstr = OperandInstr->getReverseIterator();

  llvm::DenseSet<llvm::MCRegister> DefsToFind{RegOperand.getReg()};

  while (!DefsToFind.empty()) {
  }
  llvm::MCRegAliasIterator R(PhysReg, TRI, true);

  OperandInstr->getMF()->getRegInfo().def_instr_begin()

      /// Mapping between every value in the back slice module and its
      /// corresponding machine operand
      llvm::DenseMap<const llvm::Value *, const llvm::MachineOperand *>
          ValueMap{};

  /// Mapping between the machine basic blocks in the target module and the
  /// lifted basic block in the slice module
  llvm::SmallDenseMap<llvm::MachineBasicBlock *, llvm::BasicBlock *>
      MBBToBBMap{};

  /// Mapping between the machine operands of each
  llvm::SmallDenseMap<
      llvm::BasicBlock *,
      llvm::SmallDenseMap<llvm::MachineOperand *, llvm::Value *>>
      BBEntryValueMap{};

  llvm::SmallDenseMap<
      llvm::BasicBlock *,
      llvm::SmallDenseMap<llvm::MachineOperand *, llvm::Value *>>
      BBExitValueMap{};

  auto *TRI = OperandInstr->getMF()->getSubtarget().getRegisterInfo();

  llvm::LivePhysRegs DefRegsToBeVisited{*TRI};

  DefRegsToBeVisited.addReg(RegOperand.getReg());

  const llvm::MachineBasicBlock *CurrentBlock = OperandInstr->getParent();

  const llvm::MachineInstr *CurrentInst = OperandInstr;

  llvm::SmallDenseSet<const llvm::BasicBlock *> VisitedBlocks{};

  while (!DefRegsToBeVisited.empty() && CurrentBlock != nullptr) {

    while (CurrentInst->getReverseIterator() != CurrentBlock->rend()) {
      for (const llvm::MachineOperand &DefRegOp : CurrentInst->all_defs()) {
        if (DefRegsToBeVisited.contains(DefRegOp.getReg())) {
        }
      }
    }
  }

  return llvm::Error::success();
}

} // namespace luthier