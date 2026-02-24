#include "luthier/Tooling/InstructionModel.h"
#include "luthier/Common/GenericLuthierError.h"
#include <SIRegisterInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <luthier/Tooling/TargetMachineInstrMDNode.h>

namespace luthier {

llvm::Value *MCRegisterValueMap::getValue(llvm::MCRegister Reg) {

  const llvm::TargetRegisterClass *RC = Info.getMinimalPhysRegClass(Reg);
  const auto RegSize = Info.getRegSizeInBits(*RC);

  if (RegSize > 32) {
    llvm::SmallVector<std::reference_wrapper<llvm::Value>> SubValues;
    // Split the src reg into 32-bit regs, and merge them in the
    size_t NumChannels = RegSize / 32;
    for (int i = 0; i < NumChannels; i++) {
      unsigned SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
      llvm::MCRegister SubReg = Info.getSubReg(Reg, SubIdx);
      auto It = RegValMap.find(SubReg);
      if (It == RegValMap.end()) {
        return nullptr;
      }
      SubValues.push_back(It->getSecond());
    }
    llvm::Value *OutVecVal{nullptr};
    llvm::IRBuilder Builder{Ctx};
    llvm::FixedVectorType *FixedVecType =
        llvm::FixedVectorType::get(Builder.getInt32Ty(), NumChannels);

    /// Create a merge instruction to merge the sub-regs into a single value
    for (auto [Idx, SubVal] : llvm::enumerate(SubValues)) {
      if (Idx == 0)
        OutVecVal =
            Builder.CreateInsertElement(FixedVecType, &SubVal.get(), Idx);
      else
        OutVecVal = Builder.CreateInsertElement(OutVecVal, &SubVal.get(), Idx);
    }
    return OutVecVal;
  } else if (RegSize == 32 || RegSize == 1) {
    /// If we have a direct hit on the queried register return it right away
    auto It = RegValMap.find(Reg);
    if (It == RegValMap.end()) {
      return nullptr;
    }
    return &It->getSecond().get();
  } else {
    auto SuperReg = Info.get32BitRegister(Reg);
    auto SuperRegIt = RegValMap.find(Reg);
    if (SuperRegIt == RegValMap.end()) {
      return nullptr;
    }
    auto SubIdx = Info.getSubRegIndex(SuperReg, Reg);
    assert(SubIdx <= 1 && "Sub index is out of range");
    /// Cast the register to a vector of <2 x i16> and extract the index
    llvm::IRBuilder Builder{Ctx};

    llvm::Value *DoubleI16Cast = Builder.CreateBitCast(
        &SuperRegIt->getSecond().get(),
        llvm::FixedVectorType::get(Builder.getInt16Ty(), 2));
    return Builder.CreateExtractElement(DoubleI16Cast, SubIdx);
  }
}

void MCRegisterValueMap::setValue(llvm::MCRegister Reg, llvm::Value &Val) {
  const llvm::TargetRegisterClass *RC = Info.getMinimalPhysRegClass(Reg);
  const auto RegSize = Info.getRegSizeInBits(*RC);
  llvm::IRBuilder Builder{Ctx};
  if (RegSize > 32) {
    /// Create an extract element for each sub-reg and set it in the map
    // Split the src reg into 32-bit regs, and merge them in the
    size_t NumChannels = RegSize / 32;
    for (int i = 0; i < NumChannels; i++) {
      unsigned SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
      llvm::MCRegister SubReg = Info.getSubReg(Reg, SubIdx);
      RegValMap.emplace_or_assign(
          SubReg, *Builder.CreateBitCast(Builder.CreateExtractElement(&Val, i),
                                         Builder.getInt32Ty()));
    }
  } else if (RegSize == 32 || RegSize == 1) {
    RegValMap.emplace_or_assign(
        Reg, Builder.CreateBitCast(&Val, RegSize == 32 ? Builder.getInt32Ty()
                                                       : Builder.getInt1Ty()));
  } else {
    auto SuperReg = Info.get32BitRegister(Reg);
    auto SubIdx = Info.getSubRegIndex(SuperReg, Reg);
    assert(SubIdx <= 1 && "Sub index is out of range");
    llvm::Value *SuperRegVal = [&]() -> llvm::Value * {
      auto SuperRegIt = RegValMap.find(Reg);
      if (SuperRegIt == RegValMap.end()) {
        /// Create an all zero value if we don't have the 32-bit reg initialized
        return llvm::ConstantInt::get(llvm::IntegerType::getInt32Ty(Ctx), 0);
      }
      return &SuperRegIt->getSecond().get();
    }();

    /// Cast the register to a vector of <2 x i16> and insert the sub-idx, then
    /// cast it back to i32

    llvm::Value *DoubleI16Cast = Builder.CreateBitCast(
        SuperRegVal, llvm::FixedVectorType::get(Builder.getInt16Ty(), 2));
    RegValMap.insert_or_assign(Builder.CreateBitCast(
        Builder.CreateInsertElement(DoubleI16Cast, &Val, SubIdx),
        Builder.getInt32Ty()));
  }
}

llvm::Expected<llvm::Value &>
raiseMachineInstr(const llvm::MachineInstr &MI,
                  MCRegisterValueMap &RegisterValueMap) {
  llvm::LLVMContext &Ctx = RegisterValueMap.getContext();
  llvm::IRBuilder Builder{Ctx};
  llvm::Value *Out{nullptr};
  switch (MI.getOpcode()) {
  case llvm::AMDGPU::S_GETPC_B64:
    auto *InstMD = TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
    if (!InstMD)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "GETPC instruction doesn't have a metadata to obtain its address");
    std::optional<uint64_t> TraceAddrIfPresent = InstMD->getTraceInstrAddress();
    if (!TraceAddrIfPresent.has_value()) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          "GETPC instruction doesn't have a trace address");
    }
    Out = llvm::ConstantInt::get(llvm::IntegerType::getInt64Ty(Ctx),
                                 *TraceAddrIfPresent + 4);
    RegisterValueMap.setValue(MI.getOperand(0).getReg(), *Out);
    break;
  case llvm::AMDGPU::S_ADD_I32:
    const llvm::MachineOperand &Op1 = MI.getOperand(1);
    const llvm::MachineOperand &Op2 = MI.getOperand(2);
    llvm::Value *Op1Val = Op1.isImm() ? Builder.getInt32(Op1.getImm())
                                      : RegisterValueMap.getValue(Op1.getReg());
    llvm::Value *Op2Val = Op2.isImm() ? Builder.getInt32(Op2.getImm())
                                      : RegisterValueMap.getValue(Op2.getReg());
    if (!Op1Val) {
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to create the first operand");
    }
    if (!Op2Val) {
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to create the second operand");
    }
    Out = Builder.CreateAdd(Op1Val, Op2Val);
    RegisterValueMap.setValue(MI.getOperand(0).getReg(), *Out);
    break;
  default:
  }
}
} // namespace luthier