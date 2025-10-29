#include "intrinsic/ReadReg.hpp"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/GenericLuthierError.h"
#include "luthier/common/LuthierError.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/User.h>
#include <llvm/MC/MCRegister.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
readRegIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                   const llvm::GCNTargetMachine &TM) {
  // The User must only have 1 operand
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 1,
      llvm::formatv("Expected one operand to be passed to the "
                    "luthier::readReg intrinsic '{0}', got {1}.",
                    User, User.arg_size())));

  auto *TRI = TM.getSubtargetImpl(Intrinsic)->getRegisterInfo();
  // The first argument specifies the MCRegister Enum that will be read
  // The enum value should be constant; A different intrinsic should be used
  // if reg indexing is needed at runtime
  auto *Arg = llvm::dyn_cast<llvm::ConstantInt>(User.getArgOperand(0));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Arg != nullptr, "The first argument of the luthier::readReg intrinsic is "
                      "not a constant integer."));
  // Get the MCRegister from the first argument's content
  llvm::MCRegister Reg(Arg->getZExtValue());
  // Check if the enum value is indeed a physical register
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      llvm::MCRegister::isPhysicalRegister(Reg.id()),
      llvm::formatv("The first argument of the luthier::readReg {0}"
                    "intrinsic is not an LLVM MCRegister.",
                    Reg.id())));
  // Get the type of the read register and encode its inline asm constraint
  auto *PhysRegClass = TRI->getPhysRegBaseClass(Reg);
  std::string Constraint;
  if (llvm::SIRegisterInfo::isAGPRClass(PhysRegClass))
    Constraint = "a";
  else if (llvm::SIRegisterInfo::isVGPRClass(PhysRegClass))
    Constraint = "v";
  else if (llvm::SIRegisterInfo::isSGPRClass(PhysRegClass))
    Constraint = "s";
  else
    return llvm::make_error<GenericLuthierError>(
        llvm::formatv("Unable to find a suitable register class for reading "
                      "the MC Register {0}.",
                      Reg.id()));

  luthier::IntrinsicIRLoweringInfo Out;
  // Set the output's constraint
  Out.setReturnValueInfo(&User, Constraint);
  // Save the MCReg to be encoded during MIR processing
  Out.setLoweringData(Reg);

  Out.requestAccessToPhysicalRegister(Reg);

  return Out;
}
llvm::Error readRegMIRProcessor(
    const IntrinsicIRLoweringInfo &IRLoweringInfo,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const std::function<llvm::Register(KernelArgumentType)> &,
    const llvm::MachineFunction &MF,
    const std::function<llvm::Register(llvm::MCRegister)> &PhysRegAccessor,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &PhysRegsToBeOverwritten) {
  // There should be only a single virtual register involved in the operation
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 1, llvm::formatv("Number of virtual register arguments "
                                      "involved in the MIR lowering stage of "
                                      "luthier::readReg is {0} instead of 1.",
                                      Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(),
      "The register argument of luthier::readReg is not a definition."));
  llvm::Register Output = Args[0].second;
  auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  auto *TRI = ST.getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  auto Src = IRLoweringInfo.getLoweringData<llvm::MCRegister>();
  auto SrcRegSize = TRI->getRegSizeInBits(Src, MRI);
  uint64_t DestRegSize = TRI->getRegSizeInBits(Output, MRI);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      SrcRegSize == DestRegSize || (DestRegSize == 32 && SrcRegSize == 1),
      "The input register and the destination register of "
      "luthier::readReg don't have the same size."));

  if (SrcRegSize > 32) {
    // First create a reg sequence MI
    auto Builder = MIBuilder(llvm::AMDGPU::REG_SEQUENCE);

    auto MergedReg = VirtRegBuilder(TRI->getPhysRegBaseClass(Src));
    Builder.addReg(MergedReg, llvm::RegState::Define);

    // Split the src reg into 32-bit regs, and merge them in the
    size_t NumChannels = SrcRegSize / 32;
    for (int i = 0; i < NumChannels; i++) {
      auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
      auto Reg = TRI->getSubReg(Src, SubIdx);
      Builder.addReg(PhysRegAccessor(Reg)).addImm(SubIdx);
    }
    // Do the copy
    MIBuilder(llvm::AMDGPU::COPY)
        .addReg(Output, llvm::RegState::Define)
        .addReg(MergedReg);
  } else if (SrcRegSize == 32 || SrcRegSize == 1) {
    MIBuilder(llvm::AMDGPU::COPY)
        .addReg(Output, llvm::RegState::Define)
        .addReg(PhysRegAccessor(Src));
  } else {
    auto SuperReg = TRI->get32BitRegister(Src);
    auto SubIdx = TRI->getSubRegIndex(SuperReg, Src);
    MIBuilder(llvm::AMDGPU::COPY)
        .addReg(Output, llvm::RegState::Define)
        .addReg(PhysRegAccessor(SuperReg), 0, SubIdx);
  }

  return llvm::Error::success();
}

} // namespace luthier