#include "tooling_common/intrinsic/ReadReg.hpp"
#include "common/error.hpp"
#include <AMDGPUTargetMachine.h>
#include <SIRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/User.h>
#include <llvm/MC/MCRegister.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
luthier::readRegIRProcessor(const llvm::Function &Intrinsic,
                            const llvm::CallInst &User,
                            const llvm::GCNTargetMachine &TM) {
  auto *TRI = TM.getSubtargetImpl(Intrinsic)->getRegisterInfo();
  // The first argument specifies the MCRegister Enum that will be read
  // The enum value should be constant; A different intrinsic should be used
  // if reg indexing is needed at runtime
  auto *Arg = llvm::dyn_cast<llvm::ConstantInt>(User.getArgOperand(0));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Arg != nullptr));
  // Get the MCRegister from the first argument's content
  llvm::MCRegister Reg(Arg->getZExtValue());
  // Check if the enum value is indeed a physical register
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(llvm::MCRegister::isPhysicalRegister(Reg.id())));
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
    llvm_unreachable("register type not implemented");

  luthier::IntrinsicIRLoweringInfo Out;
  // Set the output's constraint
  Out.setReturnValueInfo(User, Constraint);
  // Save the MCReg to be encoded during MIR processing
  Out.setLoweringData(Reg);

  return Out;
}

} // namespace luthier