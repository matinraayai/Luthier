#include "tooling_common/intrinsic/WriteExec.hpp"
#include "common/Error.hpp"
#include <AMDGPUTargetMachine.h>
#include <SIRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/MC/MCRegister.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
writeExecIRProcessor(const llvm::Function &Intrinsic,
                     const llvm::CallInst &User,
                     const llvm::GCNTargetMachine &TM) {
  // The User must only have 1 operand
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(User.arg_size() == 1,
                          "Expected one operand to be passed to the "
                          "luthier::writeExec intrinsic '{0}', got {1}.",
                          User, User.arg_size()));
  luthier::IntrinsicIRLoweringInfo Out;
  // Set the output's constraint
  Out.setReturnValueInfo(&User, "s");
  // The first argument specifies the source register
  auto *SrcReg = User.getArgOperand(0);
  Out.addArgInfo(SrcReg, "s");

  return Out;
}

llvm::Error writeExecMIRProcessor(
    const IntrinsicIRLoweringInfo &IRLoweringInfo,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::MachineFunction &MF,
    const std::function<llvm::Register(llvm::MCRegister)> &PhysRegAccessor,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &PhysRegsToBeOverwritten) {
  // There should be only a single virtual register involved in the operation
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Args.size() == 1,
                          "Number of virtual register arguments "
                          "involved in the MIR lowering stage of "
                          "luthier::writeExec is {0} instead of 1.",
                          Args.size()));
  // It should be of reg use kind
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      Args[0].first.isRegUseKind(),
      "The register argument of luthier::writeExec is not a definition."));
  llvm::Register InputReg = Args[0].second;

  MIBuilder(llvm::AMDGPU::COPY)
      .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Define)
      .addReg(InputReg);

  return llvm::Error::success();
}

} // namespace luthier