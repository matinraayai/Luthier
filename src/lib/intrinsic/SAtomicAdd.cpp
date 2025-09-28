#include "AMDGPUTargetMachine.h"
#include "SIRegisterInfo.h"
#include "intrinsic/WriteReg.hpp"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/GenericLuthierError.h"
#include "luthier/common/LuthierError.h"
#include "luthier/llvm/streams.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/User.h>
#include <llvm/MC/MCRegister.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
sAtomicAddIRProcessor(const llvm::Function &Intrinsic,
                      const llvm::CallInst &User,
                      const llvm::GCNTargetMachine &TM) {
  // The User must only have 2 operands
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 2,
      llvm::formatv("Expected two operands to be passed to the "
                    "luthier::sAtomicAdd intrinsic '{0}', got {1}.",
                    User, User.arg_size())));

  luthier::IntrinsicIRLoweringInfo Out;

  // The output operand will be in an SGPR
  Out.setReturnValueInfo(&User, "s");

  // Both operands of the instruction will be in SGPRs
  for (int i = 0; i < 2; i++) {
    auto *SrcReg = User.getArgOperand(i);
    Out.addArgInfo(SrcReg, "s");
  }
  return Out;
}

llvm::Error sAtomicAddMIRProcessor(
    const IntrinsicIRLoweringInfo &IRLoweringInfo,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const std::function<llvm::Register(KernelArgumentType)> &,
    const llvm::MachineFunction &MF,
    const std::function<llvm::Register(llvm::MCRegister)> &PhysRegAccessor,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &PhysRegsToBeOverwritten) {
  // There should be three virtual register involved in the operation
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 3,
      llvm::formatv(
          "Expected only a single virtual register to be passed down to the "
          "MIR lowering stage of luthier::sAtomicAdd, instead got {0}",
          Args.size())));
  // The first arg should be of reg def kind
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(), "The first virtual register argument for "
                                    "luthier::sAtomicAdd is not a def."));
  // The second and third should be of reg def kind
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[1].first.isRegUseKind(), "The second virtual register argument for "
                                    "luthier::sAtomicAdd is not a use."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[2].first.isRegUseKind(), "The third virtual register argument for "
                                    "luthier::sAtomicAdd is not a use."));
  // Get all the involved registers
  llvm::Register Output = Args[0].second;
  llvm::Register SBaseAddress = Args[1].second;
  llvm::Register SData = Args[2].second;
  auto &MRI = MF.getRegInfo();
  auto &TRI = *MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();

  auto OutputRegSize = TRI.getRegSizeInBits(Output, MRI);

  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      OutputRegSize == 64 || OutputRegSize == 32,
      llvm::formatv("Output register size of luthier::sAtomicAdd must be "
                    "either 64 or 32 bits, got {0} instead.",
                    OutputRegSize)));
  uint16_t SAtomicAddOpcode = OutputRegSize == 64
                                  ? llvm::AMDGPU::S_ATOMIC_ADD_X2_IMM_RTN
                                  : llvm::AMDGPU::S_ATOMIC_ADD_IMM_RTN;

  (void)MIBuilder(SAtomicAddOpcode)
      .addReg(Output, llvm::RegState::Define)
      .addReg(SData)
      .addReg(SBaseAddress)
      .addImm(0)
      .addImm(0);
  return llvm::Error::success();
}

} // namespace luthier