#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include "luthier/llvm/streams.h"
#include "tooling_common/intrinsic/WriteReg.hpp"
#include <AMDGPUTargetMachine.h>
#include <SIRegisterInfo.h>
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
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(User.arg_size() == 2,
                          "Expected two operands to be passed to the "
                          "luthier::sAtomicAdd intrinsic '{0}', got {1}.",
                          User, User.arg_size()));

  luthier::IntrinsicIRLoweringInfo Out;

  luthier::outs() << User << "\n";

  Out.setReturnValueInfo(&User, "s");

  for (int i = 0; i < 2; i++) {
    auto *SrcReg = User.getArgOperand(i);
    Out.addArgInfo(SrcReg, "s");
  }
  luthier::outs() << "Before returning out;\n";
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
  //  // There should be only a single virtual register involved in the
  //  operation LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
  //      Args.size() == 2,
  //      "Expected only a single virtual register to be passed down to the "
  //      "MIR lowering stage of luthier::sAtomicAdd, instead got {0}",
  //      Args.size()));
  //  // It should be of reg use kind
  //  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
  //      Args[0].first.isRegUseKind(),
  //      "The virtual register argument for luthier::writeReg is not a use."));
  //  llvm::Register Address = Args[0].second;
  luthier::outs() << "SDst operand idx: "
                  << llvm::AMDGPU::getNamedOperandIdx(
                         llvm::AMDGPU::S_ATOMIC_ADD_X2_IMM,
                         llvm::AMDGPU::OpName::sdst)
                  << "\n";

  luthier::outs() << "SData operand idx: "
                  << llvm::AMDGPU::getNamedOperandIdx(
                         llvm::AMDGPU::S_ATOMIC_ADD_X2_IMM,
                         llvm::AMDGPU::OpName::sdata)
                  << "\n";
  luthier::outs() << "Sbase operand idx: "
                  << llvm::AMDGPU::getNamedOperandIdx(
                         llvm::AMDGPU::S_ATOMIC_ADD_X2_IMM,
                         llvm::AMDGPU::OpName::sbase)
                  << "\n";

  luthier::outs() << "SOffset operand idx: "
                  << llvm::AMDGPU::getNamedOperandIdx(
                         llvm::AMDGPU::S_ATOMIC_ADD_X2_IMM,
                         llvm::AMDGPU::OpName::offset)
                  << "\n";

  luthier::outs() << "SOffset operand idx: "
                  << llvm::AMDGPU::getNamedOperandIdx(
                         llvm::AMDGPU::S_ATOMIC_ADD_X2_IMM,
                         llvm::AMDGPU::OpName::cpol)
                  << "\n";

  auto &MRI = MF.getRegInfo();
  auto &TRI = *MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
  llvm::Register OutputAddress = Args[1].second;
  if (TRI.isVGPR(MRI, Args[1].second)) {
    OutputAddress = VirtRegBuilder(&llvm::AMDGPU::SGPR_64RegClass);
    MIBuilder(llvm::AMDGPU::COPY)
        .addReg(OutputAddress, llvm::RegState::Define)
        .addReg(Args[1].second);
  }

  MIBuilder(llvm::AMDGPU::S_ATOMIC_ADD_X2_IMM)
      //      .addReg(Args[0].second, llvm::RegState::Define)
      .addReg(OutputAddress)
      .addReg(Args[2].second)
      .addImm(0)
      .addImm(1);
  MF.print(luthier::outs());
  //
  //
  //  uint64_t DestRegSize = TRI->getRegSizeInBits(Dest, MRI);
  //  uint64_t InputRegSize = TRI->getRegSizeInBits(InputReg, MRI);
  //  // Check if both the input value and the destination reg have the same
  //  size LUTHIER_RETURN_ON_ERROR(
  //      LUTHIER_ERROR_CHECK(InputRegSize == DestRegSize,
  //                          "The input register and the destination register
  //                          of " "luthier::writeReg don't have the same
  //                          size."));
  //
  //  if (DestRegSize > 32) {
  //    // Split the input reg into subregs, and set each subreg to replace the
  //    // virtual register value of the physical reg
  //    size_t NumChannels = DestRegSize / 32;
  //    const llvm::TargetRegisterClass *InputRegClass =
  //    MRI.getRegClass(InputReg); for (int i = 0; i < NumChannels; i++) {
  //      auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
  //      auto InputSubRegClass = TRI->getSubRegisterClass(InputRegClass,
  //      SubIdx); auto SubReg = VirtRegBuilder(InputSubRegClass);
  //      MIBuilder(llvm::AMDGPU::COPY)
  //          .addReg(SubReg, llvm::RegState::Define)
  //          .addReg(InputReg, 0, SubIdx);
  //      PhysRegsToBeOverwritten.insert({TRI->getSubReg(Dest, SubIdx),
  //      SubReg});
  //    }
  //  } else if (DestRegSize == 32) {
  //    PhysRegsToBeOverwritten.insert({Dest, InputReg});
  //  } else {
  //    auto SuperRegDest = TRI->get32BitRegister(Dest);
  //    auto SubIdx = TRI->getSubRegIndex(SuperRegDest, Dest);
  //    auto SuperRegVirt =
  //    VirtRegBuilder(TRI->getPhysRegBaseClass(SuperRegDest));
  //    MIBuilder(llvm::AMDGPU::INSERT_SUBREG)
  //        .addReg(SuperRegVirt, llvm::RegState::Define)
  //        .addReg(PhysRegAccessor(SuperRegDest))
  //        .addReg(InputReg)
  //        .addImm(SubIdx);
  //    PhysRegsToBeOverwritten.insert({SuperRegDest, SuperRegVirt});
  //  }
  return llvm::Error::success();
}

} // namespace luthier