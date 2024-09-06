#include "tooling_common/intrinsic/WriteReg.hpp"
#include "common/Error.hpp"
#include <AMDGPUTargetMachine.h>
#include <SIRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/User.h>
#include <llvm/MC/MCRegister.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
writeRegIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                    const llvm::GCNTargetMachine &TM) {
  auto *TRI = TM.getSubtargetImpl(Intrinsic)->getRegisterInfo();
  // The User must only have 2 operands
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(User.getNumOperands() != 2));

  luthier::IntrinsicIRLoweringInfo Out;
  // The first argument specifies the MCRegister Enum that will be read
  // The enum value should be constant; A different intrinsic should be used
  // if reg indexing is needed at runtime
  auto *DestRegEnum = llvm::dyn_cast<llvm::ConstantInt>(User.getArgOperand(0));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(DestRegEnum != nullptr));
  // Get the MCRegister from the first argument's content
  llvm::MCRegister DestReg(DestRegEnum->getZExtValue());
  // Check if the enum value is indeed a physical register
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(llvm::MCRegister::isPhysicalRegister(DestReg.id())));
  // Get the type of the dest register and encode its inline asm constraint
  auto *PhysRegClass = TRI->getPhysRegBaseClass(DestReg);
  std::string Constraint;
  if (llvm::SIRegisterInfo::isAGPRClass(PhysRegClass))
    Constraint = "a";
  else if (llvm::SIRegisterInfo::isVGPRClass(PhysRegClass))
    Constraint = "v";
  else if (llvm::SIRegisterInfo::isSGPRClass(PhysRegClass))
    Constraint = "s";
  else
    llvm_unreachable("register type not implemented");
  // Set the output's constraint
  Out.setReturnValueInfo(&User, Constraint);
  // The second argument specifies the source register
  auto *SrcReg = User.getArgOperand(1);
  Out.addArgInfo(SrcReg, Constraint);

  // Save the MCReg to be encoded during MIR processing
  Out.setLoweringData(DestReg);
  // Tell intrinsic lowering that access to the physical
  // destination is requested
  Out.requestAccessToPhysicalRegister(DestReg);

  return Out;
}

llvm::Error writeRegMIRProcessor(
    const IntrinsicIRLoweringInfo &IRLoweringInfo,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::MachineFunction &MF,
    const std::function<llvm::Register(llvm::MCRegister)> &PhysRegAccessor,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &PhysRegsToBeOverwritten) {
  // There should be only a single virtual register involved in the operation
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Args.size() == 1));
  // It should be of reg use kind
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Args[0].first.isRegUseKind()));
  llvm::Register InputReg = Args[0].second;

  auto Dest = IRLoweringInfo.getLoweringData<llvm::MCRegister>();

  auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  auto *TRI = ST.getRegisterInfo();
  auto &MRI = MF.getRegInfo();

  uint64_t DestRegSize = TRI->getRegSizeInBits(Dest, MRI);
  uint64_t InputRegSize = TRI->getRegSizeInBits(InputReg, MRI);
  // Check if both the input value and the destination reg have the same size
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(InputRegSize == DestRegSize));


  if (DestRegSize > 32) {
    // Split the input reg into subregs, and set each subreg to replace the
    // virtual register value of the physical reg
    size_t NumChannels = DestRegSize / 32;
    const llvm::TargetRegisterClass *InputRegClass = MRI.getRegClass(InputReg);
    for (int i = 0; i < NumChannels; i++) {
      auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
      auto InputSubRegClass = TRI->getSubRegisterClass(InputRegClass, SubIdx);
      auto SubReg = VirtRegBuilder(InputSubRegClass);
      MIBuilder(llvm::AMDGPU::COPY)
          .addReg(SubReg, llvm::RegState::Define)
          .addReg(InputReg, 0, SubIdx);
      PhysRegsToBeOverwritten.insert({TRI->getSubReg(Dest, SubIdx), SubReg});
    }
  } else if (DestRegSize == 32) {
    PhysRegsToBeOverwritten.insert({Dest, InputReg});
  } else {
    auto SuperRegDest = TRI->get32BitRegister(Dest);
    auto SubIdx = TRI->getSubRegIndex(SuperRegDest, Dest);
    auto SuperRegVirt = VirtRegBuilder(TRI->getPhysRegBaseClass(SuperRegDest));
    MIBuilder(llvm::AMDGPU::INSERT_SUBREG)
        .addReg(SuperRegVirt, llvm::RegState::Define)
        .addReg(PhysRegAccessor(SuperRegDest))
        .addReg(InputReg)
        .addImm(SubIdx);
    PhysRegsToBeOverwritten.insert({SuperRegDest, SuperRegVirt});
  }
  return llvm::Error::success();
}

} // namespace luthier