//===-- ReadReg.cpp -------------------------------------------------------===//
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
/// This file implements the read reg intrinsic.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/ReadReg.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/LuthierError.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Type.h>
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
  // The first argument specifies the MCRegister Enum that will be read.
  // The enum value should be constant; a different intrinsic should be used
  // if register indexing is needed at runtime.
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
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Unable to find a suitable register class for reading "
                      "the MC Register {0}.",
                      Reg.id()));

  IntrinsicIRLoweringInfo Out;
  Out.setReturnValueInfo(User, Constraint);
  // Forward the physical register enum value to the MIR lowering stage
  Out.addExtraLoweringValue(*llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(Intrinsic.getContext()), Reg.id()));
  Out.getEffects().ReadPhysRegs.push_back(Reg);

  return Out;
}

llvm::Error readRegMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *Payload,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &ReadPhysRegVRegs,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &) {
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

  // Extract the source physical register from the forwarded payload MDNode
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Payload && Payload->getNumOperands() == 1,
      "luthier::readReg MIR payload must contain exactly one operand"));
  auto *RegMeta =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(Payload->getOperand(0));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      RegMeta != nullptr,
      "luthier::readReg payload operand is not a ConstantAsMetadata"));
  llvm::MCRegister Src(
      llvm::cast<llvm::ConstantInt>(RegMeta->getValue())->getZExtValue());

  auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  auto *TRI = ST.getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  auto SrcRegSize = TRI->getRegSizeInBits(Src, MRI);
  uint64_t DestRegSize = TRI->getRegSizeInBits(Output, MRI);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      SrcRegSize == DestRegSize || (DestRegSize == 32 && SrcRegSize == 1),
      "The input register and the destination register of "
      "luthier::readReg don't have the same size."));

  auto channelVReg = [&](llvm::MCRegister Channel) -> llvm::Register {
    auto It = ReadPhysRegVRegs.find(Channel);
    return It == ReadPhysRegVRegs.end() ? llvm::Register() : It->second;
  };

  if (SrcRegSize > 32) {
    auto Builder = MIBuilder(llvm::AMDGPU::REG_SEQUENCE);
    auto MergedReg = VirtRegBuilder(TRI->getPhysRegBaseClass(Src));
    (void)Builder.addReg(MergedReg, llvm::RegState::Define);

    size_t NumChannels = SrcRegSize / 32;
    for (size_t I = 0; I < NumChannels; ++I) {
      auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(I);
      llvm::MCRegister ChannelReg = TRI->getSubReg(Src, SubIdx);
      llvm::Register VReg = channelVReg(ChannelReg);
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          VReg.isValid(),
          "luthier::readReg: channel missing from pre-staged read map (IR "
          "processor failed to declare it in Effects.ReadPhysRegs)"));
      (void)Builder.addReg(VReg).addImm(SubIdx);
    }
    (void)MIBuilder(llvm::AMDGPU::COPY)
        .addReg(Output, llvm::RegState::Define)
        .addReg(MergedReg);
  } else if (SrcRegSize == 32 || SrcRegSize == 1) {
    llvm::Register VReg = channelVReg(Src);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        VReg.isValid(),
        "luthier::readReg: channel missing from pre-staged read map"));
    (void)MIBuilder(llvm::AMDGPU::COPY)
        .addReg(Output, llvm::RegState::Define)
        .addReg(VReg);
  } else {
    auto SuperReg = TRI->get32BitRegister(Src);
    auto SubIdx = TRI->getSubRegIndex(SuperReg, Src);
    llvm::Register VReg = channelVReg(SuperReg);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        VReg.isValid(),
        "luthier::readReg: super-register channel missing from pre-staged "
        "read map (sub-32 read must declare its 32-bit super-reg)"));
    (void)MIBuilder(llvm::AMDGPU::COPY)
        .addReg(Output, llvm::RegState::Define)
        .addReg(VReg, llvm::RegState::NoFlags, SubIdx);
  }

  return llvm::Error::success();
}

} // namespace luthier
