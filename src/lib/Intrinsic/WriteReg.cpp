//===-- WriteReg.cpp ------------------------------------------------------===//
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
/// This file implements the write reg intrinsic.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/WriteReg.h"
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
writeRegIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                    const llvm::GCNTargetMachine &TM) {
  auto *TRI = TM.getSubtargetImpl(Intrinsic)->getRegisterInfo();
  // The User must only have 2 operands
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 2,
      llvm::formatv("Expected two operands to be passed to the "
                    "luthier::writeReg intrinsic '{0}', got {1}.",
                    User, User.arg_size())));

  luthier::IntrinsicIRLoweringInfo Out;
  // The first argument specifies the destination MCRegister enum value.
  auto *DestRegEnum = llvm::dyn_cast<llvm::ConstantInt>(User.getArgOperand(0));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      DestRegEnum != nullptr, "The first operand of the luthier::writeReg "
                              "intrinsic is not a constant int"));
  llvm::MCRegister DestReg(DestRegEnum->getZExtValue());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      llvm::MCRegister::isPhysicalRegister(DestReg.id()),
      llvm::formatv("The first argument of the luthier::writeReg intrinsic {0} "
                    "is not an MC Physical Register.",
                    DestReg.id())));
  // Determine the constraint for the destination register class
  auto *PhysRegClass = TRI->getPhysRegBaseClass(DestReg);
  std::string Constraint;
  if (llvm::SIRegisterInfo::isAGPRClass(PhysRegClass))
    Constraint = "a";
  else if (llvm::SIRegisterInfo::isVGPRClass(PhysRegClass))
    Constraint = "v";
  else if (llvm::SIRegisterInfo::isSGPRClass(PhysRegClass))
    Constraint = "s";
  else
    return llvm::make_error<GenericLuthierError>(llvm::formatv(
        "Unable to find a suitable register class for writing into {0}.",
        DestReg.id()));
  Out.setReturnValueInfo(User, Constraint);
  // The second argument specifies the source register
  auto *SrcReg = User.getArgOperand(1);
  Out.addArgInfo(*SrcReg, Constraint);
  // Forward the physical destination register enum to the MIR stage
  Out.addExtraLoweringValue(*llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(Intrinsic.getContext()), DestReg.id()));
  Out.getEffects().WrittenPhysRegs.push_back(DestReg);
  // Sub-32-bit destinations are written via INSERT_SUBREG into the
  // enclosing 32-bit super-register, which requires reading the
  // super-register's current value. Declare that read so the driver
  // pre-stages the channel value.
  unsigned DestSizeBits = TRI->getRegSizeInBits(*PhysRegClass);
  if (DestSizeBits < 32) {
    const auto *SITRI = static_cast<const llvm::SIRegisterInfo *>(TRI);
    llvm::MCRegister SuperReg = SITRI->get32BitRegister(DestReg);
    Out.getEffects().ReadPhysRegs.push_back(SuperReg);
  }

  return Out;
}

llvm::Error writeRegMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *Payload,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &ReadPhysRegVRegs,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &WritePhysRegSlots) {
  // The inline-asm placeholder for writeReg carries both a use (the input
  // value) and a (dead) def produced by setReturnValueInfo on the IR side.
  // We only care about the use here.
  llvm::Register InputReg;
  for (const auto &[Flag, Reg] : Args) {
    if (Flag.isRegUseKind()) {
      InputReg = Reg;
      break;
    }
  }
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      InputReg.isValid(),
      "luthier::writeReg: no register-use operand found among inline-asm "
      "args."));

  // Extract the destination physical register from the payload
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Payload && Payload->getNumOperands() == 1,
      "luthier::writeReg MIR payload must contain exactly one operand"));
  auto *RegMeta =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(Payload->getOperand(0));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      RegMeta != nullptr,
      "luthier::writeReg payload operand is not a ConstantAsMetadata"));
  llvm::MCRegister Dest(
      llvm::cast<llvm::ConstantInt>(RegMeta->getValue())->getZExtValue());

  auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  auto *TRI = ST.getRegisterInfo();
  auto &MRI = MF.getRegInfo();

  uint64_t DestRegSize = TRI->getRegSizeInBits(Dest, MRI);
  uint64_t InputRegSize = TRI->getRegSizeInBits(InputReg, MRI);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      InputRegSize == DestRegSize || (DestRegSize == 1 && InputRegSize == 32),
      "The input register and the destination register of "
      "luthier::writeReg don't have the same size."));

  if (DestRegSize > 32) {
    size_t NumChannels = DestRegSize / 32;
    const llvm::TargetRegisterClass *InputRegClass = MRI.getRegClass(InputReg);
    for (size_t I = 0; I < NumChannels; ++I) {
      auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(I);
      auto InputSubRegClass = TRI->getSubRegisterClass(InputRegClass, SubIdx);
      auto SubReg = VirtRegBuilder(InputSubRegClass);
      MIBuilder(llvm::AMDGPU::COPY)
          .addReg(SubReg, llvm::RegState::Define)
          .addReg(InputReg, llvm::RegState::NoFlags, SubIdx);
      WritePhysRegSlots.insert({TRI->getSubReg(Dest, SubIdx), SubReg});
    }
  } else if (DestRegSize == 32 || DestRegSize == 1) {
    WritePhysRegSlots.insert({Dest, InputReg});
  } else {
    auto SuperRegDest = TRI->get32BitRegister(Dest);
    auto SubIdx = TRI->getSubRegIndex(SuperRegDest, Dest);
    auto SuperRegIt = ReadPhysRegVRegs.find(SuperRegDest);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        SuperRegIt != ReadPhysRegVRegs.end(),
        "luthier::writeReg: sub-32 destination's 32-bit super-register "
        "missing from pre-staged read map (IR processor must declare it in "
        "Effects.ReadPhysRegs)"));
    auto SuperRegVirt = VirtRegBuilder(TRI->getPhysRegBaseClass(SuperRegDest));
    MIBuilder(llvm::AMDGPU::INSERT_SUBREG)
        .addReg(SuperRegVirt, llvm::RegState::Define)
        .addReg(SuperRegIt->second)
        .addReg(InputReg)
        .addImm(SubIdx);
    WritePhysRegSlots.insert({SuperRegDest, SuperRegVirt});
  }
  return llvm::Error::success();
}

} // namespace luthier
