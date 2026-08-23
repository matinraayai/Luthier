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

  IntrinsicIRLoweringInfo Out;
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
  // The value to write comes first as a register-use operand so the MIR
  // processor can find it at Args[0]. The destination phys-reg enum follows
  // as an immediate at Args[1].
  Out.addArgInfo(*User.getArgOperand(1), Constraint);
  Out.addArgInfo(*DestRegEnum, "i");

  return Out;
}

llvm::Error writeRegMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
        Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &ReadPhysRegVRegs,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &WritePhysRegSlots) {
  // Two inline-asm operands: the reg-use value at Args[0] and the
  // destination phys-reg-enum immediate at Args[1].
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 2,
      llvm::formatv("Number of arguments to the MIR lowering stage of "
                    "luthier::writeReg is {0} instead of 2.",
                    Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegUseKind(),
      "The first argument of luthier::writeReg is not a register use."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[1].first.isImmKind(),
      "The second argument of luthier::writeReg is not an immediate."));
  llvm::Register InputReg(Args[0].second->getReg());
  llvm::MCRegister Dest(Args[1].second->getImm());

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
      (void)MIBuilder(llvm::AMDGPU::COPY)
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
    (void)MIBuilder(llvm::AMDGPU::INSERT_SUBREG)
        .addReg(SuperRegVirt, llvm::RegState::Define)
        .addReg(SuperRegIt->second)
        .addReg(InputReg)
        .addImm(SubIdx);
    WritePhysRegSlots.insert({SuperRegDest, SuperRegVirt});
  }
  return llvm::Error::success();
}

} // namespace luthier
