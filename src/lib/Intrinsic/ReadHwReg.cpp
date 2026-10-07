//===-- ReadHwReg.cpp - Luthier readHwReg intrinsic -----------------------===//
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
/// This file implements the "read a hardware register" intrinsic. Hooks
/// cannot use inline \c s_getreg_b32 (Luthier treats inline assembly in a
/// payload as an intrinsic call), so this exposes it as one: an immediate
/// \c simm16 encoding in, the 32-bit register value out in an SGPR.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/ReadHwReg.h"
#include "AMDGPUTargetMachine.h"
#include "SIInstrInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/LuthierError.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
readHwRegIRProcessor(const llvm::Function &Intrinsic,
                     const llvm::CallInst &User,
                     const llvm::GCNTargetMachine &TM) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 1,
      llvm::formatv("Expected one operand to be passed to luthier::readHwReg "
                    "'{0}', got {1}.",
                    User, User.arg_size())));
  auto *Encoding = llvm::dyn_cast<llvm::ConstantInt>(User.getArgOperand(0));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Encoding != nullptr,
      "The operand of luthier::readHwReg must be a constant integer."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.getType()->isIntegerTy(32),
      llvm::formatv("Expected luthier::readHwReg '{0}' to return i32.", User)));
  IntrinsicIRLoweringInfo Out;
  // S_GETREG_B32 writes an SGPR; the encoding is an immediate.
  Out.setReturnValueInfo(User, "s");
  Out.addArgInfo(*Encoding, "i");
  return Out;
}

llvm::Error readHwRegMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
        Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &) {
  // Two inline-asm operands: the regdef output and the encoding immediate.
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 2,
      llvm::formatv("luthier::readHwReg: expected 2 args, got {0}.",
                    Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(),
      "luthier::readHwReg: first argument is not a register definition."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[1].first.isImmKind(),
      "luthier::readHwReg: second argument is not an immediate."));
  // The encoding reaches us as a sign-extended 16-bit immediate (e.g. the
  // whole-register HW_ID encoding 0xF804 arrives as -2044); S_GETREG_B32
  // takes it as an unsigned 16-bit field.
  (void)MIBuilder(llvm::AMDGPU::S_GETREG_B32)
      .addReg(Args[0].second->getReg(), llvm::RegState::Define)
      .addImm(Args[1].second->getImm() & 0xffff);
  (void)MF;
  return llvm::Error::success();
}

} // namespace luthier
