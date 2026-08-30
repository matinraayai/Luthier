//===-- LoadSVA.cpp -------------------------------------------------------===//
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
/// Minimal implementation of the luthier::loadSVA intrinsic. The IR
/// processor emits an inline-asm placeholder with a VGPR-out constraint;
/// the MIR processor materializes a \c WWM_COPY from the caller-supplied
/// SVA source register into the intrinsic's return register.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/LoadSVA.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"

#include <SIMachineFunctionInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
loadSVAIRProcessor(const llvm::Function &, const llvm::CallInst &User,
                   const llvm::GCNTargetMachine &) {
  // luthier::loadSVA takes no arguments — it materializes the whole-wave SVA
  // value in a VGPR.
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 0,
      llvm::formatv(
          "Expected zero operands to be passed to luthier::loadSVA '{0}', got "
          "{1}.",
          User, User.arg_size())));

  IntrinsicIRLoweringInfo Out;
  // loadSVA returns its value into a VGPR
  Out.setReturnValueInfo(User, "v");
  return Out;
}

llvm::Error loadSVAMIRProcessor(
    llvm::MachineFunction &MF,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
        Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    llvm::Register SVAValueSource) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 1,
      llvm::formatv("luthier::loadSVA: expected 1 arg, got {0}.",
                    Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(),
      "luthier::loadSVA: first argument is not a register definition."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      SVAValueSource.isValid(),
      llvm::formatv(
          "luthier::loadSVA: no SVA value source register resolved for MF {0}.",
          MF.getName())));
  llvm::Register Output = Args[0].second->getReg();
  MF.getInfo<llvm::SIMachineFunctionInfo>()->setFlag(
      Output, llvm::AMDGPU::VirtRegFlag::WWM_REG);
  (void)MIBuilder(llvm::AMDGPU::WWM_COPY)
      .addReg(Output, llvm::RegState::Define)
      .addReg(SVAValueSource);
  return llvm::Error::success();
}

} // namespace luthier
