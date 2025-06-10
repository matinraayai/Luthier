//===-- WriteReg.cpp ------------------------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Implements Luthier's <tt>WriteReg</tt> intrinsic.
//===----------------------------------------------------------------------===//
#include <AMDGPUTargetMachine.h>
#include <SIRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/MC/MCRegister.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/Instrumentation/AMDGPU/Intrinsics/WriteExec.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
writeExecIRProcessor(const llvm::Function &Intrinsic,
                     const llvm::CallInst &User,
                     const llvm::GCNTargetMachine &TM) {
  // The User must only have 1 operand
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 1,
      llvm::formatv("Expected one operand to be passed to the "
                    "luthier::writeExec intrinsic '{0}', got {1}.",
                    User, User.arg_size())));
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
    const std::function<llvm::Register(KernelArgumentType)> &,
    const llvm::MachineFunction &MF,
    const std::function<llvm::Register(llvm::MCRegister)> &PhysRegAccessor,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &PhysRegsToBeOverwritten) {
  // There should be only a single virtual register involved in the operation
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 1, llvm::formatv("Number of virtual register arguments "
                                      "involved in the MIR lowering stage of "
                                      "luthier::writeExec is {0} instead of 1.",
                                      Args.size())));
  // It should be of reg use kind
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegUseKind(),
      "The register argument of luthier::writeExec is not a definition."));
  llvm::Register InputReg = Args[0].second;

  (void)MIBuilder(llvm::AMDGPU::COPY)
      .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Define)
      .addReg(InputReg);

  return llvm::Error::success();
}

} // namespace luthier