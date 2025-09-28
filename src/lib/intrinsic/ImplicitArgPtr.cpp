//===-- ImplicitArgPtr.cpp - Luthier implicit arg access  -----------------===//
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
/// This file implements Luthier's <tt>ImplicitArgPtr</tt> intrinsic.
//===----------------------------------------------------------------------===//
#include "intrinsic/ImplicitArgPtr.hpp"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/User.h>
#include <llvm/MC/MCRegister.h>
#include <luthier/common/GenericLuthierError.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
implicitArgPtrIRProcessor(const llvm::Function &Intrinsic,
                          const llvm::CallInst &User,
                          const llvm::GCNTargetMachine &TM) {
  // The user must not have any operands
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 0,
      llvm::formatv("Expected no operands to be passed to the "
                    "luthier::implicitArgPtr intrinsic '{0}', got {1}.",
                    User, User.arg_size())));

  luthier::IntrinsicIRLoweringInfo Out;
  // The kernarg hidden address will be returned in an SGPR
  Out.setReturnValueInfo(&User, "s");
  // We need access to the base of the kernel argument buffer, and the offset
  // from where the hidden kernel argument starts
  Out.requestAccessToKernelArgument(HIDDEN_KERNARG_OFFSET);
  Out.requestAccessToKernelArgument(KERNARG_SEGMENT_PTR);

  return Out;
}

llvm::Error implicitArgPtrMIRProcessor(
    const IntrinsicIRLoweringInfo &IRLoweringInfo,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const std::function<llvm::Register(KernelArgumentType)> &KernArgAccessor,
    const llvm::MachineFunction &MF,
    const std::function<llvm::Register(llvm::MCRegister)> &PhysRegAccessor,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &PhysRegsToBeOverwritten) {
  // There should be only a single virtual register involved in the operation
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_GENERIC_ERROR_CHECK(Args.size() == 1,
                          llvm::formatv("Number of virtual register arguments "
                          "involved in the MIR lowering stage of "
                          "luthier::implicitArgPtr is {0} instead of 1.",
                          Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(),
      "The register argument of luthier::implicitArgPtr is not a definition."));
  llvm::Register Output = Args[0].second;
  // Get the kernel argument
  llvm::Register KernArgSGPR = KernArgAccessor(KERNARG_SEGMENT_PTR);
  // Get the offset of the hidden arg
  llvm::Register HiddenOffsetSGPR = KernArgAccessor(HIDDEN_KERNARG_OFFSET);

  llvm::Register FirstAddSGPR = VirtRegBuilder(&llvm::AMDGPU::SGPR_32RegClass);

  llvm::Register SecondAddSGPR = VirtRegBuilder(&llvm::AMDGPU::SGPR_32RegClass);

  MIBuilder(llvm::AMDGPU::S_ADD_U32)
      .addReg(FirstAddSGPR, llvm::RegState::Define)
      .addReg(KernArgSGPR, llvm::RegState::Kill,
              llvm::SIRegisterInfo::getSubRegFromChannel(0))
      .addReg(HiddenOffsetSGPR, llvm::RegState::Kill);

  MIBuilder(llvm::AMDGPU::S_ADDC_U32)
      .addReg(SecondAddSGPR, llvm::RegState::Define)
      .addReg(KernArgSGPR, llvm::RegState::Kill,
              llvm::SIRegisterInfo::getSubRegFromChannel(1))
      .addImm(0);

  // Do a reg sequence copy to the output
  (void)MIBuilder(llvm::AMDGPU::REG_SEQUENCE)
      .addReg(Output, llvm::RegState::Define)
      .addReg(SecondAddSGPR)
      .addImm(llvm::SIRegisterInfo::getSubRegFromChannel(1))
      .addReg(FirstAddSGPR)
      .addImm(llvm::SIRegisterInfo::getSubRegFromChannel(0));

  return llvm::Error::success();
}

} // namespace luthier