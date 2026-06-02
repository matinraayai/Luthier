//===-- UserArgPtr.cpp - Luthier user (explicit) arg access ---------------===//
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
/// This file implements Luthier's <tt>userArgPtr</tt> intrinsic. It returns the
/// base of the tool's explicit-argument region inside the custom kernarg
/// buffer, computed as <tt>USER_ARG_PTR + CustomKernargExplicitOffset</tt>.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/UserArgPtr.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/CustomKernargLayout.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/MC/MCRegister.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
userArgPtrIRProcessor(const llvm::Function &Intrinsic,
                      const llvm::CallInst &User,
                      const llvm::GCNTargetMachine &TM) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 0,
      llvm::formatv("Expected no operands to be passed to the "
                    "luthier::userArgPtr intrinsic '{0}', got {1}.",
                    User, User.arg_size())));

  luthier::IntrinsicIRLoweringInfo Out;
  // The explicit-arg region base is returned in an SGPR pair.
  Out.setReturnValueInfo(User, "s");
  // We need the base of the instrumentation argument buffer (the custom kernarg
  // buffer base). Declare it so the MIR-lowering driver pre-stages the SVA
  // vreg.
  Out.getEffects().ReadSVAs.push_back(USER_ARG_PTR);

  return Out;
}

llvm::Error userArgPtrMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *Payload,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 1,
      llvm::formatv(
          "Number of virtual register arguments involved in the MIR "
          "lowering stage of luthier::userArgPtr is {0} instead of 1.",
          Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(),
      "The register argument of luthier::userArgPtr is not a definition."));
  llvm::Register Output = Args[0].second;

  auto UserArgIt = SVAVRegs.find(USER_ARG_PTR);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UserArgIt != SVAVRegs.end(),
      "luthier::userArgPtr: USER_ARG_PTR missing from pre-staged SVA map (IR "
      "processor must declare it)"));
  llvm::Register UserArgSGPR = UserArgIt->second;

  llvm::Register FirstAddSGPR = VirtRegBuilder(&llvm::AMDGPU::SGPR_32RegClass);
  llvm::Register SecondAddSGPR = VirtRegBuilder(&llvm::AMDGPU::SGPR_32RegClass);

  // explicit-arg region base = USER_ARG_PTR + CustomKernargExplicitOffset
  MIBuilder(llvm::AMDGPU::S_ADD_U32)
      .addReg(FirstAddSGPR, llvm::RegState::Define)
      .addReg(UserArgSGPR, llvm::RegState::Kill,
              llvm::SIRegisterInfo::getSubRegFromChannel(0))
      .addImm(CustomKernargExplicitOffset);

  MIBuilder(llvm::AMDGPU::S_ADDC_U32)
      .addReg(SecondAddSGPR, llvm::RegState::Define)
      .addReg(UserArgSGPR, llvm::RegState::Kill,
              llvm::SIRegisterInfo::getSubRegFromChannel(1))
      .addImm(0);

  (void)MIBuilder(llvm::AMDGPU::REG_SEQUENCE)
      .addReg(Output, llvm::RegState::Define)
      .addReg(SecondAddSGPR)
      .addImm(llvm::SIRegisterInfo::getSubRegFromChannel(1))
      .addReg(FirstAddSGPR)
      .addImm(llvm::SIRegisterInfo::getSubRegFromChannel(0));

  return llvm::Error::success();
}

} // namespace luthier
