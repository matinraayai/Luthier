//===-- WorkgroupId.cpp - Luthier workgroup-id (blockIdx) intrinsic -------===//
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
/// Implements Luthier's <tt>workgroupIdX/Y/Z</tt> intrinsics. Each reads its
/// \c WORKGROUP_ID_{X,Y,Z} scalar-value argument out of the SVA — identical to
/// \c readSVA with a fixed SA. The arch-specific sourcing of the value (TTMP vs
/// preloaded system SGPR) is handled by the kernel-entry prologue in
/// TargetModulePatcherPass, so this lowering is arch-independent.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/WorkgroupId.h"
#include "AMDGPUTargetMachine.h"
#include "SIInstrInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

namespace luthier {

namespace {

llvm::Expected<IntrinsicIRLoweringInfo>
workgroupIdIRProcessorImpl(ScalarValueArgument SA, const llvm::CallInst &User) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 0,
      llvm::formatv("Expected no operands to be passed to the "
                    "luthier::workgroupId intrinsic '{0}', got {1}.",
                    User, User.arg_size())));
  IntrinsicIRLoweringInfo Out;
  // Returned in an SGPR (the SVA lane is read out via V_READLANE into an SGPR).
  Out.setReturnValueInfo(User, "s");
  // Declare the SA so StateValueArraySpecs::setModuleSVASpec allocates its lane
  // and the lowering pass pre-stages the accessor vreg.
  Out.getEffects().ReadSVAs.push_back(SA);
  return Out;
}

llvm::Error workgroupIdMIRProcessorImpl(
    ScalarValueArgument SA,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 1,
      llvm::formatv("luthier::workgroupId: expected 1 vreg arg, got {0}.",
                    Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(),
      "luthier::workgroupId: register argument is not a definition."));
  llvm::Register Output = Args[0].second;

  auto It = SVAVRegs.find(SA);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      It != SVAVRegs.end(),
      llvm::formatv("luthier::workgroupId: SA {0} is not in the pre-staged SVA "
                    "map (IR processor must declare it)",
                    static_cast<int>(SA))));
  (void)MIBuilder(llvm::AMDGPU::COPY)
      .addReg(Output, llvm::RegState::Define)
      .addReg(It->second);
  return llvm::Error::success();
}

} // namespace

llvm::Expected<IntrinsicIRLoweringInfo>
workgroupIdXIRProcessor(const llvm::Function &, const llvm::CallInst &User,
                        const llvm::GCNTargetMachine &) {
  return workgroupIdIRProcessorImpl(WORKGROUP_ID_X, User);
}
llvm::Expected<IntrinsicIRLoweringInfo>
workgroupIdYIRProcessor(const llvm::Function &, const llvm::CallInst &User,
                        const llvm::GCNTargetMachine &) {
  return workgroupIdIRProcessorImpl(WORKGROUP_ID_Y, User);
}
llvm::Expected<IntrinsicIRLoweringInfo>
workgroupIdZIRProcessor(const llvm::Function &, const llvm::CallInst &User,
                        const llvm::GCNTargetMachine &) {
  return workgroupIdIRProcessorImpl(WORKGROUP_ID_Z, User);
}

llvm::Error workgroupIdXMIRProcessor(
    const llvm::MachineFunction &,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &) {
  return workgroupIdMIRProcessorImpl(WORKGROUP_ID_X, Args, MIBuilder, SVAVRegs);
}
llvm::Error workgroupIdYMIRProcessor(
    const llvm::MachineFunction &,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &) {
  return workgroupIdMIRProcessorImpl(WORKGROUP_ID_Y, Args, MIBuilder, SVAVRegs);
}
llvm::Error workgroupIdZMIRProcessor(
    const llvm::MachineFunction &,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &) {
  return workgroupIdMIRProcessorImpl(WORKGROUP_ID_Z, Args, MIBuilder, SVAVRegs);
}

} // namespace luthier
