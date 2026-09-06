//===-- ImplicitArgPtr.cpp - Luthier implicit arg access  -----------------===//
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
/// This file implements Luthier's <tt>ImplicitArgPtr</tt> intrinsic.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/ImplicitArgPtr.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/LuthierError.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/User.h>
#include <llvm/MC/MCRegister.h>

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
  Out.setReturnValueInfo(User, "s");

  return Out;
}

llvm::Error implicitArgPtrMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Args,
    llvm::MDNode *Payload,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)>
        &VirtRegBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SVAVRegs,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &) {
  // There should be only a single virtual register involved in the operation
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 1,
      llvm::formatv("Number of virtual register arguments "
                    "involved in the MIR lowering stage of "
                    "luthier::implicitArgPtr is {0} instead of 1.",
                    Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(),
      "The register argument of luthier::implicitArgPtr is not a definition."));
  llvm::Register Output = Args[0].second;
  // The SVA's \c IMPLICIT_ARG_BUFFER lanes hold the *absolute* 64-bit base of
  // the instrumented kernel's implicit-argument block, written there by
  // \c TargetModulePatcherPass 's kernarg-buffer expansion. There is no longer
  // a \c KERNEL_ARG_PTR entry to add a hidden offset onto — the SVA is
  // saturated on GFX10 and those two lanes were reclaimed for the exec-mask
  // spill — so the intrinsic is just a copy of the preserved pointer.
  auto ImplArgIt = SVAVRegs.find(IMPLICIT_ARG_BUFFER);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      ImplArgIt != SVAVRegs.end(),
      "luthier::implicitArgPtr: IMPLICIT_ARG_BUFFER missing "
      "from pre-staged SVA map (IR processor must declare it)"));

  (void)MIBuilder(llvm::AMDGPU::COPY)
      .addReg(Output, llvm::RegState::Define)
      .addReg(ImplArgIt->second, llvm::RegState::Kill);

  return llvm::Error::success();
}

} // namespace luthier