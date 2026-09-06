//===-- SICacheInv.cpp ----------------------------------------------------===//
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
/// This file implements the "invalidate the scalar instruction cache"
/// intrinsic.
///
/// Luthier resolves some branch/call targets at run time, and a resolution can
/// be satisfied by a code object the host loads *after* the asking wave has
/// already started. The instruction bytes at the resolved target are therefore
/// younger than the wave, and the CU's instruction cache may still hold stale
/// lines covering that range. Branching there without invalidating first
/// fetches whatever those lines happen to contain and faults at the callee's
/// entry. This intrinsic gives the runtime resolver a way to drop those lines
/// before it installs the resolved handle.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/SICacheInv.h"
#include "AMDGPUTargetMachine.h"
#include "SIInstrInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/LuthierError.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
sICacheInvIRProcessor(const llvm::Function &Intrinsic,
                      const llvm::CallInst &User,
                      const llvm::GCNTargetMachine &TM) {
  // The intrinsic takes no arguments and returns nothing; there is no value
  // to lower, so the returned info stays empty.
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 0,
      llvm::formatv("Expected no operands to be passed to the "
                    "luthier::sICacheInv intrinsic '{0}', got {1}.",
                    User, User.arg_size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.getType()->isVoidTy(),
      llvm::formatv("Expected the luthier::sICacheInv intrinsic '{0}' to "
                    "return void.",
                    User)));
  luthier::IntrinsicIRLoweringInfo Out{};
  Out.setReturnValueInfo(User, "");
  return Out;
}

llvm::Error sICacheInvMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
        Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.empty(),
      llvm::formatv("Number of virtual register arguments involved in the MIR "
                    "lowering stage of luthier::sICacheInv is {0} instead of 0.",
                    Args.size())));

  // S_ICACHE_INV is a SOPP pseudo with no operands, so it retires once per
  // wave regardless of the exec mask.
  (void)MIBuilder(llvm::AMDGPU::S_ICACHE_INV);

  return llvm::Error::success();
}

} // namespace luthier
