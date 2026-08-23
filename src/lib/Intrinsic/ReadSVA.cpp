//===-- ReadSVA.cpp -------------------------------------------------------===//
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
/// Minimal implementation of the luthier::readSVA intrinsic. The IR
/// processor emits an inline-asm placeholder whose SA-enum arg is later
/// aggregated by \c StateValueArraySpecsAnalysis into the SVA layout;
/// the MIR processor copies the lowering pass's \c SVAScalarArgumentAccessor
/// vreg for that SA into the intrinsic's return register.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/ReadSVA.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Type.h>

namespace luthier {

llvm::Expected<IntrinsicIRLoweringInfo>
readSVAIRProcessor(const llvm::Function &Intrinsic, const llvm::CallInst &User,
                   const llvm::GCNTargetMachine &TM) {
  // luthier::readSVA takes a single i8 constant — the ScalarValueArgument
  // enum value of the SA being read.
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      User.arg_size() == 1,
      llvm::formatv(
          "Expected one operand to be passed to luthier::readSVA '{0}', got "
          "{1}.",
          User, User.arg_size())));
  auto *Arg = llvm::dyn_cast<llvm::ConstantInt>(User.getArgOperand(0));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Arg != nullptr,
      "The first argument of luthier::readSVA is not a constant integer."));
  uint64_t SAVal = Arg->getZExtValue();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      SAVal <= static_cast<uint64_t>(SCALAR_VALUE_ARGUMENT_LAST),
      llvm::formatv("luthier::readSVA: SA enum value {0} is out of range",
                    SAVal)));
  ScalarValueArgument SA = static_cast<ScalarValueArgument>(SAVal);

  IntrinsicIRLoweringInfo Out;
  // readSVA always returns its value into an SGPR — the SA lanes live in
  // the SVA VGPR and are read out via V_READLANE_B32 into SGPRs.
  Out.setReturnValueInfo(User, "s");
  Out.addArgInfo(*Arg, "i");

  return Out;
}

llvm::Error readSVAMIRProcessor(
    const llvm::MachineFunction &MF,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>
        Args,
    const std::function<llvm::MachineInstrBuilder(int)> &MIBuilder,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &SAAccessors) {
  // Two inline-asm operands: the regdef output and the SA-enum immediate.
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args.size() == 2,
      llvm::formatv("luthier::readSVA: expected 2 args, got {0}.",
                    Args.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[0].first.isRegDefKind(),
      "luthier::readSVA: first argument is not a register definition."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Args[1].first.isImmKind(),
      "luthier::readSVA: second argument is not an immediate."));
  llvm::Register Output = Args[0].second->getReg();
  ScalarValueArgument SA =
      static_cast<ScalarValueArgument>(Args[1].second->getImm());

  // The lowering pass pre-staged a vreg for this SA via
  // SVAScalarArgumentAccessor — find it and COPY into our return reg.
  auto It = SAAccessors.find(SA);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      It != SAAccessors.end(),
      llvm::formatv("luthier::readSVA: SA {0} is not in the per-MF accessor "
                    "map (lowering pass should have materialized it)",
                    static_cast<int>(SA))));
  (void)MIBuilder(llvm::AMDGPU::COPY)
      .addReg(Output, llvm::RegState::Define)
      .addReg(It->second);
  (void)MF;
  return llvm::Error::success();
}

} // namespace luthier
