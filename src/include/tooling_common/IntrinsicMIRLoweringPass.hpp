//===-- IntrinsicMIRLoweringPass.hpp --------------------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file describes the Intrinsic MIR Lowering Pass, in charge of
/// converting inline assembly place holder instructions with a sequence of
/// Machine Instructions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_INTRINSIC_MIR_LOWERING_PASS_HPP
#define LUTHIER_TOOLING_COMMON_INTRINSIC_MIR_LOWERING_PASS_HPP
#include "tooling_common/PrePostAmbleEmitter.hpp"
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/IR/PassManager.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>

namespace luthier {

class IntrinsicMIRLoweringPass : public llvm::MachineFunctionPass {
public:
  static char ID;

  explicit IntrinsicMIRLoweringPass() : llvm::MachineFunctionPass(ID) {};

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Intrinsic MIR Lowering";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace luthier

#endif
