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
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>

namespace luthier {

class IntrinsicMIRLoweringPass : public llvm::MachineFunctionPass {
private:
  /// List of intrinsics to be lowered by this pass; The index stored in the
  /// inline assembly string of each intrinsic can be used to find the
  /// associated \c IntrinsicIRLoweringInfo
  const llvm::ArrayRef<IntrinsicIRLoweringInfo> ToBeLoweredIntrinsics;
  /// A mapping between the intrinsic name and its processor functions;
  /// It is supplied by the \c CodeGenerator
  const llvm::StringMap<IntrinsicProcessor> &IntrinsicsProcessors;

public:
  static char ID;

  explicit IntrinsicMIRLoweringPass(
      llvm::ArrayRef<IntrinsicIRLoweringInfo> MIRLoweringMap,
      const llvm::StringMap<IntrinsicProcessor> &IntrinsicsProcessors)
      : ToBeLoweredIntrinsics(MIRLoweringMap),
        IntrinsicsProcessors(IntrinsicsProcessors),
        llvm::MachineFunctionPass(ID) {};

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Intrinsic MIR Lowering";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace luthier
