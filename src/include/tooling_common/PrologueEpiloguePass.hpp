//===-- PrologueEpiloguePass.hpp ------------------------------------------===//
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
/// This file describes Luthier's Prologue and Epilogue insertion pass,
/// which replaces the normal epilogue prologue insertion pass of frame lowering
/// in the CodeGen pipeline.
//===----------------------------------------------------------------------===//

#include "luthier/LiftedRepresentation.h"
#include "tooling_common/LRStateValueLocations.hpp"
#include "luthier/LRRegisterLiveness.h"
#include "luthier/LRCallgraph.h"
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>


namespace luthier {


class PrologueEpiloguePass : llvm::MachineFunctionPass {
public:
  static char ID;

  PrologueEpiloguePass(
      const LiftedRepresentation &LR,
      const llvm::LivePhysRegs &AccessedPhysicalRegs, const LRCallGraph &CG,
      const LRStateValueLocations &StateValueLocations,
      const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
          &HookFuncToInstPointMI,
      llvm::ArrayRef<IntrinsicIRLoweringInfo>
          InlineAsmPlaceHolderToIRLoweringInfoMap,
      const LRRegisterLiveness &RegLiveness);

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Physical Register Access Virtualization Pass";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;


  void emitHookPrologue(llvm::MachineFunction &MF,
                        llvm::SmallDenseSet<llvm::MCRegister, 4>
                            &AccessedPhysicalRegs) const override;


};



}