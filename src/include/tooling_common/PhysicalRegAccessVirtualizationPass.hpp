//===-- PhysicalRegAccessVirtualizationPass.hpp ---------------------------===//
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
/// This file describes Luthier's Physical Reg Access Virtualization Pass,
/// which is in charge of assigning virtual registers to physical registers of
/// an instrumented application, and providing access to them to the MIR
/// processing stage of Luthier intrinsics.
//===----------------------------------------------------------------------===//
#include "luthier/LRCallgraph.h"
#include "luthier/LRRegisterLiveness.h"
#include "luthier/LiftedRepresentation.h"
#include "tooling_common/LRStateValueLocations.hpp"
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>

namespace luthier {

class PhysicalRegAccessVirtualizationPass : public llvm::MachineFunctionPass {

private:
  /// The Lifted Representation being processed
  const LiftedRepresentation &LR;
  /// The State value locations selected for \c LR
  const LRStateValueLocations &StateValueLocations;
  /// Register liveness information
  const LRRegisterLiveness &RegLiveness;
  /// LR Callgraph
  const LRCallGraph &CG;
  /// Set of physical registers accessed by all the hooks that are
  /// not in the live-ins
  const llvm::LivePhysRegs &AccessedPhysicalRegs;
  /// A mapping between the hooks and their physical 32-bit live-in registers
  llvm::DenseMap<llvm::Function *, llvm::DenseSet<llvm::MCRegister>>
      PerHookLiveInRegs{};
  const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *> &HookFuncToMIMap;
  /// A mapping between the inline assembly instruction place holder indices
  /// and their IR lowering info
  llvm::ArrayRef<std::pair<llvm::Function *, IntrinsicIRLoweringInfo>>
      InlineAsmPlaceHolderToIRLoweringInfoMap;

  llvm::DenseMap<std::pair<llvm::MCRegister, const llvm::MachineBasicBlock *>,
                 llvm::Register>
      PhysRegLocationPerMBB;

public:
  static char ID;

  PhysicalRegAccessVirtualizationPass(
      const LiftedRepresentation &LR,
      const llvm::LivePhysRegs &AccessedPhysicalRegs, const LRCallGraph &CG,
      const LRStateValueLocations &StateValueLocations,
      const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
          &HookFuncToInstPointMI,
      llvm::ArrayRef<std::pair<llvm::Function *, IntrinsicIRLoweringInfo>>
          InlineAsmPlaceHolderToIRLoweringInfoMap,
      const LRRegisterLiveness &RegLiveness);

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Physical Register Access Virtualization Pass";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  llvm::Register
  getMCRegLocationInMBB(llvm::MCRegister PhysReg,
                        const llvm::MachineBasicBlock &MBB) const {
    return PhysRegLocationPerMBB.at({PhysReg, &MBB});
  }
};

} // namespace luthier
