//===-- PhysicalRegAccessVirtualizationPass.hpp ---------------------------===//
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
/// This file describes Luthier's Physical Reg Access Virtualization Pass,
/// which is in charge of assigning virtual registers to physical registers of
/// an instrumented application, and providing access to them to the MIR
/// processing stage of Luthier intrinsics.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_PHYSICAL_REG_ACCESS_VIRTUALIZATION_PASS_HPP
#define LUTHIER_TOOLING_COMMON_PHYSICAL_REG_ACCESS_VIRTUALIZATION_PASS_HPP
#include "luthier/tooling/AMDGPURegisterLiveness.h"
#include "luthier/tooling/LRCallgraph.h"
#include "luthier/tooling/LiftedRepresentation.h"
#include "tooling_common/SVStorageAndLoadLocations.hpp"
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/IR/PassManager.h>
#include <luthier/intrinsic/IntrinsicProcessor.h>

namespace luthier {

class PhysicalRegAccessVirtualizationPass : public llvm::MachineFunctionPass {

private:
  /// A mapping between the hooks and their physical 32-bit live-in registers
  llvm::DenseSet<llvm::MCRegister> PhysicalLiveInsForInjectedPayload{};

  llvm::DenseMap<std::pair<llvm::MCRegister, const llvm::MachineBasicBlock *>,
                 llvm::Register>
      PhysRegLocationPerMBB;

public:
  static char ID;

  explicit PhysicalRegAccessVirtualizationPass();

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Physical Register Access Virtualization Pass";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  [[nodiscard]] llvm::Register
  getMCRegLocationInMBB(llvm::MCRegister PhysReg,
                        const llvm::MachineBasicBlock &MBB) const;

  [[nodiscard]] const llvm::DenseSet<llvm::MCRegister> &
  get32BitLiveInRegs() const {
    return PhysicalLiveInsForInjectedPayload;
  }
};

} // namespace luthier

#endif