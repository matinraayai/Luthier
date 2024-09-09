//===-- HookPEIPass.hpp ---------------------------------------------------===//
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
/// This file describes Luthier's Hook Prologue and Epilogue insertion pass,
/// which replaces the normal prologues and epilogues inserted by the CodeGen
/// pipeline.
//===----------------------------------------------------------------------===//

#include "luthier/LRCallgraph.h"
#include "luthier/LRRegisterLiveness.h"
#include "luthier/LiftedRepresentation.h"
#include "tooling_common/LRStateValueLocations.hpp"
#include "tooling_common/PreKernelEmitter.hpp"
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>

namespace luthier {

class PhysicalRegAccessVirtualizationPass;

class HookPEIPass : public llvm::MachineFunctionPass {

private:
  /// The lifted representation being worked on
  const LiftedRepresentation &LR;
  /// Calculated locations of the state value for the current module
  const LRStateValueLocations &StateValueLocations;
  /// Mapping between a hook and its instrumentation point MI in the LR
  const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
      HookFuncToInstPointMI;
  /// LiveReg analysis for the LR
  const LRRegisterLiveness &RegLiveness;
  /// Physical registers that are not always in the Live-ins sets of the
  /// instrumentation points
  const llvm::LivePhysRegs &PhysicalRegsNotTobeClobbered;

  PhysicalRegAccessVirtualizationPass &PhysRegVirtAccessPass;

  PreKernelEmissionDescriptor &PKInfo;

public:
  static char ID;

  HookPEIPass(const LiftedRepresentation &LR,
              const LRStateValueLocations &StateValueLocations,
              PhysicalRegAccessVirtualizationPass &PhysRegVirtAccessPass,
              const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
                  &HookFuncToInstPointMI,
              const LRRegisterLiveness &RegLiveness,
              const llvm::LivePhysRegs &PhysicalRegsNotTobeClobbered,
              PreKernelEmissionDescriptor &PKInfo);

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Physical Register Access Virtualization Pass";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace luthier