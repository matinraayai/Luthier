//===-- InjectedPayloadPEIPass.hpp ----------------------------------------===//
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
#include "tooling_common/LRStateValueStorageAndLoadLocations.hpp"
#include "tooling_common/PreKernelEmitter.hpp"
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>

namespace luthier {

class PhysicalRegAccessVirtualizationPass;

class InjectedPayloadPEIPass : public llvm::MachineFunctionPass {

private:
  /// The lifted representation being worked on
  const LiftedRepresentation &LR;
  /// Calculated locations of the state value for the current module
  const LRStateValueStorageAndLoadLocations &StateValueLocations;
  /// Mapping between a hook and its instrumentation point MI in the LR
  const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
      HookFuncToInstPointMI;
  /// Physical registers that are not always in the Live-ins sets of the
  /// instrumentation points
  const llvm::LivePhysRegs &PhysicalRegsNotTobeClobbered;

  PhysicalRegAccessVirtualizationPass &PhysRegVirtAccessPass;

  FunctionPreambleDescriptor &PKInfo;

public:
  static char ID;

  InjectedPayloadPEIPass(
      const LiftedRepresentation &LR,
      const LRStateValueStorageAndLoadLocations &StateValueLocations,
      PhysicalRegAccessVirtualizationPass &PhysRegVirtAccessPass,
      const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
          &HookFuncToInstPointMI,
      const llvm::LivePhysRegs &PhysicalRegsNotTobeClobbered,
      FunctionPreambleDescriptor &PKInfo);

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Injected Payload Prologue Epilogue Insertion Pass";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace luthier