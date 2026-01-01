//===-- InjectedPayloadPEIPass.h --------------------------------*- C++ -*-===//
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
/// This file describes Luthier's Injected Payload Prologue and Epilogue
/// insertion pass, which replaces the normal prologues and epilogues insertion
/// by the CodeGen pipeline.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INJECTED_PAYLOAD_PEI_PASS_H
#define LUTHIER_TOOLING_INJECTED_PAYLOAD_PEI_PASS_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/Tooling/AMDGPURegisterLiveness.h"
#include "luthier/Tooling/LRCallgraph.h"
#include "luthier/Tooling/LiftedRepresentation.h"
#include "luthier/Tooling/PhysicalRegAccessVirtualizationPass.h"
#include "luthier/Tooling/PrePostAmbleEmitter.h"
#include "luthier/Tooling/SVStorageAndLoadLocations.h"
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class PhysicalRegAccessVirtualizationPass;

void initializeInjectedPayloadPEIPass(llvm::PassRegistry &Registry);

class InjectedPayloadPEIPass : public llvm::MachineFunctionPass {
private:
  PhysicalRegAccessVirtualizationPass &PhysRegVirtAccessPass;

public:
  static char ID;

  explicit InjectedPayloadPEIPass(
      PhysicalRegAccessVirtualizationPass &PhysRegVirtAccessPass)
      : llvm::MachineFunctionPass(ID),
        PhysRegVirtAccessPass(PhysRegVirtAccessPass) {};

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Injected Payload Prologue Epilogue Insertion Pass";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace luthier

#endif