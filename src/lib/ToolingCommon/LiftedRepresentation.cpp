//===-- LiftedRepresentation.cpp - Lifted Representation  -----------------===//
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
/// This file implements Luthier's Lifted Representation, which contains the
/// \c llvm::Module and \c llvm::MachineModuleInfo of a lifted HSA primitive (a
/// kernel or an executable), as well as a mapping between the HSA primitives
/// and LLVM IR primitives involved.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/LiftedRepresentation.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/LoadedCodeObjectDeviceFunction.h"
#include "luthier/HSA/LoadedCodeObjectExternSymbol.h"
#include "luthier/HSA/LoadedCodeObjectKernel.h"
#include "luthier/HSA/LoadedCodeObjectVariable.h"

namespace luthier {

LiftedRepresentation::LiftedRepresentation() = default;

LiftedRepresentation::~LiftedRepresentation() = default;

llvm::Error LiftedRepresentation::iterateAllDefinedFunctionTypes(
    const std::function<llvm::Error(const hsa::LoadedCodeObjectSymbol &,
                                    llvm::MachineFunction &)> &Lambda) {
  // Apply the lambda on the lifted kernel
  LUTHIER_RETURN_ON_ERROR(Lambda(*Kernel, *KernelMF));
  // Apply the lambda on the related device functions
  for (auto &[Symbol, MF] : functions()) {
    LUTHIER_RETURN_ON_ERROR(Lambda(*Symbol, *MF));
  }
  return llvm::Error::success();
}

[[nodiscard]] llvm::GlobalVariable *LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectVariable &VariableSymbol) {
  auto It = Variables.find(&VariableSymbol);
  if (It == Variables.end())
    return nullptr;
  else
    return It->second;
}

[[nodiscard]] const llvm::GlobalVariable *
LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectVariable &VariableSymbol) const {
  auto It = Variables.find(&VariableSymbol);
  if (It == Variables.end())
    return nullptr;
  else
    return It->second;
}

[[nodiscard]] const llvm::GlobalVariable *
LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectExternSymbol &ExternSymbol) const {
  auto It = Variables.find(&ExternSymbol);
  if (It == Variables.end())
    return nullptr;
  else
    return It->second;
}

[[nodiscard]] llvm::GlobalVariable *LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectExternSymbol &ExternSymbol) {
  auto It = Variables.find(&ExternSymbol);
  if (It == Variables.end())
    return nullptr;
  else
    return It->second;
}

[[nodiscard]] const llvm::GlobalValue *
LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectKernel &KernelSymbol) const {
  if (KernelSymbol == *this->Kernel)
    return &KernelMF->getFunction();
  else {
    auto It = Variables.find(&KernelSymbol);
    if (It == Variables.end())
      return nullptr;
    else
      return It->second;
  }
}

[[nodiscard]] llvm::GlobalValue *LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectKernel &KernelSymbol) {
  if (KernelSymbol == *this->Kernel)
    return &KernelMF->getFunction();
  else {
    auto It = Variables.find(&KernelSymbol);
    if (It == Variables.end())
      return nullptr;
    else
      return It->second;
  }
}

[[nodiscard]] const llvm::Function *LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectDeviceFunction &DevFunc) const {
  auto It = Functions.find(&DevFunc);
  if (It == Functions.end())
    return nullptr;
  else
    return &It->second->getFunction();
}

[[nodiscard]] llvm::Function *LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectDeviceFunction &DevFunc) {
  auto It = Functions.find(&DevFunc);
  if (It == Functions.end())
    return nullptr;
  else
    return &It->second->getFunction();
}

[[nodiscard]] const llvm::GlobalValue *
LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectSymbol &Symbol) const {
  if (auto *VarSym = llvm::dyn_cast<hsa::LoadedCodeObjectVariable>(&Symbol)) {
    return getLiftedEquivalent(*VarSym);
  } else if (auto *ExternSym =
                 llvm::dyn_cast<hsa::LoadedCodeObjectExternSymbol>(&Symbol)) {
    return getLiftedEquivalent(*ExternSym);
  } else if (auto *KernelSym =
                 llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(&Symbol)) {
    return getLiftedEquivalent(*KernelSym);
  } else if (auto *DeviceSym =
                 llvm::dyn_cast<hsa::LoadedCodeObjectDeviceFunction>(&Symbol)) {
    return getLiftedEquivalent(*DeviceSym);
  } else
    llvm_unreachable("Invalid symbol type.");
}

[[nodiscard]] llvm::GlobalValue *LiftedRepresentation::getLiftedEquivalent(
    const hsa::LoadedCodeObjectSymbol &Symbol) {
  if (auto *VarSym = llvm::dyn_cast<hsa::LoadedCodeObjectVariable>(&Symbol)) {
    return getLiftedEquivalent(*VarSym);
  } else if (auto *ExternSym =
                 llvm::dyn_cast<hsa::LoadedCodeObjectExternSymbol>(&Symbol)) {
    return getLiftedEquivalent(*ExternSym);
  } else if (auto *KernelSym =
                 llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(&Symbol)) {
    return getLiftedEquivalent(*KernelSym);
  } else if (auto *DeviceSym =
                 llvm::dyn_cast<hsa::LoadedCodeObjectDeviceFunction>(&Symbol)) {
    return getLiftedEquivalent(*DeviceSym);
  } else
    llvm_unreachable("Invalid symbol type.");
}

[[nodiscard]] const hsa::Instr *
LiftedRepresentation::getLiftedEquivalent(const llvm::MachineInstr &MI) const {
  auto It = MachineInstrToMCMap.find(&MI);
  if (It == MachineInstrToMCMap.end())
    return nullptr;
  else
    return It->second;
}

} // namespace luthier
