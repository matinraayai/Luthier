//===-- ImmutableMachineInstr.cpp -----------------------------------------===//
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
/// \file
/// Implements the \c ImmutableMachineInstr class.
//===----------------------------------------------------------------------===//
#include "luthier/tooling/ImmutableMachineInstr.h"
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/Function.h>
#include <luthier/common/GenericLuthierError.h>

namespace luthier {

llvm::Expected<ImmutableMachineInstr &>
ImmutableMachineInstr::makeImmutable(llvm::MachineInstr &MI) {
  auto *MF = MI.getMF();
  if (!MF)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "MI doesn't have a MachineFunction parent");
  llvm::LLVMContext &Ctx = MF->getFunction().getContext();
  auto *ImmutableMD = llvm::MDString::get(Ctx, ImmutableID);
  const auto *ImmutableTupleMD = llvm::MDTuple::getIfExists(Ctx, {ImmutableMD});
  MI.addOperand(*MF, llvm::MachineOperand::CreateMetadata(ImmutableTupleMD));
  return llvm::cast<ImmutableMachineInstr &>(MI);
}

} // namespace luthier