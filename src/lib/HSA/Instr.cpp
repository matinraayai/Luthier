//===-- Instr.cpp - HSA Instruction  --------------------------------------===//
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
/// This file implements the \c luthier::hsa::Instr class.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/LoadedCodeObjectDeviceFunction.h"
#include "luthier/HSA/LoadedCodeObjectKernel.h"
#include "luthier/HSA/Instr.h"

namespace luthier::hsa {

Instr::Instr(llvm::MCInst Inst, const LoadedCodeObjectKernel &Kernel,
             address_t Address, size_t Size)
    : Inst(std::move(Inst)), Symbol(Kernel), LoadedDeviceAddress(Address),
      Size(Size) {}

Instr::Instr(llvm::MCInst Inst,
             const LoadedCodeObjectDeviceFunction &DeviceFunction,
             address_t Address, size_t Size)
    : Inst(std::move(Inst)), Symbol(DeviceFunction),
      LoadedDeviceAddress(Address), Size(Size) {}

const LoadedCodeObjectSymbol &Instr::getLoadedCodeObjectSymbol() const {
  return Symbol;
}

llvm::MCInst Instr::getMCInst() const { return Inst; }

luthier::address_t Instr::getLoadedDeviceAddress() const {
  return LoadedDeviceAddress;
}

size_t Instr::getSize() const { return Size; }

} // namespace luthier::hsa
