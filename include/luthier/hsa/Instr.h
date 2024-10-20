//===-- Instr.h - HSA Instruction  ------------------------------*- C++ -*-===//
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
/// This file describes the \c Instr class under the \c luthier::hsa namespace,
/// which keeps track of an instruction disassembled by LLVM via parsing the
/// contents of the loaded contents of a \c LoadedCodeObjectSymbol of type
/// \c LoadedCodeObjectSymbol::SK_KERNEL or
/// <tt>LoadedCodeObjectSymbol::SK_DEVICE_FUNCTION</tt>
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_INSTR_H
#define LUTHIER_HSA_INSTR_H
#include <llvm/MC/MCInst.h>
#include <llvm/Support/Error.h>
#include <luthier/types.h>

namespace luthier::hsa {

class LoadedCodeObjectSymbol;

/// \brief represents an instruction that was disassembled by inspecting the
/// contents of a \c LoadedCodeObjectSymbol of type \c SK_KERNEL or
/// \c SK_DEVICE_FUNCTION loaded on device memory
/// \details \c Instr is created when calling \c luthier::disassemble or
/// <tt>luthier::lift</tt> on a function symbol. When a symbol is disassembled,
/// Luthier internally creates instances of this class to hold the disassembled
/// instructions and caches them until the \c hsa_executable_t backing the
/// symbol is destroyed by the HSA runtime.
class Instr {
private:
  /// The MC representation of the instruction
  const llvm::MCInst Inst;
  /// The address on the GPU Agent this instruction is loaded at
  const address_t LoadedDeviceAddress;
  /// The symbol this instruction belongs to
  const LoadedCodeObjectSymbol &Symbol;
  /// Size of the instruction
  const size_t Size;
  // TODO: add DWARF information when DWARF parsing is implemented in the
  // code lifter

public:
  /// Deleted default constructor
  Instr() = delete;

  /// Constructor
  /// \param Inst \c MCInst of the instruction
  /// \param Kernel the kernel this instruction belongs to
  /// \param Address the device address this instruction is loaded on
  /// \param Size size of the instruction in bytes
  Instr(llvm::MCInst Inst, const LoadedCodeObjectKernel &Kernel,
        address_t Address, size_t Size);

  /// Constructor
  /// \param Inst \c MCInst of the instruction
  /// \param DeviceFunction the device function this instruction belongs to
  /// \param Address the device address this instruction is loaded on
  /// \param Size size of the instruction in bytes
  Instr(llvm::MCInst Inst, const LoadedCodeObjectDeviceFunction &DeviceFunction,
        address_t Address, size_t Size);

  /// \return the device function/kernel that this instruction belongs to
  [[nodiscard]] const LoadedCodeObjectSymbol &getLoadedCodeObjectSymbol() const;

  /// \return the MC representation of the instruction
  [[nodiscard]] llvm::MCInst getMCInst() const;

  /// \return the loaded address of this instruction on the device
  /// \note the \c hsa_agent_t of the instruction can be queried from the
  /// this instruction's backing symbol
  [[nodiscard]] address_t getLoadedDeviceAddress() const;

  /// \return the size of the instruction in bytes
  [[nodiscard]] size_t getSize() const;
};

} // namespace luthier::hsa

#endif