//===-- Instr.h - Luthier MC Instruction Wrapper ----------------*- C++ -*-===//
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
/// This file describes the \c luthier::Instr class,
/// which keeps track of an instruction disassembled by LLVM as well as other
/// info not included in the \c llvm::MCInst class, including its symbol,
/// address, and its size in bytes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LIFT_INSTR_H
#define LUTHIER_LIFT_INSTR_H
#include <llvm/MC/MCInst.h>
#include <llvm/Object/ObjectFile.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/LuthierError.h>

namespace luthier {

/// \brief Keeps track of an instruction disassembled by LLVM as well as other
/// info not included in the \c llvm::MCInst class, including its symbol,
/// address, and its size in bytes
class Instr {
private:
  /// The MC representation of the instruction
  const llvm::MCInst Inst;
  /// The symbol it belongs to
  const llvm::object::SymbolRef Symbol;
  /// The offset from bass which this instruction resides
  const uint64_t Offset;
  /// Size of the instruction
  const size_t Size;

  /// Constructor
  /// \param Inst \c MCInst representation the instruction
  /// \param Symbol a symbol of type function which the instruction belongs to
  /// \param Offset the offset of this instruction's bytes from the
  /// start of \p Symbol
  /// \param Size size of the instruction in bytes
  Instr(llvm::MCInst Inst, llvm::object::SymbolRef Symbol, size_t Offset,
        size_t Size)
      : Inst(std::move(Inst)), Symbol(Symbol), Offset(Offset), Size(Size) {}

public:
  /// Factory method
  static llvm::Expected<Instr> create(llvm::MCInst Inst,
                                      llvm::object::SymbolRef Symbol,
                                      size_t Offset, size_t Size) {
    llvm::Expected<llvm::object::SymbolRef::Type> TypeOrErr = Symbol.getType();
    LUTHIER_RETURN_ON_ERROR(TypeOrErr.takeError());
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(*TypeOrErr != llvm::object::SymbolRef::ST_Function,
                            "Instr Symbol is not a function."));
    return Instr{std::move(Inst), Symbol, Offset, Size};
  }

  /// \return the function symbol this instruction was disassembled from
  [[nodiscard]] llvm::object::SymbolRef getSymbol() const { return Symbol; }

  /// \return the MC representation of the instruction
  [[nodiscard]] llvm::MCInst getMCInst() const { return Inst; }

  /// \return the offset of this instruction from the start of its symbol
  [[nodiscard]] uint64_t getOffset() const { return Offset; }

  /// \return the size of the instruction in bytes
  [[nodiscard]] size_t getSize() const { return Size; }
};

} // namespace luthier

#endif