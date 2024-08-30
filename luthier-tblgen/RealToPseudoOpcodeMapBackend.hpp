//===-- RealToPseudoOpcodeMapBackend.hpp - Real to Pseudo Opcode Map ------===//
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
/// Contains definitions for the real to pseudo opcode tablegen backend
/// for the Luthier tablegen. It is inspired by the InstrMap class in tablegen,
/// except modified to handle cases where real opcodes don't have a pseudo
/// equivalent, or when the same real instructions has multiple pseudo opcodes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TBLGEN_REAL_TO_PSEUDO_OPCODE_MAP_BACKEND_HPP
#define LUTHIER_TBLGEN_REAL_TO_PSEUDO_OPCODE_MAP_BACKEND_HPP
#include "RealToPseudoOpcodeMapBackend.hpp"
#include <Common/CodeGenInstruction.h>
#include <Common/CodeGenTarget.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

namespace luthier {
/// \brief Emits a map between real opcodes and their pseudo equivalent in
/// the AMDGPU backend
class RealToPseudoOpcodeMapEmitter {
private:
  /// The CodeGen target class of the AMDGPU backend; Used to emit instruction
  /// enums in order
  const llvm::CodeGenTarget &Target;

  /// A mapping between the pseudo inst string opcode and the SI Pseudo
  /// instruction
  llvm::StringMap<llvm::Record *> PseudoInsts;

  /// Emits the indexing portion of the query function; This is different from
  /// the way \c InstrMap emission in vanilla tablegen works; Instead it uses
  /// the index of the original instruction enums as the key to find its
  /// entry inside the map
  /// \param OS Output stream for the emitted file
  /// \param TableSize Size of the table
  void emitIndexing(llvm::raw_ostream &OS, unsigned TableSize);

  /// Emits the table mapping the instruction opcode to its pseudo variant
  /// \param OS Output stream for the emitted file
  /// \return Number of entries in the emitted table, used by \c emitIndexing
  /// and \c emitMapFuncBody
  unsigned emitTable(llvm::raw_ostream &OS);

  /// Emits the lookup function body
  /// \param OS Output stream for the emitted file
  /// \param TableSize Size of the table previously emitted
  void emitMapFuncBody(llvm::raw_ostream &OS, unsigned TableSize);

public:
  RealToPseudoOpcodeMapEmitter(llvm::CodeGenTarget &Target,
                               llvm::RecordKeeper &Records)
      : Target(Target) {
    auto SIInsts = Records.getAllDerivedDefinitions("SIMCInstr");
    for (auto SIInst : SIInsts) {
      // Find all pseudo SI instructions and store them in the PseudoInsts map
      bool IsPseudo =
          SIInst->getValue("isPseudo")->getValue()->getAsUnquotedString() ==
          "1";
      if (IsPseudo) {
        PseudoInsts.insert({SIInst->getValueAsString("PseudoInstr"), SIInst});
      }
    }
  }

  /// Emits the real to pseudo table and the function to query it
  /// \param OS Output stream of the emitted file
  void emitTablesWithFunc(llvm::raw_ostream &OS);
};

/// Parse the \c Records and create a mapping between real to pseudo opcodes
/// in the AMDGPU backend; This includes all instructions inherited from
/// \c SIMCInstr class
/// \param Records Records parsed by the tablegen parser
/// \param OS Output stream of the emitted file
void emitRealToPseudoOpcodeTable(llvm::RecordKeeper &Records,
                                 llvm::raw_ostream &OS);
} // namespace luthier

#endif