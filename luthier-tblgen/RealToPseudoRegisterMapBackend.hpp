//===-- RealToPseudoRegisterMapBackend.hpp - Real to Pseudo Register Map --===//
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
/// Contains definitions for the real to pseudo register tablegen backend
/// for the Luthier tablegen.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TBLGEN_REAL_TO_PSEUDO_REGISTER_MAP_BACKEND_HPP
#define LUTHIER_TBLGEN_REAL_TO_PSEUDO_REGISTER_MAP_BACKEND_HPP
#include <Common/CodeGenInstruction.h>
#include <Common/CodeGenTarget.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

namespace luthier {
/// \brief Emits a map between real register and their pseudo equivalent in
/// the AMDGPU backend
class RealToPseudoRegisterMapEmitter {
private:
  /// The CodeGen target class of the AMDGPU backend; Used to emit register
  /// enums in order
  const llvm::CodeGenTarget &Target;

  /// Emits the indexing portion of the query function; Currently works
  /// the same way as the \c InstrMap emission in vanilla tablegen
  /// \param OS Output stream for the emitted file
  /// \param TableSize Size of the table being indexed
  void emitIndexing(llvm::raw_ostream &OS, unsigned TableSize);

  /// Emits the table containing the register mappings
  /// \param OS Output stream for the emitted file
  /// \return Number of entries in the emitted table, used by \c emitIndexing
  /// and \c emitMapFuncBody
  unsigned emitTable(llvm::raw_ostream &OS);

  // Lookup functions to query the table.
  /// \param OS Output stream for the emitted file
  /// \param TableSize Size of the table previously emitted
  void emitMapFuncBody(llvm::raw_ostream &OS, unsigned TableSize);

public:
  RealToPseudoRegisterMapEmitter(llvm::CodeGenTarget &Target)
      : Target(Target) {}

  /// Emits the real to pseudo table and the function to query it
  /// \param OS Output stream of the emitted file
  void emitTablesWithFunc(llvm::raw_ostream &OS);
};

/// Parse the \c Records and create a mapping between real to pseudo registers
/// in the AMDGPU backend; This includes all instructions inherited from
/// \c SIMCInstr class
/// \param Records Records parsed by the tablegen parser
/// \param OS Output stream of the emitted file
void emitRealToPseudoRegisterTable(llvm::RecordKeeper &Records,
                                   llvm::raw_ostream &OS);
} // namespace luthier

#endif