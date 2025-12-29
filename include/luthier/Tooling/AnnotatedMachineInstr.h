//===-- AnnotatedMachineInstr.h -----------------------------------*-C++-*-===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
/// Describes the \c AnnotatedMachineInstr class, an extension to
/// \c llvm::MachineInstr that can be used to query Luthier instruction
/// metadata information.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_MACHINE_INSTR_ANNOTATED_H
#define LUTHIER_TOOLING_MACHINE_INSTR_ANNOTATED_H
#include <llvm/CodeGen/MachineInstr.h>

namespace luthier {

/// \returns \c true if \p MI contains annotations in its PC sections, \c false
/// otherwise
inline bool isAnnotated(const llvm::MachineInstr &MI) {
  return MI.getPCSections() != nullptr;
}

/// TODO: finish implementing these!

/// \returns the \c "luthier.instr.id" field of this instruction's annotation if
/// exists, \c std::nullopt otherwise
llvm::Expected<std::optional<uint64_t>>
getInstrID(const llvm::MachineInstr &MI);

llvm::Error assignInstrID(llvm::MachineInstr &MI);

llvm::Error setInstrID(llvm::MachineInstr &MI, uint64_t ID);

llvm::Error makeInstrTrace(llvm::MachineInstr &MI, uint64_t LoadAddress);

llvm::Expected<std::optional<uint64_t>>
getInstrTraceAddrIfAvailable(const llvm::MachineInstr &MI);

llvm::Error makeInstrMutable(llvm::MachineInstr &MI);

} // namespace luthier

#endif