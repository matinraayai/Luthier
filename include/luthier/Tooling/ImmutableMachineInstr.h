//===-- ImmutableMachineInstr.h -----------------------------------*-C++-*-===//
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
/// Describes the \c ImmutableMachineInstr class which annotates the machine
/// instructions manually injected inside the lifted representation to not be
/// changed during the patching phase of the instrumentation process.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_TOOLING_IMMUTABLE_MACHINE_INSTR_H
#define LUTHIER_TOOLING_IMMUTABLE_MACHINE_INSTR_H
#include <llvm/CodeGen/MachineInstr.h>

namespace luthier {

/// \brief A special type of machine instruction that when encountered by
/// Luthier's patching passes
class ImmutableMachineInstr : public llvm::MachineInstr {

  static constexpr auto ImmutableID = "Immutable";

public:
  static bool classof(const llvm::MachineInstr *MI) {
    if (const llvm::MachineOperand &LastOperand =
            MI->getOperand(MI->getNumOperands() - 1);
        LastOperand.isMetadata()) {
      if (auto *MD = llvm::dyn_cast<llvm::MDTuple>(LastOperand.getMetadata());
          MD && MD->getNumOperands() == 1 &&
          llvm::isa<llvm::MDString>(MD->getOperand(0))) {
        return llvm::cast<llvm::MDString>(MD->getOperand(0))->getString() ==
               ImmutableID;
      }
    }
    return false;
  }

  /// Makes the \p MI immutable during the patching passes of Luthier; i.e.
  /// Patching passes will not change the \p MI when they are run
  /// Use this function to make sure manually injected instructions into an
  /// instrumented lifted representation don't get changed
  /// \note Since the \p MI will get printed after the patching passes into the
  /// final binary it is the user's responsibility to make sure the machine
  /// instruction can be directly converted to a \c llvm::MCInst by the assembly
  /// printer pass
  /// \returns The same
  static llvm::Expected<ImmutableMachineInstr &>
  makeImmutable(llvm::MachineInstr &MI);
};

} // namespace luthier

#endif