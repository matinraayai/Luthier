//===-- GBinaryMIRExpr.h ------------------------------------------*-C++-*-===//
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
/// Defines the \c GBinaryMIRExpr class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXPR_G_BINARY_MIR_EXPR_H
#define LUTHIER_EXPR_G_BINARY_MIR_EXPR_H
#include "luthier/expr/GenericMIRExpr.h"

namespace luthier {

/// \brief Convenience class for constructing binary generic
/// machine instructions and identifying them via LLVM RTTI
class GBinaryMIRExpr : public GenericMIRExpr {

protected:
  GBinaryMIRExpr(uint16_t Opcode, const std::shared_ptr<GenericMIRExpr> &Left,
                 const std::shared_ptr<GenericMIRExpr> &Right)
      : GenericMIRExpr(
            Opcode, {Left->shared_from_this(), Right->shared_from_this()}) {};

public:
  const GenericMIRExpr &getLeftChild() const { return *Children[0]; }

  std::shared_ptr<GenericMIRExpr> &getLeftChild() { return Children[0]; }

  const GenericMIRExpr &getRightChild() const { return *Children[1]; }

  std::shared_ptr<GenericMIRExpr> &getRightChild() { return Children[1]; }

  static bool classof(const GenericMIRExpr *E) {
    switch (E->getOpcode()) {
    case llvm::TargetOpcode::G_ADD:
    case llvm::TargetOpcode::G_SUB:
    case llvm::TargetOpcode::G_MUL:
    case llvm::TargetOpcode::G_SDIV:
    case llvm::TargetOpcode::G_UDIV:
    case llvm::TargetOpcode::G_SREM:
    case llvm::TargetOpcode::G_UREM:
    case llvm::TargetOpcode::G_SMIN:
    case llvm::TargetOpcode::G_SMAX:
    case llvm::TargetOpcode::G_UMIN:
    case llvm::TargetOpcode::G_UMAX:
    // Floating point.
    case llvm::TargetOpcode::G_FMINNUM:
    case llvm::TargetOpcode::G_FMAXNUM:
    case llvm::TargetOpcode::G_FMINNUM_IEEE:
    case llvm::TargetOpcode::G_FMAXNUM_IEEE:
    case llvm::TargetOpcode::G_FMINIMUM:
    case llvm::TargetOpcode::G_FMAXIMUM:
    case llvm::TargetOpcode::G_FADD:
    case llvm::TargetOpcode::G_FSUB:
    case llvm::TargetOpcode::G_FMUL:
    case llvm::TargetOpcode::G_FDIV:
    case llvm::TargetOpcode::G_FPOW:
    // Logical.
    case llvm::TargetOpcode::G_AND:
    case llvm::TargetOpcode::G_OR:
    case llvm::TargetOpcode::G_XOR:
      return true;
    default:
      return false;
    }
  };
};

} // namespace luthier

#endif