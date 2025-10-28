//===-- ConstantMIRExpr.h -----------------------------------------*-C++-*-===//
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
/// Defines the constant generic machine instruction expression classes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXPR_CONSTANT_MIR_EXPR_H
#define LUTHIER_EXPR_CONSTANT_MIR_EXPR_H
#include "luthier/expr/GenericMIRExpr.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/CodeGen/TargetOpcodes.h>

#include <utility>

namespace luthier {

class ConstantMIRExpr : public GenericMIRExpr {

protected:
  explicit ConstantMIRExpr(uint16_t Opcode) : GenericMIRExpr(Opcode, {}) {}

public:
  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_CONSTANT ||
           E->getOpcode() == llvm::TargetOpcode::G_FCONSTANT;
  }
};

class GConstantMIRExpr final : public ConstantMIRExpr {

private:
  llvm::APInt Value;

  explicit GConstantMIRExpr(llvm::APInt V)
      : ConstantMIRExpr(llvm::TargetOpcode::G_CONSTANT), Value(std::move(V)) {}

public:
  unsigned int getWidth() const override { return Value.getBitWidth(); }

  /// \returns the arbitrary precision value of the constant
  const llvm::APInt &getAPValue() const { return Value; }

  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_CONSTANT;
  }

  static std::shared_ptr<GConstantMIRExpr> create(const llvm::APInt &V) {
    return std::shared_ptr<GConstantMIRExpr>(new GConstantMIRExpr(V));
  }
};

class GFConstantMIRExpr final : public ConstantMIRExpr {

private:
  llvm::APFloat Value;

  explicit GFConstantMIRExpr(llvm::APFloat V)
      : ConstantMIRExpr(llvm::TargetOpcode::G_FCONSTANT), Value(std::move(V)) {}

public:
  unsigned int getWidth() const override {
    return llvm::APFloatBase::getSizeInBits(Value.getSemantics());
  }

  /// \returns the arbitrary precision value of the constant
  const llvm::APFloat &getAPValue() const { return Value; }

  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_FCONSTANT;
  }

  static std::shared_ptr<GFConstantMIRExpr> create(const llvm::APFloat &V) {
    return std::shared_ptr<GFConstantMIRExpr>(new GFConstantMIRExpr(V));
  }
};

} // namespace luthier

#endif