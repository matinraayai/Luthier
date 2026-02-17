//===-- GMergeValuesMIRExpr.h -------------------------------------*-C++-*-===//
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
/// Defines the generic merge values expression of the generic machine IR.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_G_MERGE_VALUES_MIR_EXPR_H
#define LUTHIER_G_MERGE_VALUES_MIR_EXPR_H
#include "luthier/expr/GenericMIRExpr.h"

namespace luthier {

/// \brief Represents a merge values expression in the generic machine
/// instruction
class GMergeValuesMIRExpr final : public GenericMIRExpr {

  GMergeValuesMIRExpr(
      std::initializer_list<std::shared_ptr<GenericMIRExpr>> Values)
      : GenericMIRExpr(llvm::TargetOpcode::G_MERGE_VALUES, Values) {};

public:
  template <typename... T>
  static std::shared_ptr<GMergeValuesMIRExpr> create(const T &...Expressions) {
    return std::shared_ptr<GMergeValuesMIRExpr>(new GMergeValuesMIRExpr(
        std::initializer_list{(Expressions->shared_from_this(), ...)}));
  }

  unsigned int getWidth() const override {
    return std::transform_reduce(
        Children.begin(), Children.end(), 0, std::plus{},
        [](const std::shared_ptr<GenericMIRExpr> &Expr) {
          return Expr->getWidth();
        });
  }

  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_MERGE_VALUES;
  }
};

} // namespace luthier

#endif