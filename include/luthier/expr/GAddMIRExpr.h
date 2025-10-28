//===-- GAddMIRExpr.h ---------------------------------------------*-C++-*-===//
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
/// Defines the \c GAddMIRExpr class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXPR_G_ADD_MIR_EXPR_H
#define LUTHIER_EXPR_G_ADD_MIR_EXPR_H
#include "luthier/expr/GBinaryMIRExpr.h"

namespace luthier {

class GAddMIRExpr final : public GBinaryMIRExpr {
  GAddMIRExpr(const std::shared_ptr<GenericMIRExpr> &Left,
              const std::shared_ptr<GenericMIRExpr> &Right)
      : GBinaryMIRExpr(llvm::TargetOpcode::G_ADD, Left, Right) {}

public:
  static std::shared_ptr<GAddMIRExpr>
  create(const std::shared_ptr<GenericMIRExpr> &Left,
         const std::shared_ptr<GenericMIRExpr> &Right) {
    return std::shared_ptr<GAddMIRExpr>(new GAddMIRExpr(Left, Right));
  }

  unsigned int getWidth() const override { return getLeftChild().getWidth(); }

  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_ADD;
  }
};

} // namespace luthier

#endif