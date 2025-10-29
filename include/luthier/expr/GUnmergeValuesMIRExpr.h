//===-- GUnMergeValuesMIRExpr.h -----------------------------------*-C++-*-===//
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
/// Defines the generic unmerge values expression of the generic machine IR.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXPR_G_UNMERGE_VALUES_MIR_EXPR_H
#define LUTHIER_EXPR_G_UNMERGE_VALUES_MIR_EXPR_H
#include "luthier/expr/GenericMIRExpr.h"

namespace luthier {

class GUnmergeValuesMIRExpr : public GenericMIRExpr {
  /// Width of the unmerged values
  unsigned int Width;
  /// Index of the value extracted from the input expression; lower value
  /// corresponds to lower bits
  unsigned int Idx;

  GUnmergeValuesMIRExpr(const std::shared_ptr<GenericMIRExpr> &Input,
                        unsigned int Width, unsigned int Idx)
      : GenericMIRExpr(llvm::TargetOpcode::G_UNMERGE_VALUES,
                       {Input->shared_from_this()}),
        Width(Width), Idx(Idx) {};

public:
  std::shared_ptr<GUnmergeValuesMIRExpr>
  create(const std::shared_ptr<GenericMIRExpr> &Input, unsigned int Width,
         unsigned int Idx) {
    return std::shared_ptr<GUnmergeValuesMIRExpr>(
        new GUnmergeValuesMIRExpr(Input, Width, Idx));
  }

  unsigned int getWidth() const override { return Width; }

  unsigned int getIdx() const { return Idx; }

  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_UNMERGE_VALUES;
  }
};

} // namespace luthier

#endif