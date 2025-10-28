//===-- GBitFieldExMIRExpr.h --------------------------------------*-C++-*-===//
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
/// Defines the bit field extract expressions of the generic machine IR.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXPR_G_BIT_FIELD_EX_MIR_EXPR_H
#define LUTHIER_EXPR_G_BIT_FIELD_EX_MIR_EXPR_H
#include "luthier/expr/GenericMIRExpr.h"

namespace luthier {

/// \brief Represents a bit field extract expression in generic machine
/// instruction
class GBitFieldExMIRExpr : public GenericMIRExpr {

  /// Width of this expression; In generic MIR, this is the output reg's
  /// specified width
  unsigned OutputWidth;

protected:
  GBitFieldExMIRExpr(uint16_t Opcode,
                     const std::shared_ptr<GenericMIRExpr> &Expr,
                     const std::shared_ptr<GenericMIRExpr> &LSBExtractionStart,
                     const std::shared_ptr<GenericMIRExpr> &ExtractionWidth,
                     unsigned int OutputWidth)
      : GenericMIRExpr(Opcode, {Expr->shared_from_this(),
                                LSBExtractionStart->shared_from_this(),
                                ExtractionWidth->shared_from_this()}),
        OutputWidth(OutputWidth) {};

public:
  unsigned int getWidth() const override { return OutputWidth; }

  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_SBFX ||
           E->getOpcode() == llvm::TargetOpcode::G_UBFX;
  }
};

/// \brief Defines an unsigned bit field extract expression in generic MIR
class GUBFXMIRExpr final : public GBitFieldExMIRExpr {

  GUBFXMIRExpr(const std::shared_ptr<GenericMIRExpr> &Expr,
               const std::shared_ptr<GenericMIRExpr> &LSBExtractionStart,
               const std::shared_ptr<GenericMIRExpr> &ExtractionWidth,
               unsigned int OutputWidth)
      : GBitFieldExMIRExpr(llvm::TargetOpcode::G_UBFX, Expr, LSBExtractionStart,
                           ExtractionWidth, OutputWidth) {};

public:
  static std::shared_ptr<GUBFXMIRExpr>
  create(const std::shared_ptr<GenericMIRExpr> &Expr,
         const std::shared_ptr<GenericMIRExpr> &LSBExtractionStart,
         const std::shared_ptr<GenericMIRExpr> &ExtractionWidth,
         unsigned int OutputWidth) {
    return std::shared_ptr<GUBFXMIRExpr>(new GUBFXMIRExpr(
        Expr, LSBExtractionStart, ExtractionWidth, OutputWidth));
  }

  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_UBFX;
  }
};

/// \brief Defines a signed bit field extract expression in generic MIR
class GSBFXMIRExpr final : public GBitFieldExMIRExpr {

  GSBFXMIRExpr(const std::shared_ptr<GenericMIRExpr> &Expr,
               const std::shared_ptr<GenericMIRExpr> &LSBExtractionStart,
               const std::shared_ptr<GenericMIRExpr> &ExtractionWidth,
               unsigned int OutputWidth)
      : GBitFieldExMIRExpr(llvm::TargetOpcode::G_SBFX, Expr, LSBExtractionStart,
                           ExtractionWidth, OutputWidth) {};

public:
  static std::shared_ptr<GSBFXMIRExpr>
  create(const std::shared_ptr<GenericMIRExpr> &Expr,
         const std::shared_ptr<GenericMIRExpr> &LSBExtractionStart,
         const std::shared_ptr<GenericMIRExpr> &ExtractionWidth,
         unsigned int OutputWidth) {
    return std::shared_ptr<GSBFXMIRExpr>(new GSBFXMIRExpr(
        Expr, LSBExtractionStart, ExtractionWidth, OutputWidth));
  }

  static bool classof(const GenericMIRExpr *E) {
    return E->getOpcode() == llvm::TargetOpcode::G_SBFX;
  }
};

} // namespace luthier

#endif