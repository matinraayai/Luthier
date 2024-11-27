//===-- LiftedRepresentationAnalysis.hpp ----------------------------------===//
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
/// This file describes Luthier's Lifted Representation Analysis pass, used
/// in the code generator's instrumentation pipeline.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_LIFTED_REPRESENTATION_ANALYSIS_HPP
#define LUTHIER_TOOLING_COMMON_LIFTED_REPRESENTATION_ANALYSIS_HPP
#include "luthier/LiftedRepresentation.h"
#include <llvm/IR/PassManager.h>

namespace luthier {

class LiftedRepresentationAnalysis
    : public llvm::AnalysisInfoMixin<LiftedRepresentationAnalysis> {
private:
  friend AnalysisInfoMixin<LiftedRepresentationAnalysis>;

  static llvm::AnalysisKey Key;
  /// The \c LiftedRepresentation being worked on
  LiftedRepresentation &LR;

public:
  using Result = LiftedRepresentation &;

  explicit LiftedRepresentationAnalysis(LiftedRepresentation &LR) : LR(LR) {}

  /// Run the analysis pass and produce machine module information.
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};
} // namespace luthier

#endif