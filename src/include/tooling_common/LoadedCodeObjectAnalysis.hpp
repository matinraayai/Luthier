//===-- LoadedCodeObjectAnalysis.hpp --------------------------------------===//
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
/// This file describes Luthier's Loaded Code Object analysis, indicating
/// the current \c hsa::LoadedCodeObject of a \c LiftedRepresentation being
/// worked on inside the instrumentation pipeline.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_LOADED_CODE_OBJECT_ANALYSIS_HPP
#define LUTHIER_TOOLING_COMMON_LOADED_CODE_OBJECT_ANALYSIS_HPP

#include "hsa/LoadedCodeObject.hpp"
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief An analysis pass which returns the \c hsa::LoadedCodeObject being
/// worked on during instrumentation
class LoadedCodeObjectAnalysis
    : public llvm::AnalysisInfoMixin<LoadedCodeObjectAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<LoadedCodeObjectAnalysis>;

  static llvm::AnalysisKey Key;

  /// The \c hsa::LoadedCodeObject inside the lifted representation being worked
  /// on
  const hsa::LoadedCodeObject LCO;

public:
  using Result = const hsa::LoadedCodeObject;

  explicit LoadedCodeObjectAnalysis(hsa::LoadedCodeObject LCO)
      : LCO(std::move(LCO)) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace luthier

#endif