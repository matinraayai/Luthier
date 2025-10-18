//===-- MetadataParserAnalysis.h --------------------------------*- C++-*-===//
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
/// Describes the \c MetadataParserAnalysis class which provides access to a
/// metadata parser to other instrumentation passes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_METADATA_PARSER_ANALYSIS_H
#define LUTHIER_TOOLING_METADATA_PARSER_ANALYSIS_H
#include "luthier/hsa/Metadata.h"
#include <llvm/IR/PassManager.h>

namespace luthier {

class MetadataParserAnalysis
    : public llvm::AnalysisInfoMixin<MetadataParserAnalysis> {
  friend llvm::AnalysisInfoMixin<MetadataParserAnalysis>;

  static llvm::AnalysisKey Key;

  amdgpu::hsamd::MetadataParser &Parser;

public:
  class Result {
    friend MetadataParserAnalysis;

    amdgpu::hsamd::MetadataParser &Parser;

    explicit Result(amdgpu::hsamd::MetadataParser &Parser) : Parser(Parser) {};

  public:
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    const amdgpu::hsamd::MetadataParser &getParser() const { return Parser; }
  };

  explicit MetadataParserAnalysis(amdgpu::hsamd::MetadataParser &Parser)
      : Parser(Parser) {};

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result{Parser};
  };
};

} // namespace luthier

#endif