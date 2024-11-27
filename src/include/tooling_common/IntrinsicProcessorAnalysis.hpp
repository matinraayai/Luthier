//===-- IntrinsicProcessorAnalysis.hpp ------------------------------------===//
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
/// This file describes the <tt>IntrinsicProcessorAnalysis</tt>,
/// an analysis pass that provides IR and MIR lowering functions for each
/// intrinsic used inside the instrumentation module.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_INTRINSIC_PROCESSOR_ANALYSIS_HPP
#define LUTHIER_TOOLING_COMMON_INTRINSIC_PROCESSOR_ANALYSIS_HPP
#include <llvm/IR/PassManager.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>

namespace luthier {

class IntrinsicProcessorAnalysis
    : public llvm::AnalysisInfoMixin<IntrinsicProcessorAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<IntrinsicProcessorAnalysis>;

  static llvm::AnalysisKey Key;

  const llvm::StringMap<IntrinsicProcessor> &IntrinsicProcessorMap;

public:
  class Result {
    const llvm::StringMap<IntrinsicProcessor> &Map;
    Result(const llvm::StringMap<IntrinsicProcessor> &Map) : Map(Map) {}
    friend class IntrinsicProcessorAnalysis;

  public:
    const llvm::StringMap<IntrinsicProcessor> &getProcessors() { return Map; }

    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  explicit IntrinsicProcessorAnalysis(
      const llvm::StringMap<IntrinsicProcessor> &IntrinsicProcessorMap)
      : IntrinsicProcessorMap(IntrinsicProcessorMap) {};

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif