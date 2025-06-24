//===-- IntrinsicProcessorsAnalysis.h ---------------------------*- C++ -*-===//
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
/// Defines the \c IntrinsicProcessorsAnalysis class, which provides the
/// mapping between the names and the processors of all defined Luthier
/// intrinsics in the instrumentation process.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_INTRINSIC_PROCESSORS_ANALYSIS_H
#define LUTHIER_INSTRUMENTATION_INTRINSIC_PROCESSORS_ANALYSIS_H
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/PassManager.h>
#include <luthier/Instrumentation/IntrinsicProcessor.h>

namespace luthier {
/// \brief Produces the processors for all intrinsics used in instrumentation
class IntrinsicsProcessorsAnalysis
    : public llvm::AnalysisInfoMixin<IntrinsicsProcessorsAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<IntrinsicsProcessorsAnalysis>;

  static llvm::AnalysisKey Key;

  llvm::StringMap<IntrinsicProcessor> IntrinsicProcessors;

public:
  class Result {
    friend class IntrinsicsProcessorsAnalysis;

    llvm::StringMap<IntrinsicProcessor> &IntrinsicProcessors;

    explicit Result(llvm::StringMap<IntrinsicProcessor> &IntrinsicProcessors)
        : IntrinsicProcessors(IntrinsicProcessors) {}

  public:
    bool isIntrinsicDefined(llvm::StringRef IntrinsicName) const {
      return IntrinsicProcessors.contains(IntrinsicName);
    }

    std::optional<IntrinsicProcessor>
    getProcessor(llvm::StringRef IntrinsicName) const;

    llvm::Error defineIntrinsic(llvm::StringRef IntrinsicName,
                                IntrinsicIRProcessorFunc IRProcessor,
                                IntrinsicMIRProcessorFunc MIRProcessor);

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  IntrinsicsProcessorsAnalysis();

  Result run(llvm::Module &IModule, llvm::ModuleAnalysisManager &IMAM);
};

} // namespace luthier

#endif
