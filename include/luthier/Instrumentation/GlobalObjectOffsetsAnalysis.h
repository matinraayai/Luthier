//===-- GlobalValueOffsetsAnalysis.h ----------------------------*- C++ -*-===//
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
/// Describes the \c GlobalObjectOffsetsAnalysis which provides a
/// mapping between target object's offsets and their associated global
/// object.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_GLOBAL_OBJECT_OFFSETS_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_GLOBAL_OBJECT_OFFSETS_H
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class GlobalObjectOffsetsAnalysis
    : public llvm::AnalysisInfoMixin<GlobalObjectOffsetsAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend class GlobalObjectOffsetsAnalysis;

    llvm::DenseMap<uint64_t, const llvm::GlobalObject &> GlobalObjectOffsets{};

    Result() = default;

  public:
    llvm::GlobalObject *getObject(uint64_t Offset) const;

    void insertSymbol(uint64_t Offset, const llvm::GlobalObject &GO) {
      GlobalObjectOffsets.insert({Offset, GO});
    }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

private:
  Result GlobalObjectOffsets;

public:
  GlobalObjectOffsetsAnalysis() = default;

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return GlobalObjectOffsets;
  }
};

} // namespace luthier

#endif
