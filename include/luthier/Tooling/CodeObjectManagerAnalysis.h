//===-- CodeObjectManagerAnalysis.h -----------------------------*- C++ -*-===//
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
/// Defines the \c CodeObjectManagerAnalysis which takes ownership of
/// shared object files used with the mock loader in tests.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_CODE_OBJECT_MANAGER_ANALYSIS_H
#define LUTHIER_TOOLING_CODE_OBJECT_MANAGER_ANALYSIS_H
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

namespace luthier {

/// \brief An analysis primarily used in tests that reads and validates
/// AMDGPU code objects from file and takes ownership of them for use
/// with the \c MockLoadAMDGPUCodeObjects pass and the
/// \c MockAMDGPULoaderAnalysis
class CodeObjectManagerAnalysis
    : public llvm::AnalysisInfoMixin<CodeObjectManagerAnalysis> {
  friend llvm::AnalysisInfoMixin<CodeObjectManagerAnalysis>;

public:
  static llvm::AnalysisKey Key;

  class Result {
    friend class CodeObjectManagerAnalysis;

    /// Memory buffers associated with each code object
    llvm::SmallVector<std::unique_ptr<llvm::MemoryBuffer>> CodeObjects{};

    /// Never invalidate the analysis
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    Result() = default;

  public:
    /// Reads and validates the code object located in \c Path
    /// \returns a reference to the code object's buffer owned by the analysis
    /// result
    llvm::Expected<llvm::MemoryBuffer &>
    readCodeObjectFromFile(llvm::StringRef Path);

    /// Validates the code object and takes ownership of the \p CodeObject
    /// memory buffer
    /// \returns a reference to the code object's buffer owned by the analysis
    /// result
    llvm::Expected<llvm::MemoryBuffer &>
    takeOwnershipOfCodeObject(std::unique_ptr<llvm::MemoryBuffer> CodeObject);
  };

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    return Result{};
  }
};

} // namespace luthier

#endif