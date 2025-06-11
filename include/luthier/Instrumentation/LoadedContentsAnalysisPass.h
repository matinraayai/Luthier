//===-- LoadedContentsAnalysisPass.h ----------------------------*- C++ -*-===//
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
/// Provides access to the loaded version of the instrumented object file as
/// an analysis pass.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_INSTRUMENTATION_LOADED_CONTENTS_ANALYSIS_PASS_H
#define LUTHIER_INSTRUMENTATION_LOADED_CONTENTS_ANALYSIS_PASS_H
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief Analysis pass that indicates the instrumented object file has been
/// loaded onto a target device, and provides access to the loaded image for
/// use by other instrumentation passes
class LoadedContentsAnalysisPass
    : public llvm::AnalysisInfoMixin<LoadedContentsAnalysisPass> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend LoadedContentsAnalysisPass;

    llvm::ArrayRef<uint8_t> LoadedContents;

    explicit Result(llvm::ArrayRef<uint8_t> LoadedContents)
        : LoadedContents(LoadedContents) {};

  public:
    llvm::ArrayRef<uint8_t> getLoadedContents() const { return LoadedContents; }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

private:
  /// This is a unique pointer, because the loaded contents array ref can be
  /// non-owning (i.e., there is a host-accessible version managed by the
  /// runtime of choice) or owning (i.e., there is no host-accessible copy
  /// managed by the runtime, and we had to make a copy to host before storing
  /// it here)
  std::unique_ptr<llvm::ArrayRef<uint8_t>> LoadedContents;

public:
  /// Use this for when the loaded image is accessible on the host
  explicit LoadedContentsAnalysisPass(llvm::ArrayRef<uint8_t> NonOwningContents)
      : LoadedContents(
            std::make_unique<llvm::ArrayRef<uint8_t>>(NonOwningContents)) {};

  /// Use this for when the loaded image was copied over to the host
  explicit LoadedContentsAnalysisPass(
      llvm::OwningArrayRef<uint8_t> OwningContents)
      : LoadedContents(std::make_unique<llvm::OwningArrayRef<uint8_t>>(
            std::move(OwningContents))) {}

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result{*LoadedContents};
  }
};

} // namespace luthier

#endif
