//===-- ExecutableMemorySegmentAccessor.h -----------------------*- C++ -*-===//
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
/// Describes the \c ExecutableMemorySegmentAccessor interface class and its
/// associated analysis pass, <tt>ExecutableMemorySegmentAccessorAnalysis</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_EXECUTABLE_MEMORY_SEGMENT_ACCESSOR_H
#define LUTHIER_TOOLING_EXECUTABLE_MEMORY_SEGMENT_ACCESSOR_H
#include "luthier/hsa/ApiTable.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/PassManager.h>
#include <luthier/hsa/LoadedCodeObjectCache.h>

namespace luthier {

/// \brief Interface class used to provide information regarding executable
/// memory segments loaded on a target device to other instrumentation passes
/// \details An instance of this class is provided to other instrumentation
/// passes via the \c ExecutableMemorySegmentAccessorAnalysis in the target
/// module analysis manager. This class acts as a level of abstraction
/// over the underlying runtime for other instrumentation passes instead of
/// having them directly querying the underlying runtime. This helps to keep the
/// instrumentation passes runtime-agnostic, and makes it easier to test
/// the instrumentation passes without needing a physical GPU or the target
/// runtime
class ExecutableMemorySegmentAccessor {
public:
  typedef struct {
    /// parsed host code object used to load the segment
    const object::AMDGCNObjectFile *CodeObjectStorage{nullptr};
    /// Encapsulates the segment's copy on the host
    llvm::ArrayRef<uint8_t> SegmentOnHost{};
    /// Encapsulates the segment's location on device memory; Might not be
    /// host-accessible
    llvm::ArrayRef<uint8_t> SegmentOnDevice{};
  } SegmentDescriptor;

  /// Expects the \c SegmentDescriptor associated with the \p DeviceAddr
  [[nodiscard]] virtual llvm::Expected<SegmentDescriptor>
  getSegment(uint64_t DeviceAddr) const = 0;

  virtual ~ExecutableMemorySegmentAccessor() = default;
};

/// \brief Implementation of \c ExecutableMemorySegmentAccessor for the HSA
/// runtime
class HsaRuntimeExecutableMemorySegmentAccessor
    : public ExecutableMemorySegmentAccessor {

  const hsa::LoadedCodeObjectCache &COC;

  const hsa::ExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER>::TableType
      &VenLoaderTable;

public:
  [[nodiscard]] llvm::Expected<SegmentDescriptor>
  getSegment(uint64_t DeviceAddr) const override;

  HsaRuntimeExecutableMemorySegmentAccessor(
      const hsa::LoadedCodeObjectCache &COC,
      const hsa::ExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER>::TableType
          &VenLoaderTable)
      : COC(COC), VenLoaderTable(VenLoaderTable) {};

  ~HsaRuntimeExecutableMemorySegmentAccessor() override = default;
};

/// \brief Provides access to the \c ExecutableMemorySegmentAccessorAnalysis for
/// other instrumentation passes from the target module's analysis manager
class ExecutableMemorySegmentAccessorAnalysis
    : public llvm::AnalysisInfoMixin<ExecutableMemorySegmentAccessorAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<ExecutableMemorySegmentAccessorAnalysis>;

  static llvm::AnalysisKey Key;

  std::unique_ptr<ExecutableMemorySegmentAccessor> SegmentAccessor;

public:
  class Result {
    friend ExecutableMemorySegmentAccessorAnalysis;

    const ExecutableMemorySegmentAccessor &SegmentAccessor;

  public:
    explicit Result(const ExecutableMemorySegmentAccessor &SegmentAccessor)
        : SegmentAccessor(SegmentAccessor) {};

    /// Results should never be invalidated by the analysis manager
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    [[nodiscard]] const ExecutableMemorySegmentAccessor &getAccessor() const {
      return SegmentAccessor;
    }
  };

  explicit ExecutableMemorySegmentAccessorAnalysis(
      std::unique_ptr<ExecutableMemorySegmentAccessor> SegmentAccessor)
      : SegmentAccessor(std::move(SegmentAccessor)) {};

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result{*SegmentAccessor};
  }
};

} // namespace luthier

#endif