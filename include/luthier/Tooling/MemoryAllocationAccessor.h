//===-- MemoryAllocationAccessor.h ------------------------------*- C++ -*-===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
/// Describes the \c MemoryAllocationAccessor interface and its
/// associated analysis pass <tt>MemoryAllocationAnalysis</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_MEMORY_ALLOCATION_ACCESSOR_H
#define LUTHIER_TOOLING_MEMORY_ALLOCATION_ACCESSOR_H
#include "luthier/Object/AMDGCNObjectFile.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief Interface that provides information regarding memory allocations
/// in the target GPU runtime to passes in the Luthier code generation pipeline
/// \details An instance of this class is provided to other instrumentation
/// passes via the \c MemoryAllocationAnalysis in the target module analysis
/// manager. This class acts as a level of abstraction over the underlying
/// GPU runtime for other instrumentation passes instead of having them directly
/// query the GPU runtime. This helps to keep the instrumentation passes
/// runtime-agnostic, and makes it easier to test the instrumentation passes
/// without needing a physical GPU or the target runtime
class MemoryAllocationAccessor {
public:
  struct AllocationDescriptor {
    /// Encapsulates the allocation's base address on device memory as well as
    /// its size on the device; Note that the underlying memory might not be
    /// host-accessible
    llvm::ArrayRef<uint8_t> AllocationOnDevice{};

    /// Encapsulates the allocation's accessible "version" on the host
    /// Depending on the underlying runtime and the allocation being queried,
    /// this field can be equal to the \c AllocationOnDevice field, or it can
    /// be a separate memory containing a copy of the device allocation
    /// The lifetime of this copy can be either managed by the abstracted
    /// underlying runtime or by the accessor
    llvm::ArrayRef<uint8_t> AllocationOnHost{};

    /// If the allocation was loaded using a code object by the underlying
    /// runtime, this field will provide its parsed object
    const object::AMDGCNObjectFile *AllocationCodeObject{nullptr};

    /// TODO: Consider non-uniform loaded code object allocations (i.e., the
    /// loader loads the code object into multiple non-contiguous memory
    /// allocations)
    /// TODO: Consider adding the allocation flags (e.g., permissions)
  };

  /// Expects the \c SegmentDescriptor associated with the \p DeviceAddr
  [[nodiscard]] virtual llvm::Expected<AllocationDescriptor>
  getAllocationDescriptor(uint64_t DeviceAddr) const = 0;

  virtual ~MemoryAllocationAccessor() = default;
};

/// \brief Provides the \c MemoryAllocationAccessor to
/// other passes from the target module's analysis manager
class MemoryAllocationAnalysis
    : public llvm::AnalysisInfoMixin<MemoryAllocationAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<MemoryAllocationAnalysis>;

  static llvm::AnalysisKey Key;

  std::unique_ptr<MemoryAllocationAccessor> SegmentAccessor;

public:
  class Result {
    friend MemoryAllocationAnalysis;

    const MemoryAllocationAccessor &SegmentAccessor;

  public:
    explicit Result(const MemoryAllocationAccessor &SegmentAccessor)
        : SegmentAccessor(SegmentAccessor) {};

    /// Results should never be invalidated by the analysis manager
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    [[nodiscard]] const MemoryAllocationAccessor &getAccessor() const {
      return SegmentAccessor;
    }
  };

  explicit MemoryAllocationAnalysis(
      std::unique_ptr<MemoryAllocationAccessor> SegmentAccessor)
      : SegmentAccessor(std::move(SegmentAccessor)) {};

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result{*SegmentAccessor};
  }
};

} // namespace luthier

#endif