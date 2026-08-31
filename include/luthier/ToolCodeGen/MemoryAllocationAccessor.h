//===-- MemoryAllocationAccessor.h ------------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
#ifndef LUTHIER_TOOL_CODE_GEN_MEMORY_ALLOCATION_ACCESSOR_H
#define LUTHIER_TOOL_CODE_GEN_MEMORY_ALLOCATION_ACCESSOR_H
#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/ToolCodeGen/DriverAllocationResolver.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/PassManager.h>

#include <memory>

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
  private:
    /// Encapsulates the allocation's base address on device memory as well as
    /// its size on the device; Note that the underlying memory might not be
    /// host-accessible
    const std::byte *DeviceAllocation{nullptr};

    /// Encapsulates the allocation's accessible "version" on the host
    /// Depending on the underlying runtime and the allocation being queried,
    /// this field can be equal to the \c AllocationOnDevice field, or it can
    /// be a separate memory containing a copy of the device allocation
    /// The lifetime of this copy can be either managed by the abstracted
    /// underlying runtime or by the accessor
    const std::byte *HostAccessibleAllocation{nullptr};

    size_t Size{0};

    /// If the allocation was loaded using a code object by the underlying
    /// runtime, this field will provide its parsed object
    const object::AMDGCNObjectFile *AllocationCodeObject{nullptr};

  public:
    AllocationDescriptor() = default;

    AllocationDescriptor(
        const std::byte &DeviceAllocation, const std::byte &HostAllocation,
        size_t Size,
        const object::AMDGCNObjectFile *AllocationCodeObject = nullptr)
        : DeviceAllocation(&DeviceAllocation),
          HostAccessibleAllocation(&HostAllocation), Size(Size),
          AllocationCodeObject(AllocationCodeObject) {}

    [[nodiscard]] bool empty() const { return Size == 0; }

    [[nodiscard]] size_t getSize() const { return Size; }

    llvm::ArrayRef<uint8_t> getDeviceAllocation() const {
      return {reinterpret_cast<const uint8_t *>(DeviceAllocation), Size};
    }

    llvm::ArrayRef<uint8_t> getHostAllocation() const {
      return {reinterpret_cast<const uint8_t *>(HostAccessibleAllocation),
              Size};
    }

    const object::AMDGCNObjectFile *getAllocationCodeObject() const {
      return AllocationCodeObject;
    }

    /// \brief Translate a device address inside this allocation into the
    /// corresponding address in the host-readable view.
    ///
    /// Exists because doing this by hand is easy to get wrong in a way nothing
    /// catches: the two bases are equal for every accessor that reads memory the
    /// host already owns, so an expression that subtracts the device base and adds
    /// it back -- cancelling to the device address -- behaves correctly until an
    /// accessor appears whose host view lives elsewhere. Allocations tracked
    /// through KFD ioctls are exactly that case.
    ///
    /// \param DeviceAddr an address inside this allocation. Callers already know
    /// it is inside, because that is how they obtained the descriptor.
    [[nodiscard]] uint64_t hostAddressFor(uint64_t DeviceAddr) const {
      const auto DeviceBase = reinterpret_cast<uint64_t>(DeviceAllocation);
      const auto HostBase = reinterpret_cast<uint64_t>(HostAccessibleAllocation);
      return HostBase + (DeviceAddr - DeviceBase);
    }

    /// \brief One past the last host-readable byte of this allocation.
    ///
    /// Anchored to the allocation's own host base, not to whatever address a
    /// caller happens to be reading from. Adding \c Size to a mid-allocation
    /// address instead overruns the end by that address's offset into the
    /// allocation -- and a caller that started at an entry point rather than at
    /// the base is the normal case, not the exception.
    [[nodiscard]] uint64_t hostEndAddress() const {
      return reinterpret_cast<uint64_t>(HostAccessibleAllocation) + Size;
    }

    /// TODO: Consider adding the allocation flags (e.g., permissions)
  };

  /// Provides the allocation descriptor of the \p DeviceAddr
  /// \returns The allocation descriptor of the \p DeviceAddr if exists, \c
  /// std::nullopt if there are not allocation associated with the address, and
  /// an \c llvm::Error if any other issue was encountered in the process
  [[nodiscard]] virtual llvm::Expected<AllocationDescriptor>
  getAllocationDescriptor(uint64_t DeviceAddr) const = 0;

  virtual ~MemoryAllocationAccessor() = default;
};

/// \brief The \c MemoryAllocationAccessor for a process with no GPU runtime above
/// the driver.
///
/// \par When this is the right accessor
/// An application that issues KFD ioctls itself cannot have HSA in its process:
/// it holds the DRM virtual address space for its GPUs, the kernel permits one
/// per GPU per process, and \c hsa_init therefore fails there. So there is no
/// runtime to ask, and the driver-level record is not a fallback but the only
/// source. \c HsaMemoryAllocationAccessor would also work -- its HSA half detects
/// that it cannot run and defers -- but it would carry three API-table snapshots
/// and a loaded-code-object cache that can never be populated, and a reader would
/// reasonably wonder what they were for.
///
/// \par What it necessarily cannot report
/// A parsed code object, because there is no loader below the driver to have
/// parsed one. \c CodeDiscoveryPass handles that by naming the kernel after its
/// address (\c CodeDiscoveryPass.cpp:761) rather than by failing.
class DriverOnlyMemoryAllocationAccessor final : public MemoryAllocationAccessor {
  std::unique_ptr<DriverAllocationResolver> Resolver;

public:
  /// \param Resolver the driver-level source. May be null, which makes every
  /// lookup report no allocation -- the same answer an available resolver gives
  /// for an address it has not seen, because a caller walking a disassembly must
  /// not be aborted either way.
  explicit DriverOnlyMemoryAllocationAccessor(
      std::unique_ptr<DriverAllocationResolver> Resolver)
      : Resolver(std::move(Resolver)) {}

  /// \brief Turn a driver-level allocation into a descriptor.
  ///
  /// The one place this conversion is written. Two accessors need it, and the
  /// thing they must not disagree about is which of the two bases goes in which
  /// slot: they are equal for every source that reads memory the host already
  /// owns, so swapping them is invisible until a source appears whose host view
  /// lives elsewhere -- which is exactly what a driver-level resolver is.
  [[nodiscard]] static AllocationDescriptor
  descriptorFor(const DriverAllocationResolver::Allocation &A) {
    return AllocationDescriptor{*A.DeviceBase, *A.HostBase, A.Size, nullptr};
  }

  [[nodiscard]] llvm::Expected<AllocationDescriptor>
  getAllocationDescriptor(uint64_t DeviceAddr) const override;

  /// \brief Whether a driver-level source is present and has records.
  [[nodiscard]] bool hasSource() const {
    return Resolver != nullptr && Resolver->isAvailable();
  }
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