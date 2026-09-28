//===-- HSADispatcher.h -----------------------------------------*- C++ -*-===//
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
/// Describes the \c HsaMemoryAllocationAccessor class which implements the
/// \c MemoryAllocationAccessor interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_HSA_DISPATCHER_H
#define LUTHIER_HSA_TOOLING_HSA_DISPATCHER_H
#include <cstddef>
#include <cstdint>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <string>

namespace luthier::test {

/// Manages HSA runtime lifecycle, GPU detection, memory allocation, and kernel
/// dispatch for the instruction semantic fuzzer.
class HSADispatcher {
public:
  HSADispatcher() = default;
  ~HSADispatcher();

  HSADispatcher(const HSADispatcher &) = delete;
  HSADispatcher &operator=(const HSADispatcher &) = delete;

  /// Initialize the HSA runtime, detect a GPU agent, create a queue, and
  /// discover memory pools.
  llvm::Error init();

  /// Shut down the HSA runtime and release all resources.
  void shutdown();

  /// \returns The GPU target string (e.g. "gfx908", "gfx90a", "gfx1030").
  llvm::StringRef getGpuTarget() const { return GpuTarget; }

  /// Allocate \p Size bytes from the kernarg memory pool.
  /// The returned pointer is CPU-writable and GPU-readable.
  llvm::Expected<void *> allocKernarg(size_t Size);

  /// Allocate \p Size bytes from the fine-grained system pool.
  /// Suitable for output buffers readable from both CPU and GPU.
  llvm::Expected<void *> allocFinegrained(size_t Size);

  /// Allocate \p Size bytes from coarse-grained VRAM (device-local).
  /// Suitable for data buffers accessed by the GPU.
  llvm::Expected<void *> allocDeviceLocal(size_t Size);

  /// Free memory previously allocated via any of the alloc* methods.
  void freeMem(void *Ptr);

  /// Holds the state needed to dispatch a single kernel and read back results.
  struct DispatchResult {
    hsa_signal_value_t SignalValue; ///< Final signal value (0 = success).
  };

  /// Load an ELF code object, look up the kernel symbol \p KernelName,
  /// and dispatch it with the given parameters.
  ///
  /// \param ELF           Raw bytes of the AMDGPU ELF code object.
  /// \param KernelName    Mangled symbol name of the kernel.
  /// \param KernargPtr    Pointer to the filled kernarg buffer.
  /// \param KernargSize   Size of the kernarg buffer in bytes.
  /// \param GridSizeX     Total number of work-items in the X dimension.
  /// \param WorkgroupSizeX Number of work-items per workgroup in X.
  /// \param GroupSegSize   Requested LDS size in bytes.
  /// \param PrivateSegSize Requested scratch size per work-item in bytes.
  llvm::Expected<DispatchResult>
  dispatch(llvm::ArrayRef<char> ELF, llvm::StringRef KernelName,
           void *KernargPtr, uint32_t KernargSize, uint32_t GridSizeX = 1,
           uint32_t WorkgroupSizeX = 1, uint32_t GroupSegSize = 0,
           uint32_t PrivateSegSize = 0);

private:
  bool Initialized = false;
  hsa_agent_t GpuAgent{};
  hsa_agent_t CpuAgent{};
  hsa_queue_t *Queue = nullptr;
  std::string GpuTarget;

  // Memory pools.
  hsa_amd_memory_pool_t KernargPool{};
  hsa_amd_memory_pool_t FinePool{};
  hsa_amd_memory_pool_t DevicePool{};
  bool HasKernargPool = false;
  bool HasFinePool = false;
  bool HasDevicePool = false;

  /// Discover memory pools for the GPU and CPU agents.
  llvm::Error discoverMemoryPools();
};

} // namespace luthier::test

#endif // LUTHIER_TEST_HSA_DISPATCHER_H
