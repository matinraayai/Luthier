//===-- SharedBuffers.h - Pinned host buffers the GPU writes into -*- C++ -*-===//
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
///
/// \file
/// Owns the memory shared between the GPU and the CPU: the message buffer,
/// per-sub-buffer sizes, the host hand-off flags and the error bits.
///
/// dh_comms allocates these with
/// <tt>hipHostMalloc(&p, size, hipHostMallocCoherent)</tt>. A Luthier tool
/// talks to HSA rather than HIP, so this does what that HIP call does
/// underneath: allocate from the CPU agent's fine-grained global memory pool
/// (coherent between CPU and GPU, no explicit copies) and grant every GPU
/// agent access to it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXAMPLES_DH_COMMS_SHARED_BUFFERS_H
#define LUTHIER_EXAMPLES_DH_COMMS_SHARED_BUFFERS_H

#include "dh_comms/Protocol.h"

#include <hsa/hsa_api_trace.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <luthier/HSA/ApiTable.h>

#include <memory>

namespace luthier::dh_comms {

class SharedBuffers {
public:
  struct Config {
    uint32_t NumSubBuffers;     ///< Power of two, <= MaxSubBuffers.
    uint64_t SubBufferCapacity; ///< Bytes per sub-buffer.
  };

  /// Allocates every shared region and makes it accessible to all GPUs.
  static llvm::Expected<std::unique_ptr<SharedBuffers>>
  create(const hsa::ApiTableContainer<::CoreApiTable> &Core,
         const hsa::ApiTableContainer<::AmdExtTable> &AmdExt, Config Cfg);

  ~SharedBuffers();
  SharedBuffers(const SharedBuffers &) = delete;
  SharedBuffers &operator=(const SharedBuffers &) = delete;

  /// Empties every sub-buffer and clears flags and error bits. Only valid
  /// while no kernel is writing to the buffers.
  void reset();

  /// Writes the buffer half of the descriptor the GPU uses.
  void describe(DeviceDescriptor &D) const;

  uint32_t numSubBuffers() const { return Cfg.NumSubBuffers; }
  uint64_t subBufferCapacity() const { return Cfg.SubBufferCapacity; }
  const char *subBuffer(uint32_t I) const {
    return Buffer + uint64_t(I) * Cfg.SubBufferCapacity;
  }
  uint64_t *subBufferSizes() const { return SubBufferSizes; }
  uint32_t *hostFlags() const { return HostFlags; }
  uint32_t errorBits() const { return *ErrorBits; }

private:
  SharedBuffers(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                Config Cfg)
      : AmdExt(AmdExt), Cfg(Cfg) {}

  /// One zero-initialised allocation from the fine-grained host pool.
  llvm::Expected<void *> allocate(size_t Bytes);

  hsa::ApiTableContainer<::AmdExtTable> AmdExt;
  Config Cfg;
  hsa_amd_memory_pool_t HostPool{};
  llvm::SmallVector<hsa_agent_t, 4> GpuAgents;
  llvm::SmallVector<void *, 4> Allocations;

  char *Buffer = nullptr;
  uint64_t *SubBufferSizes = nullptr;
  uint32_t *HostFlags = nullptr;
  uint32_t *ErrorBits = nullptr;
};

} // namespace luthier::dh_comms

#endif
