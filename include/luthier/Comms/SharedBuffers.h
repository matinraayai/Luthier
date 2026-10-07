//===-- SharedBuffers.h - Host memory the device writes into ----*- C++ -*-===//
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
/// Owns the memory a \c ChannelDescriptor points at: the message buffer,
/// per-sub-buffer sizes, host hand-off flags and error bits.
///
/// Allocated the way <tt>hipHostMalloc(..., hipHostMallocCoherent)</tt> does
/// it underneath, since a Luthier tool talks to HSA rather than HIP: from the
/// CPU agent's fine-grained pool (coherent with the GPUs, no copies), with
/// every GPU agent granted access.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_SHARED_BUFFERS_H
#define LUTHIER_COMMS_SHARED_BUFFERS_H

#include "luthier/Comms/Protocol.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/HSA/Agent.h>
#include <luthier/HSA/ApiTable.h>
#include <luthier/HSA/MemoryPool.h>

#include <cstring>
#include <memory>
#include <optional>

namespace luthier::comms {

class SharedBuffers {
public:
  struct Config {
    uint32_t NumSubBuffers;     ///< Power of two, <= MaxSubBuffers.
    uint64_t SubBufferCapacity; ///< Bytes per sub-buffer.
  };

  /// Allocates every shared region and makes it accessible to all GPUs.
  static llvm::Expected<std::unique_ptr<SharedBuffers>>
  create(const hsa::ApiTableContainer<::CoreApiTable> &Core,
         const hsa::ApiTableContainer<::AmdExtTable> &AmdExt, Config Cfg) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        Cfg.NumSubBuffers != 0 && Cfg.NumSubBuffers <= MaxSubBuffers &&
            (Cfg.NumSubBuffers & (Cfg.NumSubBuffers - 1)) == 0,
        "comms: the number of sub-buffers must be a power of two no larger "
        "than MaxSubBuffers"));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        Cfg.SubBufferCapacity >= sizeof(WaveHeader) + 64 * 4 * 4,
        "comms: a sub-buffer must hold at least one full-wave message"));

    std::unique_ptr<SharedBuffers> SB(new SharedBuffers(AmdExt, Cfg));
    llvm::SmallVector<hsa_agent_t, 8> Agents;
    LUTHIER_RETURN_ON_ERROR(hsa::getAllAgents(Core, Agents));
    std::optional<hsa_agent_t> Cpu;
    for (hsa_agent_t Agent : Agents) {
      auto TypeOrErr = hsa::agentGetDeviceType(Core, Agent);
      LUTHIER_RETURN_ON_ERROR(TypeOrErr.takeError());
      if (*TypeOrErr == HSA_DEVICE_TYPE_GPU)
        SB->GpuAgents.push_back(Agent);
      else if (*TypeOrErr == HSA_DEVICE_TYPE_CPU && !Cpu)
        Cpu = Agent;
    }
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        Cpu.has_value() && !SB->GpuAgents.empty(),
        "comms: need a CPU agent and at least one GPU agent"));

    auto PoolOrErr = hsa::agentFindFineGrainedPool(AmdExt, *Cpu);
    LUTHIER_RETURN_ON_ERROR(PoolOrErr.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        PoolOrErr->has_value(),
        "comms: the CPU agent exposes no fine-grained memory pool"));
    SB->HostPool = **PoolOrErr;

    const uint64_t N = Cfg.NumSubBuffers;
    LUTHIER_RETURN_ON_ERROR(SB->allocate(N * Cfg.SubBufferCapacity, SB->Buffer));
    LUTHIER_RETURN_ON_ERROR(
        SB->allocate(N * sizeof(uint64_t), SB->SubBufferSizes));
    LUTHIER_RETURN_ON_ERROR(SB->allocate(N * sizeof(uint32_t), SB->HostFlags));
    LUTHIER_RETURN_ON_ERROR(SB->allocate(sizeof(uint32_t), SB->ErrorBits));
    return SB;
  }

  ~SharedBuffers() {
    for (void *Ptr : Allocations)
      llvm::consumeError(hsa::memoryPoolFree(AmdExt, Ptr));
  }
  SharedBuffers(const SharedBuffers &) = delete;
  SharedBuffers &operator=(const SharedBuffers &) = delete;

  /// Empties every sub-buffer and clears flags and error bits. Only valid
  /// while no kernel is writing to the buffers.
  void reset() {
    std::memset(SubBufferSizes, 0, Cfg.NumSubBuffers * sizeof(uint64_t));
    std::memset(HostFlags, 0, Cfg.NumSubBuffers * sizeof(uint32_t));
    *ErrorBits = 0;
  }

  /// The descriptor the device writes through.
  ChannelDescriptor describe() const {
    return {Cfg.NumSubBuffers, Cfg.SubBufferCapacity, Buffer,
            SubBufferSizes,    HostFlags,             ErrorBits};
  }

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

  /// One zero-filled allocation from the fine-grained host pool.
  template <typename T> llvm::Error allocate(size_t Bytes, T *&Out) {
    auto PtrOrErr = hsa::memoryPoolAllocate(AmdExt, HostPool, Bytes);
    LUTHIER_RETURN_ON_ERROR(PtrOrErr.takeError());
    Allocations.push_back(*PtrOrErr);
    LUTHIER_RETURN_ON_ERROR(
        hsa::agentsAllowAccess(AmdExt, GpuAgents, *PtrOrErr));
    std::memset(*PtrOrErr, 0, Bytes); // host memory: the CPU writes it directly
    Out = static_cast<T *>(*PtrOrErr);
    return llvm::Error::success();
  }

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

} // namespace luthier::comms

#endif
