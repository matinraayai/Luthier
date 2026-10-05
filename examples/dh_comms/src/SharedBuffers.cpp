//===-- SharedBuffers.cpp - Pinned host buffers the GPU writes into -------===//
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
#include "dh_comms/SharedBuffers.h"

#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/HSA/Agent.h>
#include <luthier/HSA/MemoryPool.h>

#include <cstring>

namespace luthier::dh_comms {

llvm::Expected<std::unique_ptr<SharedBuffers>>
SharedBuffers::create(const hsa::ApiTableContainer<::CoreApiTable> &Core,
                      const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                      Config Cfg) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Cfg.NumSubBuffers != 0 && Cfg.NumSubBuffers <= MaxSubBuffers &&
          (Cfg.NumSubBuffers & (Cfg.NumSubBuffers - 1)) == 0,
      "dh_comms: the number of sub-buffers must be a power of two no larger "
      "than MaxSubBuffers"));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Cfg.SubBufferCapacity >= sizeof(WaveHeader) + 64 * sizeof(AccessRecord),
      "dh_comms: a sub-buffer must hold at least one full-wave message"));

  std::unique_ptr<SharedBuffers> SB(new SharedBuffers(AmdExt, Cfg));

  // Split agents into the CPU that owns the memory and the GPUs that write it.
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
      "dh_comms: need a CPU agent and at least one GPU agent"));

  // Fine-grained == coherent: the hipHostMallocCoherent equivalent.
  auto PoolOrErr = hsa::agentFindFineGrainedPool(AmdExt, *Cpu);
  LUTHIER_RETURN_ON_ERROR(PoolOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      PoolOrErr->has_value(),
      "dh_comms: the CPU agent exposes no fine-grained memory pool"));
  SB->HostPool = **PoolOrErr;

  const uint64_t N = Cfg.NumSubBuffers;
  auto Take = [&](size_t Bytes, auto *&Out) -> llvm::Error {
    auto PtrOrErr = SB->allocate(Bytes);
    LUTHIER_RETURN_ON_ERROR(PtrOrErr.takeError());
    Out = static_cast<std::remove_reference_t<decltype(Out)>>(*PtrOrErr);
    return llvm::Error::success();
  };
  LUTHIER_RETURN_ON_ERROR(Take(N * Cfg.SubBufferCapacity, SB->Buffer));
  LUTHIER_RETURN_ON_ERROR(Take(N * sizeof(uint64_t), SB->SubBufferSizes));
  LUTHIER_RETURN_ON_ERROR(Take(N * sizeof(uint32_t), SB->HostFlags));
  LUTHIER_RETURN_ON_ERROR(Take(sizeof(uint32_t), SB->ErrorBits));
  return SB;
}

llvm::Expected<void *> SharedBuffers::allocate(size_t Bytes) {
  auto PtrOrErr = hsa::memoryPoolAllocate(AmdExt, HostPool, Bytes);
  LUTHIER_RETURN_ON_ERROR(PtrOrErr.takeError());
  void *Ptr = *PtrOrErr;
  Allocations.push_back(Ptr);
  LUTHIER_RETURN_ON_ERROR(hsa::agentsAllowAccess(AmdExt, GpuAgents, Ptr));
  std::memset(Ptr, 0, Bytes); // host memory: the CPU can touch it directly
  return Ptr;
}

SharedBuffers::~SharedBuffers() {
  for (void *Ptr : Allocations)
    llvm::consumeError(hsa::memoryPoolFree(AmdExt, Ptr));
}

void SharedBuffers::reset() {
  std::memset(SubBufferSizes, 0, Cfg.NumSubBuffers * sizeof(uint64_t));
  std::memset(HostFlags, 0, Cfg.NumSubBuffers * sizeof(uint32_t));
  *ErrorBits = 0;
}

void SharedBuffers::describe(DeviceDescriptor &D) const {
  D.NumSubBuffers = Cfg.NumSubBuffers;
  D.SubBufferCapacity = Cfg.SubBufferCapacity;
  D.Buffer = Buffer;
  D.SubBufferSizes = SubBufferSizes;
  D.HostFlags = HostFlags;
  D.ErrorBits = ErrorBits;
}

} // namespace luthier::dh_comms
