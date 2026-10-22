//===-- SharedBuffers.cpp -------------------------------------------------===//
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
/// Implements \c SharedBuffers.
//===----------------------------------------------------------------------===//

#include "luthier/Comms/SharedBuffers.h"

llvm::Expected<std::unique_ptr<luthier::comms::SharedBuffers>>
luthier::comms::SharedBuffers::create(
    const hsa::ApiTableContainer<CoreApiTable> &Core,
    const hsa::ApiTableContainer<AmdExtTable> &AmdExt, Config Cfg) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Cfg.NumSubBuffers != 0 && Cfg.NumSubBuffers <= MaxSubBuffers &&
          (Cfg.NumSubBuffers & (Cfg.NumSubBuffers - 1)) == 0,
      "comms: the number of sub-buffers must be a power of two no larger "
      "than MaxSubBuffers"));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Cfg.SubBufferCapacity >= 4096 && Cfg.SubBufferCapacity % 8 == 0,
      "comms: a sub-buffer must be at least 4 KiB and a multiple of 8 "
      "bytes"));

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