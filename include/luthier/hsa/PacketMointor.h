//===-- HsaPacketMonitor.h --------------------------------------*- C++ -*-===//
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
/// Describes the \c PacketMonitor interface and its instance singleton, in
/// charge of monitoring packets submitted to all devices at runtime, modifying
/// them, and providing event handlers to wait on packet's doorbells.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_RUNTIME_PACKET_MONITOR_H
#define LUTHIER_HSA_RUNTIME_PACKET_MONITOR_H
#include "luthier/common/Singleton.h"
#include "luthier/hsa/AqlPacket.h"
#include "luthier/hsa/ExecutableSymbol.h"
#include "luthier/hsa/HsaError.h"
#include "luthier/rocprofiler-sdk/ApiTableSnapshot.h"
#include <hsa/hsa_api_trace.h>
#include <luthier/rocprofiler-sdk/ApiTableWrapperInstaller.h>

namespace luthier::hsa {

class PacketMonitor : public Singleton<PacketMonitor> {
public:
  typedef std::function<void(const hsa_queue_t &, uint64_t,
                             llvm::ArrayRef<AqlPacket>,
                             hsa_amd_queue_intercept_packet_writer)>
      CallbackType;

private:
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot;

  const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtSnapshot;

  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &LoaderApiSnapshot;

  const CallbackType CB;

  std::unique_ptr<
      const rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>
      HsaApiTableInterceptor;

  static ROCPROFILER_HIDDEN_API decltype(hsa_queue_create)
      *UnderlyingHsaQueueCreateFn;

  static ROCPROFILER_HIDDEN_API hsa_status_t hsaQueueCreateWrapper(
      hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
      void (*Callback)(hsa_status_t Status, hsa_queue_t *Source, void *Data),
      void *Data, uint32_t PrivateSegmentSize, uint32_t GroupSegmentSize,
      hsa_queue_t **Queue);

  static ROCPROFILER_HIDDEN_API void
  interceptQueuePacketHandler(const void *Packets, uint64_t PacketCount,
                              uint64_t UserPacketIdx, void *Data,
                              hsa_amd_queue_intercept_packet_writer Writer);

public:
  PacketMonitor(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
      const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtSnapshot,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &LoaderApiSnapshot,
      CallbackType CB, llvm::Error &Err)
      : CoreApiSnapshot(CoreApiSnapshot), AmdExtSnapshot(AmdExtSnapshot),
        LoaderApiSnapshot(LoaderApiSnapshot), CB(std::move(CB)) {
    HsaApiTableInterceptor = std::make_unique<
        rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>(
        Err, std::make_tuple(&::CoreApiTable::hsa_queue_create_fn,
                             std::ref(UnderlyingHsaQueueCreateFn),
                             hsaQueueCreateWrapper));
    if (Err) {
      return;
    }
  };
};

} // namespace luthier::hsa

#endif