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
/// Describes the \c HsaPacketMonitor singleton, in charge of monitoring
/// packets submitted to all devices at runtime, modifying them, and providing
/// event handlers to wait on packet's doorbells.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_RUNTIME_HSA_PACKET_MONITOR_H
#define LUTHIER_RUNTIME_HSA_PACKET_MONITOR_H
#include "HsaError.h"

#include <hsa/hsa_api_trace.h>
#include <luthier/common/Singleton.h>
#include <luthier/hsa/AqlPacket.h>
#include <luthier/hsa/HsaApiTableInterceptor.h>

namespace luthier::hsa {

template <size_t Idx>
class ROCPROFILER_HIDDEN_API HsaPacketMonitorInstance
    : public Singleton<HsaPacketMonitorInstance<Idx>> {
private:
  std::function<void(const hsa_queue_t &, uint64_t,
                     llvm::ArrayRef<hsa::AqlPacket>,
                     hsa_amd_queue_intercept_packet_writer)>
      CB;

  const std::unique_ptr<HsaApiTableInterceptor> HsaApiTableInterceptor;

  static ROCPROFILER_HIDDEN_API decltype(hsa_queue_create)
      *UnderlyingHsaQueueCreateFn;

  decltype(hsa_amd_queue_intercept_create)
      *UnderlyingHsaAmdQueueInterceptCreateFn = nullptr;

  decltype(hsa_amd_queue_intercept_register)
      *UnderlyingHsaAmdQueueInterceptRegisterFn = nullptr;

  decltype(hsa_queue_destroy) *UnderlyingHsaQueueDestroyFn = nullptr;

  static ROCPROFILER_HIDDEN_API hsa_status_t hsaQueueCreateWrapper(
      hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
      void (*Callback)(hsa_status_t Status, hsa_queue_t *Source, void *Data),
      void *Data, uint32_t PrivateSegmentSize, uint32_t GroupSegmentSize,
      hsa_queue_t **Queue);

  static ROCPROFILER_HIDDEN_API void
  InterceptQueuePacketHandler(const void *Packets, uint64_t PacketCount,
                              uint64_t UserPacketIdx, void *Data,
                              hsa_amd_queue_intercept_packet_writer Writer);

public:
  static llvm::Expected<std::unique_ptr<HsaPacketMonitorInstance>> create() {
    auto Out = std::make_unique<HsaPacketMonitorInstance>();

    auto HsaApiTableInterceptorOrErr =
        hsa::HsaApiTableInterceptor::requestApiTable(
            [PacketMonitor = Out.get()](::HsaApiTable &Table) {
              /// Save the needed underlying function
              UnderlyingHsaQueueCreateFn = Table.core_->hsa_queue_create_fn;
              PacketMonitor->UnderlyingHsaAmdQueueInterceptCreateFn =
                  Table.amd_ext_->hsa_amd_queue_intercept_create_fn;
              PacketMonitor->UnderlyingHsaAmdQueueInterceptRegisterFn =
                  Table.amd_ext_->hsa_amd_queue_intercept_register_fn;
              PacketMonitor->UnderlyingHsaQueueDestroyFn =
                  Table.core_->hsa_queue_destroy_fn;
              /// Install wrappers
              Table.core_->hsa_queue_create_fn = hsaQueueCreateWrapper;
            });
    LUTHIER_RETURN_ON_ERROR(HsaApiTableInterceptorOrErr.takeError());

    Out->HsaApiTableInterceptor = std::move(*HsaApiTableInterceptorOrErr);
    return Out;
  }
};

template <size_t Idx>
decltype(hsa_queue_create)
    *HsaPacketMonitorInstance<Idx>::UnderlyingHsaQueueCreateFn = nullptr;

template <size_t Idx>
hsa_status_t HsaPacketMonitorInstance<Idx>::hsaQueueCreateWrapper(
    hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
    void (*Callback)(hsa_status_t, hsa_queue_t *, void *), void *Data,
    uint32_t PrivateSegmentSize, uint32_t GroupSegmentSize,
    hsa_queue_t **Queue) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHsaQueueCreateFn != nullptr,
      llvm::formatv("The underlying hsa_queue_create function for "
                    "HsaPacketMonitorInstance {0} is nullptr",
                    Idx)));
  /// Allow the application to create its queue
  hsa_status_t Out =
      UnderlyingHsaQueueCreateFn(Agent, Size, Type, Callback, Data,
                                 PrivateSegmentSize, GroupSegmentSize, Queue);
  /// If the packet monitor is not initialized or if the queue creation
  /// encountered an issue, then return right away
  if (!HsaPacketMonitorInstance::isInitialized() || Out != HSA_STATUS_SUCCESS) {
    return Out;
  }
  auto &PacketMonitor = HsaPacketMonitorInstance::instance();
  /// Make sure the intercept queue creation and event handler register
  /// functions are not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      PacketMonitor.UnderlyingHsaAmdQueueInterceptCreateFn != nullptr,
      llvm::formatv("The underlying hsa_amd_queue_intercept_create "
                    "function of hsaPacketMonitorInstance {0} is nullptr",
                    Idx)));
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      PacketMonitor.UnderlyingHsaAmdQueueInterceptRegister != nullptr,
      llvm::formatv("The underlying hsa_amd_queue_intercept_register "
                    "function of hsaPacketMonitorInstance {0} is nullptr",
                    Idx)));

  /// Try to install an event handler on the newly created queue
  hsa_status_t EventHandlerStatus =
      PacketMonitor.UnderlyingHsaAmdQueueInterceptRegister(
          *Queue, InterceptQueuePacketHandler, *Queue);
  /// If we fail to install an event handler, the queue was
  /// a normal queue; Destroy it, and recreate an intercept queue in its place
  if (EventHandlerStatus == HSA_STATUS_ERROR_INVALID_QUEUE) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        PacketMonitor.UnderlyingHsaQueueDestroyFn(*Queue)));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        PacketMonitor.UnderlyingHsaAmdQueueInterceptCreateFn(
            Agent, Size, Type, Callback, Data, PrivateSegmentSize,
            GroupSegmentSize, Queue)));
    LUTHIER_REPORT_FATAL_ON_ERROR(
        PacketMonitor.UnderlyingHsaAmdQueueInterceptRegister(
            *Queue, InterceptQueuePacketHandler, *Queue));
  } else {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        EventHandlerStatus, "Failed to install HSA queue intercept handler to "
                            "monitor its packets"));
  }
  return Out;
}

template <size_t Idx>
void HsaPacketMonitorInstance<Idx>::InterceptQueuePacketHandler(
    const void *Packets, uint64_t PacketCount, uint64_t UserPacketIdx,
    void *Data, hsa_amd_queue_intercept_packet_writer Writer) {
  // Call the writer directly and return if the packet monitor is not
  // initialized
  if (!HsaPacketMonitorInstance::isInitialized()) {
    Writer(Packets, PacketCount);
    return;
  }
  auto &PacketMonitor = HsaPacketMonitorInstance::instance();
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Data != nullptr, "Failed to get the queue used to dispatch packets."));
  auto &Queue = *static_cast<hsa_queue_t *>(Data);

  PacketMonitor.CB(
      Queue, UserPacketIdx,
      llvm::ArrayRef(static_cast<const AqlPacket *>(Packets), PacketCount),
      Writer);
}

} // namespace luthier::hsa

#endif
