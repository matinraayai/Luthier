//===-- HsaPacketMonitor.cpp ----------------------------------------------===//
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
/// Implements the \c PacketMonitor interface.
//===----------------------------------------------------------------------===//
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/GenericLuthierError.h"
#include "luthier/hsa/PacketMointor.h"

namespace luthier {

template <>
hsa::PacketMonitor *Singleton<hsa::PacketMonitor>::Instance{nullptr};

namespace hsa {
decltype(hsa_queue_create) *PacketMonitor::UnderlyingHsaQueueCreateFn = nullptr;

hsa_status_t PacketMonitor::hsaQueueCreateWrapper(
    hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
    void (*Callback)(hsa_status_t, hsa_queue_t *, void *), void *Data,
    uint32_t PrivateSegmentSize, uint32_t GroupSegmentSize,
    hsa_queue_t **Queue) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHsaQueueCreateFn != nullptr,
      "The underlying hsa_queue_create function for "
      "PacketMonitor is nullptr"));
  /// Allow the application to create its queue
  hsa_status_t Out =
      UnderlyingHsaQueueCreateFn(Agent, Size, Type, Callback, Data,
                                 PrivateSegmentSize, GroupSegmentSize, Queue);
  /// If the packet monitor is not initialized or if the queue creation
  /// encountered an issue, then return right away
  if (!isInitialized() || Out != HSA_STATUS_SUCCESS) {
    return Out;
  }
  auto &PacketMonitor = instance();

  /// Try to install an event handler on the newly created queue
  const hsa_status_t EventHandlerStatus =
      PacketMonitor.AmdExtSnapshot.getTable()
          .callFunction<hsa_amd_queue_intercept_register>(
              *Queue, interceptQueuePacketHandler, *Queue);
  /// If we fail to install an event handler, the queue was
  /// a normal queue; Destroy it, and recreate an intercept queue in its place
  if (EventHandlerStatus == HSA_STATUS_ERROR_INVALID_QUEUE) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        PacketMonitor.CoreApiSnapshot.getTable()
            .callFunction<hsa_queue_destroy>(*Queue),
        "Failed to destroy the application's queue"));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        PacketMonitor.AmdExtSnapshot.getTable()
            .callFunction<hsa_amd_queue_intercept_create>(
                Agent, Size, Type, Callback, Data, PrivateSegmentSize,
                GroupSegmentSize, Queue),
        "Failed to create an intercept queue"));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        PacketMonitor.AmdExtSnapshot.getTable()
            .callFunction<hsa_amd_queue_intercept_register>(
                *Queue, interceptQueuePacketHandler, *Queue),
        "Failed to assign a packet handler to the intercept queue"));
  } else {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        EventHandlerStatus, "Failed to install HSA queue intercept handler to "
                            "monitor its packets"));
  }
  return Out;
}

void PacketMonitor::interceptQueuePacketHandler(
    const void *Packets, uint64_t PacketCount, uint64_t UserPacketIdx,
    void *Data, hsa_amd_queue_intercept_packet_writer Writer) {
  // Call the writer directly and return if the packet monitor is not
  // initialized
  if (!isInitialized()) {
    Writer(Packets, PacketCount);
    return;
  }
  auto &PacketMonitor = instance();
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Data != nullptr, "Failed to get the queue used to dispatch packets."));
  auto &Queue = *static_cast<hsa_queue_t *>(Data);

  PacketMonitor.CB(
      Queue, UserPacketIdx,
      llvm::ArrayRef(static_cast<const AqlPacket *>(Packets), PacketCount),
      Writer);
}
} // namespace hsa

} // namespace luthier