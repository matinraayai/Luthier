//===-- PacketMonitorTrait.h - Tool packet-monitor trait --------*- C++ -*-===//
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
/// Header-only CRTP trait that intercepts \c hsa_queue_create so packets
/// submitted to all queues can be observed and rewritten by the tool. The
/// per-packet callback is no longer a \c std::function passed at
/// construction; it's a method \c Derived must implement:
/// \code
///   void onPackets(const hsa_queue_t &Q, uint64_t UserPacketIdx,
///                  llvm::ArrayRef<hsa::AqlPacket> Packets,
///                  hsa_amd_queue_intercept_packet_writer Writer);
/// \endcode
///
/// Wrappers are installed in the trait constructor and never uninstalled.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_PACKET_MONITOR_TRAIT_H
#define LUTHIER_TOOLING_PACKET_MONITOR_TRAIT_H

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/Singleton.h"
#include "luthier/HSA/AqlPacket.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"
#include <hsa/hsa_api_trace.h>
#include <llvm/ADT/ArrayRef.h>
#include <memory>

namespace luthier {

template <typename Derived> class PacketMonitorTrait {
private:
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot;
  const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtSnapshot;
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &LoaderApiSnapshot;

  std::unique_ptr<rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>
      HsaApiTableInterceptor;

  inline static decltype(hsa_queue_create) *UnderlyingHsaQueueCreateFn{};

  static hsa_status_t
  hsaQueueCreateWrapper(hsa_agent_t Agent, uint32_t Size,
                        hsa_queue_type32_t Type,
                        void (*Callback)(hsa_status_t, hsa_queue_t *, void *),
                        void *Data, uint32_t PrivateSegmentSize,
                        uint32_t GroupSegmentSize, hsa_queue_t **Queue) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        UnderlyingHsaQueueCreateFn != nullptr,
        "The underlying hsa_queue_create function for "
        "PacketMonitorTrait is nullptr"));
    hsa_status_t Out =
        UnderlyingHsaQueueCreateFn(Agent, Size, Type, Callback, Data,
                                   PrivateSegmentSize, GroupSegmentSize, Queue);
    if (Out != HSA_STATUS_SUCCESS)
      return Out;

    (void)Singleton<Derived>::withInstance([&](Derived &Self) {
      auto &Trait = static_cast<PacketMonitorTrait<Derived> &>(Self);

      /// Try to install an event handler on the newly-created queue.
      const hsa_status_t EventHandlerStatus =
          Trait.AmdExtSnapshot.getTable()
              .template callFunction<hsa_amd_queue_intercept_register>(
                  *Queue, interceptQueuePacketHandler, *Queue);
      /// If we failed to install an event handler, the queue was a normal
      /// queue; destroy it and recreate an intercept queue in its place.
      if (EventHandlerStatus == HSA_STATUS_ERROR_INVALID_QUEUE) {
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            Trait.CoreApiSnapshot.getTable()
                .template callFunction<hsa_queue_destroy>(*Queue),
            "Failed to destroy the application's queue"));
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            Trait.AmdExtSnapshot.getTable()
                .template callFunction<hsa_amd_queue_intercept_create>(
                    Agent, Size, Type, Callback, Data, PrivateSegmentSize,
                    GroupSegmentSize, Queue),
            "Failed to create an intercept queue"));
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            Trait.AmdExtSnapshot.getTable()
                .template callFunction<hsa_amd_queue_intercept_register>(
                    *Queue, interceptQueuePacketHandler, *Queue),
            "Failed to assign a packet handler to the intercept queue"));
      } else {
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            EventHandlerStatus,
            "Failed to install HSA queue intercept handler"));
      }
    });
    return Out;
  }

  static void
  interceptQueuePacketHandler(const void *Packets, uint64_t PacketCount,
                              uint64_t UserPacketIdx, void *Data,
                              hsa_amd_queue_intercept_packet_writer Writer) {
    bool Handled = Singleton<Derived>::withInstance([&](Derived &Self) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          Data != nullptr,
          "Failed to get the queue used to dispatch packets."));
      auto &Queue = *static_cast<hsa_queue_t *>(Data);

      Self.onPackets(
          Queue, UserPacketIdx,
          llvm::ArrayRef(static_cast<const hsa::AqlPacket *>(Packets),
                         PacketCount),
          Writer);
    });
    if (!Handled)
      Writer(Packets, PacketCount);
  }

public:
  PacketMonitorTrait(
      const DynamicLibrary &RocprofilerLib,
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
      const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &Loader,
      llvm::Error &Err)
      : CoreApiSnapshot(CoreApi), AmdExtSnapshot(AmdExt),
        LoaderApiSnapshot(Loader) {
    llvm::ErrorAsOutParameter EAO(Err);
    HsaApiTableInterceptor = std::make_unique<
        rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>(
        RocprofilerLib, Err,
        std::make_tuple(&::CoreApiTable::hsa_queue_create_fn,
                        std::ref(UnderlyingHsaQueueCreateFn),
                        hsaQueueCreateWrapper));
  }

  /// Wrappers are intentionally NOT uninstalled. See file header.
  ~PacketMonitorTrait() = default;

  PacketMonitorTrait(const PacketMonitorTrait &) = delete;
  PacketMonitorTrait &operator=(const PacketMonitorTrait &) = delete;
};

} // namespace luthier

#endif // LUTHIER_TOOLING_PACKET_MONITOR_TRAIT_H
