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
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <luthier/Common/Singleton.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/ExecutableSymbol.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/Rocprofiler/HSAApiTable.h>

namespace luthier::hsa {

class PacketMonitor {
protected:
  const hsa::ApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot;

  const hsa::ExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &LoaderApiSnapshot;

  std::unique_ptr<hsa_ven_amd_loader_1_03_pfn_t> LoaderTable{nullptr};

  PacketMonitor(const hsa::ApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
                const hsa::ExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
                    &LoaderApiSnapshot)
      : CoreApiSnapshot(CoreApiSnapshot),
        LoaderApiSnapshot(LoaderApiSnapshot) {};

public:
  /// \return the executable, the loaded code object, and the executable
  /// symbol associated with the \p KernelObject
  [[nodiscard]] llvm::Expected<std::tuple<
      hsa_executable_t, hsa_loaded_code_object_t, hsa_executable_symbol_t>>
  getKernelObjectDefinition(uint64_t KernelObject) const;

  template <typename... Args>
  llvm::Expected<hsa_kernel_dispatch_packet_t>
  overrideLaunchParameters(const hsa_kernel_dispatch_packet_t &OriginalPacket,
                           hsa_executable_symbol_t NewKernel) {
    // hsa_kernel_dispatch_packet_t OutPacket = OriginalPacket;
    //
    // /// Override the kernel
    // llvm::Expected<uint64_t> NewKDOrErr = hsa::executableSymbolGetAddress(
    //     NewKernel,
    //     CoreApiSnapshot
    //         .getFunction<&::CoreApiTable::hsa_executable_symbol_get_info_fn>());
    // LUTHIER_RETURN_ON_ERROR(NewKDOrErr.takeError());
    // OutPacket.kernel_object = *NewKDOrErr;
    //
    // auto InstrumentedKernelMD = InstrumentedKernel->getKernelMetadata();
    //
    // Packet.private_segment_size = InstrumentedKernelMD.PrivateSegmentFixedSize;
    //
    // return OutPacket;
  }
};

template <size_t Idx>
class ROCPROFILER_HIDDEN_API PacketMonitorInstance
    : public PacketMonitor,
      public Singleton<PacketMonitorInstance<Idx>> {
private:
  typedef std::function<void(const hsa_queue_t &, uint64_t,
                             llvm::ArrayRef<hsa::AqlPacket>,
                             hsa_amd_queue_intercept_packet_writer)>
      CallbackType;

  CallbackType CB;

  const std::unique_ptr<ApiTableWrapperInstaller> HsaApiTableInterceptor;

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

  PacketMonitorInstance(
      const hsa::ApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
      const hsa::ExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &LoaderApiSnapshot,
      CallbackType CB, llvm::Error &Err)
      : PacketMonitor(CoreApiSnapshot, LoaderApiSnapshot), CB(std::move(CB)),
        HsaApiTableInterceptor([&] -> decltype(HsaApiTableInterceptor) {
          auto WrapperInstallerOrErr =
              hsa::ApiTableWrapperInstaller::requestWrapperInstallation(
                  {&::CoreApiTable::hsa_queue_create_fn,
                   UnderlyingHsaQueueCreateFn, hsaQueueCreateWrapper});
          Err = WrapperInstallerOrErr.takeError();
          if (Err) {
            Err = std::move(Err);
            return nullptr;
          }
          return std::move(*WrapperInstallerOrErr);
        }()) {};

public:
  static llvm::Expected<std::unique_ptr<PacketMonitorInstance>>
  create(const hsa::ApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
         const hsa::ExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
             &LoaderApiSnapshot,
         const CallbackType &CB) {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<PacketMonitorInstance>(
        CoreApiSnapshot, LoaderApiSnapshot, CB, Err);
    if (Err)
      return std::move(Err);

    return Out;
  }
};

template <size_t Idx>
decltype(hsa_queue_create)
    *PacketMonitorInstance<Idx>::UnderlyingHsaQueueCreateFn = nullptr;

template <size_t Idx>
hsa_status_t PacketMonitorInstance<Idx>::hsaQueueCreateWrapper(
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
  if (!PacketMonitorInstance::isInitialized() || Out != HSA_STATUS_SUCCESS) {
    return Out;
  }
  auto &PacketMonitor = PacketMonitorInstance::instance();

  /// Try to install an event handler on the newly created queue
  const hsa_status_t EventHandlerStatus =
      PacketMonitor.HsaApiTable.template getFunction<
          &::AmdExtTable::hsa_amd_queue_intercept_register_fn>()(
          *Queue, interceptQueuePacketHandler, *Queue);
  /// If we fail to install an event handler, the queue was
  /// a normal queue; Destroy it, and recreate an intercept queue in its place
  if (EventHandlerStatus == HSA_STATUS_ERROR_INVALID_QUEUE) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        PacketMonitor.HsaApiTable
            .template getFunction<&::CoreApiTable::hsa_queue_destroy_fn>()(
                *Queue)));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        PacketMonitor.template getFunction<
            &::AmdExtTable::hsa_amd_queue_intercept_create_fn>()(
            Agent, Size, Type, Callback, Data, PrivateSegmentSize,
            GroupSegmentSize, Queue)));
    LUTHIER_REPORT_FATAL_ON_ERROR(
        PacketMonitor.template getFunction<
            &::AmdExtTable::hsa_amd_queue_intercept_register_fn>()(
            *Queue, interceptQueuePacketHandler, *Queue));
  } else {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        EventHandlerStatus, "Failed to install HSA queue intercept handler to "
                            "monitor its packets"));
  }
  return Out;
}

template <size_t Idx>
void PacketMonitorInstance<Idx>::interceptQueuePacketHandler(
    const void *Packets, uint64_t PacketCount, uint64_t UserPacketIdx,
    void *Data, hsa_amd_queue_intercept_packet_writer Writer) {
  // Call the writer directly and return if the packet monitor is not
  // initialized
  if (!PacketMonitorInstance::isInitialized()) {
    Writer(Packets, PacketCount);
    return;
  }
  auto &PacketMonitor = PacketMonitorInstance::instance();
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Data != nullptr, "Failed to get the queue used to dispatch packets."));
  auto &Queue = *static_cast<hsa_queue_t *>(Data);

  PacketMonitor.CB(
      Queue, UserPacketIdx,
      llvm::ArrayRef(static_cast<const AqlPacket *>(Packets), PacketCount),
      Writer);
}

#define LUTHIER_CREATE_NEW_HSA_PACKET_MONITOR_INSTANCE(...)                    \
  luthier::hsa::PacketMonitorInstance<__COUNTER__>::create(__VA_ARGS__)

} // namespace luthier::hsa

#endif
