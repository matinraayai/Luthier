//===-- KfdPacketMonitorTrait.h ---------------------------------*- C++ -*-===//
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
/// Delivers the application's AQL packets to a tool, from below the runtime.
///
/// \par The KFD counterpart of \c PacketMonitorTrait
/// The HSA trait installs queue interception through the HSA API tables. That is
/// unavailable to a tool attached to an application which drives KFD itself, for
/// the usual reason: such an application holds the DRM virtual address space for
/// its GPUs, only one such VM is permitted per GPU per process, and \c hsa_init
/// therefore fails there. The packets are still observable -- the preloaded
/// wrapper substitutes each queue's ring buffer and copies packets through -- so
/// this trait registers with that instead.
///
/// \par Why the registration is reached by \c dlsym
/// The same reason the allocation tracker is: the chain lives in whichever
/// module intercepted the ioctls, which is \c libluthier-kfd-queue-wrapper.so,
/// preloaded into the application and not something a tool can link against.
/// Linking the wrapper's logic in instead would give the tool a second, empty
/// chain that no packet ever reaches.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_KFD_PACKET_MONITOR_TRAIT_H
#define LUTHIER_KFD_KFD_PACKET_MONITOR_TRAIT_H
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/Singleton.h"
#include "luthier/KFD/QueueWrapper.h"

#include <llvm/Support/Error.h>

#include <dlfcn.h>

namespace luthier {

/// \brief CRTP trait that forwards the application's AQL packets to \p Derived.
///
/// \p Derived must provide
/// <tt>onDispatchPacket(const kfd::QueueInfo &, uint64_t, hsa::AqlPacket &)</tt>.
template <typename Derived> class KfdPacketMonitorTrait {
public:
  /// \param Err receives an error when no wrapper is present in the process.
  /// That is a real failure rather than a quiet no-op: a tool that attaches and
  /// then observes nothing looks exactly like an application that dispatched
  /// nothing, which is the failure mode this whole module is most prone to.
  explicit KfdPacketMonitorTrait(llvm::Error &Err) {
    llvm::ErrorAsOutParameter EAO(&Err);

    auto Add = reinterpret_cast<int (*)(kfd::PacketCallback, void *)>(
        dlsym(RTLD_DEFAULT, "luthierKfdAddPacketCallback"));
    if (Add == nullptr) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(
          "luthierKfdAddPacketCallback was not found in this process, so no "
          "AQL packets can be observed. Preload "
          "libluthier-kfd-queue-wrapper.so, which is what substitutes each "
          "queue's ring buffer and copies packets through.");
      return;
    }
    Remove = reinterpret_cast<void (*)(int)>(
        dlsym(RTLD_DEFAULT, "luthierKfdRemovePacketCallback"));

    Handle = Add(&onPacket, nullptr);
    if (Handle < 0) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(
          "The KFD packet callback chain is full, so this tool could not "
          "attach. Packets would be delivered to the components already "
          "registered and not to this one.");
      return;
    }
    Err = llvm::Error::success();
  }

  ~KfdPacketMonitorTrait() {
    if (Handle >= 0 && Remove != nullptr)
      Remove(Handle);
  }

  KfdPacketMonitorTrait(const KfdPacketMonitorTrait &) = delete;
  KfdPacketMonitorTrait &operator=(const KfdPacketMonitorTrait &) = delete;

  /// \brief The GPU the packet currently being handled was submitted to.
  ///
  /// Exists because the pipeline reaches the tool as
  /// <tt>buildTargetMachineForKD(KD)</tt>, and a kernel descriptor does not say
  /// which device it will run on. On the HSA path that is recovered from the
  /// descriptor's owning agent; below HSA nothing owns it, so the queue is the
  /// only place the device is named. Held per thread rather than per tool
  /// because it is scoped to one callback, not to the tool's lifetime.
  ///
  /// \return the \c gpu_id, or 0 outside a packet callback -- 0 is never a GPU
  /// in KFD's topology, so it cannot be mistaken for one.
  static uint32_t getDispatchGpuId() { return CurrentGpuId; }

private:
  static inline thread_local uint32_t CurrentGpuId{0};

  int Handle{-1};
  void (*Remove)(int){nullptr};

  static void onPacket(const kfd::QueueInfo &Q, uint64_t Index,
                       hsa::AqlPacket &Packet, void *) {
    // Reached through the singleton rather than through the pointer captured at
    // registration, for the two reasons the HSA packet trait does the same
    // (PacketMonitorTrait.h:74). Registration happens inside this trait's
    // constructor, so a dispatch arriving before the tool finishes constructing
    // would otherwise reach a half-built Derived; and withInstance holds the
    // instance alive for the call, so a concurrent teardown cannot destroy the
    // tool underneath a callback that is already running.
    (void)Singleton<Derived>::withInstance([&](Derived &Self) {
      const uint32_t Saved = CurrentGpuId;
      CurrentGpuId = Q.GpuId;
      Self.onDispatchPacket(Q, Index, Packet);
      CurrentGpuId = Saved;
    });
  }
};

} // namespace luthier

#endif // LUTHIER_KFD_KFD_PACKET_MONITOR_TRAIT_H
