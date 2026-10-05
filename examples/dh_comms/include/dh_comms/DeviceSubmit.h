//===-- DeviceSubmit.h - GPU side of dh_comms -------------------*- C++ -*-===//
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
/// Device code that runs inside the instrumented kernel: turns an address
/// into a memory index and submits it to the host. A port of dh_comms'
/// \c generic_submit_message (dh_comms_dev.h); the protocol is unchanged:
///
///   1. pick a sub-buffer from the workgroup id
///   2. take that sub-buffer's lock (serialises waves)
///   3. if the message does not fit, hand the sub-buffer to the host and
///      wait for it to drain
///   4. write the wave header and every active lane's record
///   5. publish the new size and release the lock
///
/// What changes is how the waiting is written. This code runs as a Luthier
/// injected payload, where a spin loop executed by a divergent subset of
/// lanes deadlocks: the structurizer retires the winning lane from EXEC
/// before it reaches the release. dh_comms guards its spins with
/// `if (lane == 0) while (...)`, which is exactly that shape. Here every spin
/// loop is entered by all active lanes and exits on a wave-uniform condition
/// (broadcast with readfirstlane), so it compiles to a scalar branch and no
/// lane is ever peeled away. Only the side-effecting atomic inside the loop
/// is guarded to a single lane, and it is straight-line code.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXAMPLES_DH_COMMS_DEVICE_SUBMIT_H
#define LUTHIER_EXAMPLES_DH_COMMS_DEVICE_SUBMIT_H

#include "dh_comms/Protocol.h"

#include <hip/hip_runtime.h>
#include <luthier/Intrinsic/Intrinsics.h>

#define DH_COMMS_DEVICE __attribute__((device, always_inline)) inline

namespace luthier::dh_comms::device {

//===----------------------------------------------------------------------===//
// Wave helpers
//===----------------------------------------------------------------------===//

/// Rank of this lane among the active lanes (0 for the first active lane).
DH_COMMS_DEVICE uint32_t activeLaneRank(uint64_t Exec) {
  return __builtin_amdgcn_mbcnt_hi(
      uint32_t(Exec >> 32), __builtin_amdgcn_mbcnt_lo(uint32_t(Exec), 0u));
}

/// Broadcasts the first active lane's value to the whole wave. The result is
/// wave-uniform, so branching on it is a scalar branch.
DH_COMMS_DEVICE uint32_t broadcast(uint32_t V) {
  return uint32_t(__builtin_amdgcn_readfirstlane(int(V)));
}

DH_COMMS_DEVICE uint8_t currentArch() {
#if defined(__gfx906__)
  return gcn_arch::Gfx906;
#elif defined(__gfx908__)
  return gcn_arch::Gfx908;
#elif defined(__gfx90a__)
  return gcn_arch::Gfx90a;
#elif defined(__gfx940__)
  return gcn_arch::Gfx940;
#elif defined(__gfx941__)
  return gcn_arch::Gfx941;
#elif defined(__gfx942__)
  return gcn_arch::Gfx942;
#else
  return gcn_arch::Unsupported;
#endif
}

//===----------------------------------------------------------------------===//
// Memory indexing
//===----------------------------------------------------------------------===//

/// Maps an address to (buffer, element index). Never sends the address
/// itself: the host only ever sees where in which buffer the access landed.
DH_COMMS_DEVICE AccessRecord resolveIndex(const DeviceDescriptor &D,
                                          uint64_t Address) {
  AccessRecord Rec{0u, UnresolvedBuffer};
  // Uniform trip count and no early exit; buffers never overlap, so at most
  // one of them matches.
  for (uint32_t I = 0; I < D.NumTrackedBuffers; ++I) {
    const uint64_t Base = D.BufferBase[I];
    if (Address >= Base && Address < D.BufferEnd[I]) {
      Rec.ElementIndex = uint32_t((Address - Base) >> D.ElementSizeLog2);
      Rec.BufferId = D.BufferArgSlot[I];
    }
  }
  return Rec;
}

//===----------------------------------------------------------------------===//
// Sub-buffer ownership
//===----------------------------------------------------------------------===//

/// Sub-buffer for this wave. dh_comms uses the flattened workgroup id; all it
/// needs is a wave-uniform value that spreads waves across sub-buffers, since
/// the lock makes any choice correct. Reading the workgroup id inside a
/// Luthier payload requires \c luthier::readSVA, which this Luthier build
/// leaves unlowered in hook code (see README), so the wave hashes the page of
/// its first active lane's address instead.
DH_COMMS_DEVICE uint32_t pickSubBuffer(const DeviceDescriptor &D,
                                       uint64_t Address) {
  const uint32_t Page = broadcast(uint32_t(Address >> 12));
  return (Page * 2654435761u >> 8) & uint32_t(D.NumSubBuffers - 1);
}

/// Takes the sub-buffer's lock on behalf of the whole wave (dh_comms'
/// \c wave_acquire). Exits only once the leader holds the lock.
DH_COMMS_DEVICE void acquire(uint32_t *Lock, bool IsLeader) {
  while (true) {
    uint32_t Won = 0;
    if (IsLeader) {
      uint32_t Expected = 0;
      Won = __hip_atomic_compare_exchange_strong(
          Lock, &Expected, 1u, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
    }
    if (broadcast(Won))
      return;
    __builtin_amdgcn_s_sleep(1);
  }
}

/// Releases the lock (dh_comms' \c wave_release). An exchange rather than a
/// plain store: dh_comms found the compiler could drop a store-release.
DH_COMMS_DEVICE void release(uint32_t *Lock, bool IsLeader) {
  if (IsLeader)
    (void)__hip_atomic_exchange(Lock, 0u, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_AGENT);
}

/// The sub-buffer is full: give it to the host and wait until the host has
/// drained it and cleared the flag (dh_comms' \c wave_signal_host). The lock
/// is still held, so no other wave touches the sub-buffer meanwhile.
DH_COMMS_DEVICE void handOffToHost(uint32_t *HostFlag, bool IsLeader) {
  if (IsLeader)
    __hip_atomic_store(HostFlag, 1u, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  while (true) {
    uint32_t Pending = 0;
    if (IsLeader)
      Pending = __hip_atomic_load(HostFlag, __ATOMIC_ACQUIRE,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
    if (broadcast(Pending) == 0)
      return;
    __builtin_amdgcn_s_sleep(2);
  }
}

//===----------------------------------------------------------------------===//
// Submission
//===----------------------------------------------------------------------===//

/// Submits one memory access from every active lane: a wave header plus one
/// \c AccessRecord per active lane.
DH_COMMS_DEVICE void submitMemoryIndex(const DeviceDescriptor &D,
                                       uint32_t *Locks, uint64_t Address,
                                       uint32_t InstrId, uint32_t AccessInfo) {
  // Read the clock first so the submission itself barely perturbs it.
  const uint64_t Timestamp = __builtin_readcyclecounter();
  if (D.NumSubBuffers == 0)
    return; // descriptor not configured for this dispatch

  const AccessRecord Rec = resolveIndex(D, Address);

  const uint64_t Exec = __builtin_amdgcn_read_exec();
  const uint32_t ActiveLanes = uint32_t(__builtin_popcountll(Exec));
  const uint32_t Rank = activeLaneRank(Exec);
  const bool IsLeader = Rank == 0;

  const uint32_t DataSize = ActiveLanes * uint32_t(sizeof(AccessRecord));
  const uint32_t MessageSize = uint32_t(sizeof(WaveHeader)) + DataSize;
  const uint32_t Capacity = uint32_t(D.SubBufferCapacity);
  const uint32_t Sb = pickSubBuffer(D, Address);

  acquire(&Locks[Sb], IsLeader);

  // Can never fit, even in an empty sub-buffer: record and drop.
  if (MessageSize > Capacity) {
    if (IsLeader)
      __hip_atomic_fetch_or(D.ErrorBits, 1u, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_SYSTEM);
    release(&Locks[Sb], IsLeader);
    return;
  }

  // Current fill level. Only the leader reads it; broadcasting makes it
  // uniform, which the branch below needs to stay a scalar branch.
  uint32_t Used = 0;
  if (IsLeader)
    Used = uint32_t(__hip_atomic_load(&D.SubBufferSizes[Sb], __ATOMIC_ACQUIRE,
                                      __HIP_MEMORY_SCOPE_SYSTEM));
  Used = broadcast(Used);

  if (Used + MessageSize > Capacity) {
    handOffToHost(&D.HostFlags[Sb], IsLeader);
    // The host emptied the sub-buffer and we still hold the lock, so it is
    // known to be empty without re-reading it (dh_comms does the same).
    Used = 0;
  }

  char *const Message = D.Buffer + uint64_t(Sb) * Capacity + Used;

  if (IsLeader) {
    WaveHeader H{};
    H.Exec = Exec;
    H.DataSize = DataSize;
    H.Timestamp = Timestamp;
    H.DwarfFileHash = 0;
    H.InstrId = InstrId;
    H.DwarfColumn = 0xffffffffu;
    H.UserType = message_type::MemoryIndex;
    H.UserData = AccessInfo;
    // Workgroup ids need luthier::readSVA (see pickSubBuffer): unknown.
    H.BlockIdxX = H.BlockIdxY = H.BlockIdxZ = 0xffff;
    // dh_comms reads the SE/CU ids with s_getreg inline asm. Luthier lowers
    // inline asm in a payload as an intrinsic and has none for s_getreg, so
    // the ids are not reported.
    H.HwIds = 0;
    H.ActiveLaneCount = uint8_t(ActiveLanes);
    H.Flags = WaveHeader::VectorMessageFlag;
    H.Arch = currentArch();
    *reinterpret_cast<WaveHeader *>(Message) = H;
  }

  // Every active lane writes its record, dword-major so the wave's stores
  // to each dword row are consecutive (coalesced), as in dh_comms.
  auto *Data = reinterpret_cast<uint32_t *>(Message + sizeof(WaveHeader));
  Data[0 * ActiveLanes + Rank] = Rec.ElementIndex;
  Data[1 * ActiveLanes + Rank] = Rec.BufferId;

  // Make the header and lane data visible to the host before the new size
  // is: a system-scope release covers every lane's stores.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  if (IsLeader)
    __hip_atomic_store(&D.SubBufferSizes[Sb], uint64_t(Used + MessageSize),
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

  release(&Locks[Sb], IsLeader);
}

} // namespace luthier::dh_comms::device

#undef DH_COMMS_DEVICE

#endif
