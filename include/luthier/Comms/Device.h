//===-- Device.h - Device side of the device-to-host channel ----*- C++ -*-===//
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
/// Device code that sends a message from inside an injected payload:
///
///   1. take the sub-buffer's lock (serialises waves sharing it)
///   2. if the message does not fit, hand the sub-buffer to the host and
///      wait for it to be drained
///   3. write the wave header and every active lane's dwords
///   4. publish the new size and release the lock
///
/// In a Luthier payload a spin loop run by a divergent subset of lanes
/// deadlocks: the structurizer retires the winning lane from \c EXEC before
/// it reaches the release. So every loop here is entered by all active lanes
/// and exits on a wave-uniform condition (broadcast with readfirstlane);
/// only the side-effecting atomic inside it is guarded to one lane.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_DEVICE_H
#define LUTHIER_COMMS_DEVICE_H

#include "luthier/Comms/Protocol.h"

#include <hip/hip_runtime.h>

#define LUTHIER_COMMS_DEVICE __attribute__((device, always_inline)) inline

namespace luthier::comms::device {

/// Rank of this lane among the active lanes (0 for the first active lane).
LUTHIER_COMMS_DEVICE uint32_t activeLaneRank(uint64_t Exec) {
  return __builtin_amdgcn_mbcnt_hi(
      uint32_t(Exec >> 32), __builtin_amdgcn_mbcnt_lo(uint32_t(Exec), 0u));
}

/// The first active lane's value, for the whole wave. Branching on the
/// result is a scalar branch.
LUTHIER_COMMS_DEVICE uint32_t broadcast(uint32_t V) {
  return uint32_t(__builtin_amdgcn_readfirstlane(int(V)));
}

/// Takes \p Lock for the whole wave; returns once the leader holds it.
LUTHIER_COMMS_DEVICE void acquire(uint32_t *Lock, bool IsLeader) {
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

/// Releases \p Lock. An exchange, not a store-release, which the compiler
/// has been seen to drop.
LUTHIER_COMMS_DEVICE void release(uint32_t *Lock, bool IsLeader) {
  if (IsLeader)
    (void)__hip_atomic_exchange(Lock, 0u, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_AGENT);
}

/// Hands a full sub-buffer to the host and waits until it is drained. The
/// lock is still held, so no other wave touches the sub-buffer meanwhile.
LUTHIER_COMMS_DEVICE void handOffToHost(uint32_t *HostFlag, bool IsLeader) {
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

/// Shader engine, shader array and compute unit of the executing wave.
struct HwLocation {
  uint8_t SeId = UnknownHwId;
  uint8_t SaId = UnknownHwId;
  uint8_t CuId = UnknownHwId;
};

/// Decodes the hardware-id register the payload read with
/// \c luthier::readHwReg: \c HW_REG_HW_ID on GFX9 (CDNA),
/// \c HW_REG_HW_ID1 on GFX10+ (RDNA), where \c CuId is the WGP id.
LUTHIER_COMMS_DEVICE HwLocation decodeHwId(uint32_t R) {
#if defined(__GFX9__)
  // CU_ID [11:8], SH_ID [12], SE_ID [15:13].
  return {uint8_t(R >> 13 & 0x7u), uint8_t(R >> 12 & 0x1u),
          uint8_t(R >> 8 & 0xfu)};
#elif defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
  // WGP_ID [13:10], SA_ID [16], SE_ID [20:18].
  return {uint8_t(R >> 18 & 0x7u), uint8_t(R >> 16 & 0x1u),
          uint8_t(R >> 10 & 0xfu)};
#else
  (void)R;
  return {};
#endif
}

/// Fields of the \c WaveHeader that the sender provides.
struct MessageInfo {
  uint32_t Tag;
  uint32_t Site;
  uint32_t Info;
  uint32_t BlockIdxX = UnknownBlockIdx;
  uint32_t BlockIdxY = UnknownBlockIdx;
  uint32_t BlockIdxZ = UnknownBlockIdx;
  HwLocation Hw = {};
};

/// Sends \p Lane (this lane's \p N dwords) from every active lane as one
/// message, through sub-buffer \p SubBuffer, which must be wave-uniform.
template <unsigned N>
LUTHIER_COMMS_DEVICE void submit(const ChannelDescriptor &C, uint32_t *Locks,
                                 uint32_t SubBuffer, const MessageInfo &M,
                                 const uint32_t (&Lane)[N]) {
  // Read the clock first so the submission itself barely perturbs it.
  const uint64_t Timestamp = __builtin_readcyclecounter();
  if (C.NumSubBuffers == 0)
    return; // channel not configured for this dispatch

  const uint64_t Exec = __builtin_amdgcn_read_exec();
  const uint32_t ActiveLanes = uint32_t(__builtin_popcountll(Exec));
  const uint32_t Rank = activeLaneRank(Exec);
  const bool IsLeader = Rank == 0;
  const uint32_t DataSize = ActiveLanes * N * uint32_t(sizeof(uint32_t));
  const uint32_t MessageSize = uint32_t(sizeof(WaveHeader)) + DataSize;
  const uint32_t Capacity = uint32_t(C.SubBufferCapacity);
  const uint32_t Sb = SubBuffer & uint32_t(C.NumSubBuffers - 1);

  acquire(&Locks[Sb], IsLeader);

  if (MessageSize > Capacity) { // never fits: record and drop
    if (IsLeader)
      __hip_atomic_fetch_or(C.ErrorBits, 1u, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_SYSTEM);
    release(&Locks[Sb], IsLeader);
    return;
  }

  // Fill level, made wave-uniform so the branch below stays scalar.
  uint32_t Used = 0;
  if (IsLeader)
    Used = uint32_t(__hip_atomic_load(&C.SubBufferSizes[Sb], __ATOMIC_ACQUIRE,
                                      __HIP_MEMORY_SCOPE_SYSTEM));
  Used = broadcast(Used);
  if (Used + MessageSize > Capacity) {
    handOffToHost(&C.HostFlags[Sb], IsLeader);
    Used = 0; // drained, and we still hold the lock
  }

  char *const Message = C.Buffer + uint64_t(Sb) * Capacity + Used;
  if (IsLeader) {
    WaveHeader H{};
    H.Exec = Exec;
    H.Timestamp = Timestamp;
    H.DataSize = DataSize;
    H.Tag = M.Tag;
    H.Site = M.Site;
    H.Info = M.Info;
    H.BlockIdxX = M.BlockIdxX;
    H.BlockIdxY = M.BlockIdxY;
    H.BlockIdxZ = M.BlockIdxZ;
    H.SeId = M.Hw.SeId;
    H.SaId = M.Hw.SaId;
    H.CuId = M.Hw.CuId;
    // 32 or 64, folded from the subtarget the payload is compiled for (it
    // follows -mwavefrontsize64 on RDNA).
    H.WaveSize = uint8_t(__builtin_amdgcn_wavefrontsize());
    H.ActiveLanes = uint8_t(ActiveLanes);
    H.DwordsPerLane = uint8_t(N);
    *reinterpret_cast<WaveHeader *>(Message) = H;
  }
  auto *Data = reinterpret_cast<uint32_t *>(Message + sizeof(WaveHeader));
  for (unsigned D = 0; D < N; ++D)
    Data[D * ActiveLanes + Rank] = Lane[D];

  // Header and lane data must reach the host before the new size does.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  if (IsLeader)
    __hip_atomic_store(&C.SubBufferSizes[Sb], uint64_t(Used + MessageSize),
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  release(&Locks[Sb], IsLeader);
}

} // namespace luthier::comms::device

#undef LUTHIER_COMMS_DEVICE

#endif
