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
/// The lock and host hand-off functions are taken from dh_comms (see below).
/// The code is taken by: dh_comms (https://github.com/AMDResearch/dh_comms)
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_DEVICE_H
#define LUTHIER_COMMS_DEVICE_H

#include "luthier/Comms/Protocol.h"

#include <hip/hip_runtime.h>

#define LUTHIER_COMMS_DEVICE __attribute__((device, always_inline)) inline

// Function bodies are compiled for the device only (__HIP_DEVICE_COMPILE__);
// the host pass sees the declarations, which hooks need, and empty bodies.

namespace luthier::comms::device {

/// Rank of this lane among the active lanes (0 for the first active lane).
LUTHIER_COMMS_DEVICE uint32_t activeLaneRank(uint64_t Exec) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_mbcnt_hi(
      uint32_t(Exec >> 32), __builtin_amdgcn_mbcnt_lo(uint32_t(Exec), 0u));
#else
  return 0;
#endif
}

/// The first active lane's value, for the whole wave. Branching on the
/// result is a scalar branch.
LUTHIER_COMMS_DEVICE uint32_t broadcast(uint32_t V) {
#if defined(__HIP_DEVICE_COMPILE__)
  return uint32_t(__builtin_amdgcn_readfirstlane(int(V)));
#else
  return V;
#endif
}

// Have the calling wave spin on an atomic to get exclusive read/write access to a sub-buffer.
LUTHIER_COMMS_DEVICE void wave_acquire(uint32_t *atomic_flags_d, size_t sub_buf_idx, unsigned int active_lane_id) {
#if defined(__HIP_DEVICE_COMPILE__)
  if (active_lane_id == 0) // only one lane acquires the lock on behalf of the wave
  {
    uint32_t expected = 0; // 0 -> sub-buffer is not locked
    uint32_t desired = 1;  // 1 -> sub-buffer is locked by (this) wave
    bool weak = false;
    while (not __atomic_compare_exchange(&atomic_flags_d[sub_buf_idx], &expected, &desired, weak, __ATOMIC_ACQUIRE,
                                         __ATOMIC_RELAXED)) {
      // __atomic_compare_exchange returns the value at the address (first argument) in
      // the second argument (expected), so we need to reset it in this spin-loop
      expected = 0;
    }
  }
#endif
}

// Have the calling wave release its exclusive read/write access to a sub-buffer
// Note: we should be able to do this with an __atomic_store, but in some cases, the
// compiler implemented the __atomic_store without a memory operation, leading to an
// infinite spin loop for the next call to wave_acquire. Implementing this function
// with an __atomic_compare_exchange instead resolves the issue for the cases we
// have seen so far.
LUTHIER_COMMS_DEVICE void wave_release(uint32_t *atomic_flags_d, size_t sub_buf_idx, unsigned int active_lane_id) {
#if defined(__HIP_DEVICE_COMPILE__)
  if (active_lane_id == 0) // only one lane acquires the lock on behalf of the wave
  {
    uint32_t expected = 1; // 1 -> sub-buffer is locked by this wave
    uint32_t desired = 0;  // 0 -> sub-buffer is unlocked
    bool weak = false;
    while (not __atomic_compare_exchange(&atomic_flags_d[sub_buf_idx], &expected, &desired, weak, __ATOMIC_RELEASE,
                                         __ATOMIC_RELAXED)) {
      // __atomic_compare_exchange returns the value at the address (first argument) in
      // the second argument (expected), so we need to reset it in this spin-loop
      expected = 1;
    }
  }
#endif
}

// Have the calling wave, which has exclusive read/write access to a sub-buffer, pass control
// of the sub-buffer to the host by setting a flag to 1, and then spin on the flag until the host
// resets it to zero, indicating that it is done processing the sub-buffer.
LUTHIER_COMMS_DEVICE void wave_signal_host(uint32_t *atomic_flags_hd, size_t sub_buf_idx, unsigned int active_lane_id) {
#if defined(__HIP_DEVICE_COMPILE__)
  if (active_lane_id == 0) // only one lane passes control of the sub-buffer to the host on behalf of the wave
  {
    uint32_t flag = 1;
    __atomic_store_n(&atomic_flags_hd[sub_buf_idx], flag, __ATOMIC_RELEASE);
    while (__atomic_load_n(&atomic_flags_hd[sub_buf_idx], __ATOMIC_ACQUIRE) != 0) {
    }
  }
#endif
}

/// Shader engine, shader array and compute unit of the executing wave.
struct HwLocation {
  uint8_t SeId = UnknownHwId;
  uint8_t SaId = UnknownHwId;
  uint8_t CuId = UnknownHwId;
};

/// Shader engine, shader array and CU of the executing wave, read with
/// \c __builtin_amdgcn_s_getreg using HIP's \c HW_ID field macros (as
/// \c __smid does): \c HW_REG_HW_ID on GFX9 (CDNA), \c HW_REG_HW_ID1 on
/// GFX10/11 (RDNA), where \c CuId is the WGP id. HIP defines no layout for
/// GFX12, so the ids are reported unknown there.
LUTHIER_COMMS_DEVICE HwLocation readHwLocation() {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX10__) || defined(__GFX11__))
  return {uint8_t(__builtin_amdgcn_s_getreg(GETREG_IMMED(
              HW_ID_SE_ID_SIZE - 1, HW_ID_SE_ID_OFFSET, HW_ID))),
          uint8_t(__builtin_amdgcn_s_getreg(GETREG_IMMED(
              HW_ID_SA_ID_SIZE - 1, HW_ID_SA_ID_OFFSET, HW_ID))),
          uint8_t(__builtin_amdgcn_s_getreg(GETREG_IMMED(
              HW_ID_WGP_ID_SIZE - 1, HW_ID_WGP_ID_OFFSET, HW_ID)))};
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  // SH_ID [12] is the shader array; HIP has no macro for it.
  return {uint8_t(__builtin_amdgcn_s_getreg(GETREG_IMMED(
              HW_ID_SE_ID_SIZE - 1, HW_ID_SE_ID_OFFSET, HW_ID))),
          uint8_t(__builtin_amdgcn_s_getreg(GETREG_IMMED(0, 12, HW_ID))),
          uint8_t(__builtin_amdgcn_s_getreg(GETREG_IMMED(
              HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID)))};
#else
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
#if defined(__HIP_DEVICE_COMPILE__)
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

  wave_acquire(Locks, Sb, Rank);

  if (MessageSize > Capacity) { // never fits: record and drop
    if (IsLeader)
      __hip_atomic_fetch_or(C.ErrorBits, 1u, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_SYSTEM);
    wave_release(Locks, Sb, Rank);
    return;
  }

  // Fill level, made wave-uniform so the branch below stays scalar.
  uint32_t Used = 0;
  if (IsLeader)
    Used = uint32_t(__hip_atomic_load(&C.SubBufferSizes[Sb], __ATOMIC_ACQUIRE,
                                      __HIP_MEMORY_SCOPE_SYSTEM));
  Used = broadcast(Used);
  if (Used + MessageSize > Capacity) {
    wave_signal_host(C.HostFlags, Sb, Rank);
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
  wave_release(Locks, Sb, Rank);
#endif
}

} // namespace luthier::comms::device

#undef LUTHIER_COMMS_DEVICE

#endif
