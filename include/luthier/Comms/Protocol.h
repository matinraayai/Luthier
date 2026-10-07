//===-- Protocol.h - Device-to-host message format --------------*- C++ -*-===//
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
/// Byte layout of the messages instrumented kernels send to the host, and of
/// the descriptor that tells the device where to put them. Shared by device
/// and host code, so nothing here uses bitfields.
///
/// A message is one \c WaveHeader followed by \c DwordsPerLane dwords for
/// each active lane, written dword-major so a wave's stores coalesce:
///
///   dword 0 of active lane 0..N-1, dword 1 of active lane 0..N-1, ...
///
/// The i-th lane's data belongs to the i-th set bit of \c WaveHeader::Exec.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_PROTOCOL_H
#define LUTHIER_COMMS_PROTOCOL_H

#include <cstddef>
#include <cstdint>

namespace luthier::comms {

/// Upper bound on sub-buffers; sizes the device-side lock array.
inline constexpr uint32_t MaxSubBuffers = 1024;

/// \c WaveHeader::BlockIdx* value when the workgroup id is not reported.
inline constexpr uint32_t UnknownBlockIdx = 0xffffffffu;

/// Information that applies to the whole wave.
struct WaveHeader {
  uint64_t Exec;         ///< Execution mask at the instrumentation point.
  uint64_t Timestamp;    ///< Cycle counter when the wave submitted.
  uint32_t DataSize;     ///< Bytes of lane data after this header.
  uint32_t Tag;          ///< What the lane data is; defined by the sender.
  uint32_t Site;         ///< Which instrumentation point sent it.
  uint32_t Info;         ///< Per-site data; defined by the sender.
  uint32_t BlockIdxX;    ///< Workgroup id, or \c UnknownBlockIdx.
  uint32_t BlockIdxY;
  uint32_t BlockIdxZ;
  uint32_t HwId;         ///< Raw \c HW_REG_HW_ID, or 0 if not reported.
  uint8_t ActiveLanes;   ///< Number of set bits in \c Exec.
  uint8_t DwordsPerLane; ///< Lane data is ActiveLanes * DwordsPerLane dwords.
  uint8_t Reserved[6];
};

static_assert(sizeof(WaveHeader) == 56, "WaveHeader layout drifted");
static_assert(offsetof(WaveHeader, DataSize) == 16);
static_assert(offsetof(WaveHeader, BlockIdxX) == 32);
static_assert(offsetof(WaveHeader, ActiveLanes) == 48);

/// Where the device writes messages: fine-grained host memory split into
/// sub-buffers. Filled by \c SharedBuffers::describe.
struct ChannelDescriptor {
  uint64_t NumSubBuffers;     ///< Power of two, <= MaxSubBuffers.
  uint64_t SubBufferCapacity; ///< Bytes per sub-buffer.
  char *Buffer;               ///< NumSubBuffers * SubBufferCapacity bytes.
  uint64_t *SubBufferSizes;   ///< Bytes in use, per sub-buffer.
  uint32_t *HostFlags;        ///< 1: "full, please drain"; host resets to 0.
  uint32_t *ErrorBits;        ///< [0]: a message exceeded a sub-buffer.
};

} // namespace luthier::comms

#endif
