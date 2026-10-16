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
/// and host code.
///
/// What is sent is up to the tool: a struct for the whole wave (\c WaveT)
/// and a struct for each active lane (\c LaneT). A message is
///
///   uint64_t Exec          which lanes were active (the only fixed field)
///   WaveT                  written once, by the first active lane
///   LaneT[A]               one per active lane, A = popcount(Exec)
///
/// with \c WaveT padded and the whole message rounded up to 8 bytes, so each
/// part is 8-byte aligned. The i-th \c LaneT belongs to the i-th set bit of
/// \c Exec. No sizes or types travel with the message: both sides know
/// \c WaveT and \c LaneT, and the length follows from \c Exec.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_PROTOCOL_H
#define LUTHIER_COMMS_PROTOCOL_H

#include <cstddef>
#include <cstdint>

namespace luthier::comms {

/// Upper bound on sub-buffers; sizes the device-side lock array.
inline constexpr uint32_t MaxSubBuffers = 1024;

/// \c HwLocation value of an id the target does not report.
inline constexpr uint8_t UnknownHwId = 0xff;

constexpr uint32_t alignTo8(uint32_t Bytes) { return (Bytes + 7u) & ~7u; }

/// Byte offsets of one message's parts, for \p ActiveLanes lanes.
template <typename WaveT, typename LaneT> struct MessageLayout {
  static_assert(alignof(WaveT) <= 8 && alignof(LaneT) <= 8,
                "message structs must not need more than 8-byte alignment");
  static constexpr uint32_t WaveOffset = sizeof(uint64_t);
  static constexpr uint32_t LaneOffset = WaveOffset + alignTo8(sizeof(WaveT));
  static constexpr uint32_t bytes(uint32_t ActiveLanes) {
    return alignTo8(LaneOffset + ActiveLanes * uint32_t(sizeof(LaneT)));
  }
};

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
