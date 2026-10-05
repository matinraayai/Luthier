//===-- Protocol.h - dh_comms wire format shared by GPU and CPU -*- C++ -*-===//
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
/// The byte layout of every message the GPU writes into the shared buffers,
/// plus the descriptor the GPU uses to find those buffers.
///
/// This header is compiled twice: by clang for the GPU (inside the tool's
/// .hip file) and by the host C++ compiler (GCC) for the CPU-side readers.
/// Both must agree on every byte, so nothing here uses C bitfields, whose
/// layout is implementation-defined. Packed fields are plain integers with
/// accessor functions, and \c static_assert pins every offset.
///
/// Ported from AMD's dh_comms (data_headers.h, message.h); the field set and
/// semantics are the same, only the packing is made explicit.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXAMPLES_DH_COMMS_PROTOCOL_H
#define LUTHIER_EXAMPLES_DH_COMMS_PROTOCOL_H

#include <cstddef>
#include <cstdint>

namespace luthier::dh_comms {

//===----------------------------------------------------------------------===//
// Limits
//===----------------------------------------------------------------------===//

/// Upper bound on sub-buffers; sizes the device-side lock array.
inline constexpr uint32_t MaxSubBuffers = 1024;

/// Upper bound on kernel-argument buffers resolved for memory indexing.
inline constexpr uint32_t MaxTrackedBuffers = 16;

/// \c AccessRecord::BufferId for an address that falls in no tracked buffer.
inline constexpr uint32_t UnresolvedBuffer = 0xffffffffu;

//===----------------------------------------------------------------------===//
// Message tags (dh_comms message.h)
//===----------------------------------------------------------------------===//

namespace message_type {
enum : uint32_t {
  Address = 0,       ///< dh_comms: raw address per lane
  TimeInterval = 1,  ///< dh_comms: start/stop timestamps
  BasicBlockStart = 2,
  MemoryIndex = 3,   ///< this tool: (buffer, element index) per lane
};
} // namespace message_type

namespace memory_access {
enum : uint8_t { Undefined = 0, Read = 1, Write = 2, ReadWrite = 3 };
} // namespace memory_access

namespace address_space {
enum : uint8_t { Flat = 0, Global = 1, Gds = 2, Shared = 3, Constant = 4,
                 Scratch = 5, Undefined = 0xf };
} // namespace address_space

namespace gcn_arch {
enum : uint8_t { Unsupported = 0, Gfx906 = 1, Gfx908 = 2, Gfx90a = 3,
                 Gfx940 = 4, Gfx941 = 5, Gfx942 = 6 };
} // namespace gcn_arch

/// Packs a memory access's kind into \c WaveHeader::UserData, using the same
/// bit layout as dh_comms' \c v_submit_address:
///   [1:0] read/write kind, [5:2] address space, [21:6] access width in bytes.
constexpr uint32_t packAccessInfo(uint8_t RwKind, uint8_t AddressSpace,
                                  uint16_t WidthBytes) {
  return (RwKind & 0x3u) | ((AddressSpace & 0xfu) << 2) |
         (uint32_t(WidthBytes) << 6);
}
constexpr uint8_t accessRwKind(uint32_t UserData) { return UserData & 0x3u; }
constexpr uint8_t accessAddressSpace(uint32_t UserData) {
  return (UserData >> 2) & 0xfu;
}
constexpr uint16_t accessWidthBytes(uint32_t UserData) {
  return uint16_t(UserData >> 6);
}

//===----------------------------------------------------------------------===//
// Messages
//===----------------------------------------------------------------------===//
//
// A message is one wave header followed by one AccessRecord per active lane:
//
//   +------------+------------------------------------------------+
//   | WaveHeader |  record data for ActiveLaneCount lanes          |
//   |  64 bytes  |  laid out dword-major (see below)               |
//   +------------+------------------------------------------------+
//
// As in dh_comms, lane data is written one dword at a time so that the
// active lanes of a wave store to consecutive addresses (coalesced):
//
//   dword 0 of lane 0, dword 0 of lane 1, ..., dword 0 of lane N-1,
//   dword 1 of lane 0, dword 1 of lane 1, ..., dword 1 of lane N-1
//
// The i-th record belongs to the i-th set bit of WaveHeader::Exec.

/// Information that applies to the whole wave. Same fields as dh_comms'
/// \c wave_header_t; the two flag bits move out of the size field into
/// \c Flags, and the hardware ids are packed by hand.
struct WaveHeader {
  uint64_t Exec;          ///< Execution mask at the instrumented instruction.
  uint64_t DataSize;      ///< Bytes of lane data that follow this header.
  uint64_t Timestamp;     ///< s_memtime when the wave submitted.
  uint64_t DwarfFileHash; ///< 0: lifted code carries no DWARF.
  uint32_t InstrId;       ///< Which instrumented instruction (dh_comms'
                          ///< dwarf_line slot: "location of the access").
  uint32_t DwarfColumn;   ///< Unused; 0xffffffff.
  uint32_t UserType;      ///< A \c message_type tag.
  uint32_t UserData;      ///< \c packAccessInfo() of the access.
  uint16_t BlockIdxX;
  uint16_t BlockIdxY;
  uint16_t BlockIdxZ;
  uint16_t HwIds;         ///< [2:0] shader engine, [6:3] compute unit.
  uint8_t ActiveLaneCount;
  uint8_t Flags;          ///< [0] is_vector_message, [1] has_lane_headers.
  uint8_t Arch;           ///< A \c gcn_arch value.
  uint8_t Reserved[5];

  static constexpr uint8_t VectorMessageFlag = 1u << 0;
  static constexpr uint8_t LaneHeadersFlag = 1u << 1;

  uint8_t shaderEngine() const { return HwIds & 0x7u; }
  uint8_t computeUnit() const { return (HwIds >> 3) & 0xfu; }
  bool isVectorMessage() const { return Flags & VectorMessageFlag; }
};

static_assert(sizeof(WaveHeader) == 64, "WaveHeader layout drifted");
static_assert(offsetof(WaveHeader, DataSize) == 8);
static_assert(offsetof(WaveHeader, InstrId) == 32);
static_assert(offsetof(WaveHeader, UserData) == 44);
static_assert(offsetof(WaveHeader, BlockIdxX) == 48);
static_assert(offsetof(WaveHeader, ActiveLaneCount) == 56);

/// What one lane reports for one memory access: not the address, but where
/// the address lands — which kernel-argument buffer, and which element of it.
struct AccessRecord {
  uint32_t ElementIndex; ///< (address - buffer start) / element size.
  uint32_t BufferId;     ///< Kernel-argument slot the buffer was passed in,
                         ///< or \c UnresolvedBuffer.
};

static_assert(sizeof(AccessRecord) == 8, "AccessRecord layout drifted");
inline constexpr uint32_t AccessRecordDwords =
    sizeof(AccessRecord) / sizeof(uint32_t);

//===----------------------------------------------------------------------===//
// Descriptor
//===----------------------------------------------------------------------===//

/// Everything the GPU needs to submit messages. The host fills it in before
/// each instrumented dispatch and copies it into the tool's device global;
/// the pointers inside point at fine-grained pinned host memory, which the
/// GPU writes to directly. Same role as dh_comms' \c dh_comms_descriptor.
struct DeviceDescriptor {
  // ---- Shared buffers (dh_comms) ----------------------------------------
  uint64_t NumSubBuffers;     ///< Power of two, <= MaxSubBuffers.
  uint64_t SubBufferCapacity; ///< Bytes per sub-buffer.
  char *Buffer;               ///< NumSubBuffers * SubBufferCapacity bytes.
  uint64_t *SubBufferSizes;   ///< Bytes in use, one entry per sub-buffer.
  uint32_t *HostFlags;        ///< 1: "full, please drain"; host resets to 0.
  uint32_t *ErrorBits;        ///< [0]: a message exceeded a sub-buffer.

  // ---- Grid shape ---------------------------------------------------------
  // Filled by the host but not read yet: dh_comms picks the sub-buffer from
  // the flattened workgroup id, which needs luthier::readSVA in the hook (see
  // README). Until then DeviceSubmit.h hashes the access address instead.
  uint32_t NumGroupsX;
  uint32_t NumGroupsY;

  // ---- Memory indexing --------------------------------------------------
  uint32_t ElementSizeLog2;   ///< Index = byte offset >> ElementSizeLog2.
  uint32_t NumTrackedBuffers;
  uint64_t BufferBase[MaxTrackedBuffers];
  uint64_t BufferEnd[MaxTrackedBuffers];      ///< One past the last byte.
  uint32_t BufferArgSlot[MaxTrackedBuffers];  ///< Kernel-argument slot.
};

} // namespace luthier::dh_comms

#endif
