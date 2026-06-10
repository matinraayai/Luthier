//===-- HostcallABI.h - Device-libs hostcall buffer ABI ---------*- C++ -*-===//
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
/// Host-side view of the AMD device-libs hostcall buffer ABI — the shared
/// memory protocol that the device function \c __ockl_hostcall_internal uses to
/// request services from a host listener thread. Luthier provisions its own
/// hostcall buffer + consumer for instrumented kernels (the ROCclr runtime only
/// does so when the application itself uses hostcall), so this layout MUST stay
/// bit-compatible with what \c __ockl_hostcall_internal compiles against.
///
/// Ported verbatim (layout, free-stack tagged-pointer scheme, sizing) from
/// ROCclr \c device/devhostcall.{hpp,cpp} and the service-id enum from
/// \c device/devhcmessages.hpp. Kept standalone (no ROCclr/\c amd:: deps) so it
/// can live in the Luthier HSATooling layer.
///
/// Lifecycle (mirrors \c amd::enableHostcalls): allocate a fine-grain +
/// atomics-capable buffer of \c getHostcallBufferSize(numPackets) bytes aligned
/// to \c getHostcallBufferAlignment(), \c initialize() it, point a doorbell
/// \c hsa_signal_t at it, run a consumer thread that waits on the doorbell and
/// drains the ready stack, and publish the buffer pointer into the kernel's
/// COV5 hostcall implicit arg (\c cov5::HostcallPtr). See the Luthier hostcall
/// consumer for the host half.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_HOSTCALL_ABI_H
#define LUTHIER_HSA_TOOLING_HOSTCALL_ABI_H
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <llvm/Support/MathExtras.h>
#include <type_traits>

namespace luthier::hostcall {

/// Doorbell signal values. The kernel sets the doorbell to \c SIGNAL_DONE (0)
/// to wake the listener; the listener resets it to \c SIGNAL_INIT (1).
enum SignalValue : uint64_t { SIGNAL_DONE = 0, SIGNAL_INIT = 1 };

/// Service requested by a wave in \c PacketHeader::Service. Matches ROCclr
/// \c device/devhcmessages.hpp. Luthier's indirect-branch resolver rides
/// \c SERVICE_FUNCTION_CALL (the payload carries a host function pointer the
/// listener invokes directly).
enum ServiceID : uint32_t {
  SERVICE_RESERVED = 0,
  SERVICE_FUNCTION_CALL = 1,
  SERVICE_PRINTF = 2,
  SERVICE_DEVMEM = 3,
  SERVICE_SANITIZER = 4,
};

/// The \c SERVICE_FUNCTION_CALL contract: \c Payload::Slots[lane][0] is a host
/// function pointer of this type; the listener calls it with \c Output pointing
/// at up to two result ulongs and \c Input at up to seven argument ulongs (the
/// remaining slots of that lane).
using HostcallFunctionCall = void (*)(uint64_t *Output, const uint64_t *Input);

/// Packet payload: 64 slots (one per wavefront workitem) of 8 ulongs each. Slot
/// \c i holds valid data iff bit \c i of \c PacketHeader::ActiveMask is set.
struct Payload {
  uint64_t Slots[64][8];
};

/// Packet header. Field order/types are ABI — \c __ockl_hostcall_internal reads
/// and atomically updates these in place.
struct PacketHeader {
  /// Tagged pointer to the next packet in an intrusive stack.
  uint64_t Next;
  /// Bitmask of payload slots carrying valid data.
  uint64_t ActiveMask;
  /// Service ID requested by the wave (see \c ServiceID).
  uint32_t Service;
  /// Control bits; bit 0 (\c CONTROL_OFFSET_READY_FLAG) = packet awaiting a
  /// host response.
  std::atomic<uint32_t> Control;
};

static_assert(std::is_standard_layout_v<PacketHeader>,
              "the hostcall packet must be usable from device code");

/// Field offsets within \c PacketHeader::Control.
enum ControlOffset : uint32_t {
  CONTROL_OFFSET_READY_FLAG = 0,
  CONTROL_OFFSET_RESERVED0 = 1,
};

/// Field widths within \c PacketHeader::Control.
enum ControlWidth : uint32_t {
  CONTROL_WIDTH_READY_FLAG = 1,
  CONTROL_WIDTH_RESERVED0 = 31,
};

/// Shared buffer holding hostcall packets for one device queue. Packets are
/// referenced by 64-bit tagged pointers (low bits = index into the packet
/// arrays via \c IndexMask, high bits = an ABA tag bumped on every pop). Field
/// order/types are ABI: \c __ockl_hostcall_internal indexes this struct
/// directly, so it must match ROCclr's \c amd::HostcallBuffer exactly.
struct HostcallBuffer {
  /// Array of \c NumPackets packet headers (within this buffer allocation).
  PacketHeader *Headers;
  /// Array of \c NumPackets packet payloads (within this buffer allocation).
  Payload *Payloads;
  /// Doorbell signal the kernel pulses to announce new work (opaque handle;
  /// the host stores the \c hsa_signal_t's underlying pointer here).
  void *Doorbell;
  /// Stack of free packets (tagged pointer).
  uint64_t FreeStack;
  /// Stack of ready packets awaiting host service (tagged pointer).
  std::atomic<uint64_t> ReadyStack;
  /// Mask extracting the packet index from a tagged pointer.
  uint64_t IndexMask;
  /// Opaque device handle some services need; unused by Luthier services.
  const void *Device;

  /// Resolve a tagged pointer to its packet header.
  [[nodiscard]] PacketHeader *getHeader(uint64_t Ptr) const {
    return &Headers[Ptr & IndexMask];
  }
  /// Resolve a tagged pointer to its packet payload.
  [[nodiscard]] Payload *getPayload(uint64_t Ptr) const {
    return &Payloads[Ptr & IndexMask];
  }

  /// Lay out the header/payload arrays inside this allocation and seed the free
  /// stack. \p NumPackets must be > 1. Verbatim port of
  /// \c amd::HostcallBuffer::initialize.
  void initialize(uint32_t NumPackets);
};

static_assert(std::is_standard_layout_v<HostcallBuffer>,
              "the hostcall buffer must be usable from device code");

/// Offset of the packet-header array within a \c HostcallBuffer allocation.
inline uintptr_t getHeaderStart() {
  return llvm::alignTo(sizeof(HostcallBuffer), alignof(PacketHeader));
}

/// Offset of the payload array within a \c HostcallBuffer allocation.
inline uintptr_t getPayloadStart(uint32_t NumPackets) {
  uintptr_t HeaderEnd = getHeaderStart() + sizeof(PacketHeader) * NumPackets;
  return llvm::alignTo(HeaderEnd, alignof(Payload));
}

/// Total bytes to allocate for a buffer holding \p NumPackets packets.
inline size_t getHostcallBufferSize(uint32_t NumPackets) {
  return getPayloadStart(NumPackets) + size_t{NumPackets} * sizeof(Payload);
}

/// Required allocation alignment for a hostcall buffer.
inline uint32_t getHostcallBufferAlignment() { return alignof(Payload); }

/// Index mask for \p NumPackets (rounded up to a power of two minus one).
inline uint64_t getIndexMask(uint32_t NumPackets) {
  // Callers guarantee NumPackets > 1 (>= the device's max concurrent waves),
  // so the zero/one border cases do not arise.
  if (!llvm::isPowerOf2_32(NumPackets))
    NumPackets = llvm::PowerOf2Ceil(NumPackets);
  return uint64_t{NumPackets} - 1;
}

inline void HostcallBuffer::initialize(uint32_t NumPackets) {
  auto *Base = reinterpret_cast<uint8_t *>(this);
  Headers = reinterpret_cast<PacketHeader *>(Base + getHeaderStart());
  Payloads = reinterpret_cast<Payload *>(Base + getPayloadStart(NumPackets));
  IndexMask = getIndexMask(NumPackets);

  // The null tagged pointer is (uint64_t)0; to keep the tag+index from both
  // being zero, the first link gets a tag of 1 (= IndexMask + 1).
  uint64_t Next = IndexMask + 1;
  Headers[0].Next = 0;
  for (uint32_t I = 1; I != NumPackets; ++I) {
    Headers[I].Next = Next;
    Next = I;
  }
  FreeStack = Next;
  ReadyStack = 0;
}

} // namespace luthier::hostcall

#endif // LUTHIER_HSA_TOOLING_HOSTCALL_ABI_H
