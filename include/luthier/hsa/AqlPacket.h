//===-- AqlPacket.h - HSA AQL Packet POD Wrapper ----------------*- C++ -*-===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file the \c AqlPacket struct under the \c luthier::hsa namespace,
/// which is a plain-old-data struct that provides an abstraction over the
/// AQL packet.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_HSA_AQL_PACKET_H
#define LUTHIER_HSA_AQL_PACKET_H
#include <hsa/hsa.h>

namespace hsa {

/// \brief AMD Vendor Packet POD struct
typedef struct {
  uint16_t Header;
  uint8_t Format;
  uint8_t Rest[61];
} amd_vendor_packet_t;

/// \brief POD struct to provide an abstraction over HSA AQL packets as well
/// as some convenience methods
/// \details To inspect the content of a packet using this struct simply
/// use use <tt>reinterpret_cast</tt> over the address of the packet:
/// \code
/// auto& Packet = *reinterpret_cast<HsaAqlPacket*>(PacketAddress);
/// \endcode
struct AqlPacket {
  /// \brief Generic packet struct
  struct {
    uint16_t Header;
    struct {
      uint8_t UserData[62];
    } Body;
  } Packet;

  /// \return the type of the packet
  /// \sa hsa_packet_type_t
  [[nodiscard]] hsa_packet_type_t getPacketType() const {
    return static_cast<hsa_packet_type_t>(
        (Packet.Header >> HSA_PACKET_HEADER_TYPE) &
        ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
  }

  /// \return a non-const reference to the content of this packet viewed
  /// as a AMD vendor packet
  /// \sa hsa_amd_vendor_packet_t
  [[nodiscard]] amd_vendor_packet_t &asAMDVendor() {
    return *reinterpret_cast<amd_vendor_packet_t *>(&Packet);
  }

  /// \return a non-const reference to the content of this packet viewed as
  /// a AMD vendor packet
  /// \sa hsa_amd_vendor_packet_t
  [[nodiscard]] const amd_vendor_packet_t &asAMDVendor() const {
    return *reinterpret_cast<const amd_vendor_packet_t *>(&Packet);
  }

  /// \return a non-const reference to the contents of this packet viewed as a
  /// kernel dispatch packet
  /// \sa hsa_kernel_dispatch_packet_t
  [[nodiscard]] hsa_kernel_dispatch_packet_t &asKernelDispatch() {
    return *reinterpret_cast<hsa_kernel_dispatch_packet_t *>(&Packet);
  }

  /// \return a const reference to this packet viewed as a kernel dispatch packet
  /// \sa hsa_kernel_dispatch_packet_t
  [[nodiscard]] const hsa_kernel_dispatch_packet_t &asKernelDispatch() const {
    return *reinterpret_cast<const hsa_kernel_dispatch_packet_t *>(&Packet);
  }

  /// \return a non-const reference to this packet viewed as a Barrier-And packet
  /// \sa hsa_barrier_and_packet_t
  [[nodiscard]] hsa_barrier_and_packet_t &asBarrierAnd() {
    return *reinterpret_cast<hsa_barrier_and_packet_t *>(&Packet);
  }

  /// \return a const reference to this packet viewed as a Barrier-And packet
  /// \sa hsa_barrier_and_packet_t
  [[nodiscard]] const hsa_barrier_and_packet_t &asBarrierAnd() const {
    return *reinterpret_cast<const hsa_barrier_and_packet_t *>(&Packet);
  }

  /// \return a non-const reference to this packet viewed as a Barrier-Or packet
  /// \sa hsa_barrier_or_packet_t
  [[nodiscard]] hsa_barrier_or_packet_t &asBarrierOr() {
    return *reinterpret_cast<hsa_barrier_or_packet_t *>(&Packet);
  }

  /// \return a const reference to this packet viewed as a Barrier-Or packet
  /// \sa hsa_barrier_or_packet_t
  [[nodiscard]] const hsa_barrier_or_packet_t &asBarrierOr() const {
    return *reinterpret_cast<const hsa_barrier_or_packet_t *>(&Packet);
  }

  /// \return a non-const reference to this packet viewed as a Agent packet
  /// \sa hsa_agent_dispatch_packet_t
  [[nodiscard]] hsa_agent_dispatch_packet_t &asAgentDispatch() {
    return *reinterpret_cast<hsa_agent_dispatch_packet_t *>(&Packet);
  }

  /// \return a const reference to this packet viewed as a Agent packet
  /// \sa hsa_agent_dispatch_packet_t
  [[nodiscard]] const hsa_agent_dispatch_packet_t &asAgentDispatch() const {
    return *reinterpret_cast<const hsa_agent_dispatch_packet_t *>(&Packet);
  }
};

} // namespace hsa


#endif