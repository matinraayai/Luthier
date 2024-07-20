//===-- types.h - Luthier Types  --------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
/// \file
/// This file describes simple types and structs used by Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TYPES_H
#define LUTHIER_TYPES_H

#include <hsa/hsa.h>

namespace luthier {

#define HOOK_HANDLE_PREFIX __luthier_hook_handle_

#define RESERVED_MANAGED_VAR __luthier_reserved

#define STRINGIFY(S) #S

/// A macro to delay expansion of \c STRINGIFY
#define X_STRINGIFY(S) STRINGIFY(S)

constexpr const char *HookHandlePrefix = X_STRINGIFY(HOOK_HANDLE_PREFIX);

constexpr const char *ReservedManagedVar = X_STRINGIFY(RESERVED_MANAGED_VAR);

#define LUTHIER_HOOK_ATTRIBUTE "luthier_hook"

#define LUTHIER_INTRINSIC_ATTRIBUTE "luthier_intrinsic"

#undef X_STRINGIFY

#undef STRINGIFY

/// Luthier address type
typedef unsigned long address_t;

/// Phase of the API/Event callback
enum ApiEvtPhase : unsigned short {
  /// Before API/Event occurs
  API_EVT_PHASE_BEFORE = 0,
  /// After API/Event has occurred
  API_EVT_PHASE_AFTER = 1
};

/// Points to before/after an \c llvm::MachineInstr
enum InstrPoint : unsigned short {
  INSTR_POINT_BEFORE = 0,
  INSTR_POINT_AFTER = 1
};

/// \brief AMD Vendor Packet POD struct
typedef struct {
  uint16_t Header;
  uint8_t Format;
  uint8_t Rest[61];
} hsa_amd_vendor_packet_t;

/// \brief POD struct to provide an abstraction over HSA AQL packets as well
/// as some convenience methods
/// \details This should not be constructed directly. It should only be obtained
/// by using <tt>reinterpret_cast</tt> over the address of a packet:
/// \code
/// auto& Packet = *reinterpret_cast<HsaAqlPacket*>(PacketAddress);
/// \endcode
struct HsaAqlPacket {
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

  /// \return a non-const reference to this as a AMD vendor packet
  /// \sa hsa_amd_vendor_packet_t
  [[nodiscard]] hsa_amd_vendor_packet_t &asAMDVendor() {
    return *reinterpret_cast<hsa_amd_vendor_packet_t *>(&Packet);
  }

  /// \return a non-const reference to this a AMD vendor packet
  /// \sa hsa_amd_vendor_packet_t
  [[nodiscard]] const hsa_amd_vendor_packet_t &asAMDVendor() const {
    return *reinterpret_cast<const hsa_amd_vendor_packet_t *>(&Packet);
  }

  /// \return a non-const reference to this as a kernel dispatch packet
  /// \sa hsa_kernel_dispatch_packet_t
  [[nodiscard]] hsa_kernel_dispatch_packet_t &asKernelDispatch() {
    return *reinterpret_cast<hsa_kernel_dispatch_packet_t *>(&Packet);
  }

  /// \return a const reference to this as a kernel dispatch packet
  /// \sa hsa_kernel_dispatch_packet_t
  [[nodiscard]] const hsa_kernel_dispatch_packet_t &asKernelDispatch() const {
    return *reinterpret_cast<const hsa_kernel_dispatch_packet_t *>(&Packet);
  }

  /// \return a non-const reference to this as a Barrier-And packet
  /// \sa hsa_barrier_and_packet_t
  [[nodiscard]] hsa_barrier_and_packet_t &asBarrierAnd() {
    return *reinterpret_cast<hsa_barrier_and_packet_t *>(&Packet);
  }

  /// \return a const reference to this as a Barrier-And packet
  /// \sa hsa_barrier_and_packet_t
  [[nodiscard]] const hsa_barrier_and_packet_t &asBarrierAnd() const {
    return *reinterpret_cast<const hsa_barrier_and_packet_t *>(&Packet);
  }

  /// \return a non-const reference to this as a Barrier-Or packet
  /// \sa hsa_barrier_or_packet_t
  [[nodiscard]] hsa_barrier_or_packet_t &asBarrierOr() {
    return *reinterpret_cast<hsa_barrier_or_packet_t *>(&Packet);
  }

  /// \return a const reference to this as a Barrier-Or packet
  /// \sa hsa_barrier_or_packet_t
  [[nodiscard]] const hsa_barrier_or_packet_t &asBarrierOr() const {
    return *reinterpret_cast<const hsa_barrier_or_packet_t *>(&Packet);
  }

  /// \return a non-const reference to this as a Agent packet
  /// \sa hsa_agent_dispatch_packet_t
  [[nodiscard]] hsa_agent_dispatch_packet_t &asAgentDispatch() {
    return *reinterpret_cast<hsa_agent_dispatch_packet_t *>(&Packet);
  }

  /// \return a const reference to this as a Agent packet
  /// \sa hsa_agent_dispatch_packet_t
  [[nodiscard]] const hsa_agent_dispatch_packet_t &asAgentDispatch() const {
    return *reinterpret_cast<const hsa_agent_dispatch_packet_t *>(&Packet);
  }
};
namespace hsa {

/// \brief HSA Symbol kinds; Use this instead of \c hsa_symbol_kind_t
/// \details At the time of writing HSA only supports kernel and variable symbols;
/// Support for GNU IFunc (i.e. <tt>INDIRECT_FUNCTION</tt> is not yet implemented,
/// and device functions are not recognized by HSA but emitted in LLVM. Since
/// Luthier needs to recognize device functions as an
/// <tt>hsa_executable_symbol_t</tt>, it implements an abstraction layer which
/// generates handles for each device function and exposes them to a tool writer.
/// In order to do HSA operations on an executable symbol, one must make
/// sure it is of type <tt>KERNEL</tt> or <tt>VARIABLE</tt>. Luthier implements
/// and exposes device function symbol functionality separately
enum SymbolKind : uint8_t { VARIABLE = 0, KERNEL = 1, DEVICE_FUNCTION = 2 };

} // namespace hsa
} // namespace luthier

#endif
