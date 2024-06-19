#ifndef LUTHIER_TYPES_H
#define LUTHIER_TYPES_H

#include <hsa/hsa.h>

namespace luthier {

#define HOOK_HANDLE_PREFIX __luthier_hook_handle

#define RESERVED_MANAGED_VAR __luthier_reserved

#define STRINGIFY(S) #S

/// A macro to delay expansion of \c STRINGIFY
/// TODO: Is there a better way to include this?
#define X_STRINGIFY(S) STRINGIFY(S)

constexpr const char *HookHandlePrefix = X_STRINGIFY(HOOK_HANDLE_PREFIX);

constexpr const char *ReservedManagedVar = X_STRINGIFY(RESERVED_MANAGED_VAR);

#undef X_STRINGIFY

#undef STRINGIFY

typedef unsigned long address_t;

enum ApiEvtPhase : unsigned short {
  API_EVT_PHASE_ENTER = 0,
  API_EVT_PHASE_EXIT = 1
};

enum InstrPoint : unsigned short {
  INSTR_POINT_BEFORE = 0,
  INSTR_POINT_AFTER = 1
};

typedef struct {
  uint16_t Header;
  uint8_t Format;
  uint8_t Rest[61];
} hsa_amd_vendor_packet_t;

/**
 * POD struct to provide an abstraction over HSA AQL packets and some
 * convenience methods
 * This should not be constructed directly. It should be constructed using
 * reinterpret_cast over the address of a packet:
 * \code
 * auto& Packet = *reinterpret_cast<HsaAqlPacket*>(PacketAddress);
 * \endcode
 */
struct HsaAqlPacket {
  struct {
    uint16_t Header;
    struct {
      uint8_t UserData[62];
    } Body;
  } Packet;

  /**
   * \returns the HSA packet type of the AQL packet
   */
  [[nodiscard]] hsa_packet_type_t getPacketType() const {
    return static_cast<hsa_packet_type_t>(
        (Packet.Header >> HSA_PACKET_HEADER_TYPE) &
        ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
  }
  /*
   *
   */
  [[nodiscard]] hsa_amd_vendor_packet_t &asAMDVendor() {
    return *reinterpret_cast<hsa_amd_vendor_packet_t *>(&Packet);
  }
  [[nodiscard]] const hsa_amd_vendor_packet_t &asAMDVendor() const {
    return *reinterpret_cast<const hsa_amd_vendor_packet_t *>(&Packet);
  }
  [[nodiscard]] hsa_kernel_dispatch_packet_t &asKernelDispatch() {
    return *reinterpret_cast<hsa_kernel_dispatch_packet_t *>(&Packet);
  }
  [[nodiscard]] const hsa_kernel_dispatch_packet_t &asKernelDispatch() const {
    return *reinterpret_cast<const hsa_kernel_dispatch_packet_t *>(&Packet);
  }

  [[nodiscard]] hsa_barrier_and_packet_t &asBarrierAnd() {
    return *reinterpret_cast<hsa_barrier_and_packet_t *>(&Packet);
  }

  [[nodiscard]] const hsa_barrier_and_packet_t &asBarrierAnd() const {
    return *reinterpret_cast<const hsa_barrier_and_packet_t *>(&Packet);
  }

  [[nodiscard]] hsa_barrier_or_packet_t &asBarrierOr() {
    return *reinterpret_cast<hsa_barrier_or_packet_t *>(&Packet);
  }

  [[nodiscard]] const hsa_barrier_or_packet_t &asBarrierOr() const {
    return *reinterpret_cast<const hsa_barrier_or_packet_t *>(&Packet);
  }

  [[nodiscard]] hsa_agent_dispatch_packet_t &asAgentDispatch() {
    return *reinterpret_cast<hsa_agent_dispatch_packet_t *>(&Packet);
  }

  [[nodiscard]] const hsa_agent_dispatch_packet_t &asAgentDispatch() const {
    return *reinterpret_cast<const hsa_agent_dispatch_packet_t *>(&Packet);
  }
};
namespace hsa {

enum SymbolKind : uint8_t { VARIABLE = 0, KERNEL = 1, DEVICE_FUNCTION = 2 };

} // namespace hsa
} // namespace luthier

#endif
