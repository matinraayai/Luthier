
#ifndef LUTHIER_TOOLING_ENTRY_POINT_H
#define LUTHIER_TOOLING_ENTRY_POINT_H
#include <bits/refwrap.h>
#include <cassert>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <variant>

namespace luthier {

/// \brief Represents the different entry point types in the lifting passes
/// \details An entry point is  that is reached
/// different type of entry points encountered during the
/// code discovery pass. An entry point can either be the kernel descriptor
/// that's about to get launched, or a device address reached by the code
/// via an indirect jump or call
class EntryPoint {
  using KernelRefWrapperType =
      std::reference_wrapper<const llvm::amdhsa::kernel_descriptor_t>;

  std::variant<KernelRefWrapperType, uint64_t> EP;

public:
  explicit EntryPoint(const llvm::amdhsa::kernel_descriptor_t &KD) : EP(KD) {};

  explicit EntryPoint(const KernelRefWrapperType &K) : EP(K) {};

  explicit EntryPoint(uint64_t DeviceAddress) : EP(DeviceAddress) {};

  /// \returns \c true if the entry point is a kernel, \c false otherwise
  [[nodiscard]] bool isKernel() const {
    return std::holds_alternative<
        std::reference_wrapper<const llvm::amdhsa::kernel_descriptor_t>>(EP);
  }

  /// \returns \c true if the entry point is a device address, \c false
  /// otherwise
  [[nodiscard]] bool isDeviceAddress() const {
    return std::holds_alternative<uint64_t>(EP);
  }

  /// \returns if the entry point is a kernel, returns a pointer to the kernel
  /// descriptor; \c nullptr otherwise
  [[nodiscard]] const llvm::amdhsa::kernel_descriptor_t *
  getKernelDescriptor() const {
    if (isKernel()) {
      return &std::get<KernelRefWrapperType>(EP).get();
    }
    return nullptr;
  }

  /// \returns the kernel descriptor's entry address if the entry point is a
  /// kernel descriptor, otherwise the device address
  [[nodiscard]] uint64_t getEntryPointAddress() const {
    if (isDeviceAddress()) {
      return std::get<uint64_t>(EP);
    } else {
      const auto *KD = getKernelDescriptor();

      const auto KDAddress = reinterpret_cast<uint64_t>(KD);
      const auto ByteOffset =
          static_cast<uint64_t>(KD->kernel_code_entry_byte_offset);

      assert(KDAddress > ByteOffset &&
             "kernel descriptor's entry byte offset is greater than its base "
             "address");
      return KD->kernel_code_entry_byte_offset > 0 ? KDAddress + ByteOffset
                                                   : KDAddress - ByteOffset;
    }
  }
};

} // namespace luthier

#endif