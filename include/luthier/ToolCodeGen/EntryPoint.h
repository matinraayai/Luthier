//===-- EntryPoint.h ---------------------------------------------*- C++-*-===//
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
/// \file EntryPoint.h
/// Describes the \c EntryPoint class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_ENTRY_POINT_H
#define LUTHIER_TOOL_CODE_GEN_ENTRY_POINT_H
#include <cassert>
#include <llvm/ADT/DenseMapInfo.h>
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

  std::variant<const llvm::amdhsa::kernel_descriptor_t *, uint64_t> EP;

public:
  explicit EntryPoint(const llvm::amdhsa::kernel_descriptor_t &KD) : EP(&KD) {};

  explicit EntryPoint(uint64_t DeviceAddress) : EP(DeviceAddress) {};

  EntryPoint() : EP(0U) {}

  /// \returns \c true if the entry point is a kernel, \c false otherwise
  [[nodiscard]] bool isKernel() const {
    return std::holds_alternative<const llvm::amdhsa::kernel_descriptor_t *>(
        EP);
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
      return std::get<const llvm::amdhsa::kernel_descriptor_t *>(EP);
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

  [[nodiscard]] uint64_t getRawAddress() const {
    if (auto *KD = getKernelDescriptor()) {
      return reinterpret_cast<uint64_t>(KD);
    } else {
      return std::get<uint64_t>(EP);
    }
  }

  bool operator==(const EntryPoint &Other) const { return EP == Other.EP; }
};

} // namespace luthier

template <> struct llvm::DenseMapInfo<luthier::EntryPoint> {
  static luthier::EntryPoint getEmptyKey() {
    return luthier::EntryPoint(DenseMapInfo<uint64_t>::getEmptyKey());
  }

  static unsigned getHashValue(const luthier::EntryPoint &EP) {
    if (const amdhsa::kernel_descriptor_t *KD = EP.getKernelDescriptor()) {
      return DenseMapInfo<llvm::amdhsa::kernel_descriptor_t *>::getHashValue(
          KD);
    } else {
      return DenseMapInfo<uint64_t>::getHashValue(EP.getEntryPointAddress());
    }
  }

  static bool isEqual(const luthier::EntryPoint &Lhs,
                      const luthier::EntryPoint &Rhs) {
    return (Lhs.getEntryPointAddress() == Rhs.getEntryPointAddress()) &&
           (Lhs.isKernel() == Rhs.isKernel());
  }
};

#endif