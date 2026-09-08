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

  /// Size of the kernarg-preload backward-compatibility prologue that
  /// \c AMDGPUPreloadKernArgProlog emits ahead of a preloading kernel's real
  /// entry point.
  ///
  /// A kernel that asks the CP to preload kernargs is emitted with *two* entry
  /// points. The one named by \c kernel_code_entry_byte_offset is a
  /// compatibility block that re-materializes the preloaded kernargs with
  /// \c s_load instructions (for firmware that doesn't implement preloading)
  /// and then branches to the real entry; it is padded out to this size so
  /// that firmware which *does* implement preloading can skip it by starting
  /// the wave exactly this many bytes further in.
  ///
  /// See \c llvm/lib/Target/AMDGPU/AMDGPUPreloadKernArgProlog.cpp.
  static constexpr uint64_t KernargPreloadBackCompatPrologSize = 256;

  /// \returns the number of kernarg dwords this entry point's kernel asks the
  /// CP to preload into SGPRs; 0 if this is not a kernel, or if the kernel
  /// does not use kernarg preloading
  [[nodiscard]] unsigned getKernargPreloadLength() const {
    const auto *KD = getKernelDescriptor();
    if (!KD)
      return 0;
    return AMDHSA_BITS_GET(KD->kernarg_preload,
                           llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH);
  }

  /// \returns the address the wave actually begins executing at: the device
  /// address if this entry point is one, otherwise the start of the kernel's
  /// code as derived from its descriptor
  ///
  /// \param TargetNeedsKernargPreloadProlog whether the subtarget emits the
  /// backward-compatibility prologue for preloading kernels — i.e.
  /// \c GCNSubtarget::needsKernArgPreloadProlog(), true for every target with
  /// the kernarg-preload feature except GFX1250+. Defaults to \c true, which
  /// is correct for every such target; pass the subtarget's answer explicitly
  /// when it is available.
  ///
  /// For a preloading kernel the code does *not* start at
  /// \c kernel_code_entry_byte_offset — that names the compatibility prologue,
  /// which the CP skips (see \c KernargPreloadBackCompatPrologSize). Reporting
  /// the prologue as the kernel's start makes the lifted entry block the
  /// compatibility block, whose \c s_load instructions *define* the preload
  /// SGPRs and so make the kernel's live-in kernargs look dead at its entry.
  [[nodiscard]] uint64_t
  getEntryPointAddress(bool TargetNeedsKernargPreloadProlog = true) const {
    if (isDeviceAddress())
      return std::get<uint64_t>(EP);

    const auto *KD = getKernelDescriptor();
    const auto KDAddress = reinterpret_cast<uint64_t>(KD);

    /// \c kernel_code_entry_byte_offset is a *signed* displacement from the
    /// descriptor's own address — the code may sit either after (positive) or
    /// before (negative) the descriptor — so it has to be applied with signed
    /// arithmetic rather than branching on its sign.
    uint64_t Addr = KDAddress + static_cast<uint64_t>(static_cast<int64_t>(
                                    KD->kernel_code_entry_byte_offset));

    if (TargetNeedsKernargPreloadProlog && getKernargPreloadLength() != 0)
      Addr += KernargPreloadBackCompatPrologSize;

    return Addr;
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