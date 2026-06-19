//===-- DeviceToolCodeFatBinaryParser.h --------------------------*- C++-*-===//
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
/// \file
/// Defines \c DeviceToolCodeFatBinaryParser, a CRTP class that allows embedding
/// the HIP device logic of the derived as an offload bundle inside the
/// derived TU's host object.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_DEVICE_TOOL_CODE_FAT_BINARY_PARSER_H
#define LUTHIER_TOOLING_DEVICE_TOOL_CODE_FAT_BINARY_PARSER_H

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSATooling/DeviceToolCodeParser.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace luthier {

/// The IR pass for \c DeviceToolCodeFatBinaryParser writes a \c { ptr, i64 }
/// struct constant into each placeholder slot, matching \c llvm::ArrayRef<T>'s
/// ABI. If LLVM ever rearranges \c ArrayRef's members, these asserts trip at
/// compile time and the pass needs to be updated in lockstep.
static_assert(sizeof(llvm::ArrayRef<void *>) ==
                  sizeof(void *) + sizeof(uint64_t),
              "llvm::ArrayRef ABI changed: expected { ptr, i64 } layout "
              "matching the IR pass's ConstantStruct initializer.");
static_assert(alignof(llvm::ArrayRef<void *>) == alignof(void *),
              "llvm::ArrayRef alignment changed.");
static_assert(sizeof(decltype(std::declval<llvm::ArrayRef<void *>>().size())) ==
                  sizeof(uint64_t),
              "llvm::ArrayRef length is no longer 64-bit.");

/// \brief CRTP class that adds to the \c Derived its own set of \c inline
/// static annotated slots, which in turn gives \c Derived direct access to the
/// HIP FAT binary registration slots otherwise only accessible by the HIP
/// runtime. With the help of the \c luthier_create_offload_bundle cmake helper,
/// and the \c DeviceToolCodeParser base, this class ultimately give the
/// Derived's host logic seamless access to its device logic's LLVM IR at
/// runtime. This is very useful for writing instrumentation passes, allowing
/// the \c Derived to express complex device-side logic in the same HIP source
/// code that performs instrumentation as an LLVM pass on the host side.
///
/// A tool MUST be instantiated from exactly one host translation unit. Each TU
/// that ODR-uses the class emits a \c linkonce_odr definition of every
/// annotated slot (forced live by \c [[gnu::used]]).
template <typename Derived>
class DeviceToolCodeFatBinaryParser : public DeviceToolCodeParser {
private:
  /// Sentinel \c __managed__ variable that forces Clang to emit "host-visible"
  /// device code
  static inline __attribute__((managed, used)) char DeviceCodeMarker = 0;

protected:
  /// Mapping between the void * host shadow handles from the globals in the
  /// device logic to their names
  llvm::DenseMap<const void *, std::string> HandleToName;

public:
  /// \brief Per-fat-binary entry produced from a \c __hipRegisterFatBinary
  /// call. \c Bundle points at the raw Clang offload bundle; \c Size is the
  /// bundle's byte extent and might be zero if it was not provided at compile
  /// time.
  /// TODO: Remove the \c Size field
  struct HipFatBinaryInfo {
    const void *Bundle{nullptr};
    size_t Size{0};
  };

  /// Per-kernel entry produced from a \c __hipRegisterFunction call.
  /// (Despite HIP's API name, \c __hipRegisterFunction registers
  /// \c __global__ kernels — user-launchable entry points whose host
  /// shadow is what \c hipLaunchKernel takes as its first argument.)
  struct HipKernelInfo {
    void *HostHandle{nullptr};
    const char *DeviceName{nullptr};
  };

  /// Per-device-function entry produced by the Luthier CXX plugin's
  /// \c [[luthier::export_function_handle]] machinery: a synthesized
  /// \c __host__ sibling for a tagged \c __device__ function.
  struct HipDeviceFunctionInfo {
    void *HostHandle{nullptr};
    const char *DeviceName{nullptr};
  };

  /// Per-device-variable entry produced from a \c __hipRegisterVar call
  struct HipDeviceVarInfo {
    void *HostHandle{nullptr};
    const char *DeviceName{nullptr};
  };

  /// Per-managed-variable entry produced from a \c __hipRegisterManagedVar call
  struct HipManagedVarInfo {
    void **Pointer{nullptr};
    void *InitValue{nullptr};
    const char *Name{nullptr};
    unsigned long long Size{0};
    unsigned Align{0};
  };

  /// Per-texture entry produced from a \c __hipRegisterTexture call
  struct HipTextureInfo {
    void *HostHandle{nullptr};
    const char *DeviceName{nullptr};
  };

  /// Per-surface entry produced from a \c __hipRegisterSurface call
  struct HipSurfaceInfo {
    void *HostHandle{nullptr};
    const char *DeviceName{nullptr};
  };

  //===--------------------------------------------------------------------===//
  /// HIP's __Register* slots populated by \c LoadHIPFATBinaryInfoPass at
  /// host IR-compile time.
  //===--------------------------------------------------------------------===//
protected:
  inline static __attribute__((used)) LUTHIER_ANNOTATE_VARIABLE(
      LUTHIER_HIP_FAT_BINARIES_ATTR) HipFatBinaryInfo HipFatBinary{};

  inline static __attribute__((used)) LUTHIER_ANNOTATE_VARIABLE(
      LUTHIER_HIP_KERNELS_ATTR) llvm::ArrayRef<HipKernelInfo> HipKernels{};

  inline static __attribute__((used))
  LUTHIER_ANNOTATE_VARIABLE(LUTHIER_HIP_DEVICE_FUNCTIONS_ATTR)
      llvm::ArrayRef<HipDeviceFunctionInfo> HipDeviceFunctions{};

  inline static __attribute__((used))
  LUTHIER_ANNOTATE_VARIABLE(LUTHIER_HIP_DEVICE_VARS_ATTR)
      llvm::ArrayRef<HipDeviceVarInfo> HipDeviceVars{};

  inline static __attribute__((used))
  LUTHIER_ANNOTATE_VARIABLE(LUTHIER_HIP_MANAGED_VARS_ATTR)
      llvm::ArrayRef<HipManagedVarInfo> HipManagedVars{};

  inline static __attribute__((used))
  LUTHIER_ANNOTATE_VARIABLE(LUTHIER_HIP_TEXTURE_VARS_ATTR)
      llvm::ArrayRef<HipTextureInfo> HipTextureVars{};

  inline static __attribute__((used))
  LUTHIER_ANNOTATE_VARIABLE(LUTHIER_HIP_SURFACE_VARS_ATTR)
      llvm::ArrayRef<HipSurfaceInfo> HipSurfaceVars{};

  /// Build a non-owning \c MemoryBuffer wrapper around the fat binary recorded
  /// in \p Slot. Returns \c nullptr if \p Slot has no bundle (host-only code).
  /// A zero \c Size (the split-compile case, where the host TU
  /// sees \c __hip_fatbin as an opaque \c extern) is recovered from the bundle
  /// header via \c DeviceToolCodeParser::discoverBundleSize; sets \p Err if the
  /// size still can't be determined.
  static std::unique_ptr<llvm::MemoryBuffer>
  buildBundleBuffer(const HipFatBinaryInfo &Slot, llvm::Error &Err) {
    llvm::ErrorAsOutParameter EAO(&Err);
    if (Err)
      return nullptr;
    if (Slot.Bundle == nullptr)
      return nullptr;
    uint64_t Size = Slot.Size;
    if (Size == 0)
      Size = discoverBundleSize(Slot.Bundle);
    if (Size == 0) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(
          "Cannot determine fat-bin bundle size: the HipFatBinary slot "
          "recorded size 0 and the bundle header was not a recognized "
          "(uncompressed or CCOB) Clang offload bundle.");
      return nullptr;
    }
    return llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(static_cast<const char *>(Slot.Bundle), Size), "fatbin",
        /*RequiresNullTerminator=*/false);
  }

public:
  explicit DeviceToolCodeFatBinaryParser(llvm::Error &Err)
      : DeviceToolCodeParser(buildBundleBuffer(HipFatBinary, Err), Err) {
    /// Use of managed variable to ensure the host side of the tool gets a
    /// HIP FAT binary slot regardless of the device logic written in the
    /// tool
    (void)&DeviceCodeMarker;
    llvm::ErrorAsOutParameter EAO(&Err);
    if (Err)
      return;
    auto Record = [&](const void *Handle, const char *Name) {
      if (Handle != nullptr && Name != nullptr)
        HandleToName[Handle] = std::string(Name);
    };
    for (const auto &E : HipKernels)
      Record(E.HostHandle, E.DeviceName);
    for (const auto &E : HipDeviceFunctions)
      Record(E.HostHandle, E.DeviceName);
    for (const auto &E : HipDeviceVars)
      Record(E.HostHandle, E.DeviceName);
    for (const auto &E : HipTextureVars)
      Record(E.HostHandle, E.DeviceName);
    for (const auto &E : HipSurfaceVars)
      Record(E.HostHandle, E.DeviceName);
    for (const auto &MV : HipManagedVars)
      Record(MV.Pointer, MV.Name);
  }

  /// Resolve a HIP host shadow handle (the \c __hipRegister* host-side pointer,
  /// e.g. \c &MyTool::MyDeviceVar) to its device-side symbol name
  template <typename T>
  llvm::Expected<llvm::StringRef> lookupNameByHandle(T *Handle) {
    std::lock_guard Lock(Mutex);
    auto It = HandleToName.find(reinterpret_cast<const void *>(Handle));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        It != HandleToName.end(),
        "No device-side symbol registered for the given host handle."));
    return llvm::StringRef{It->second};
  }
};

} // namespace luthier

#endif // LUTHIER_TOOLING_DEVICE_TOOL_CODE_FAT_BINARY_PARSER_H
