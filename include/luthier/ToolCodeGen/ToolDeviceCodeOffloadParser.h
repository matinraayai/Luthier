//===-- ToolDeviceCodeOffloadParser.h --------------------------*- C++-*-===//
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
/// Defines \c ToolDeviceCodeOffloadParser and its associated CRTP class, which,
/// with the help of its companion IR pass \c ToolDeviceCodeOffloadParserPass,
/// allows the derived to have access to its defining translation unit (TU)'s
/// offload bundle and device code shadow host handles.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODEGEN_TOOL_DEVICE_CODE_OFFLOAD_PARSER_H
#define LUTHIER_TOOL_CODEGEN_TOOL_DEVICE_CODE_OFFLOAD_PARSER_H

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeParser.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <mutex>
#include <utility>

namespace luthier {
/// \brief Base class for \c ToolDeviceCodeOffloadParserTrait; Contains all
/// logic for the trait that doesn't require a \c static field
class ToolDeviceCodeOffloadParser : public ToolDeviceCodeParser {
  /// Mapping between \b ALL <tt>void *</tt> host shadow handles from the
  /// globals in the device logic to their names.
  llvm::DenseMap<const void *, llvm::StringRef> HandleToName;

public:
  /// Struct that holds the \c void * HIP shadow host handle of the globals in
  /// the HIP code and its associated variable name
  struct HipHandleInfo {
    void *HostHandle{nullptr};
    const char *DeviceName{nullptr};
  };

protected:
  ToolDeviceCodeOffloadParser(llvm::MemoryBufferRef Bundle,
                              llvm::ArrayRef<HipHandleInfo> HipHandles,
                              llvm::Error &Err)
      : ToolDeviceCodeParser(Bundle, Err) {
    for (const auto &E : HipHandles)
      if (E.HostHandle != nullptr && E.DeviceName != nullptr)
        HandleToName[E.HostHandle] = E.DeviceName;
  }

public:
  /// Resolve a HIP host shadow handle (the \c __hipRegister* host-side pointer,
  /// e.g. \c &MyTool::MyDeviceVar) to its device-side symbol name
  template <typename T>
  llvm::Expected<llvm::StringRef> lookupHandleName(T *Handle) {
    std::lock_guard Lock(Mutex);
    auto It = HandleToName.find(reinterpret_cast<const void *>(Handle));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        It != HandleToName.end(),
        "No device-side symbol registered for the given host handle."));
    return llvm::StringRef{It->second};
  }
};

/// \brief CRTP class that adds to the \c Derived its own set of static slots,
/// which in turn gives \c Derived direct access to the HIP FAT binary
/// registration slots otherwise only accessible by the HIP runtime. With the
/// help of the \c luthier_create_offload_bundle cmake helper, and the \c
/// DeviceToolCodeParser base, this class ultimately give the Derived's host
/// logic seamless access to its device logic's LLVM IR at runtime. This is very
/// useful for writing instrumentation passes, allowing the \c Derived to
/// express complex device-side logic in the same HIP source code that performs
/// instrumentation as an LLVM pass on the host side.
///
/// A tool MUST be instantiated from exactly one host translation unit. Each TU
/// that ODR-uses the class emits a \c linkonce_odr definition of every
/// annotated slot (forced live by \c [[gnu::used]]).
template <typename Derived>
class ToolDeviceCodeOffloadParserTrait : public ToolDeviceCodeOffloadParser {
  /// The IR pass for \c ToolDeviceCodeOffloadParser writes a \c { ptr, i64 }
  /// struct constant into each placeholder slot, matching \c
  /// llvm::ArrayRef<T>'s ABI. If LLVM ever rearranges \c ArrayRef's members,
  /// these asserts trip at compile time and the pass needs to be updated in
  /// lockstep.
  static_assert(sizeof(llvm::ArrayRef<void *>) ==
                    sizeof(void *) + sizeof(uint64_t),
                "llvm::ArrayRef ABI changed: expected { ptr, i64 } layout "
                "matching the IR pass's ConstantStruct initializer.");
  static_assert(alignof(llvm::ArrayRef<void *>) == alignof(void *),
                "llvm::ArrayRef alignment changed.");
  static_assert(
      sizeof(decltype(std::declval<llvm::ArrayRef<void *>>().size())) ==
          sizeof(uint64_t),
      "llvm::ArrayRef length is no longer 64-bit.");

  /// Sentinel \c __managed__ variable that forces Clang to emit "host-visible"
  /// device code in the derived parser's TU
  static __attribute__((managed)) char DeviceCodeMarker;

  //===--------------------------------------------------------------------===//
  /// Slots populated by \c ToolDeviceCodeOffloadParserPass at host IR-compile
  /// time.
  //===--------------------------------------------------------------------===//

  /// Start and boundary (one-past-the-end) load addresses of the embedded tool
  /// fat binary. The pass references the linker's \c luthier_fatbin
  /// section-boundary symbols (\c __start_luthier_fatbin /
  /// \c __stop_luthier_fatbin) and stores their addresses into these slots.
  static const char *FatBinaryStart;
  static const char *FatBinaryStop;

  static llvm::ArrayRef<HipHandleInfo> HipHandles;

public:
  explicit ToolDeviceCodeOffloadParserTrait(llvm::Error &Err)
      : ToolDeviceCodeOffloadParser(
            llvm::MemoryBufferRef{
                llvm::StringRef{
                    FatBinaryStart,
                    static_cast<uint64_t>(FatBinaryStop - FatBinaryStart)},
                ""},
            HipHandles, Err) {
    /// Spurious use of the \c DeviceCodeMarker to force emission of
    /// registration functions in the host code.
    (void)&DeviceCodeMarker;
  }
};

#define LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(DERIVED)                    \
  __attribute__((managed, used)) char ::luthier::                              \
      ToolDeviceCodeOffloadParserTrait<DERIVED>::DeviceCodeMarker;             \
  __attribute__((used)) const char                                             \
      * ::luthier::ToolDeviceCodeOffloadParserTrait<DERIVED>::FatBinaryStart;  \
  __attribute__((used)) const char                                             \
      * ::luthier::ToolDeviceCodeOffloadParserTrait<DERIVED>::FatBinaryStop;   \
  __attribute__((used)) llvm::ArrayRef<HipHandleInfo>::luthier::               \
      ToolDeviceCodeOffloadParserTrait<DERIVED>::HipHandles;

} // namespace luthier

#endif // LUTHIER_TOOLING_DEVICE_TOOL_CODE_FAT_BINARY_PARSER_H
