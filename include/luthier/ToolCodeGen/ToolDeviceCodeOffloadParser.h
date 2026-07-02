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

//===----------------------------------------------------------------------===//
/// Compiler annotations for the \c ToolDeviceCodeOffloadParser.
/// Note that these attributes can be used outside
/// \c ToolDeviceCodeOffloadParser, and the tool compiler plugin should honor
/// them.
//===----------------------------------------------------------------------===//

/// Attribute indicating that a static global pointer must be populated with the
/// starting address of the clang offload section (FAT binary) embedded
/// inside the current translation unit
#define LUTHIER_CLANG_OFFLOAD_SECTION_BEGIN luthier_clang_offload_section_begin

/// Attribute Indicating a static global pointer must be populated with the
/// ending address of the clang offload section (FAT binary) embedded inside
/// the current translation unit
#define LUTHIER_CLANG_OFFLOAD_SECTION_END luthier_clang_offload_section_end

/// Attribute indicating that a static global pointer must be populated with
/// the starting address of the HIP handle section (see \c HipHandleInfo)
#define LUTHIER_HIP_HANDLE_SECTION_BEGIN luthier_hip_handle_section_begin

/// Attribute indicating that a static global pointer must be populated with
/// the ending address of the HIP handle section (see \c HipHandleInfo)
#define LUTHIER_HIP_HANDLE_SECTION_END luthier_hip_handle_section_end

static constexpr llvm::StringLiteral OffloadSectionBeginAnnotation{
    LUTHIER_STRINGIFY(LUTHIER_CLANG_OFFLOAD_SECTION_BEGIN)};

static constexpr llvm::StringLiteral OffloadSectionEndAnnotation{
    LUTHIER_STRINGIFY(LUTHIER_CLANG_OFFLOAD_SECTION_END)};

static constexpr llvm::StringLiteral HipHandleSectionBeginAnnotation{
    LUTHIER_STRINGIFY(LUTHIER_HIP_HANDLE_SECTION_BEGIN)};

static constexpr llvm::StringLiteral HipHandleSectionEndAnnotation{
    LUTHIER_STRINGIFY(LUTHIER_HIP_HANDLE_SECTION_END)};

/// A single handle info entry; Holds the \c void * HIP shadow host handle of
/// the globals in the HIP code, plus its associated name.
struct HipHandleInfo {
  void *HostHandle{nullptr};
  const char *DeviceName{nullptr};
};

/// \brief Base class for \c ToolDeviceCodeOffloadParserTrait; Contains all
/// logic for the trait that doesn't require a \c static field
class ToolDeviceCodeOffloadParser : public ToolDeviceCodeParser {
  /// Mapping between \b ALL <tt>void *</tt> host shadow handles from the
  /// globals in the device logic to their names.
  llvm::DenseMap<const void *, llvm::StringRef> HandleToName;

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

  /// Sentinel \c __managed__ variable that forces Clang to emit "host-visible"
  /// device code in the derived parser's TU
  static char DeviceCodeMarker;

  static const char *FatBinarySectionBegin;

  static const char *FatBinarySectionEnd;

  static HipHandleInfo *HipHandleSectionBegin;

  static HipHandleInfo *HipHandleSectionEnd;

public:
  explicit ToolDeviceCodeOffloadParserTrait(llvm::Error &Err)
      : ToolDeviceCodeOffloadParser(
            llvm::MemoryBufferRef{
                llvm::StringRef{FatBinarySectionBegin,
                                static_cast<uint64_t>(FatBinarySectionEnd -
                                                      FatBinarySectionBegin)},
                ""},
            llvm::ArrayRef<HipHandleInfo>{HipHandleSectionBegin,
                                          HipHandleSectionEnd},
            Err) {}
};

#define LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(DERIVED)                    \
  __attribute__((managed, used)) char ::luthier::                              \
      ToolDeviceCodeOffloadParserTrait<DERIVED>::DeviceCodeMarker;             \
  __attribute__((used, annotate(LUTHIER_CLANG_OFFLOAD_SECTION_BEGIN)))         \
  const char * ::luthier::ToolDeviceCodeOffloadParserTrait<                    \
      DERIVED>::FatBinarySectionBegin;                                         \
  __attribute__((used, annotate(LUTHIER_CLANG_OFFLOAD_SECTION_END)))           \
  const char * ::luthier::ToolDeviceCodeOffloadParserTrait<                    \
      DERIVED>::FatBinarySectionEnd;                                           \
  __attribute__((used, annotate(LUTHIER_HIP_HANDLE_SECTION_BEGIN)))            \
  const HipHandleInfo * ::luthier::ToolDeviceCodeOffloadParserTrait<           \
      DERIVED>::HipHandleSectionBegin;                                         \
  __attribute__((used, annotate(LUTHIER_HIP_HANDLE_SECTION_END)))              \
  const HipHandleInfo * ::luthier::ToolDeviceCodeOffloadParserTrait<           \
      DERIVED>::HipHandleSectionEnd;

} // namespace luthier

#endif // LUTHIER_TOOLING_DEVICE_TOOL_CODE_FAT_BINARY_PARSER_H
