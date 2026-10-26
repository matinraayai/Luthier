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
#include <luthier/ToolCodeGen/FunctionAnnotations.h>
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
  /// Looks up the name associated with the HIP host shadow \p Handle registered
  /// with this parser; An empty \c llvm::StringRef is returned if the \p
  /// Handle is not managed by the parser
  template <typename T> llvm::StringRef lookupHandleName(T *Handle) const {
    std::lock_guard Lock(Mutex);
    const auto It = HandleToName.find(reinterpret_cast<const void *>(Handle));
    return It != HandleToName.end() ? llvm::StringRef{It->second}
                                    : llvm::StringRef{};
  }

  /// Looks up the LLVM Named Global handle associated with the \p Handle inside
  /// the \c llvm::Module \p M; If found, also casts it to \c NamedGlobalTy.
  /// \c nullptr is returned if the \p Handle is not managed by the parser,
  /// if \p M doesn't have any named globals associated with the \p Handle's
  /// name, or if the type of the named global found in \p M doesn't match
  /// \c NamedGlobalTy
  template <typename NamedGlobalTy, typename HandleTy>
  NamedGlobalTy *getLLVMHandle(const llvm::Module &M, HandleTy *Handle) const {
    llvm::StringRef HandleName = lookupHandleName(Handle);
    return HandleName.empty() ? nullptr
                              : llvm::dyn_cast_or_null<NamedGlobalTy>(
                                    M.getNamedGlobal(HandleName));
  }

  template <typename NamedGlobalTy, typename HandleTy>
  NamedGlobalTy *getLLVMHandle(llvm::Module &M, HandleTy *Handle) const {
    llvm::StringRef HandleName = lookupHandleName(Handle);
    return HandleName.empty() ? nullptr
                              : llvm::dyn_cast_or_null<NamedGlobalTy>(
                                    M.getNamedGlobal(HandleName));
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
template <typename Derived>
class ToolDeviceCodeOffloadParserTrait : public ToolDeviceCodeOffloadParser {

  /// Sentinel \c __managed__ variable that forces Clang to emit "host-visible"
  /// device code in the der=ived parser's TU
  static char DeviceCodeMarker;

  static const char *FatBinarySectionBegin;

  static const char *FatBinarySectionEnd;

  static const HipHandleInfo *HipHandleSectionBegin;

  static const HipHandleInfo *HipHandleSectionEnd;

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

/// Defines the static fields of \c ToolDeviceCodeOffloadParserTrait's static
/// fields in the target translation unit, which in turn should give it access
/// to the TU's FAT binary and its host shadow handles
#define LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(DERIVED)                    \
  template <>                                                                  \
  __attribute__((managed, used)) char(::luthier::                             \
      ToolDeviceCodeOffloadParserTrait<DERIVED>::DeviceCodeMarker){};          \
  template <>                                                                  \
  __attribute__((                                                              \
      used, annotate(LUTHIER_STRINGIFY(LUTHIER_CLANG_OFFLOAD_SECTION_BEGIN)))) \
  const char * ::luthier::ToolDeviceCodeOffloadParserTrait<                    \
      DERIVED>::FatBinarySectionBegin{nullptr};                                \
  template <>                                                                  \
  __attribute__((                                                              \
      used, annotate(LUTHIER_STRINGIFY(LUTHIER_CLANG_OFFLOAD_SECTION_END))))   \
  const char * ::luthier::ToolDeviceCodeOffloadParserTrait<                    \
      DERIVED>::FatBinarySectionEnd{nullptr};                                  \
  template <>                                                                  \
  __attribute__((                                                              \
      used, annotate(LUTHIER_STRINGIFY(LUTHIER_HIP_HANDLE_SECTION_BEGIN))))    \
  const ::luthier::HipHandleInfo                                               \
      * ::luthier::ToolDeviceCodeOffloadParserTrait<                           \
          DERIVED>::HipHandleSectionBegin{nullptr};                            \
  template <>                                                                  \
  __attribute__((used,                                                         \
                 annotate(LUTHIER_STRINGIFY(LUTHIER_HIP_HANDLE_SECTION_END)))) \
  const ::luthier::HipHandleInfo * ::luthier::                                 \
      ToolDeviceCodeOffloadParserTrait<DERIVED>::HipHandleSectionEnd{nullptr};

} // namespace luthier

#endif // LUTHIER_TOOLING_DEVICE_TOOL_CODE_FAT_BINARY_PARSER_H
