//===-- CustomKernargLayout.h -----------------------------------*- C++ -*-===//
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
///
/// \file
/// Describes the layout of the Luthier-managed *custom kernel argument buffer*
/// used by instrumented kernels, and the ABI of the \c .luthier.kernarg_layout
/// ELF section the patcher emits and the device tool code loader consumes.
///
/// Instead of reusing the application's kernarg buffer, an instrumented kernel
/// that needs instrumentation arguments is launched with a Luthier-managed
/// buffer pointed at by the AQL dispatch packet's \c kernarg_address. Its
/// layout (offsets relative to the buffer base, which the kernel sees as the
/// preloaded
/// \c KERNARG_SEGMENT_PTR and which the prologue saves into the \c USER_ARG_PTR
/// SVA) is:
///
/// \verbatim
///   [0 .. 7)                : uint64 original_kernarg_ptr  (the app's buffer)
///   [ExplicitOffset .. +E)  : tool-declared explicit args  (filled at launch)
///   [ImplicitOffset .. +256): ROCclr COV5 implicit/hidden args (filled
///   at launch)
/// \endverbatim
///
/// \c OrigKernargPtrOffset is fixed at 0 and \c ExplicitOffset at 8.
/// \c luthier::userArgPtr() returns <tt>base + ExplicitOffset</tt> (the
/// explicit region) and \c luthier::implicitArgPtr() returns <tt>base +
/// IMPLICIT_ARG_OFFSET</tt> with the \c IMPLICIT_ARG_OFFSET SVA set to
/// \c ImplicitOffset.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_CUSTOM_KERNARG_LAYOUT_H
#define LUTHIER_TOOL_CODE_GEN_CUSTOM_KERNARG_LAYOUT_H
#include <cstdint>

namespace luthier {

/// Name of the ELF section that carries the per-object \c CustomKernargLayout.
inline constexpr const char *CustomKernargLayoutSectionName =
    ".luthier.kernarg_layout";

/// Name of the global variable anchoring that section in the emitted module.
inline constexpr const char *CustomKernargLayoutGlobalName =
    "__luthier_kernarg_layout";

/// Byte offsets, relative to the start of the COV5 implicit-args block, of each
/// hidden argument. Sourced from LLVM's
/// \c AMDGPU::ImplicitArg::Offset_COV5 (SIDefines.h) and
/// \c AMDGPULowerKernelAttributes.cpp, cross-checked against emitted
/// \c amdhsa.kernels metadata. The block is 256 bytes total.
namespace cov5 {
inline constexpr uint32_t ImplicitArgsBlockSize = 256;

inline constexpr uint32_t BlockCountX = 0;        // u32
inline constexpr uint32_t BlockCountY = 4;        // u32
inline constexpr uint32_t BlockCountZ = 8;        // u32
inline constexpr uint32_t GroupSizeX = 12;        // u16
inline constexpr uint32_t GroupSizeY = 14;        // u16
inline constexpr uint32_t GroupSizeZ = 16;        // u16
inline constexpr uint32_t RemainderX = 18;        // u16
inline constexpr uint32_t RemainderY = 20;        // u16
inline constexpr uint32_t RemainderZ = 22;        // u16
inline constexpr uint32_t GlobalOffsetX = 40;     // u64
inline constexpr uint32_t GlobalOffsetY = 48;     // u64
inline constexpr uint32_t GlobalOffsetZ = 56;     // u64
inline constexpr uint32_t GridDims = 64;          // u16
inline constexpr uint32_t HostcallPtr = 80;       // u64
inline constexpr uint32_t MultigridSyncArg = 88;  // u64
inline constexpr uint32_t HeapV1 = 96;            // u64
inline constexpr uint32_t DefaultQueue = 104;     // u64
inline constexpr uint32_t CompletionAction = 112; // u64
inline constexpr uint32_t PrivateBase = 192;      // u32
inline constexpr uint32_t SharedBase = 196;       // u32
inline constexpr uint32_t QueuePtr = 200;         // u64
} // namespace cov5

/// Fixed offset of the original-kernarg-pointer slot within the custom buffer.
inline constexpr uint32_t CustomKernargOrigPtrOffset = 0;
/// Fixed offset of the tool's explicit-arg region within the custom buffer.
inline constexpr uint32_t CustomKernargExplicitOffset = 8;

/// \brief Self-describing record serialized verbatim into the
/// \c .luthier.kernarg_layout section.
///
/// One record per instrumented code object (the loader enforces exactly one
/// kernel per object). All fields are little-endian; the struct is a fixed-size
/// POD so emitter and loader (de)serialize it with a plain copy.
struct CustomKernargLayout {
  /// Magic = "LKAL" (0x4C414B4C little-endian as bytes 'L','K','A','L').
  uint32_t Magic;
  /// Format version.
  uint32_t Version;
  /// Total size of the custom kernarg buffer in bytes.
  uint32_t TotalSize;
  /// Required alignment of the custom kernarg buffer in bytes.
  uint32_t Align;
  /// Offset of the original-kernarg-ptr slot (always 0).
  uint32_t OrigKernargPtrOffset;
  /// Offset of the explicit-arg region (always 8).
  uint32_t ExplicitOffset;
  /// Size in bytes of the tool's explicit-arg region (may be 0).
  uint32_t ExplicitSize;
  /// Offset of the COV5 implicit-args block (= IMPLICIT_ARG_OFFSET SVA value).
  uint32_t ImplicitOffset;
  /// Size in bytes of the COV5 implicit-args block (= 256).
  uint32_t ImplicitSize;
};

inline constexpr uint32_t CustomKernargLayoutMagic = 0x4C414B4Cu; // 'LKAL'
inline constexpr uint32_t CustomKernargLayoutVersion = 1u;

/// Round \p Value up to the next multiple of \p Align (a power of two).
inline constexpr uint32_t alignUpTo(uint32_t Value, uint32_t Align) {
  return (Value + Align - 1u) & ~(Align - 1u);
}

/// \brief Compute the custom kernarg buffer layout for a kernel whose
/// instrumentation declares an explicit-arg region of \p ExplicitSize bytes
/// with
/// \p ExplicitAlign alignment (both 0 / 1 when there are no explicit args).
inline CustomKernargLayout
computeCustomKernargLayout(uint32_t ExplicitSize, uint32_t ExplicitAlign = 8) {
  CustomKernargLayout L{};
  L.Magic = CustomKernargLayoutMagic;
  L.Version = CustomKernargLayoutVersion;
  L.OrigKernargPtrOffset = CustomKernargOrigPtrOffset;
  L.ExplicitOffset = CustomKernargExplicitOffset;
  L.ExplicitSize = ExplicitSize;
  // The COV5 implicit block must start 8-aligned; honor a larger explicit-arg
  // alignment if requested.
  const uint32_t ImplAlign = ExplicitAlign < 8u ? 8u : ExplicitAlign;
  L.ImplicitOffset =
      alignUpTo(CustomKernargExplicitOffset + ExplicitSize, ImplAlign);
  L.ImplicitSize = cov5::ImplicitArgsBlockSize;
  L.TotalSize = L.ImplicitOffset + L.ImplicitSize;
  L.Align = ImplAlign;
  return L;
}

} // namespace luthier

#endif
