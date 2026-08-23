//===-- Intrinsics.h - Luthier's built-in Intrinsics ------------*- C++ -*-===//
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
/// This file describes utilities to write device code bindings to Luthier
/// intrinsics, as well as a set of bindings to Luthier's built-in intrinsics.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_INTRINSICS_H
#define LUTHIER_INTRINSIC_INTRINSICS_H
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <llvm/MC/MCRegister.h>
#include <type_traits>

namespace luthier {

/// \brief All bindings to Luthier intrinsics must be annotated using this macro
/// \details This macro defines the binding as a device function, adds a
/// noinline attribute as  well as a \c LUTHIER_INTRINSIC_ATTRIBUTE attribute
/// to be recognized by Luthier as an intrinsic
#define LUTHIER_INTRINSIC_ANNOTATE                                             \
  extern __attribute__((                                                       \
      device, annotate(LUTHIER_STRINGIFY(LUTHIER_INTRINSIC_ATTRIBUTE))))

#if defined(__HIPCC__)

namespace detail {

/// \brief Type trait admitting any scalar type that can be held in one or more
/// AMD GPU general-purpose registers:
///   - integral types (\c bool, \c int8_t / \c uint8_t, ..., \c int64_t /
///     \c uint64_t) covered by \c std::is_integral_v
///   - floating point types (\c float, \c double, and — where recognized by
///     the standard library — \c _Float16) covered by \c
///     std::is_floating_point_v
///   - \c _Float16 and \c __bf16 half-precision extensions used by HIP fp16 /
///     bf16 headers (these are not always caught by \c
///     std::is_floating_point_v)
///   - the AMDGPU built-in opaque type \c __amdgpu_buffer_rsrc_t, which is a
///     128-bit buffer resource descriptor commonly held in an SGPR quad
template <typename T>
struct is_amdgpu_register_compatible
    : std::integral_constant<
          bool, std::is_integral_v<T> || std::is_floating_point_v<T> ||
                    std::is_same_v<T, _Float16> || std::is_same_v<T, __bf16> ||
                    std::is_same_v<T, __amdgpu_buffer_rsrc_t>> {};

template <typename T>
inline constexpr bool is_amdgpu_register_compatible_v =
    is_amdgpu_register_compatible<T>::value;

} // namespace detail

/// \brief Intrinsic to read the value of a register
/// \details The readReg intrinsic reads the value of the \p Reg and returns it
/// \tparam T the return type of the output; Must be a type that can be held
/// in one or more AMD GPU registers (see
/// \c detail::is_amdgpu_register_compatible): an integral type, a floating
/// point type (including \c _Float16 / \c __bf16), or the AMDGPU built-in
/// opaque type \c __amdgpu_buffer_rsrc_t. The size of \p T must match the
/// size of \p Reg (for example, reading \c llvm::AMDGPU::SGPR4_SGPR5
/// requires a 64-bit \p T such as \c uint64_t or \c double; reading a 128-bit
/// SGPR quad can produce an \c __amdgpu_buffer_rsrc_t).
/// \param Reg the ID of the register to be read; It will be removed during
/// the IR processing stage from the IR; Must be a constant value
/// \returns the value of the read register
template <typename T, typename = std::enable_if_t<
                          detail::is_amdgpu_register_compatible_v<T>>>
LUTHIER_INTRINSIC_ANNOTATE T readReg(llvm::MCRegister Reg);

/// \brief Intrinsic to write the value of a register
/// \details The writeReg intrinsic writes \p Val into the register named \p Reg
/// \tparam T the type of value to be written; Must be a type that can be held
/// in one or more AMD GPU registers (see
/// \c detail::is_amdgpu_register_compatible): an integral type, a floating
/// point type (including \c _Float16 / \c __bf16), or the AMDGPU built-in
/// opaque type \c __amdgpu_buffer_rsrc_t. The size of \p T must match the
/// size of \p Reg (for example, writing to \c llvm::AMDGPU::SGPR4_SGPR5
/// requires a 64-bit \p Val such as \c uint64_t or \c double; writing a
/// 128-bit SGPR quad can take an \c __amdgpu_buffer_rsrc_t).
/// \param Reg the ID of the register to be written; It will be removed during
/// the IR processing stage from the IR; Must be a constant value
/// \param Val the value to write into the register
template <typename T, typename = std::enable_if_t<
                          detail::is_amdgpu_register_compatible_v<T>>>
LUTHIER_INTRINSIC_ANNOTATE void writeReg(llvm::MCRegister Reg, T Val);

LUTHIER_INTRINSIC_ANNOTATE void writeExec(uint64_t Val);

template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
              std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>>>
LUTHIER_INTRINSIC_ANNOTATE T sAtomicAdd(T *Address, T Value);

#endif

} // namespace luthier
#endif