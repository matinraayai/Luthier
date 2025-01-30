//===-- Intrinsics.h - Luthier's built-in Intrinsics ------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
#include "luthier/consts.h"
#include <llvm/MC/MCRegister.h>

namespace luthier {

/// \brief All bindings to Luthier intrinsics must be annotated using this macro
/// \details This macro defines the binding as a device function, adds a
/// noinline attribute as  well as a \c LUTHIER_INTRINSIC_ATTRIBUTE attribute
/// to be recognized by Luthier as an intrinsic
#define LUTHIER_INTRINSIC_ANNOTATE                                             \
  __attribute__((device, noinline,                                             \
                 annotate(LUTHIER_STRINGIFY(LUTHIER_INTRINSIC_ATTRIBUTE))))

#if defined(__HIPCC__)

/// \brief Macro to use to prevent the compiler from optimizing a code region
/// away
/// \details This macro places an empty volatile inline assembly with "memory"
/// side-effects to prevent the compiler from dead-code eliminating a basic
/// block
#define LUTHIER_DONT_OPTIMIZE __asm__ __volatile__("" : : : "memory");

/// \brief Macro to use on all values involved in an intrinsic device
/// binding, to prevent their elimination from the binding prototype by the
/// compiler
/// \details This macro places an empty volatile inline assembly with
/// the arbitrary constraint on the passed <tt>Value</t>>, which will prevent
/// the compiler from optimizing it away
/// \note These operations will not show up in the IR of a Luthier device
/// module, as the body of Luthier intrinsic bindings are removed at the end of
/// the LLVM IR pipeline
/// \param Value the L-value to prevent optimization on
template <typename T>
__attribute__((device, always_inline)) void doNotOptimize(T const &Value) {
  __asm__ __volatile__("" : : "X"(Value) : "memory");
}

/// \brief Intrinsic to read the value of a register
/// \details The readReg intrinsic reads the value of the \p Reg and returns it
/// \tparam T the return type of the output; Must be of integral type and be
/// compatible with the size of \p Reg; For example reading \c
/// llvm::AMDGPU::SGPR4_SGPR5 must return a <tt>uint64_t</tt>
/// \param Reg the ID of the register to be read; It will be removed during
/// the IR processing stage from the IR; Must be a constant value,
/// and the register must be at most 64-bit wide
/// \returns the value of the read register
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
LUTHIER_INTRINSIC_ANNOTATE T readReg(llvm::MCRegister Reg) {
  T Out;
  doNotOptimize(Reg);
  doNotOptimize(Out);
  return Out;
}

/// \brief Intrinsic to write the value of a register
/// \details The writeReg intrinsic writes \p Val into the register named \p Reg
/// \tparam T the type of value to be written output; Must be of integral type
/// and be compatible with the size of \p Reg; For example writing to
// \c llvm::AMDGPU::SGPR4_SGPR5 requires a <tt>uint64_t</tt> \p Val
/// \param Reg the ID of the register to be read; It will be removed during
/// the IR processing stage from the IR; Must be a constant value,
/// and the register must be at most 64-bit wide
/// \param Val the value to write into the register
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
LUTHIER_INTRINSIC_ANNOTATE void writeReg(llvm::MCRegister Reg, T Val) {
  doNotOptimize(Reg);
  doNotOptimize(Val);
}

LUTHIER_INTRINSIC_ANNOTATE void writeExec(uint64_t Val) { doNotOptimize(Val); }

/// \return the address of the implicit argument segment
LUTHIER_INTRINSIC_ANNOTATE uint32_t *implicitArgPtr() {
  uint32_t *Out;
  doNotOptimize(Out);
  return Out;
}

LUTHIER_INTRINSIC_ANNOTATE uint32_t workgroupIdX() {
  uint32_t Out;
  doNotOptimize(Out);
  return Out;
}

LUTHIER_INTRINSIC_ANNOTATE uint32_t workgroupIdY() {
  uint32_t Out;
  doNotOptimize(Out);
  return Out;
}

LUTHIER_INTRINSIC_ANNOTATE uint32_t workgroupIdZ() {
  uint32_t Out;
  doNotOptimize(Out);
  return Out;
}

template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
              std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>>>
LUTHIER_INTRINSIC_ANNOTATE T *sAtomicAdd(T *Address, T Value) {
  T *Out;
  doNotOptimize(Out);
  doNotOptimize(Address);
  doNotOptimize(Value);
  return Out;
}

#endif

} // namespace luthier
#endif