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

namespace luthier {

/// \brief All bindings to Luthier intrinsics must be annotated using this macro
/// \details This macro defines the binding as a device function, adds a
/// noinline attribute as  well as a \c LUTHIER_INTRINSIC_ATTRIBUTE attribute
/// to be recognized by Luthier as an intrinsic
#define LUTHIER_INTRINSIC_ANNOTATE                                             \
  __attribute__((device,                                                       \
                 annotate(LUTHIER_STRINGIFY(LUTHIER_INTRINSIC_ATTRIBUTE))))

#if defined(__HIPCC__)

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
LUTHIER_INTRINSIC_ANNOTATE T readReg(llvm::MCRegister Reg);

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