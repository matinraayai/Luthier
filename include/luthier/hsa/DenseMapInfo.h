//===-- DenseMapInfo.h - HSA Primitive DenseMapInfo Definitions --*- C++-*-===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// \file Includes the <tt>DenseMapInfo</tt> for HSA primitives used by Luthier,
/// so that they can be stored in LLVM-based maps without additional code on
/// the tool side.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_DENSE_MAP_INFO
#define LUTHIER_HSA_DENSE_MAP_INFO
#include <hsa/hsa.h>
#include <llvm/ADT/DenseMapInfo.h>

namespace llvm {

/// \brief defines an \c llvm::DenseMapInfo for a HSA opaque handle primitive
#define HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE(T)                                 \
  template <> struct DenseMapInfo<T> {                                         \
    static inline T getEmptyKey() {                                            \
      return (T){DenseMapInfo<decltype(T::handle)>::getEmptyKey()};            \
    }                                                                          \
                                                                               \
    static inline T getTombstoneKey() {                                        \
      return (T){DenseMapInfo<decltype(T::handle)>::getTombstoneKey()};        \
    }                                                                          \
                                                                               \
    static unsigned getHashValue(const T &P) {                                 \
      return DenseMapInfo<decltype(T::handle)>::getHashValue(P.handle);        \
    }                                                                          \
    static bool isEqual(const T &LHS, const T &RHS) {                          \
      return LHS.handle == RHS.handle;                                         \
    }                                                                          \
  };

HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE(hsa_executable_t);
HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE(hsa_loaded_code_object_t);
HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE(hsa_executable_symbol_t);

#undef HSA_PRIMITIVE_DENSE_MAP_INFO_DEFINE

} // namespace llvm

#endif