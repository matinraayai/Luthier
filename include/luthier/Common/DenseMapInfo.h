//===-- DenseMapInfo.h ------------------------------------------*- C++ -*-===//
// Copyright 2026 @ Northeastern University Computer Architecture Lab
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
/// \file DenseMapInfo.h
/// Defines \c DenseMapInfo specializations for types not implemented in LLVM.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_DENSE_MAP_INFO_H
#define LUTHIER_COMMON_DENSE_MAP_INFO_H
#include <llvm/ADT/DenseMapInfo.h>

/// Info for storing reference wrappers in an LLVM Map

namespace llvm {
template <typename T> struct DenseMapInfo<std::reference_wrapper<T>> {

  static constexpr std::reference_wrapper<T> getEmptyKey() {
    return std::reference_wrapper<T>(*DenseMapInfo<T *>::getEmptyKey());
  }

  static constexpr std::reference_wrapper<T> getTombstoneKey() {
    return std::reference_wrapper<T>(*DenseMapInfo<T *>::getTombstoneKey());
  }

  static unsigned getHashValue(const std::reference_wrapper<T> PtrVal) {
    return DenseMapInfo<T *>::getHashValue(&PtrVal.get());
  }

  static bool isEqual(const std::reference_wrapper<T> LHS,
                      const std::reference_wrapper<T> RHS) {
    return &LHS.get() == &RHS.get();
  }
};

} // namespace llvm

#endif