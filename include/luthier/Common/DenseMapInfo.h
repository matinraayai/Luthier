//===-- DenseMapInfo.h ------------------------------------------*- C++ -*-===//
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
/// Defines \c DenseMapInfo specializations for types not implemented in LLVM.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_DENSE_MAP_INFO_H
#define LUTHIER_COMMON_DENSE_MAP_INFO_H
#include <llvm/ADT/DenseMapInfo.h>

namespace llvm {

/// \c DenseMapInfo specialization that lets \c std::reference_wrapper<T> be
/// used as a DenseMap/DenseSet key, keyed on the referenced object's address.
template <typename T> struct DenseMapInfo<std::reference_wrapper<T>> {
  // NOTE: the empty key binds a reference to a dereferenced sentinel
  // pointer. This is safe in practice because getHashValue/isEqual immediately
  // take the address again (&.get()), recovering the exact sentinel bits.
  static std::reference_wrapper<T> getEmptyKey() {
    return std::reference_wrapper<T>(*DenseMapInfo<T *>::getEmptyKey());
  }

  static unsigned getHashValue(const std::reference_wrapper<T> &Val) {
    return DenseMapInfo<T *>::getHashValue(&Val.get());
  }

  static bool isEqual(const std::reference_wrapper<T> &LHS,
                      const std::reference_wrapper<T> &RHS) {
    return &LHS.get() == &RHS.get();
  }
};

} // namespace llvm

#endif