//===-- DenseMapInfo.h ------------------------------------------*- C++ -*-===//
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
/// This file implements any \c llvm::DenseMapInfo structs for
/// frequently-used LLVM class/objects not included in upstream LLVM.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LLVM_DENSE_MAP_INFO_H
#define LUTHIER_LLVM_DENSE_MAP_INFO_H
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/Object/ELFObjectFile.h>

namespace llvm {

template <> struct DenseMapInfo<object::ELFSymbolRef> {

  static inline object::ELFSymbolRef getEmptyKey() {
    object::DataRefImpl EmptyDR;
    EmptyDR.d.a = DenseMapInfo<decltype(EmptyDR.d.a)>::getEmptyKey();
    EmptyDR.d.b = DenseMapInfo<decltype(EmptyDR.d.b)>::getEmptyKey();
    EmptyDR.p = DenseMapInfo<decltype(EmptyDR.p)>::getEmptyKey();

    return object::ELFSymbolRef{object::SymbolRef{EmptyDR, nullptr}};
  }

  static inline object::ELFSymbolRef getTombstoneKey() {
    object::DataRefImpl EmptyDR;
    EmptyDR.d.a = DenseMapInfo<decltype(EmptyDR.d.a)>::getTombstoneKey();
    EmptyDR.d.b = DenseMapInfo<decltype(EmptyDR.d.b)>::getTombstoneKey();
    EmptyDR.p = DenseMapInfo<decltype(EmptyDR.p)>::getTombstoneKey();

    return object::ELFSymbolRef{object::SymbolRef{EmptyDR, nullptr}};
  }

  static unsigned getHashValue(const object::ELFSymbolRef &S) {
    return hash_combine(S.getRawDataRefImpl().d.a, S.getRawDataRefImpl().d.b,
                        S.getRawDataRefImpl().p);
  }

  static bool isEqual(const object::ELFSymbolRef &LHS,
                      const object::ELFSymbolRef &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif