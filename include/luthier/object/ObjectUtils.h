//===-- ObjectUtils.h - Luthier object file utilities  ----------*- C++ -*-===//
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
/// This file defines a set of utilites for the LLVM object file class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_OBJECT_OBJECT_UTILS_H
#define LUTHIER_OBJECT_OBJECT_UTILS_H
#include <llvm/Object/ObjectFile.h>

namespace llvm {

/// Similar to \c llvm::DenseMapInfo for \c llvm::object::SectionRef
template <> struct DenseMapInfo<object::SymbolRef> {
  static bool isEqual(const object::SymbolRef &A, const object::SymbolRef &B) {
    return A == B;
  }

  static object::SymbolRef getEmptyKey() {
    return object::SymbolRef{{}, nullptr};
  }

  static object::SymbolRef getTombstoneKey() {
    object::DataRefImpl TS;
    TS.p = (uintptr_t)-1;
    return object::SymbolRef{TS, nullptr};
  }
  static unsigned getHashValue(const object::SymbolRef &Sym) {
    object::DataRefImpl Raw = Sym.getRawDataRefImpl();
    return hash_combine(Raw.p, Raw.d.a, Raw.d.b);
  }
};

}; // namespace llvm

#endif