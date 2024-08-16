//===-- LoadedCodeObjectExternSymbol.h - LCO External Symbol ----*- C++ -*-===//
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
/// \file
/// This file defines the \c LoadedCodeObjectExternSymbol under the
/// \c luthier::hsa namespace, which represents all symbols declared
/// inside a \c hsa::LoadedCodeObject but not defined.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LOADED_CODE_OBJECT_EXTERN_SYMBOL_H
#define LUTHIER_LOADED_CODE_OBJECT_EXTERN_SYMBOL_H
#include "LoadedCodeObjectSymbol.h"

namespace luthier::hsa {

/// \brief a \c LoadedCodeObjectSymbol of type
/// \c LoadedCodeObjectSymbol::SK_EXTERNAL
class LoadedCodeObjectExternSymbol final : public LoadedCodeObjectSymbol {

public:
  /// Constructor
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param ExternSymbol the external symbol,
  /// cached internally by Luthier
  LoadedCodeObjectExternSymbol(hsa_loaded_code_object_t LCO,
                               const llvm::object::ELFSymbolRef *ExternSymbol)
      : LoadedCodeObjectSymbol(LCO, ExternSymbol, SymbolKind::SK_EXTERNAL) {}

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const LoadedCodeObjectSymbol *S) {
    return S->getType() == SK_EXTERNAL;
  }
};

} // namespace luthier::hsa

#endif