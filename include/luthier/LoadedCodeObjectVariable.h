//===-- LoadedCodeObjectVariable.h - LCO Variable Symbol --------*- C++ -*-===//
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
/// This file defines the \c LoadedCodeObjectVariable under the
/// \c luthier::hsa namespace, which represents all device variable symbols
/// inside a \c hsa::LoadedCodeObject.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LOADED_CODE_OBJECT_VARIABLE_H
#define LUTHIER_LOADED_CODE_OBJECT_VARIABLE_H
#include "LoadedCodeObjectSymbol.h"

namespace luthier::hsa {

/// \brief a \c LoadedCodeObjectSymbol of type
/// \c LoadedCodeObjectSymbol::ST_DEVICE_FUNCTION
class LoadedCodeObjectVariable final : public LoadedCodeObjectSymbol {

public:
  /// Constructor
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param VarSymbol the symbol of the variable,
  /// cached internally by Luthier
  LoadedCodeObjectVariable(hsa_loaded_code_object_t LCO,
                                 const llvm::object::ELFSymbolRef *VarSymbol)
      : LoadedCodeObjectSymbol(LCO, VarSymbol,
                               SymbolKind::SK_VARIABLE) {}

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const LoadedCodeObjectSymbol *S) {
    return S->getType() == SK_VARIABLE;
  }
};

} // namespace luthier::hsa

#endif