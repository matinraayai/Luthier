//===-- LoadedCodeObjectVariable.h - LCO Variable Symbol --------*- C++ -*-===//
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
/// This file defines the \c LoadedCodeObjectVariable under the
/// \c luthier::hsa namespace, which represents all device variable symbols
/// inside a <tt>hsa::LoadedCodeObject</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LOADED_CODE_OBJECT_VARIABLE_H
#define LUTHIER_LOADED_CODE_OBJECT_VARIABLE_H
#include "LoadedCodeObjectSymbol.h"

namespace luthier::hsa {

/// \brief a \c LoadedCodeObjectSymbol of type
/// \c LoadedCodeObjectSymbol::ST_DEVICE_FUNCTION
class LoadedCodeObjectVariable final : public LoadedCodeObjectSymbol {

private:
  /// Constructor
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param VarSymbol the symbol of the variable,
  /// cached internally by Luthier
  /// \param ExecutableSymbol the \c hsa_executable_symbol_t equivalent of
  /// the variable symbol, if exists
  LoadedCodeObjectVariable(
      hsa_loaded_code_object_t LCO, llvm::object::ELFSymbolRef VarSymbol,
      std::optional<hsa_executable_symbol_t> ExecutableSymbol)
      : LoadedCodeObjectSymbol(LCO, VarSymbol, SymbolKind::SK_VARIABLE,
                               ExecutableSymbol) {}

public:
  static llvm::Expected<std::unique_ptr<LoadedCodeObjectVariable>>
  create(hsa_loaded_code_object_t LCO, llvm::object::ELFSymbolRef VarSymbol);

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const LoadedCodeObjectSymbol *S) {
    return S->getType() == SK_VARIABLE;
  }
};

} // namespace luthier::hsa

#endif