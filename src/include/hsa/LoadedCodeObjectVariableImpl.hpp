//===-- LoadedCodeObjectVariableImpl.hpp ----------------------------------===//
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
/// This file defines the \c hsa::LoadedCodeObjectVariableImpl class, the
/// concrete implementation of the \c hsa::LoadedCodeObjectVariable interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_VARIABLE_IMPL_HPP
#define LUTHIER_HSA_LOADED_CODE_OBJECT_VARIABLE_IMPL_HPP
#include "hsa/LoadedCodeObjectSymbolImpl.hpp"
#include <luthier/hsa/LoadedCodeObjectVariable.h>

namespace luthier::hsa {

/// \brief the concrete implementation of the \c LoadedCodeObjectVariable of
/// type
class LoadedCodeObjectVariableImpl
    : public llvm::RTTIExtends<LoadedCodeObjectVariableImpl,
                               LoadedCodeObjectSymbolImpl,
                               LoadedCodeObjectVariable> {
public:
  static char ID;

private:
  /// Constructor
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param VarSymbol the symbol of the variable,
  /// cached internally by Luthier
  /// \param ExecutableSymbol the \c hsa_executable_symbol_t equivalent of
  /// the variable symbol, if exists
  LoadedCodeObjectVariableImpl(
      hsa_loaded_code_object_t LCO, llvm::object::ELFSymbolRef VarSymbol,
      std::optional<hsa_executable_symbol_t> ExecutableSymbol)
      : llvm::RTTIExtends<LoadedCodeObjectVariableImpl,
                          LoadedCodeObjectSymbolImpl, LoadedCodeObjectVariable>(
            LCO, VarSymbol, ExecutableSymbol) {}

public:
  static llvm::Expected<std::unique_ptr<LoadedCodeObjectVariableImpl>>
  create(hsa_loaded_code_object_t LCO, llvm::object::ELFSymbolRef VarSymbol);
};

} // namespace luthier::hsa

#endif