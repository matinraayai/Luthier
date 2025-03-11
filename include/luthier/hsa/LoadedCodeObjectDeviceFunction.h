//===-- LoadedCodeObjectDeviceFunction.h - LCO Device Function --*- C++ -*-===//
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
/// This file defines the \c LoadedCodeObjectDeviceFunction under the
/// \c luthier::hsa namespace, which represents all device (non-kernel)
/// functions inside a \c hsa::LoadedCodeObject.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LOADED_CODE_OBJECT_DEVICE_FUNCTION_H
#define LUTHIER_LOADED_CODE_OBJECT_DEVICE_FUNCTION_H
#include "LoadedCodeObjectSymbol.h"

namespace luthier::hsa {

/// \brief a \c LoadedCodeObjectSymbol of type
/// \c LoadedCodeObjectSymbol::ST_DEVICE_FUNCTION
class LoadedCodeObjectDeviceFunction final : public LoadedCodeObjectSymbol {

private:
  /// Constructor
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param FuncSymbol the function symbol of the device function,
  /// cached internally by Luthier
  LoadedCodeObjectDeviceFunction(hsa_loaded_code_object_t LCO,
                                 llvm::object::ELF64LEObjectFile &StorageElf,
                                 llvm::object::ELFSymbolRef FuncSymbol)
      : LoadedCodeObjectSymbol(LCO, StorageElf, FuncSymbol,
                               SymbolKind::SK_DEVICE_FUNCTION, std::nullopt) {}

public:
  /// Factory method used internally by Luthier
  /// Symbols created using this method will be cached, and a reference to them
  /// will be returned to the tool writer when queried
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param FuncSymbol the function symbol of the device function,
  /// cached internally by Luthier
  static llvm::Expected<std::unique_ptr<LoadedCodeObjectDeviceFunction>>
  create(hsa_loaded_code_object_t LCO,
         llvm::object::ELF64LEObjectFile &StorageElf,
         llvm::object::ELFSymbolRef FuncSymbol);

  /// method for providing LLVM RTTI
  __attribute__((used)) static bool classof(const LoadedCodeObjectSymbol *S) {
    return S->getType() == SK_DEVICE_FUNCTION;
  }
};

} // namespace luthier::hsa

#endif