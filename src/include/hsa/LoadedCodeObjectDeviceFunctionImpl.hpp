//===-- LoadedCodeObjectDeviceFunctionImpl.hpp ----------------------------===//
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
/// This file defines the \c hsa::LoadedCodeObjectDeviceFunctionImpl class,
/// the concrete implementation of the \c hsa::LoadedCodeObjectFunction
/// interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LOADED_CODE_OBJECT_DEVICE_FUNCTION_IMPL_HPP
#define LUTHIER_LOADED_CODE_OBJECT_DEVICE_FUNCTION_IMPL_HPP
#include "hsa/LoadedCodeObjectSymbolImpl.hpp"
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>

namespace luthier::hsa {

/// \brief concrete implementation of \c hsa::LoadedCodeObjectDeviceFunction
class LoadedCodeObjectDeviceFunctionImpl final
    : public llvm::RTTIExtends<LoadedCodeObjectDeviceFunctionImpl,
                               LoadedCodeObjectSymbolImpl,
                               LoadedCodeObjectDeviceFunction> {
public:
  static char ID;

private:
  /// Constructor
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param FuncSymbol the function symbol of the device function,
  /// cached internally by Luthier
  LoadedCodeObjectDeviceFunctionImpl(
      hsa_loaded_code_object_t LCO, llvm::object::ELF64LEObjectFile &StorageElf,
      llvm::object::ELFSymbolRef FuncSymbol)
      : llvm::RTTIExtends<LoadedCodeObjectDeviceFunctionImpl,
                          LoadedCodeObjectSymbolImpl,
                          LoadedCodeObjectDeviceFunction>(
            LCO, FuncSymbol, std::nullopt) {}

public:
  /// Factory method used internally by Luthier
  /// Symbols created using this method will be cached, and a reference to them
  /// will be returned to the tool writer when queried
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param FuncSymbol the function symbol of the device function,
  /// cached internally by Luthier
  static llvm::Expected<std::unique_ptr<LoadedCodeObjectDeviceFunction>>
  create(hsa_loaded_code_object_t LCO,
         llvm::object::ELFSymbolRef FuncSymbol);

};

} // namespace luthier::hsa

#endif