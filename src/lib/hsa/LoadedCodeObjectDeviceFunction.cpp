//===-- LoadedCodeObjectDeviceFunction.cpp --------------------------------===//
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
/// This file implements the \c LoadedCodeObjectDeviceFunction under the \c
/// luthier::hsa namespace.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/LoadedCodeObjectDeviceFunction.h"
#include "common/ObjectUtils.hpp"

namespace luthier::hsa {

llvm::Expected<std::unique_ptr<LoadedCodeObjectDeviceFunction>>
LoadedCodeObjectDeviceFunction::create(hsa_loaded_code_object_t LCO,
                                       luthier::AMDGCNObjectFile &StorageElf,
                                       llvm::object::ELFSymbolRef FuncSymbol) {
  return std::unique_ptr<LoadedCodeObjectDeviceFunction>(
      new LoadedCodeObjectDeviceFunction(LCO, StorageElf, FuncSymbol));
}

} // namespace luthier::hsa