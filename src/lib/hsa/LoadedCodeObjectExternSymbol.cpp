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

#include <hsa/hsa.h>

namespace luthier::hsa {

llvm::Expected<std::unique_ptr<LoadedCodeObjectExternSymbol>>
LoadedCodeObjectExternSymbol::create(
    const ApiTableContainer<::CoreApiTable> &CoreApiTable,
    const hsa_ven_amd_loader_1_01_pfn_t &VenLoaderApi,
    hsa_loaded_code_object_t LCO, llvm::object::ELF64LEObjectFile &StorageElf,
    llvm::object::ELFSymbolRef ExternSymbol) {

}

} // namespace luthier::hsa