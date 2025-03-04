//===-- LoadedCodeObjectVariable.cpp - Loaded Code Object Variable --------===//
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
/// This file implements the \c LoadedCodeObjectVariable under the \c
/// luthier::hsa namespace.
//===----------------------------------------------------------------------===//

#include "hsa/Executable.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/LoadedCodeObject.hpp"

#include <hsa/hsa.h>
#include <luthier/hsa/LoadedCodeObjectVariable.h>

namespace luthier::hsa {

llvm::Expected<std::unique_ptr<LoadedCodeObjectVariable>>
LoadedCodeObjectVariable::create(
    hsa_loaded_code_object_s LCO,
    std::shared_ptr<llvm::object::ELF64LEObjectFile> StorageElf,
    llvm::object::ELFSymbolRef VarSymbol) {
  hsa::LoadedCodeObject LCOWrapper(LCO);
  // Get the kernel symbol associated with this kernel
  auto Exec = LCOWrapper.getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());

  auto Agent = LCOWrapper.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  auto Name = VarSymbol.getName();
  LUTHIER_RETURN_ON_ERROR(Name.takeError());

  auto ExecSymbol = Exec->getExecutableSymbolByName(*Name, *Agent);
  LUTHIER_RETURN_ON_ERROR(ExecSymbol.takeError());

  auto ExecSymbolAsOptionalHandle =
      ExecSymbol->has_value() ? std::make_optional<hsa_executable_symbol_t>(
                                    ExecSymbol.get()->asHsaType())
                              : std::nullopt;

  return std::unique_ptr<LoadedCodeObjectVariable>(new LoadedCodeObjectVariable(
      LCO, std::move(StorageElf), VarSymbol, ExecSymbolAsOptionalHandle));
}

} // namespace luthier::hsa