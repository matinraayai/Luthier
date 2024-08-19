//===-- LoadedCodeObjectDeviceFunction.cpp --------------------------------===//
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
/// This file implements the \c LoadedCodeObjectDeviceFunction under the \c
/// luthier::hsa namespace.
//===----------------------------------------------------------------------===//
#include "hsa/Executable.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/LoadedCodeObject.hpp"

#include <luthier/hsa/LoadedCodeObjectVariable.h>

namespace luthier::hsa {

llvm::Expected<std::unique_ptr<LoadedCodeObjectExternSymbol>>
LoadedCodeObjectExternSymbol::create(
    hsa_loaded_code_object_t LCO,
    llvm::object::ELFSymbolRef ExternSymbol) {
  hsa::LoadedCodeObject LCOWrapper(LCO);
  // Get the executable symbol associated with this external symbol
  auto Exec = LCOWrapper.getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());

  auto Agent = LCOWrapper.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  auto Name = ExternSymbol.getName();
  LUTHIER_RETURN_ON_ERROR(Name.takeError());

  auto ExecSymbol = Exec->getExecutableSymbolByName(*Name, *Agent);
  LUTHIER_RETURN_ON_ERROR(ExecSymbol.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ExecSymbol->has_value()));

  return std::unique_ptr<LoadedCodeObjectExternSymbol>(
      new LoadedCodeObjectExternSymbol(LCO, ExternSymbol,
                                       ExecSymbol.get()->asHsaType()));
}

} // namespace luthier::hsa