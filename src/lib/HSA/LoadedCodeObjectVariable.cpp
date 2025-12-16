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

#include "luthier/HSA/LoadedCodeObjectVariable.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/Executable.h"
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include <hsa/hsa.h>

namespace luthier::hsa {

llvm::Expected<std::unique_ptr<LoadedCodeObjectVariable>>
LoadedCodeObjectVariable::create(
    const ApiTableContainer<::CoreApiTable> &CoreApiTable,
    const hsa_ven_amd_loader_1_03_pfn_t &VenLoaderApi,
    hsa_loaded_code_object_s LCO, luthier::object::AMDGCNObjectFile &StorageElf,
    llvm::object::ELFSymbolRef VarSymbol) {
  llvm::Expected<hsa_executable_t> ExecOrErr =
      hsa::loadedCodeObjectGetExecutable(VenLoaderApi, LCO);
  LUTHIER_RETURN_ON_ERROR(ExecOrErr.takeError());

  llvm::Expected<hsa_agent_t> AgentOrErr =
      hsa::loadedCodeObjectGetAgent(VenLoaderApi, LCO);
  LUTHIER_RETURN_ON_ERROR(AgentOrErr.takeError());

  auto Name = VarSymbol.getName();
  LUTHIER_RETURN_ON_ERROR(Name.takeError());

  auto ExecSymbol = hsa::executableGetSymbolByName(CoreApiTable, *ExecOrErr,
                                                   *Name, *AgentOrErr);
  LUTHIER_RETURN_ON_ERROR(ExecSymbol.takeError());

  return std::unique_ptr<LoadedCodeObjectVariable>(
      new LoadedCodeObjectVariable(LCO, StorageElf, VarSymbol, *ExecSymbol));
}

} // namespace luthier::hsa