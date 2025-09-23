//===-- LoadedCodeObjectKernel.cpp - Loaded Code Object Kernel ------------===//
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
/// This file defines the \c LoadedCodeObjectKernel under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/LoadedCodeObjectKernel.h"
#include "luthier/hsa/Agent.h"
#include "luthier/hsa/Executable.h"
#include "luthier/hsa/ExecutableSymbol.h"
#include "luthier/hsa/KernelDescriptor.h"
#include "luthier/hsa/LoadedCodeObject.h"
#include <hsa/hsa.h>

namespace luthier::hsa {

llvm::Expected<std::unique_ptr<LoadedCodeObjectKernel>>
LoadedCodeObjectKernel::create(
    const ApiTableContainer<::CoreApiTable> &CoreApiTable,
    const hsa_ven_amd_loader_1_03_pfn_t &VenLoaderApi,
    hsa_loaded_code_object_s LCO, llvm::object::ELF64LEObjectFile &StorageElf,
    std::unique_ptr<amdgpu::hsamd::Kernel::Metadata> MD,
    llvm::object::ELFSymbolRef KFuncSymbol,
    llvm::object::ELFSymbolRef KDSymbol) {
  // Get the kernel symbol associated with this kernel
  llvm::Expected<hsa_executable_t> ExecOrErr =
      hsa::loadedCodeObjectGetExecutable(VenLoaderApi, LCO);
  LUTHIER_RETURN_ON_ERROR(ExecOrErr.takeError());

  llvm::Expected<hsa_agent_t> AgentOrErr =
      hsa::loadedCodeObjectGetAgent(VenLoaderApi, LCO);
  LUTHIER_RETURN_ON_ERROR(AgentOrErr.takeError());

  auto NameWithKDSuffixed = KDSymbol.getName();
  LUTHIER_RETURN_ON_ERROR(NameWithKDSuffixed.takeError());

  auto ExecSymbol = hsa::executableGetSymbolByName(
      CoreApiTable, *ExecOrErr, *NameWithKDSuffixed, *AgentOrErr);
  LUTHIER_RETURN_ON_ERROR(ExecSymbol.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      ExecSymbol->has_value(),
      llvm::formatv("Failed to query the HSA executable symbol of kernel "
                    "{0} from its executable using its name.",
                    *NameWithKDSuffixed)));

  llvm::Expected<llvm::StringRef> KernelNameOrErr = KDSymbol.getName();
  LUTHIER_RETURN_ON_ERROR(KernelNameOrErr.takeError());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      MD->Name == *KernelNameOrErr,
      "Name of the kernel in HSA and Metadata doesn't match"));

  return std::unique_ptr<LoadedCodeObjectKernel>(new LoadedCodeObjectKernel(
      LCO, StorageElf, KFuncSymbol, KDSymbol, **ExecSymbol, std::move(MD)));
}

llvm::Expected<const KernelDescriptor *>
LoadedCodeObjectKernel::getKernelDescriptor(
    const ApiTableContainer<::CoreApiTable> &CoreApiTable) const {
  llvm::Expected<uint64_t> KernelObjectOrErr =
      executableSymbolGetAddress(CoreApiTable, *ExecutableSymbol);
  LUTHIER_RETURN_ON_ERROR(KernelObjectOrErr.takeError());
  return reinterpret_cast<const KernelDescriptor *>(*KernelObjectOrErr);
}

} // namespace luthier::hsa