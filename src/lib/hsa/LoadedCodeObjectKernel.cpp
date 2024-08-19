//===-- LoadedCodeObjectKernel.cpp - Loaded Code Object Kernel ------------===//
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
/// This file defines the \c LoadedCodeObjectKernel under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//

#include "hsa/Executable.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include "hsa/LoadedCodeObject.hpp"

#include <luthier/hsa/KernelDescriptor.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>

namespace luthier::hsa {

llvm::Expected<std::unique_ptr<LoadedCodeObjectKernel>>
LoadedCodeObjectKernel::create(hsa_loaded_code_object_t LCO,
                               llvm::object::ELFSymbolRef KFuncSymbol,
                               llvm::object::ELFSymbolRef KDSymbol,
                               const md::Kernel::Metadata &Metadata) {
  hsa::LoadedCodeObject LCOWrapper(LCO);
  // Get the kernel symbol associated with this kernel
  auto Exec = LCOWrapper.getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());

  auto Agent = LCOWrapper.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  auto NameWithKDSuffixed = KDSymbol.getName();
  LUTHIER_RETURN_ON_ERROR(NameWithKDSuffixed.takeError());

  auto ExecSymbol =
      Exec->getExecutableSymbolByName(*NameWithKDSuffixed, *Agent);
  LUTHIER_RETURN_ON_ERROR(ExecSymbol.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ExecSymbol->has_value()));

  return std::unique_ptr<LoadedCodeObjectKernel>(new LoadedCodeObjectKernel(
      LCO, KFuncSymbol, KDSymbol, Metadata, ExecSymbol.get()->asHsaType()));
}

llvm::Expected<const KernelDescriptor *>
LoadedCodeObjectKernel::getKernelDescriptor() const {
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(
      luthier::address_t, KernelObject,
      hsa::ExecutableSymbol(*ExecutableSymbol).getAddress());
  return reinterpret_cast<const KernelDescriptor *>(KernelObject);
}

} // namespace luthier::hsa