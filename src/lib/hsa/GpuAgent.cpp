//===-- GpuAgent.cpp - HSA GPU Agent Wrapper Implementation ---------------===//
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
/// This file implements the \c GpuAgent class under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//

#include "hsa/GpuAgent.hpp"
#include "hsa/hsa_isa.hpp"

namespace luthier::hsa {

llvm::Error
GpuAgent::getSupportedISAs(llvm::SmallVectorImpl<ISA> &IsaList) const {
  auto Iterator = [](hsa_isa_t Isa, void *Data) {
    auto SupportedIsaList =
        reinterpret_cast<llvm::SmallVectorImpl<ISA> *>(Data);
    SupportedIsaList->emplace_back(Isa);
    return HSA_STATUS_SUCCESS;
  };

  return LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_agent_iterate_isas_fn(
      this->asHsaType(), Iterator, &IsaList));
}

} // namespace luthier::hsa
