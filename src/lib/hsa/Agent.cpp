//===-- GpuAgent.cpp - HSA GPU Agent Wrapper Implementation ---------------===//
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
/// Implements a set of commonly used functionality around the \c hsa_agent_t
/// handle in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/Agent.h"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

llvm::Error agentGetSupportedISAs(
    hsa_agent_t Agent,
    const decltype(hsa_agent_iterate_isas) &HsaAgentIterateISAsFn,
    llvm::SmallVectorImpl<hsa_isa_t> &ISAList) {
  auto Iterator = [](hsa_isa_t Isa, void *Data) {
    auto SupportedIsaList =
        static_cast<llvm::SmallVectorImpl<hsa_isa_t> *>(Data);
    SupportedIsaList->emplace_back(Isa);
    return HSA_STATUS_SUCCESS;
  };

  if (auto Status = HsaAgentIterateISAsFn(Agent, Iterator, &ISAList);
      Status != HSA_STATUS_SUCCESS)
    return llvm::make_error<HsaError>(
        llvm::formatv("Failed to iterate over ISAs of Agent {0:x}"), Status);
  else
    return llvm::Error::success();
}

} // namespace luthier::hsa
