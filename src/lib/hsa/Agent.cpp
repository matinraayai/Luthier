//===-- GpuAgent.cpp ------------------------------------------------------===//
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
#include <luthier/common/ErrorCheck.h>
#include <luthier/hsa/Agent.h>
#include <luthier/hsa/HsaError.h>

namespace luthier::hsa {

llvm::Error agentGetSupportedISAs(
    const hsa_agent_t Agent,
    const decltype(hsa_agent_iterate_isas) &HsaAgentIterateISAsFn,
    llvm::SmallVectorImpl<hsa_isa_t> &ISAList) {
  auto Iterator = [](hsa_isa_t Isa, void *Data) {
    auto SupportedIsaList =
        static_cast<llvm::SmallVectorImpl<hsa_isa_t> *>(Data);
    SupportedIsaList->emplace_back(Isa);
    return HSA_STATUS_SUCCESS;
  };
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      HsaAgentIterateISAsFn(Agent, Iterator, &ISAList),
      llvm::formatv("Failed to iterate over ISAs of Agent {0:x}"));
}

llvm::Error
agentIterateISAs(const hsa_agent_t Agent,
                 decltype(hsa_agent_iterate_isas) &HsaAgentIterateISAsFn,
                 const std::function<llvm::Error(hsa_isa_t)> &Callback) {
  struct CallbackDataType {
    decltype(Callback) CB;
    llvm::Error Err;
  } CBData{Callback, llvm::Error::success()};

  auto Iterator = [](const hsa_isa_t ISA, void *D) -> hsa_status_t {
    auto *Data = static_cast<CallbackDataType *>(D);
    if (!Data) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    Data->Err = Data->CB(ISA);
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t Out = HsaAgentIterateISAsFn(Agent, Iterator, &CBData);
      Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK)
    return std::move(CBData.Err);

  return llvm::make_error<HsaError>(
      llvm::formatv("Failed to iterate over the ISAs of agent "
                    "{0:x}.",
                    Agent.handle));
}

llvm::Expected<std::optional<hsa_isa_t>> agentFindFirstISA(
    hsa_agent_t Agent,
    const decltype(hsa_agent_iterate_isas) &HsaAgentIterateISAsFn,
    const std::function<llvm::Expected<bool>(hsa_isa_t)> &Predicate) {
  struct CallbackDataType {
    decltype(Predicate) CB;
    std::optional<hsa_isa_t> ISA;
    llvm::Error Err;
  } CBData{Predicate, std::nullopt, llvm::Error::success()};

  auto Iterator = [](const hsa_isa_t ISA, void *D) -> hsa_status_t {
    auto *Data = static_cast<CallbackDataType *>(D);
    if (!Data) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    llvm::Expected<bool> Res = Data->CB(ISA);
    Data->Err = Res.takeError();
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    if (*Res) {
      Data->ISA = ISA;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t Out = HsaAgentIterateISAsFn(Agent, Iterator, &CBData);
      Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK) {
    LUTHIER_RETURN_ON_ERROR(CBData.Err);
    return CBData.ISA;
  }

  return llvm::make_error<HsaError>(
      llvm::formatv("Failed to iterate over the ISAs of agent "
                    "{0:x}.",
                    Agent.handle));
}

} // namespace luthier::hsa
