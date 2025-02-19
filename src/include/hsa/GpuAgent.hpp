//===-- GpuAgent.hpp - HSA GPU Agent Wrapper Interface --------------------===//
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
/// This file defines the \c GpuAgent interface under the \c luthier::hsa
/// namespace, representing a wrapper around the \c hsa_agent_t of type
/// \c HSA_DEVICE_TYPE_GPU and its related functionality.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_GPU_AGENT_HPP
#define LUTHIER_HSA_GPU_AGENT_HPP
#include "hsa/HandleType.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include "hsa/ISA.hpp"
#include <hsa/hsa.h>
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

/// \brief Interface for a wrapper around the \c hsa_agent_t handles of type
/// \c HSA_DEVICE_TYPE_GPU
class GpuAgent : public HandleType<hsa_agent_t> {
public:
  /// Constructor
  /// \param Agent the HSA handle of the GPU Agent
  explicit GpuAgent(hsa_agent_t Agent) : HandleType<hsa_agent_t>(Agent) {};

  virtual ~GpuAgent() = default;

  /// Queries all the <tt>hsa::ISA</tt>s supported by this GPU Agent.
  /// \param [out] Isa list of ISAs supported by the Agent
  /// \return an llvm::ErrorSuccess if the operation was successful, or
  /// a \c luthier::HsaError in case of failure
  /// \sa hsa_agent_iterate_isas
  virtual llvm::Error
  getSupportedISAs(llvm::SmallVectorImpl<std::unique_ptr<ISA>> &Isa) const = 0;
};

} // namespace luthier::hsa

DECLARE_LLVM_MAP_INFO_STRUCTS_FOR_HANDLE_TYPE(luthier::hsa::GpuAgent,
                                              hsa_agent_t)

#endif