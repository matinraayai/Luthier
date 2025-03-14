//===-- GpuAgent.hpp ------------------------------------------------------===//
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
/// This file defines the \c hsa::GpuAgent interface, representing a GPU Agent
/// inside the HSA standard.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_GPU_AGENT_HPP
#define LUTHIER_HSA_GPU_AGENT_HPP
#include "hsa/ISA.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ExtensibleRTTI.h>

namespace luthier::hsa {

/// \brief Wrapper around the \c hsa_agent_t handles of type
/// \c HSA_DEVICE_TYPE_CPU
class GpuAgent : public llvm::RTTIExtends<GpuAgent, llvm::RTTIRoot> {
public:
  static char ID;
  /// Queries all the <tt>hsa::ISA</tt>s supported by this GPU Agent.
  /// \param [out] Isa list of ISAs supported by the Agent
  /// \return an llvm::ErrorSuccess if the operation was successful, or
  /// a \c luthier::HsaError in case of failure
  /// \sa hsa_agent_iterate_isas
  llvm::Error
  getSupportedISAs(llvm::SmallVectorImpl<std::unique_ptr<ISA>> &Isa) const;
};

} // namespace luthier::hsa

#endif