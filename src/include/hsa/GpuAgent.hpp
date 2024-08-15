//===-- GpuAgent.hpp - HSA GPU Agent Wrapper ------------------------------===//
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
/// This file defines the \c GpuAgent class under the \c luthier::hsa
/// namespace, representing a wrapper around the \c hsa_agent_t of type
/// \c HSA_DEVICE_TYPE_CPU and its related functionality.
//===----------------------------------------------------------------------===//

#ifndef HSA_AGENT_HPP
#define HSA_AGENT_HPP
#include "hsa/hsa_handle_type.hpp"
#include "hsa/hsa_intercept.hpp"
#include "hsa/hsa_isa.hpp"
#include <hsa/hsa.h>
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

/// \brief Wrapper around the \c hsa_agent_t handles of type
/// \c HSA_DEVICE_TYPE_CPU
class GpuAgent : public HandleType<hsa_agent_t> {
public:
  /// Constructor
  /// \param Agent the HSA handle of the GPU Agent
  explicit GpuAgent(hsa_agent_t Agent) : HandleType<hsa_agent_t>(Agent){};

  /// Queries all the <tt>hsa::ISA</tt>s supported by this GPU Agent.
  /// \param [out] Isa list of ISAs supported by the Agent
  /// \return an llvm::ErrorSuccess if the operation was successful, or
  /// a \c luthier::HsaError in case of failure
  /// \sa hsa_agent_iterate_isas
  llvm::Error getSupportedISAs(llvm::SmallVectorImpl<ISA> &Isa) const;
};

} // namespace luthier::hsa


//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::GpuAgent> {
  static inline luthier::hsa::GpuAgent getEmptyKey() {
    return luthier::hsa::GpuAgent(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::GpuAgent getTombstoneKey() {
    return luthier::hsa::GpuAgent(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::GpuAgent &Agent) {
    return DenseMapInfo<decltype(hsa_agent_t::handle)>::getHashValue(
        Agent.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::GpuAgent &lhs,
                      const luthier::hsa::GpuAgent &rhs) {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<luthier::hsa::GpuAgent> {
  size_t operator()(const luthier::hsa::GpuAgent &obj) const {
    return hash<unsigned long>()(obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() < rhs.hsaHandle();
  }
};

template <> struct less_equal<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() <= rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

template <> struct not_equal_to<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() != rhs.hsaHandle();
  }
};

template <> struct greater<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() > rhs.hsaHandle();
  }
};

template <> struct greater_equal<luthier::hsa::GpuAgent> {
  bool operator()(const luthier::hsa::GpuAgent &lhs,
                  const luthier::hsa::GpuAgent &rhs) const {
    return lhs.hsaHandle() >= rhs.hsaHandle();
  }
};

} // namespace std


#endif