//===-- GpuAgent.h ------------------------------===//
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
/// Defines a set of commonly used functionality for the
/// \c hsa_agent_t handle in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_AGENT_H
#define LUTHIER_HSA_AGENT_H
#include <hsa/hsa.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// Queries all the <tt>hsa_isa_t</tt>s supported by the \p Agent
/// \param [in] Agent the \c hsa_agent_t being queried
/// \param [in] HsaAgentIterateISAsFn the underlying \c hsa_agent_iterate_isas
/// function used to carry out this operation
/// \param [out] ISAList list of ISAs supported by the Agent
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_agent_iterate_isas
llvm::Error
getSupportedISAsOfAgent(hsa_agent_t Agent,
                 const decltype(hsa_agent_iterate_isas) &HsaAgentIterateISAsFn,
                 llvm::SmallVectorImpl<hsa_isa_t> &ISAList);

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct DenseMapInfo<hsa_agent_t> {
  static inline hsa_agent_t getEmptyKey() {
    return hsa_agent_t(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getEmptyKey()});
  }

  static inline hsa_agent_t getTombstoneKey() {
    return hsa_agent_t(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const hsa_agent_t &Agent) {
    return DenseMapInfo<decltype(hsa_agent_t::handle)>::getHashValue(
        Agent.handle);
  }

  static bool isEqual(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_agent_t> {
  size_t operator()(const hsa_agent_t &Obj) const noexcept {
    return hash<unsigned long>()(Obj.handle);
  }
};

template <> struct less<hsa_agent_t> {
  bool operator()(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) const {
    return Lhs.handle < Rhs.handle;
  }
};

template <> struct less_equal<hsa_agent_t> {
  bool operator()(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) const {
    return Lhs.handle <= Rhs.handle;
  }
};

template <> struct equal_to<hsa_agent_t> {
  bool operator()(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

template <> struct not_equal_to<hsa_agent_t> {
  bool operator()(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) const {
    return Lhs.handle != Rhs.handle;
  }
};

template <> struct greater<hsa_agent_t> {
  bool operator()(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) const {
    return Lhs.handle > Rhs.handle;
  }
};

template <> struct greater_equal<hsa_agent_t> {
  bool operator()(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) const {
    return Lhs.handle >= Rhs.handle;
  }
};

} // namespace std

#endif