//===-- Agent.h -------------------------------------------------*- C++ -*-===//
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
/// Defines a set of commonly used functionality for the \c hsa_agent_t handle
/// in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_AGENT_H
#define LUTHIER_HSA_AGENT_H
#include "luthier/hsa/ApiTable.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// Queries all the <tt>hsa_isa_t</tt>s supported by the \p Agent
/// \param [in] CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param [in] Agent the \c hsa_agent_t being queried
/// \param [out] ISAList list of ISAs supported by the Agent
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_agent_iterate_isas
llvm::Error
agentGetSupportedISAs(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_agent_t Agent,
                      llvm::SmallVectorImpl<hsa_isa_t> &ISAList);

/// Iterates over the supported \c hsa_isa_t list of \p Agent and invokes the
/// \p Callback
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Agent the agent being queried
/// \param Callback a callback function invoked for each \c hsa_isa_t of the
/// <tt>Agent</tt>. If the callback doesn't return a success error value, it
/// will halt the iteration, and the error will be returned
/// \return \c llvm::Error indication the success or failure of the operation
llvm::Error
agentIterateISAs(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_agent_t Agent,
                 const std::function<llvm::Error(hsa_isa_t)> &Callback);

/// Iterates over the supported \c hsa_isa_t list of \p Agent and finds and
/// returns the first \c hsa_isa_t of the \p Agent that the \p Predicate returns
/// \c true on
/// \param Agent the agent being queried
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Predicate a predicate function invoked for each \c hsa_isa_t of the
/// <tt>Agent</tt>. If the callback returns a failure error value, it
/// will halt the iteration, and the error will be returned
/// \return Expects the first \c hsa_isa_t entry found by the predicate; Expects
/// a \c std::nullopt if no ISA entry was found by the predicate
llvm::Expected<std::optional<hsa_isa_t>> agentFindFirstISA(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_agent_t Agent,
    const std::function<llvm::Expected<bool>(hsa_isa_t)> &Predicate);

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

template <> struct llvm::DenseMapInfo<hsa_agent_t> {
  static hsa_agent_t getEmptyKey() {
    return hsa_agent_t(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getEmptyKey()});
  }

  static hsa_agent_t getTombstoneKey() {
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
}; // namespace llvm

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

template <> struct equal_to<hsa_agent_t> {
  bool operator()(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

} // namespace std

#endif