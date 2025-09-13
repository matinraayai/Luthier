//===-- ExecutableSymbol.h ---------------------------------------*- C++-*-===//
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
/// \c hsa_executable_symbol_t handle in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_SYMBOL_H
#define LUTHIER_HSA_EXECUTABLE_SYMBOL_H
#include "luthier/hsa/ApiTable.h"
#include <llvm/Support/Error.h>
#include <string>

namespace luthier::hsa {

/// Queries the type of the \p Symbol
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Symbol the \c hsa_executable_symbol_t being queried
/// \return Expects the type of the symbol on success
/// \sa HSA_EXECUTABLE_SYMBOL_INFO_TYPE
/// \sa hsa_executable_symbol_get_info
[[nodiscard]] llvm::Expected<hsa_symbol_kind_t>
executableSymbolGetType(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        hsa_executable_symbol_t Symbol);

/// Queries the name of the \p Symbol
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Symbol the \c hsa_executable_symbol_t being queried
/// \return Expects the name of the symbol on operation success
/// \sa HSA_EXECUTABLE_SYMBOL_INFO_NAME
/// \sa HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH
/// \sa hsa_executable_symbol_get_info
[[nodiscard]] llvm::Expected<std::string>
executableSymbolGetName(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        hsa_executable_symbol_t Symbol);

/// Queries the size of the \p Symbol
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Symbol the \c hsa_executable_symbol_t being queried
/// \return on success, size of the symbol if symbol is a variable
/// (0 if the variable is external)
/// \sa HSA_EXECUTABLE_SYMBOL_INFO_SIZE
/// \sa hsa_executable_symbol_get_info
[[nodiscard]] llvm::Expected<size_t>
executableSymbolGetSymbolSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                              hsa_executable_symbol_t Symbol);

/// Queries the loaded address of the \p Symbol
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Symbol the \c hsa_executable_symbol_t being queried
/// \return on success, the address of the symbol if the symbol is a variable,
/// or the address of the kernel descriptor if the symbol is a kernel
/// \sa HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT
/// \sa HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS
/// \sa hsa_executable_symbol_get_info
[[nodiscard]] llvm::Expected<uint64_t>
executableSymbolGetAddress(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           hsa_executable_symbol_t Symbol);

/// Queries the \c hsa_agent_t associated with the \p Symbol
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Symbol the \c hsa_executable_symbol_t being queried
/// \return Expects the agent of the symbol on operation success
/// \note This is safe to use since all symbols used in HSA are of
/// agent allocation and not of program allocation
/// \sa HSA_EXECUTABLE_SYMBOL_INFO_AGENT
/// \sa hsa_executable_symbol_get_info
[[nodiscard]] llvm::Expected<hsa_agent_t>
executableSymbolGetAgent(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         hsa_executable_symbol_t Symbol);

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

template <> struct llvm::DenseMapInfo<hsa_executable_symbol_t> {
  static hsa_executable_symbol_t getEmptyKey() {
    return hsa_executable_symbol_t{{DenseMapInfo<
        decltype(hsa_executable_symbol_t::handle)>::getEmptyKey()}};
  }

  static hsa_executable_symbol_t getTombstoneKey() {
    return hsa_executable_symbol_t{{DenseMapInfo<
        decltype(hsa_executable_symbol_t::handle)>::getTombstoneKey()}};
  }

  static unsigned getHashValue(const hsa_executable_symbol_t &Symbol) {
    return DenseMapInfo<
        decltype(hsa_executable_symbol_t::handle)>::getHashValue(Symbol.handle);
  }

  static bool isEqual(const hsa_executable_symbol_t &Lhs,
                      const hsa_executable_symbol_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
}; // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_executable_symbol_t> {
  size_t operator()(const hsa_executable_symbol_t &Obj) const noexcept {
    return hash<unsigned long>()(Obj.handle);
  }
};

template <> struct equal_to<hsa_executable_symbol_t> {
  bool operator()(const hsa_executable_symbol_t &Lhs,
                  const hsa_executable_symbol_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

} // namespace std

#endif