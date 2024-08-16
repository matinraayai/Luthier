//===-- ExecutableSymbol.hpp - HSA Executable Symbol Wrapper --------------===//
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
/// This file defines the \c ExecutableSymbol class under the \c luthier::hsa
/// namespace, representing a wrapper around the \c hsa_executable_symbol handle
/// type.
//===----------------------------------------------------------------------===//

#ifndef HSA_EXECUTABLE_SYMBOL_HPP
#define HSA_EXECUTABLE_SYMBOL_HPP
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>

#include <optional>
#include <string>

#include "hsa/hsa_handle_type.hpp"
#include <luthier/types.h>

namespace luthier::hsa {

class GpuAgent;

/// \brief wrapper around \c hsa_executable_symbol_t
/// \details By design <tt>hsa::LoadedCodeObjectSymbol</tt> is the primary
/// symbol representation used by Luthier, and this wrapper is reserved for
/// only straight-forward queries for global-facing symbols (e.g. address of
/// a global variable, especially when the variable is externally defined).
class ExecutableSymbol final : public HandleType<hsa_executable_symbol_t> {

public:
  /// Wrapper constructor
  /// \param Handle HSA handle of the \c hsa_executable_symbol_t
  explicit ExecutableSymbol(hsa_executable_symbol_t Handle)
      : HandleType<hsa_executable_symbol_t>(Handle) {}

  /// \return the type of the symbol on success, a \c luthier::HsaError on
  /// failure
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_TYPE
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] llvm::Expected<hsa_symbol_kind_t> getType() const;

  /// \return the name of the symbol on success, a \c luthier::HsaError on
  /// failure
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_NAME
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] llvm::Expected<llvm::StringRef> getName() const;

  /// \return size of the symbol if symbol is a variable
  /// (0 if the variable is external), otherwise a \c luthier::HsaError on
  /// failure
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_SIZE
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] llvm::Expected<size_t> getSize() const;

  /// \return the address of the symbol if the symbol is a variable, or the
  /// address of the kernel descriptor if the symbol is a kernel; Otherwise,
  /// a \c luthier::HsaError
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] llvm::Expected<luthier::address_t> getAddress() const;

  /// \return the agent of the symbol;
  /// \note This is safe to use since all symbols used in HSA are of
  /// agent allocation and not of program allocation
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_AGENT
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;
};

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::ExecutableSymbol> {
  static inline luthier::hsa::ExecutableSymbol getEmptyKey() {
    return luthier::hsa::ExecutableSymbol{{DenseMapInfo<
        decltype(hsa_executable_symbol_t::handle)>::getEmptyKey()}};
  }

  static inline luthier::hsa::ExecutableSymbol getTombstoneKey() {
    return luthier::hsa::ExecutableSymbol{{DenseMapInfo<
        decltype(hsa_executable_symbol_t::handle)>::getTombstoneKey()}};
  }

  static unsigned getHashValue(const luthier::hsa::ExecutableSymbol &Symbol) {
    return DenseMapInfo<decltype(hsa_executable_symbol_t::handle)>::
        getHashValue(Symbol.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::ExecutableSymbol &Lhs,
                      const luthier::hsa::ExecutableSymbol &Rhs) {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<luthier::hsa::ExecutableSymbol> {
  size_t operator()(const luthier::hsa::ExecutableSymbol &Obj) const {
    return hash<unsigned long>()(Obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::ExecutableSymbol> {
  bool operator()(const luthier::hsa::ExecutableSymbol &Lhs,
                  const luthier::hsa::ExecutableSymbol &Rhs) const {
    return Lhs.hsaHandle() < Rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::ExecutableSymbol> {
  bool operator()(const luthier::hsa::ExecutableSymbol &Lhs,
                  const luthier::hsa::ExecutableSymbol &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace std

#endif