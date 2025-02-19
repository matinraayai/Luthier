//===-- ExecutableSymbol.hpp - HSA Executable Symbol Interface ------------===//
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
/// This file defines the \c hsa::ExecutableSymbol interface, representing a
/// wrapper around the \c hsa_executable_symbol handle operations in ROCr.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_SYMBOL_HPP
#define LUTHIER_HSA_EXECUTABLE_SYMBOL_HPP
#include "hsa/HandleType.hpp"
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <luthier/types.h>
#include <optional>
#include <string>

namespace luthier::hsa {

class GpuAgent;

/// \brief wrapper around \c hsa_executable_symbol_t
/// \details By design <tt>hsa::LoadedCodeObjectSymbol</tt> is the primary
/// symbol representation used by Luthier, and this wrapper is reserved for
/// only straight-forward queries for global-facing symbols (e.g. address of
/// a global variable, especially when the variable is externally defined).
class ExecutableSymbol : public HandleType<hsa_executable_symbol_t> {

public:
  /// Wrapper constructor
  /// \param Handle HSA handle of the \c hsa_executable_symbol_t
  explicit ExecutableSymbol(hsa_executable_symbol_t Handle)
      : HandleType<hsa_executable_symbol_t>(Handle) {}

  virtual ~ExecutableSymbol() = default;

  [[nodiscard]] virtual std::unique_ptr<ExecutableSymbol> clone() const = 0;

  /// \return the type of the symbol on success, a \c luthier::HsaError on
  /// failure
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_TYPE
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] virtual llvm::Expected<hsa_symbol_kind_t> getType() const = 0;

  /// \return the name of the symbol on success, a \c luthier::HsaError on
  /// failure
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_NAME
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] virtual llvm::Expected<llvm::StringRef> getName() const = 0;

  /// \return size of the symbol if symbol is a variable
  /// (0 if the variable is external), otherwise a \c luthier::HsaError on
  /// failure
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_SIZE
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] virtual llvm::Expected<size_t> getSize() const = 0;

  /// \return the address of the symbol if the symbol is a variable, or the
  /// address of the kernel descriptor if the symbol is a kernel; Otherwise,
  /// a \c luthier::HsaError
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] virtual llvm::Expected<luthier::address_t>
  getAddress() const = 0;

  /// \return the agent of the symbol;
  /// \note This is safe to use since all symbols used in HSA are of
  /// agent allocation and not of program allocation
  /// \sa HSA_EXECUTABLE_SYMBOL_INFO_AGENT
  /// \sa hsa_executable_symbol_get_info
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<GpuAgent>>
  getAgent() const = 0;
};

} // namespace luthier::hsa

DECLARE_LLVM_MAP_INFO_STRUCTS_FOR_HANDLE_TYPE(luthier::hsa::ExecutableSymbol,
                                              hsa_executable_symbol_t)

#endif