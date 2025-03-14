//===-- LoadedCodeObjectSymbolInternal.hpp --------------------------------===//
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
/// This file defines the \c hsa::LoadedCodeObjectSymbolInternal interface,
/// which is a version of the \c hsa::LoadedCodeObjectSymbol used
/// by the internal components of Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_INTERNAL_HPP
#define LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_INTERNAL_HPP
#include "hsa/GpuAgent.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>
#include <luthier/hsa/DenseMapInfo.h>
#include <luthier/hsa/LoadedCodeObjectSymbol.h>
#include <luthier/types.h>
#include <optional>
#include <string>

namespace luthier::hsa {

/// \brief Represents a symbol inside the ELF of an \c hsa_loaded_code_object_t
/// \details Unlike <tt>hsa_executable_symbol_t</tt> where only global facing
/// symbols are enumerated by the backing <tt>hsa_executable_t</tt>, objects
/// encapsulated by this class have both <tt>STB_GLOBAL</tt> and
/// <tt>STB_LOCAL</tt> bindings. This allows for representation of symbols of
/// interest, including device functions and variables with local bindings (e.g.
/// strings used in host call print operations).
class LoadedCodeObjectSymbolInternal
    : public llvm::RTTIExtends<LoadedCodeObjectSymbolInternal,
                               LoadedCodeObjectSymbol> {
public:
  static char ID;
  /// Default constructor
  LoadedCodeObjectSymbolInternal() = default;

  /// \return GPU Agent of this symbol on success, an \c llvm::Error
  /// on failure
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<GpuAgent>>
  getAgentAsInternalHandle() const = 0;

  /// \return Loaded Code Object of this symbol
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<LoadedCodeObject>>
  getLoadedCodeObjectAsInternalHandle() const = 0;

  /// \return the executable this symbol was loaded into
  [[nodiscard]] virtual llvm::Expected<hsa_executable_t>
  getExecutable() const = 0;

  /// \return the name of the symbol on success, or an \c llvm::Error on
  /// failure
  [[nodiscard]] virtual llvm::Expected<llvm::StringRef> getName() const = 0;

  /// \return the size of the symbol
  [[nodiscard]] virtual size_t getSize() const = 0;

  /// \return the binding of the symbol
  [[nodiscard]] virtual uint8_t getBinding() const = 0;

  /// \return an \c llvm::ArrayRef<uint8_t> encapsulating the contents of
  /// this symbol on the \c GpuAgent it was loaded onto
  [[nodiscard]] virtual llvm::Expected<llvm::ArrayRef<uint8_t>>
  getLoadedSymbolContents() const = 0;

  [[nodiscard]] virtual llvm::Expected<luthier::address_t>
  getLoadedSymbolAddress() const = 0;

  /// \return the \c hsa_executable_symbol_t associated with
  /// this LCO Symbol if exists (i.e the symbol has a \c llvm::ELF::STB_GLOBAL
  /// binding), or an \c std::nullopt otherwise
  [[nodiscard]] virtual std::optional<hsa_executable_symbol_t>
  getExecutableSymbol() const = 0;

  /// Print the symbol in human-readable form.
  void virtual print(llvm::raw_ostream &OS) const = 0;

  LLVM_DUMP_METHOD void dump() const;

  [[nodiscard]] virtual size_t hash() const = 0;
};

} // namespace luthier::hsa

#endif
