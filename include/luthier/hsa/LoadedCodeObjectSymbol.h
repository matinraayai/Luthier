//===-- LoadedCodeObjectSymbol.h - Loaded Code Object Symbol ----*- C++ -*-===//
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
/// This file defines the \c LoadedCodeObjectSymbol under the \c luthier::hsa
/// namespace. It represents all symbols inside an \c hsa_loaded_code_object_t
/// with \b all ELF bindings (not just \c STB_GLOBAL like those represented by
/// <tt>hsa_executable_symbol_t</tt>).
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LOADED_CODE_OBJECT_SYMBOL_H
#define LUTHIER_LOADED_CODE_OBJECT_SYMBOL_H
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Object/ELFObjectFile.h>

#include <optional>
#include <string>

#include <luthier/types.h>

namespace luthier::hsa {

/// \brief Represents a symbol inside the ELF of an \c hsa_loaded_code_object_t
/// \details Unlike <tt>hsa_executable_symbol_t</tt> where only global facing
/// symbols are enumerated by the backing <tt>hsa_executable_t</tt>, objects
/// encapsulated by this class have both <tt>STB_GLOBAL</tt> and
/// <tt>STB_LOCAL</tt> bindings. This allows for representation of symbols of
/// interest, including device functions and variables with local bindings (e.g.
/// strings used in host call print operations).
class LoadedCodeObjectSymbol {
public:
  enum SymbolKind { SK_KERNEL, SK_DEVICE_FUNCTION, SK_VARIABLE, SK_EXTERNAL };

protected:
  /// The HSA Loaded Code Object this symbol belongs to
  hsa_loaded_code_object_t BackingLCO;
  /// The LLVM Object ELF symbol of this LCO symbol;
  /// Backed by parsing the storage ELF of the LCO
  const llvm::object::ELFSymbolRef Symbol;
  /// LLVM RTTI
  SymbolKind Kind;
  /// The HSA executable symbol equivalent, if exists
  std::optional<hsa_executable_symbol_t> ExecutableSymbol;

  /// Constructor used by sub-classes
  /// \param LCO the \c hsa_loaded_code_object_t which the symbol belongs to
  /// \param Symbol a reference to the \c llvm::object::ELFSymbolRef
  /// that was obtained from parsing the storage ELF of the \p LCO and cached
  /// \param Kind the type of the symbol being constructed
  /// \param ExecutableSymbol the \c hsa_executable_symbol_t equivalent of
  /// the <tt>LoadedCodeObjectSymbol</tt> if exists
  LoadedCodeObjectSymbol(
      hsa_loaded_code_object_t LCO, llvm::object::ELFSymbolRef Symbol,
      SymbolKind Kind, std::optional<hsa_executable_symbol_t> ExecutableSymbol);

public:
  /// Disallowed copy construction
  LoadedCodeObjectSymbol(const LoadedCodeObjectSymbol &) = delete;

  /// Disallowed assignment operation
  LoadedCodeObjectSymbol &operator=(const LoadedCodeObjectSymbol &) = delete;

  /// Factory method which returns the \c LoadedCodeObjectSymbol given its
  /// \c hsa_executable_symbol_t
  /// \param Symbol the \c hsa_executable_symbol_t being queried
  /// \return on success, a const reference to a cached
  /// \c LoadedCodeObjectSymbol of the HSA executable symbol, or an
  /// \c llvm::Error on failure
  static llvm::Expected<const hsa::LoadedCodeObjectSymbol &>
  fromExecutableSymbol(hsa_executable_symbol_t Symbol);

  /// Queries if a \c hsa::LoadedCodeObjectSymbol is
  /// loaded on device memory at \p LoadedAddress
  /// \param LoadedAddress the device loaded address being queried
  /// \return \c nullptr if no symbol is loaded at the given address, or
  /// a \c const pointer to the symbol loaded at the given address
  static const hsa::LoadedCodeObjectSymbol *
  fromLoadedAddress(luthier::address_t LoadedAddress);

  /// \return the \c SymbolKind of this symbol
  [[nodiscard]] SymbolKind getType() const { return Kind; }

  /// \return GPU Agent of this symbol on success, an \c llvm::Error
  /// on failure
  [[nodiscard]] llvm::Expected<hsa_agent_t> getAgent() const;

  /// \return Loaded Code Object of this symbol
  [[nodiscard]] hsa_loaded_code_object_t getLoadedCodeObject() const {
    return BackingLCO;
  }

  /// \return the executable this symbol was loaded into
  [[nodiscard]] llvm::Expected<hsa_executable_t> getExecutable() const;

  /// \return the name of the symbol on success, or an \c llvm::Error on
  /// failure
  [[nodiscard]] llvm::Expected<llvm::StringRef> getName() const;

  /// \return the size of the symbol
  [[nodiscard]] size_t getSize() const;

  /// \return the binding of the symbol
  [[nodiscard]] uint8_t getBinding() const;

  /// \return an \c llvm::ArrayRef<uint8_t> encapsulating the contents of
  /// this symbol on the \c GpuAgent it was loaded onto
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getLoadedSymbolContents() const;


  [[nodiscard]] llvm::Expected<luthier::address_t>
      getLoadedSymbolAddress() const;

  /// \return the \c hsa_executable_symbol_t associated with
  /// this LCO Symbol if exists (i.e the symbol has a \c llvm::ELF::STB_GLOBAL
  /// binding), or an \c std::nullopt otherwise
  [[nodiscard]] std::optional<hsa_executable_symbol_t>
  getExecutableSymbol() const;


  /// Print the symbol in human-readable form.
  void print(llvm::raw_ostream &OS) const;

  void dump() const;
};

} // namespace luthier::hsa

#endif