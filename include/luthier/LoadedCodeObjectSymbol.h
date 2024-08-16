//===-- LoadedCodeObjectSymbol.h - Loaded Code Object Symbol ----*- C++ -*-===//
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

#include <optional>
#include <string>

#include "common/object_utils.hpp"
#include "hsa/hsa_handle_type.hpp"
#include "hsa/hsa_loaded_code_object.hpp"
#include "hsa/hsa_platform.hpp"
#include <luthier/kernel_descriptor.h>
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
  enum SymbolKind {
    SK_KERNEL,
    SK_DEVICE_FUNCTION,
    SK_VARIABLE,
    SK_EXTERNAL,
    SK_INDIRECT_FUNCTION
  };

protected:
  /// The HSA Loaded Code Object this symbol belongs to
  hsa_loaded_code_object_t BackingLCO;
  /// The LLVM Object ELF symbol of this LCO symbol;
  /// Backed by parsing the storage ELF of the LCO
  const llvm::object::ELFSymbolRef *Symbol;
  /// LLVM RTTI
  SymbolKind Kind;

  /// Protected constructor used by sub-classes
  /// \param LCO the \c hsa_loaded_code_object_t which the symbol belongs to
  /// \param Symbol a reference to the \c llvm::object::ELFSymbolRef
  /// that was obtained from parsing the storage ELF of the \p LCO and cached
  LoadedCodeObjectSymbol(hsa_loaded_code_object_t LCO,
                         const llvm::object::ELFSymbolRef &Symbol,
                         SymbolKind Kind)
      : BackingLCO(LCO), Symbol(&Symbol), Kind(Kind){};

public:
  /// Copy constructor
  LoadedCodeObjectSymbol(const LoadedCodeObjectSymbol &Other) = default;

  /// Copy assignment constructor
  LoadedCodeObjectSymbol &
  operator=(const LoadedCodeObjectSymbol &Other) = default;

  /// Move constructor
  LoadedCodeObjectSymbol &operator=(LoadedCodeObjectSymbol &&Other) = default;

  /// Factory method which constructs a \c LoadedCodeObjectSymbol from its
  /// \c hsa_executable_symbol_t
  /// \param Symbol the \c hsa_executable_symbol_t being queried
  /// \return on success, the \c LoadedCodeObjectSymbol of the HSA executable
  /// symbol, or an \c llvm::Error on failure
  static llvm::Expected<LoadedCodeObjectSymbol>
  fromExecutableSymbol(hsa_executable_symbol_t Symbol);

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
  [[nodiscard]] llvm::Expected<llvm::StringRef> getName() const {
    return Symbol->getName();
  }

  /// \return the size of the symbol
  [[nodiscard]] size_t getSize() const { return Symbol->getSize(); }

  /// \return the binding of the symbol
  [[nodiscard]] uint8_t getBinding() const { return Symbol->getBinding(); }

  /// \return an \c llvm::ArrayRef<uint8_t> encapsulating the contents of
  /// this symbol on the \c GpuAgent it was loaded onto
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getSymbolContentsOnDevice() const;

  /// \return on success, the \c hsa_executable_symbol_t associated with
  /// this LCO Symbol if it has a \c llvm::ELF::STB_GLOBAL binding, or
  /// an \c std::nullopt otherwise; On failure, an llvm::Error if an issue
  /// occurred during the process
  [[nodiscard]] llvm::Expected<std::optional<hsa_executable_symbol_t>>
  getExecutableSymbol();

  /// \return a unique hash value associated with this symbol
  [[nodiscard]] unsigned getHashValue() {
    return llvm::DenseMapInfo<std::tuple<hsa_loaded_code_object_t::handle,
                                         const llvm::object::ELFSymbolRef *>>::
        getHashValue({BackingLCO.handle, Symbol});
  }
};

} // namespace luthier::hsa

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::LoadedCodeObjectSymbol> {
  static inline luthier::hsa::LoadedCodeObjectSymbol getEmptyKey() {
    return luthier::hsa::LoadedCodeObjectSymbol{
        {DenseMapInfo<
            decltype(hsa_loaded_code_object_t::handle)>::getEmptyKey()},
        DenseMapInfo<
            decltype(const llvm::object::ELFSymbolRef *)>::getEmptyKey(),
        luthier::hsa::LoadedCodeObjectSymbol::SK_KERNEL}
        .getHashValue();
  }

  static inline luthier::hsa::LoadedCodeObjectSymbol getTombstoneKey() {
    return luthier::hsa::LoadedCodeObjectSymbol{
        {DenseMapInfo<
            decltype(hsa_loaded_code_object_t::handle)>::getTombstoneKey()},
        DenseMapInfo<
            decltype(const llvm::object::ELFSymbolRef *)>::getTombstoneKey(),
        luthier::hsa::LoadedCodeObjectSymbol::SK_KERNEL}
        .getHashValue();
  }

  static unsigned
  getHashValue(const luthier::hsa::LoadedCodeObjectSymbol &Symbol) {
    return symbol.getHashValue();
  }

  static bool isEqual(const luthier::hsa::LoadedCodeObjectSymbol &Lhs,
                      const luthier::hsa::LoadedCodeObjectSymbol &Rhs) {
    return Lhs.getHashValue() == Rhs.getHashValue();
  }
};

} // namespace llvm

namespace std {

template <> struct hash<luthier::hsa::LoadedCodeObjectSymbol> {
  size_t operator()(const luthier::hsa::LoadedCodeObjectSymbol &Obj) const {
    return hash<unsigned long>()(Obj.getHashValue());
  }
};

template <> struct less<luthier::hsa::LoadedCodeObjectSymbol> {
  bool operator()(const luthier::hsa::LoadedCodeObjectSymbol &Lhs,
                  const luthier::hsa::LoadedCodeObjectSymbol &Rhs) const {
    return Lhs.getHashValue() < Rhs.getHashValue();
  }
};

template <> struct equal_to<luthier::hsa::LoadedCodeObjectSymbol> {
  bool operator()(const luthier::hsa::LoadedCodeObjectSymbol &Lhs,
                  const luthier::hsa::LoadedCodeObjectSymbol &Rhs) const {
    return Lhs.getHashValue() == Rhs.getHashValue();
  }
};

} // namespace std

#endif