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
/// This file defines the \c hsa::LoadedCodeObjectSymbol interface.
/// It represents all symbols of interest inside an \c hsa_loaded_code_object_t
/// regardless of their binding type, unlike <tt>hsa_executable_symbol_t</tt>
/// which only include symbols with a \c STB_GLOBAL binding.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_H
#define LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_H
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/Support/ExtensibleRTTI.h>
#include <luthier/hsa/DenseMapInfo.h>
#include <luthier/types.h>
#include <string>

namespace luthier::hsa {

/// \brief Represents a symbol inside the ELF of an \c hsa_loaded_code_object_t
/// \details Unlike <tt>hsa_executable_symbol_t</tt> where only global facing
/// symbols are enumerated by the backing <tt>hsa_executable_t</tt>, objects
/// encapsulated by this class have both <tt>STB_GLOBAL</tt> and
/// <tt>STB_LOCAL</tt> bindings. This allows for representation of symbols of
/// interest, including device functions and variables with local bindings (e.g.
/// strings used in host call print operations).
class LoadedCodeObjectSymbol
    : public llvm::RTTIExtends<LoadedCodeObjectSymbol, llvm::RTTIRoot> {
public:
  static char ID;

  /// \return a deep clone copy of the symbol
  [[nodiscard]] virtual std::unique_ptr<LoadedCodeObjectSymbol>
  clone() const = 0;

  /// Equality operator
  virtual bool operator==(const LoadedCodeObjectSymbol &Other) const = 0;

  /// Factory method which returns the \c LoadedCodeObjectSymbol given its
  /// \c hsa_executable_symbol_t
  /// \param Symbol the \c hsa_executable_symbol_t being queried
  /// \return on success, a const reference to a cached
  /// \c LoadedCodeObjectSymbol of the HSA executable symbol, or an
  /// \c llvm::Error on failure
  virtual llvm::Error fromExecutableSymbol(hsa_executable_symbol_t Symbol) = 0;

  /// Queries if a \c hsa::LoadedCodeObjectSymbol is
  /// loaded on device memory at \p LoadedAddress
  /// \param LoadedAddress the device loaded address being queried
  /// \return \c nullptr if no symbol is loaded at the given address, or
  /// a \c const pointer to the symbol loaded at the given address
  virtual llvm::Error fromLoadedAddress(luthier::address_t LoadedAddress) = 0;

  /// \return GPU Agent of this symbol on success, an \c llvm::Error
  /// on failure
  [[nodiscard]] virtual llvm::Expected<hsa_agent_t> getAgent() const = 0;

  /// \return Loaded Code Object of this symbol
  [[nodiscard]] virtual llvm::Expected<hsa_loaded_code_object_t>
  getLoadedCodeObject() const = 0;

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

/// Equal-to struct used to allow convenient look-ups of symbols inside
/// STL containers
template <
    typename SymbolType,
    std::enable_if_t<std::is_base_of_v<LoadedCodeObjectSymbol, SymbolType>,
                     bool> = true>
struct LoadedCodeObjectSymbolEqualTo {
  using is_transparent = void;

  template <typename Dt, typename Dt2>
  bool operator()(const std::unique_ptr<SymbolType, Dt> &Lhs,
                  const std::unique_ptr<SymbolType, Dt2> &Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs.get() == Rhs.get();
  }

  template <typename Dt>
  bool operator()(const std::unique_ptr<SymbolType, Dt> &Lhs,
                  const std::shared_ptr<SymbolType> &Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs.get() == Rhs.get();
  }

  template <typename Dt>
  bool operator()(const std::unique_ptr<SymbolType, Dt> &Lhs,
                  const SymbolType *Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs.get() == Rhs;
  }

  template <typename Dt>
  bool operator()(const std::unique_ptr<SymbolType, Dt> &Lhs,
                  const SymbolType &Rhs) const {
    return Lhs && *Lhs == Rhs;
  }

  bool operator()(const std::shared_ptr<SymbolType> &Lhs,
                  const std::shared_ptr<SymbolType> &Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs.get() == Rhs.get();
  }

  template <typename Dt>
  bool operator()(const std::shared_ptr<SymbolType> &Lhs,
                  const std::unique_ptr<SymbolType, Dt> &Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs.get() == Rhs.get();
  }

  bool operator()(const std::shared_ptr<SymbolType> &Lhs,
                  const SymbolType *Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs.get() == Rhs;
  }

  bool operator()(const std::shared_ptr<SymbolType> &Lhs,
                  const SymbolType &Rhs) const {
    return Lhs && *Lhs == Rhs;
  }

  bool operator()(const SymbolType *Lhs, const SymbolType *Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs == Rhs;
  }

  bool operator()(const SymbolType *Lhs,
                  const std::unique_ptr<SymbolType> &Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs == Rhs.get();
  }

  bool operator()(const SymbolType *Lhs,
                  const std::shared_ptr<SymbolType> &Rhs) const {
    if (Lhs && Rhs)
      return *Lhs == *Rhs;
    else
      return Lhs == Rhs.get();
  }

  bool operator()(const SymbolType *Lhs, const SymbolType &Rhs) const {
    return Lhs && *Lhs == Rhs;
  }

  bool operator()(const SymbolType &Lhs, const SymbolType &Rhs) const {
    return Lhs == Rhs;
  }

  template <typename Dt>
  bool operator()(const SymbolType &Lhs,
                  const std::unique_ptr<SymbolType, Dt> &Rhs) const {
    return Rhs && Lhs == *Rhs;
  }

  bool operator()(const SymbolType &Lhs,
                  const std::shared_ptr<SymbolType> &Rhs) const {
    return Rhs && Lhs == *Rhs;
  }

  bool operator()(const SymbolType &Lhs, const SymbolType *Rhs) const {
    return Rhs && Lhs == *Rhs;
  }
};

/// \brief Hash struct to allow convenient look-up of symbols inside STL
/// containers
template <
    typename SymbolType,
    std::enable_if_t<std::is_base_of_v<LoadedCodeObjectSymbol, SymbolType>,
                     bool> = true>
struct LoadedCodeObjectSymbolHash {
  using is_transparent = void;

  using transparent_key_equal = LoadedCodeObjectSymbolEqualTo<SymbolType>;

  std::size_t operator()(const std::unique_ptr<SymbolType> &Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value((SymbolType *)nullptr);
  }

  std::size_t
  operator()(const std::unique_ptr<const SymbolType> &Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value((const SymbolType *)nullptr);
  }

  std::size_t operator()(const std::shared_ptr<SymbolType> &Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value((SymbolType *)nullptr);
  }

  std::size_t
  operator()(const std::shared_ptr<const SymbolType> &Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value((const SymbolType *)nullptr);
  }

  std::size_t operator()(const SymbolType *Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value((SymbolType *)nullptr);
  }

  std::size_t operator()(const SymbolType &Symbol) const {
    return Symbol.hash();
  }
};

} // namespace luthier::hsa

#endif
