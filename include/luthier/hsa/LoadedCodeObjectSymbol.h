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
/// This file defines the \c hsa::LoadedCodeObjectSymbol class.
/// It represents all symbols of interest inside an \c hsa_loaded_code_object_t
/// regardless of their binding type, unlike <tt>hsa_executable_symbol_t</tt>
/// which only include symbols with a \c STB_GLOBAL binding.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_H
#define LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_H
#include "luthier/hsa/ApiTable.h"
#include "luthier/hsa/LoadedCodeObject.h"
#include "luthier/object/AMDGCNObjectFile.h"
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/Object/ELFObjectFile.h>
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
class LoadedCodeObjectSymbol {
public:
  enum SymbolKind { SK_KERNEL, SK_DEVICE_FUNCTION, SK_VARIABLE, SK_EXTERNAL };

protected:
  /// The HSA Loaded Code Object this symbol belongs to
  hsa_loaded_code_object_t BackingLCO{};
  /// Parsed storage ELF of the LCO, to ensure \c Symbol stays valid
  luthier::object::AMDGCNObjectFile &StorageELF;
  /// The LLVM Object ELF symbol of this LCO symbol;
  /// Backed by parsing the storage ELF of the LCO
  llvm::object::ELFSymbolRef Symbol;
  /// LLVM RTTI
  SymbolKind Kind;
  /// The HSA executable symbol equivalent, if exists
  std::optional<hsa_executable_symbol_t> ExecutableSymbol;

  /// Constructor used by subclasses
  /// \param LCO the \c hsa_loaded_code_object_t which the symbol belongs to
  /// \param StorageELF the \c luthier::AMDGCNObjectFile of \p Symbol
  /// \param Symbol a reference to the \c llvm::object::ELFSymbolRef
  /// that was obtained from parsing the storage ELF of the \p LCO and cached
  /// \param Kind the type of the symbol being constructed
  /// \param ExecutableSymbol the \c hsa_executable_symbol_t equivalent of
  /// the <tt>LoadedCodeObjectSymbol</tt> if exists
  LoadedCodeObjectSymbol(
      hsa_loaded_code_object_t LCO,
      luthier::object::AMDGCNObjectFile &StorageELF,
      llvm::object::ELFSymbolRef Symbol, SymbolKind Kind,
      std::optional<hsa_executable_symbol_t> ExecutableSymbol);

public:
  /// Disallowed copy construction
  LoadedCodeObjectSymbol(const LoadedCodeObjectSymbol &) = delete;

  /// Disallowed assignment operation
  LoadedCodeObjectSymbol &operator=(const LoadedCodeObjectSymbol &) = delete;

  virtual ~LoadedCodeObjectSymbol() = default;

  /// \return a deep clone copy of the
  [[nodiscard]] virtual std::unique_ptr<LoadedCodeObjectSymbol> clone() const {
    return std::unique_ptr<LoadedCodeObjectSymbol>(new LoadedCodeObjectSymbol(
        this->BackingLCO, this->StorageELF, this->Symbol, this->Kind,
        this->ExecutableSymbol));
  }

  /// Equality operator
  bool operator==(const LoadedCodeObjectSymbol &Other) const {
    bool Out = Symbol == Other.Symbol &&
               BackingLCO.handle == Other.BackingLCO.handle &&
               Kind == Other.Kind;
    if (ExecutableSymbol.has_value()) {
      return Out &&
             (Other.ExecutableSymbol.has_value() &&
              ExecutableSymbol->handle == Other.ExecutableSymbol->handle);
    } else
      return Out && !Other.ExecutableSymbol.has_value();
  }

  /// Factory method which returns the \c LoadedCodeObjectSymbol given its
  /// \c hsa_executable_symbol_t
  /// \param Symbol the \c hsa_executable_symbol_t being queried
  /// \return on success, a const reference to a cached
  /// \c LoadedCodeObjectSymbol of the HSA executable symbol, or an
  /// \c llvm::Error on failure
  static llvm::Expected<std::unique_ptr<hsa::LoadedCodeObjectSymbol>>
  fromExecutableSymbol(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       const hsa_ven_amd_loader_1_03_pfn_t &LoaderApi,
                       hsa_executable_symbol_t Symbol);

  /// Queries if a \c hsa::LoadedCodeObjectSymbol is
  /// loaded on device memory at \p LoadedAddress
  /// \param LoadedAddress the device loaded address being queried
  /// \return \c nullptr if no symbol is loaded at the given address, or
  /// a \c const pointer to the symbol loaded at the given address
  static llvm::Expected<std::unique_ptr<hsa::LoadedCodeObjectSymbol>>
  fromLoadedAddress(const ApiTableContainer<::CoreApiTable> &CoreApi,
                    const hsa_ven_amd_loader_1_03_pfn_t &LoaderApi,
                    luthier::address_t LoadedAddress);

  /// \return the \c SymbolKind of this symbol
  [[nodiscard]] SymbolKind getType() const { return Kind; }

  /// \return GPU Agent of this symbol on success, an \c llvm::Error
  /// on failure
  template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
  [[nodiscard]] llvm::Expected<hsa_agent_t>
  getAgent(const LoaderTableType &VenLoaderTable) const {
    return hsa::loadedCodeObjectGetAgent(VenLoaderTable, BackingLCO);
  }

  /// \return Loaded Code Object of this symbol
  [[nodiscard]] hsa_loaded_code_object_t getLoadedCodeObject() const {
    return BackingLCO;
  }

  /// \return the executable this symbol was loaded into
  template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
  [[nodiscard]] llvm::Expected<hsa_executable_t>
  getExecutable(const LoaderTableType &VenLoaderTable) const {
    return hsa::loadedCodeObjectGetExecutable(VenLoaderTable, BackingLCO);
  }

  /// \return the name of the symbol on success, or an \c llvm::Error on
  /// failure
  [[nodiscard]] llvm::Expected<llvm::StringRef> getName() const;

  /// \return the size of the symbol
  [[nodiscard]] size_t getSize() const;

  /// \return the binding of the symbol
  [[nodiscard]] uint8_t getBinding() const;

  /// \return an \c llvm::ArrayRef<uint8_t> encapsulating the contents of
  /// this symbol on the \c GpuAgent it was loaded onto
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getLoadedSymbolContents(
      const hsa_ven_amd_loader_1_03_pfn_t &VenLoaderTable) const;

  [[nodiscard]] llvm::Expected<luthier::address_t> getLoadedSymbolAddress(
      const hsa_ven_amd_loader_1_03_pfn_t &VenLoaderTable) const;

  /// \return the \c hsa_executable_symbol_t associated with
  /// this LCO Symbol if exists (i.e the symbol has a \c llvm::ELF::STB_GLOBAL
  /// binding), or an \c std::nullopt otherwise
  [[nodiscard]] std::optional<hsa_executable_symbol_t>
  getExecutableSymbol() const;

  /// Print the symbol in human-readable form.
  void print(llvm::raw_ostream &OS) const;

  void dump() const;

  [[nodiscard]] inline size_t hash() const {
    llvm::object::DataRefImpl Raw = Symbol.getRawDataRefImpl();
    return llvm::hash_combine(
        BackingLCO.handle, Raw.p, Raw.d.a, Raw.d.b, Kind,
        ExecutableSymbol.has_value() ? ExecutableSymbol->handle : 0);
  }
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
      return llvm::hash_value(static_cast<SymbolType *>(nullptr));
  }

  std::size_t
  operator()(const std::unique_ptr<const SymbolType> &Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value(static_cast<const SymbolType *>(nullptr));
  }

  std::size_t operator()(const std::shared_ptr<SymbolType> &Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value(static_cast<SymbolType *>(nullptr));
  }

  std::size_t
  operator()(const std::shared_ptr<const SymbolType> &Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value(static_cast<const SymbolType *>(nullptr));
  }

  std::size_t operator()(const SymbolType *Symbol) const {
    if (Symbol)
      return Symbol->hash();
    else
      return llvm::hash_value(static_cast<SymbolType *>(nullptr));
  }

  std::size_t operator()(const SymbolType &Symbol) const {
    return Symbol.hash();
  }
};

} // namespace luthier::hsa

#endif
