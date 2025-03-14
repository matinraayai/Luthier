//===-- LoadedCodeObjectSymbolImpl.hpp ------------------------------------===//
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
/// This file defines the \c hsa::LoadedCodeObjectSymbolImpl class, which
/// implements the concrete version of the \c hsa::LoadedCodeObjectSymbol
/// interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_IMPL_HPP
#define LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_IMPL_HPP
#include "LoadedCodeObjectImpl.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/Object/ELFObjectFile.h>
#include <luthier/hsa/DenseMapInfo.h>
#include <luthier/hsa/LoadedCodeObjectSymbol.h>
#include <luthier/types.h>
#include <optional>
#include <string>

namespace luthier::hsa {

/// \brief concrete implementation of the \c LoadedCodeObjectSymbol
class LoadedCodeObjectSymbolImpl
    : public llvm::RTTIExtends<LoadedCodeObjectSymbolImpl,
                               LoadedCodeObjectSymbolInternal> {
public:
  static char ID;

protected:
  /// The HSA Loaded Code Object this symbol belongs to
  hsa::LoadedCodeObjectImpl BackingLCO{};
  /// The LLVM Object ELF symbol of this LCO symbol;
  /// Backed by parsing the storage ELF of the LCO
  std::unique_ptr<llvm::object::ELFSymbolRef> Symbol{};
  /// The HSA executable symbol equivalent, if exists
  std::optional<hsa::ExecutableSymbol> ExecutableSymbol;

  /// Constructor used by sub-classes
  /// \param LCO the \c hsa_loaded_code_object_t which the symbol belongs to
  /// \param Symbol a reference to the \c llvm::object::ELFSymbolRef
  /// that was obtained from parsing the storage ELF of the \p LCO and cached
  /// \param ExecutableSymbol the \c hsa_executable_symbol_t equivalent of
  /// the <tt>LoadedCodeObjectSymbol</tt> if exists
  LoadedCodeObjectSymbolImpl(
      hsa::LoadedCodeObjectImpl LCO, llvm::object::ELFSymbolRef Symbol,
      std::optional<hsa::ExecutableSymbol> ExecutableSymbol);

public:
  LoadedCodeObjectSymbolImpl() = default;

  [[nodiscard]] std::unique_ptr<LoadedCodeObjectSymbol> clone() const override {
    return std::unique_ptr<LoadedCodeObjectSymbolImpl>(
        new LoadedCodeObjectSymbolImpl(
            this->BackingLCO, llvm::object::ELFSymbolRef(*this->Symbol),
            this->ExecutableSymbol));
  }

  bool operator==(const LoadedCodeObjectSymbol &Other) const override {
    if (auto *OtherAsSymImpl =
            llvm::dyn_cast<LoadedCodeObjectSymbolImpl>(&Other)) {
      bool Out = Symbol == OtherAsSymImpl->Symbol &&
                 BackingLCO.handle == OtherAsSymImpl->BackingLCO.handle;
      if (ExecutableSymbol.has_value()) {
        return Out && (OtherAsSymImpl->ExecutableSymbol.has_value() &&
                       ExecutableSymbol->handle ==
                           OtherAsSymImpl->ExecutableSymbol->handle);
      } else
        return Out && !OtherAsSymImpl->ExecutableSymbol.has_value();
    } else
      return false;
  }

  llvm::Error
  fromExecutableSymbol(hsa_executable_symbol_t Symbol) override final;

  llvm::Error
  fromLoadedAddress(luthier::address_t LoadedAddress) override final;

  [[nodiscard]] llvm::Expected<hsa_agent_t> getAgent() const override;

  [[nodiscard]] llvm::Expected<hsa_loaded_code_object_t>
  getLoadedCodeObject() const override {
    return BackingLCO;
  }

  [[nodiscard]] llvm::Expected<hsa_executable_t> getExecutable() const override;

  [[nodiscard]] llvm::Expected<llvm::StringRef> getName() const override;

  [[nodiscard]] size_t getSize() const override;

  [[nodiscard]] uint8_t getBinding() const override;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getLoadedSymbolContents() const override;

  [[nodiscard]] llvm::Expected<luthier::address_t>
  getLoadedSymbolAddress() const override;

  [[nodiscard]] std::optional<hsa_executable_symbol_t>
  getExecutableSymbol() const override;

  /// Print the symbol in human-readable form.
  void print(llvm::raw_ostream &OS) const override;

  [[nodiscard]] inline size_t hash() const override {
    llvm::object::DataRefImpl Raw = Symbol->getRawDataRefImpl();
    return llvm::hash_combine(
        BackingLCO.handle, Raw.p, Raw.d.a, Raw.d.b,
        ExecutableSymbol.has_value() ? ExecutableSymbol->handle : 0);
  }
};

} // namespace luthier::hsa

#endif
