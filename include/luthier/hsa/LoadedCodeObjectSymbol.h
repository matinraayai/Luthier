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
/// This file defines the \c LoadedCodeObjectSymbol interface under
/// the \c luthier::hsa namespace. It represents all symbols inside an
/// \c hsa_loaded_code_object_t with \b all ELF bindings (not just
/// \c STB_GLOBAL like those represented by <tt>hsa_executable_symbol_t</tt>).
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_H
#define LUTHIER_HSA_LOADED_CODE_OBJECT_SYMBOL_H
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/ExtensibleRTTI.h>
#include <luthier/types.h>
#include <optional>
#include <string>

namespace luthier::hsa {

/// \brief Represents a symbol inside the ELF of an \c hsa_loaded_code_object_t
/// \details Unlike <tt>hsa_executable_symbol_t</tt> where only global facing
/// symbols are enumerated by the backing <tt>hsa_executable_t</tt>, objects
/// encapsulated by this class have both <tt>STB_GLOBAL</tt> and
/// <tt>STB_LOCAL</tt> bindings. This allows for representation of symbols of
/// interest, including device functions and variables with local bindings
class LoadedCodeObjectSymbol
    : public llvm::RTTIExtends<LoadedCodeObjectSymbol, llvm::RTTIRoot> {
public:
  static char ID;

  /// Factory method which returns the \c LoadedCodeObjectSymbol given its
  /// \c hsa_executable_symbol_t
  /// \param Symbol the \c hsa_executable_symbol_t being queried
  /// \return on success, a unique pointer of \c LoadedCodeObjectSymbol
  /// of the HSA executable symbol, or an \c llvm::Error on failure
  static llvm::Expected<std::unique_ptr<hsa::LoadedCodeObjectSymbol>>
  fromExecutableSymbol(hsa_executable_symbol_t Symbol);

  /// Queries if a \c hsa::LoadedCodeObjectSymbol is
  /// loaded on device memory at \p LoadedAddress
  /// \param LoadedAddress the device loaded address being queried
  /// \return \c nullptr if no symbol is loaded at the given address, or
  /// a unique pointer to the loaded code object symbol loaded at the
  /// given address
  static std::unique_ptr<hsa::LoadedCodeObjectSymbol>
  fromLoadedAddress(luthier::address_t LoadedAddress);

  /// Creates a deep-clone of this symbol
  virtual std::unique_ptr<hsa::LoadedCodeObjectSymbol> clone() const = 0;

  /// \return GPU Agent of this symbol on success, an \c llvm::Error
  /// on failure
  [[nodiscard]] virtual llvm::Expected<hsa_agent_t> getAgent() const = 0;

  /// \return Loaded Code Object of this symbol
  [[nodiscard]] virtual hsa_loaded_code_object_t
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
  virtual void print(llvm::raw_ostream &OS) const = 0;

  /// Prints the symbol in human-readable form onto the debug stream
  /// Useful for debugging
  virtual void dump() const = 0;
};

} // namespace luthier::hsa

#endif