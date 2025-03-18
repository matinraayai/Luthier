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
#include <luthier/hsa/LoadedCodeObjectSymbol.h>

namespace luthier::hsa {

/// \brief a version of \c hsa::LoadedCodeObjectSymbol used internally by
/// Luthier components
/// \details By design, internal Luthier components cannot work or interact with
/// HSA handles directly; However, the getter methods of
/// \c hsa::LoadedCodeObjectSymbol for agent, loaded code object, and executable
/// have to return an hsa handle since they are used externally by Luthier
/// tools.
/// Therefore, this interface exists to define an "internal" version of the
/// public-facing HSA handle getter methods of a \c hsa::LoadedCodeObjectSymbol
/// for use by Luthier's internal components
class LoadedCodeObjectSymbolInternal
    : public llvm::RTTIExtends<LoadedCodeObjectSymbolInternal,
                               LoadedCodeObjectSymbol> {
public:
  static char ID;

  /// \return GPU Agent of this symbol on success, an \c llvm::Error
  /// on failure
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<GpuAgent>>
  getAgentAsInternalHandle() const = 0;

  /// \return Loaded Code Object of this symbol
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<LoadedCodeObject>>
  getLoadedCodeObjectAsInternalHandle() const = 0;

  /// \return the executable this symbol was loaded into
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<Executable>>
  getExecutableAsInternalHandle() const = 0;
};

} // namespace luthier::hsa

#endif
