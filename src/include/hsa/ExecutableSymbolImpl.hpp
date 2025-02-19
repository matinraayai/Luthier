//===-- ExecutableSymbol.hpp - HSA Executable Symbol Wrapper --------------===//
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
/// This file defines the \c hsa::ExecutableSymbolImpl which provides the
/// concrete implementation for the \c hsa::ExecutableSymbol interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_SYMBOL_IMPL_HPP
#define LUTHIER_HSA_EXECUTABLE_SYMBOL_IMPL_HPP
#include "hsa/ExecutableSymbol.hpp"
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <luthier/types.h>
#include <optional>
#include <string>

namespace luthier::hsa {

class ExecutableSymbolImpl : public ExecutableSymbol {

public:
  /// Wrapper constructor
  /// \param Handle HSA handle of the \c hsa_executable_symbol_t
  explicit ExecutableSymbolImpl(hsa_executable_symbol_t Handle)
      : ExecutableSymbol(Handle) {}

  [[nodiscard]] std::unique_ptr<ExecutableSymbol> clone() const override;

  [[nodiscard]] llvm::Expected<hsa_symbol_kind_t> getType() const override;

  [[nodiscard]] llvm::Expected<llvm::StringRef> getName() const override;

  [[nodiscard]] llvm::Expected<size_t> getSize() const override;

  [[nodiscard]] llvm::Expected<luthier::address_t> getAddress() const override;

  [[nodiscard]] llvm::Expected<std::unique_ptr<GpuAgent>>
  getAgent() const override;
};

} // namespace luthier::hsa

#endif