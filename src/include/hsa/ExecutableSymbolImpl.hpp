//===-- ExecutableSymbolImpl.hpp ------------------------------------------===//
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
/// This file defines the \c hsa::ExecutableSymbolImpl class, which implements
/// the \c hsa::ExecutableSymbol as a wrapper around the
/// \c hsa_executable_symbol handle type.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_SYMBOL_IMPL_HPP
#define LUTHIER_HSA_EXECUTABLE_SYMBOL_IMPL_HPP
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/HandleType.hpp"
#include <hsa/hsa.h>

namespace luthier::hsa {

/// \brief wrapper around \c hsa_executable_symbol_t implementing the
/// \c hsa::ExecutableSymbol interface
class ExecutableSymbolImpl
    : public llvm::RTTIExtends<ExecutableSymbolImpl, ExecutableSymbol>,
      public HandleType<hsa_executable_symbol_t> {
public:
  static char ID;

  /// Wrapper constructor
  /// \param Handle HSA handle of the \c hsa_executable_symbol_t
  explicit ExecutableSymbol(hsa_executable_symbol_t Handle)
      : HandleType<hsa_executable_symbol_t>(Handle) {};

  [[nodiscard]] llvm::Expected<hsa_symbol_kind_t> getType() const override;

  [[nodiscard]] llvm::Expected<llvm::StringRef> getName() const override;

  [[nodiscard]] llvm::Expected<size_t> getSize() const override;

  [[nodiscard]] llvm::Expected<luthier::address_t> getAddress() const override;

  [[nodiscard]] llvm::Expected<std::unique_ptr<GpuAgent>>
  getAgent() const override;
};

} // namespace luthier::hsa

#endif