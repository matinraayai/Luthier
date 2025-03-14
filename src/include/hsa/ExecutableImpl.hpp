//===-- ExecutableImpl.hpp ------------------------------------------------===//
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
/// This file defines the \c hsa::ExecutableImpl class, which is the concrete
/// implementation of the \c hsa::Executable interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_IMPL_HPP
#define LUTHIER_HSA_EXECUTABLE_IMPL_HPP
#include "hsa/Executable.hpp"
#include "hsa/HandleType.hpp"
#include <hsa/hsa.h>

namespace luthier::hsa {

/// \brief Concrete implementation of the \c Executable interface
class ExecutableImpl : public llvm::RTTIExtends<ExecutableImpl, Executable>,
                       public HandleType<hsa_executable_t> {
public:
  static char ID;

  /// Default constructor
  ExecutableImpl() : HandleType<hsa_executable_t>({0}) {};

  /// Constructor using an already created \c hsa_executable_t \p Exec handle
  /// \warning This constructor must only be used with handles already created
  /// by HSA. To create executables from scratch, use \c create instead.
  explicit ExecutableImpl(hsa_executable_t Exec);

  llvm::Error
  create(hsa_profile_t Profile,
         hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) override;

  llvm::Expected<std::unique_ptr<hsa::LoadedCodeObject>>
  loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                      const hsa::GpuAgent &Agent,
                      llvm::StringRef LoaderOptions) override;

  llvm::Error defineExternalAgentGlobalVariable(const hsa::GpuAgent &Agent,
                                                llvm::StringRef SymbolName,
                                                const void *Address) override;

  llvm::Error freeze() override;

  llvm::Expected<bool> validate() override;

  llvm::Error destroy() override;

  llvm::Expected<hsa_profile_t> getProfile() override;

  [[nodiscard]] llvm::Expected<hsa_executable_state_t>
  getState() const override;

  llvm::Expected<hsa_default_float_rounding_mode_t> getRoundingMode() override;

  llvm::Error getLoadedCodeObjects(
      llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObject>> &LCOs)
      const override;

  llvm::Expected<std::unique_ptr<ExecutableSymbol>>
  getExecutableSymbolByName(llvm::StringRef Name, const hsa::GpuAgent &Agent);
};

} // namespace luthier::hsa

#endif