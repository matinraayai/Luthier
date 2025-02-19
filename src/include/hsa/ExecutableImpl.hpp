//===-- ExecutableImpl.hpp - Concrete HSA Executable Wrapper --------------===//
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
/// This file defines the \c hsa::ExecutableImpl class which implements the
/// \c hsa::Executable interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_IMPL_HPP
#define LUTHIER_HSA_EXECUTABLE_IMPL_HPP
#include "hsa/CodeObjectReader.hpp"
#include "hsa/Executable.hpp"
#include "hsa/HandleType.hpp"
#include <llvm/ADT/DenseMapInfo.h>
#include <optional>
#include <vector>

namespace luthier::hsa {

/// \brief concrete implementation of \c hsa::Executable interface
class ExecutableImpl : public Executable {
public:
  /// Constructor using a \c hsa_executable_t handle
  explicit ExecutableImpl(hsa_executable_t Exec) : Executable(Exec) {};

  [[nodiscard]] std::unique_ptr<Executable> clone() const override;

  llvm::Expected<std::unique_ptr<hsa::LoadedCodeObject>>
  loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                      const hsa::GpuAgent &Agent,
                      llvm::StringRef LoaderOptions) override;

  llvm::Error defineExternalAgentGlobalVariable(const hsa::GpuAgent &Agent,
                                                llvm::StringRef SymbolName,
                                                const void *Address) override;

  llvm::Error freeze() override;

  llvm::Error destroy() override;

  llvm::Expected<hsa_profile_t> getProfile() override;

  [[nodiscard]] llvm::Expected<hsa_executable_state_t>
  getState() const override;

  llvm::Expected<hsa_default_float_rounding_mode_t> getRoundingMode() override;

  llvm::Error getLoadedCodeObjects(
      llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObject>> &LCOs)
      const override;

  llvm::Expected<std::unique_ptr<ExecutableSymbol>>
  getExecutableSymbolByName(llvm::StringRef Name,
                            const hsa::GpuAgent &Agent) override;
};

} // namespace luthier::hsa

#endif