//===-- ISAImpl.hpp -------------------------------------------------------===//
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
/// This file defines the \c hsa::ISAImpl class, which is the concrete
/// implementation of the \c hsa::ISA interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_ISA_IMPL_HPP
#define LUTHIER_HSA_ISA_IMPL_HPP
#include "hsa/ISA.hpp"
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>

namespace luthier::hsa {

class ISAImpl : public ISA {
public:
  /// Constructor
  /// \param Isa the \c hsa_isa_t handle being encapsulated
  explicit ISAImpl(hsa_isa_t Isa) : ISA(Isa) {};

  llvm::Error fromName(llvm::StringRef FullIsaName) override;

  llvm::Error fromLLVM(const llvm::Triple &TT, llvm::StringRef GPUName,
                       const llvm::SubtargetFeatures &Features) override;

  [[nodiscard]] llvm::Expected<std::string> getName() const override;

  [[nodiscard]] llvm::Expected<std::string> getArchitecture() const override;

  [[nodiscard]] llvm::Expected<std::string> getVendor() const override;

  [[nodiscard]] llvm::Expected<std::string> getOS() const override;

  [[nodiscard]] llvm::Expected<std::string> getEnvironment() const override;

  [[nodiscard]] llvm::Expected<std::string> getGPUName() const override;

  [[nodiscard]] llvm::Expected<bool> isXNACKSupported() const override;

  [[nodiscard]] llvm::Expected<bool> isSRAMECCSupported() const override;

  [[nodiscard]] llvm::Expected<llvm::Triple> getTargetTriple() const override;

  [[nodiscard]] llvm::Expected<llvm::SubtargetFeatures>
  getSubTargetFeatures() const override;
};

} // namespace luthier::hsa

#endif