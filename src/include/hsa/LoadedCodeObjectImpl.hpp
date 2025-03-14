//===-- LoadedCodeObjectImpl.hpp ------------------------------------------===//
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
/// This file defines the \c LoadedCodeObject class under the \c luthier::hsa
/// namespace, representing a wrapper around the \c hsa_loaded_code_object_t HSA
/// type.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_IMPL_HPP
#define LUTHIER_HSA_LOADED_CODE_OBJECT_IMPL_HPP
#include "hsa/HandleType.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include <luthier/hsa/Metadata.h>

namespace luthier::hsa {

/// \brief Wraps functionality related to \c hsa_loaded_code_object_t in Luthier
///
/// \note This wrapper is implemented with the following assumptions based on
/// the current state of ROCr:
/// 1. Even though the ROCr HSA vendor loader API
/// (under <hsa/hsa_ven_amd_loader.h>) acknowledge that both file-backed and
/// memory-backed Loaded Code Objects exists, only memory-backed ones are
/// actually implemented. Therefore, querying the storage type and querying the
/// FD of the storage is not included in the API. L
/// Luthier assumes all Loaded Code Objects have a memory storage in order to
/// return its associated ELF.
/// In the event that file-backed storage is implemented in the loader, the
/// code needs to be updated.
/// 2. Program Loaded Code Objects has been deprecated and are not used anywhere
/// in the ROCm stack. ROCr does not even allow using Loaded Code Objects with
/// program allocations. Therefore, it is safe to assume all Loaded Code Objects
/// are backed by a \c GpuAgent.
/// 3. Internally, the vendor loader API keeps track of loaded segments, and
/// allows for querying these segments via the
/// \c hsa_ven_amd_loader_query_segment_descriptors function. As of right now,
/// Luthier does not use this information to locate the load address of the
/// symbols, and instead relies on the \c luthier::getLoadedMemoryOffset
/// function to calculate the load address.
///
/// \note This wrapper relies on cached functionality as described by the
/// \c hsa::ExecutableBackedCachable interface and backed by the \c
/// hsa::Platform Singleton.
class LoadedCodeObjectImpl
    : public llvm::RTTIExtends<LoadedCodeObjectImpl, LoadedCodeObject>,
      public HandleType<hsa_loaded_code_object_t> {
public:
  static char ID;

  /// Primary constructor
  /// \param LCO HSA handle of the \c hsa_loaded_code_object_t
  explicit LoadedCodeObjectImpl(hsa_loaded_code_object_t LCO);

  [[nodiscard]] llvm::Expected<std::unique_ptr<Executable>>
  getExecutable() const override;

  [[nodiscard]] llvm::Expected<std::unique_ptr<GpuAgent>>
  getAgent() const override;

  [[nodiscard]] llvm::Expected<llvm::object::ELF64LEObjectFile &>
  getStorageELF() const override;

  [[nodiscard]] llvm::Expected<long> getLoadDelta() const override;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getLoadedMemory() const override;

  [[nodiscard]] llvm::Expected<std::string> getUri() const override;

  [[nodiscard]] llvm::Expected<std::unique_ptr<ISA>> getISA() const override;

  [[nodiscard]] llvm::Expected<const hsa::md::Metadata &> getMetadata() const;

  [[nodiscard]] llvm::Error getKernelSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const override;

  [[nodiscard]] llvm::Error getVariableSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const override;

  [[nodiscard]] llvm::Error getDeviceFunctionSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const override;

  [[nodiscard]] llvm::Error getExternalSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const override;

  [[nodiscard]] llvm::Error getLoadedCodeObjectSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObjectSymbol>> &Out)
      const override;

  [[nodiscard]] llvm::Expected<std::unique_ptr<LoadedCodeObjectSymbol>>
  getLoadedCodeObjectSymbolByName(llvm::StringRef Name) const override;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getStorageMemory() const override;
};

} // namespace luthier::hsa

#endif