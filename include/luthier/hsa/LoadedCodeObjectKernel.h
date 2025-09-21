//===-- LoadedCodeObjectKernel.h - Loaded Code Object Kernel ----*- C++ -*-===//
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
/// This file defines the \c LoadedCodeObjectKernel under the \c luthier::hsa
/// namespace, which represents all kernels inside a \c hsa::LoadedCodeObject.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_KERNEL_H
#define LUTHIER_HSA_LOADED_CODE_OBJECT_KERNEL_H
#include "luthier/hsa/LoadedCodeObjectSymbol.h"
#include "luthier/hsa/Metadata.h"

namespace luthier::hsa {

class KernelDescriptor;

/// \brief a \c LoadedCodeObjectSymbol of type
/// \c LoadedCodeObjectSymbol::ST_KERNEL
class LoadedCodeObjectKernel final : public LoadedCodeObjectSymbol {

private:
  /// A reference to the Kernel descriptor symbol of the kernel
  /// The original \c Symbol will hold the kernel function's symbol
  const llvm::object::ELFSymbolRef KDSymbol;

  std::shared_ptr<md::Metadata> LCOMeta;

  /// The Kernel metadata
  md::Kernel::Metadata &MD;

  /// Constructor
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param KFuncSymbol the function symbol of the kernel, cached internally
  /// by Luthier
  /// \param KDSymbol the kernel descriptor symbol of the kernel, cached
  /// internally by Luthier
  /// \param Metadata the Metadata of the kernel, cached internally by Luthier
  /// \param ExecutableSymbol the \c hsa_executable_symbol_t equivalent of
  /// the kernel
  LoadedCodeObjectKernel(hsa_loaded_code_object_t LCO,
                         llvm::object::ELF64LEObjectFile &StorageElf,
                         llvm::object::ELFSymbolRef KFuncSymbol,
                         llvm::object::ELFSymbolRef KDSymbol,
                         hsa_executable_symbol_t ExecutableSymbol,
                         std::shared_ptr<md::Metadata> LCOMeta,
                         md::Kernel::Metadata &MD)
      : LoadedCodeObjectSymbol(LCO, StorageElf, KFuncSymbol,
                               SymbolKind::SK_KERNEL, ExecutableSymbol),
        KDSymbol(KDSymbol), LCOMeta(std::move(LCOMeta)), MD(MD) {}

public:
  /// Factory method used internally by Luthier
  /// Symbols created using this method will be cached, and a reference to them
  /// will be returned to the tool writer when queried
  /// \param LCO the \c hsa_loaded_code_object_t wrapper handle, only accessible
  /// internally to Luthier
  /// \param KFuncSymbol the function symbol of the kernel, cached internally
  /// by Luthier
  /// \param KDSymbol the kernel descriptor symbol of the kernel, cached
  /// internally by Luthier
  /// \param Metadata the Metadata of the kernel, cached internally by Luthier
  /// \return on success
  static llvm::Expected<std::unique_ptr<LoadedCodeObjectKernel>>
  create(const ApiTableContainer<::CoreApiTable> &CoreApiTable,
         const hsa_ven_amd_loader_1_01_pfn_t &VenLoaderApi,
         hsa_loaded_code_object_t LCO,
         llvm::object::ELF64LEObjectFile &StorageElf,
         std::shared_ptr<md::Metadata> LCOMeta,
         llvm::object::ELFSymbolRef KFuncSymbol,
         llvm::object::ELFSymbolRef KDSymbol);

  [[nodiscard]] std::unique_ptr<LoadedCodeObjectSymbol> clone() const override {
    return std::unique_ptr<LoadedCodeObjectKernel>(new LoadedCodeObjectKernel(
        this->BackingLCO, this->StorageELF, this->Symbol, this->KDSymbol,
        *this->ExecutableSymbol, this->LCOMeta, this->MD));
  }

  /// \return a pointer to the \c hsa::KernelDescriptor of the kernel on the
  /// agent it is loaded on
  [[nodiscard]] llvm::Expected<const KernelDescriptor *> getKernelDescriptor(
      const ApiTableContainer<::CoreApiTable> &CoreApiTable) const;

  /// \return the parsed \c hsa::md::Kernel::Metadata of the kernel
  [[nodiscard]] const hsa::md::Kernel::Metadata &getKernelMetadata() const {
    return MD;
  }

  /// method for providing LLVM RTTI
  __attribute__((used)) static bool classof(const LoadedCodeObjectSymbol *S) {
    return S->getType() == SK_KERNEL;
  }
};

} // namespace luthier::hsa

#endif