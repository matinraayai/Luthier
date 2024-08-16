//===-- LoadedCodeObjectKernel.h - Loaded Code Object Kernel ----*- C++ -*-===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
#ifndef LUTHIER_LOADED_CODE_OBJECT_KERNEL
#define LUTHIER_LOADED_CODE_OBJECT_KERNEL
#include "LoadedCodeObjectSymbol.h"
#include "Metadata.h"

namespace luthier::hsa {

/// \brief a \c LoadedCodeObjectSymbol of type
/// \c LoadedCodeObjectSymbol::ST_KERNEL
class LoadedCodeObjectKernel final : public LoadedCodeObjectSymbol {
private:
  /// A reference to the Kernel descriptor symbol of the kernel,
  /// cached internally by Luthier
  /// The original \c Symbol will hold the kernel function's symbol
  const llvm::object::ELFSymbolRef *KDSymbol;
  /// The Kernel metadata, cached internally by Luthier
  const md::Kernel::Metadata *MD;

public:
  /// Constructor
  /// \param LCO the \c hsa_loaded_code_object_t this symbol belongs to
  /// \param KFuncSymbol the function symbol of the kernel, cached internally
  /// by Luthier
  /// \param KDSymbol the kernel descriptor symbol of the kernel, cached
  /// internally by Luthier
  /// \param Metadata the Metadata of the kernel, cached internally by Luthier
  LoadedCodeObjectKernel(hsa_loaded_code_object_t LCO,
                         const llvm::object::ELFSymbolRef *KFuncSymbol,
                         const llvm::object::ELFSymbolRef *KDSymbol,
                         const md::Kernel::Metadata &Metadata)
      : LoadedCodeObjectSymbol(LCO, KFuncSymbol, SymbolKind::SK_KERNEL),
        KDSymbol(KDSymbol), MD(&Metadata) {}

  /// \return a pointer to the \c hsa::KernelDescriptor of the kernel on the
  /// agent it is loaded on
  [[nodiscard]] llvm::Expected<const KernelDescriptor *>
  getKernelDescriptor() const;

  /// \return the parsed \c hsa::md::Kernel::Metadata of the kernel
  [[nodiscard]] const hsa::md::Kernel::Metadata &getKernelMetadata() const {
    return *MD;
  }

  /// method for providing LLVM RTTI
  [[nodiscard]] static bool classof(const LoadedCodeObjectSymbol *S) {
    return S->getType() == SK_KERNEL;
  }
};

} // namespace luthier::hsa

#endif