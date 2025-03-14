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
/// This file defines the \c luthier::hsa::LoadedCodeObjectKernel interface,
/// which represents all kernels inside a <tt>hsa_loaded_code_object_t</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_KERNEL_H
#define LUTHIER_HSA_LOADED_CODE_OBJECT_KERNEL_H
#include "LoadedCodeObjectSymbol.h"
#include "Metadata.h"

namespace luthier::hsa {

class LoadedCodeObject;

class KernelDescriptor;

/// \brief interface representing a \c hsa::LoadedCodeObjectSymbol of kernel
/// type
class LoadedCodeObjectKernel
    : public llvm::RTTIExtends<LoadedCodeObjectKernel, LoadedCodeObjectSymbol> {
public:
  static char ID;

  /// \return a pointer to the \c hsa::KernelDescriptor of the kernel on the
  /// agent it is loaded on
  [[nodiscard]] virtual llvm::Expected<const KernelDescriptor *>
  getKernelDescriptor() const = 0;

  /// \return the parsed \c hsa::md::Kernel::Metadata of the kernel
  [[nodiscard]] virtual const hsa::md::Kernel::Metadata &
  getKernelMetadata() const = 0;
};

} // namespace luthier::hsa

#endif
