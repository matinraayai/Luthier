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
/// This file defines the \c LoadedCodeObjectKernel interface under the
/// \c luthier::hsa namespace, which represents all kernels inside a
/// \c hsa::LoadedCodeObject.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_KERNEL_H
#define LUTHIER_HSA_LOADED_CODE_OBJECT_KERNEL_H
#include "LoadedCodeObjectSymbol.h"
#include "Metadata.h"

namespace luthier::hsa {

class LoadedCodeObject;

class KernelDescriptor;

class LoadedCodeObjectKernel
    : public llvm::RTTIExtends<LoadedCodeObjectKernel, LoadedCodeObjectSymbol> {
public:
  static char ID;

  /// \return a pointer to the \c hsa::KernelDescriptor of the kernel on the
  /// agent it is loaded on
  [[nodiscard]] virtual llvm::Expected<const KernelDescriptor *>
  getKernelDescriptor() const = 0;

  /// \return the parsed \c hsa::md::Kernel::Metadata of the kernel
  [[nodiscard]] virtual std::unique_ptr<hsa::md::Kernel::Metadata>
  getKernelMetadata() const = 0;
};

} // namespace luthier::hsa

#endif