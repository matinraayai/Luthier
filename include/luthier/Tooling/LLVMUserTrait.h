//===-- LLVMUserTrait.h - HSATool LLVM-init trait ---------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file LLVMUserTrait.h
/// Header-only CRTP trait that initializes the LLVM AMDGPU target on
/// initialization, and calls \c llvm::llvm_shutdown on finalization.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_LLVM_USER_TRAIT_H
#define LUTHIER_TOOLING_LLVM_USER_TRAIT_H

#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>

namespace luthier {

/// \brief CRTP trait that initializes the LLVM AMDGPU target on construction
/// and finalizes LLVM on destruction
template <typename Derived> class LLVMUserTrait {

public:
  LLVMUserTrait() {
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTargetMCA();
  }

  ~LLVMUserTrait() {
    llvm::llvm_shutdown();
  }

  LLVMUserTrait(const LLVMUserTrait &) = delete;
  LLVMUserTrait &operator=(const LLVMUserTrait &) = delete;

};

} // namespace luthier

#endif // LUTHIER_TOOLING_LLVM_USER_TRAIT_H
