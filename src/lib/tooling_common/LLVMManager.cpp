//===-- LLVMManager.cpp - LLVM Library Lifetime Management ----------------===//
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
/// This file implements Luthier's LLVM Manager.
//===----------------------------------------------------------------------===//
#include "tooling_common/LLVMManager.hpp"
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>

namespace luthier {

template <> LLVMManager *Singleton<LLVMManager>::Instance{nullptr};

LLVMManager::LLVMManager() : Singleton<LLVMManager>() {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUTargetMCA();
}

LLVMManager::~LLVMManager() {
  llvm::llvm_shutdown();
  Singleton<LLVMManager>::~Singleton();
}

} // namespace luthier
