//===-- EmbedInstrumentationModuleBitcodePass.hpp -------------------------===//
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
/// This file describes a LLVM compile plugin for embedding Luthier
/// instrumentation module bitcode inside tool device code.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_COMPILE_PLUGINS_EMBED_INSTRUMENTATION_MODULE_BITCODE_PASS_HPP
#define LUTHIER_COMPILE_PLUGINS_EMBED_INSTRUMENTATION_MODULE_BITCODE_PASS_HPP
#include <llvm/IR/PassManager.h>

namespace llvm {
class Module;
} // namespace llvm

namespace luthier {

/// \brief This pass intercepts the optimized HIP device module of a Luthier
/// tool, clones and pre-process it, and then embeds it inside the device
/// code's ELF
class EmbedInstrumentationModuleBitcodePass
    : public llvm::PassInfoMixin<EmbedInstrumentationModuleBitcodePass> {

public:
  EmbedInstrumentationModuleBitcodePass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);

  static bool isRequired() { return true; }
};

} // namespace luthier

#endif