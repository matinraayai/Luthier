//===-- AddSVAToFuncArgsPass.h -----------------------------------*- C++-*-===//
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
/// \file
/// Defines the \c AddSVAToFuncArgsPass class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_IR_COMPILATION_ADD_SVA_TO_FUNC_ARGS_PASS_H
#define LUTHIER_TOOL_IR_COMPILATION_ADD_SVA_TO_FUNC_ARGS_PASS_H
#include <llvm/IR/PassManager.h>

namespace llvm {
class Module;
}

namespace luthier {

/// \brief Prepends a \c "luthier.sva"-tagged \c i32 first parameter to every
/// eligible function in the instrumentation module and rewrites every direct
/// call site to pass a fresh \c luthier::loadSVA() as the new first argument.
///
/// Ineligible functions (skipped): intrinsic and builtin declarations
/// (\c luthier.intrinsic / \c luthier.builtin), injected payloads
/// (\c luthier.function.injected_payload), and functions whose first parameter
/// already carries the SVA attribute (idempotence).
class AddSVAToFuncArgsPass
    : public llvm::PassInfoMixin<AddSVAToFuncArgsPass> {
public:
  AddSVAToFuncArgsPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }

  static llvm::StringRef name() { return "luthier-add-sva-to-func-args"; }
};

} // namespace luthier

#endif
