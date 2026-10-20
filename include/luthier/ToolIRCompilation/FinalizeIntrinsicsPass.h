//===-- FinalizeIntrinsicsPass.h ---------------------------------*- C++-*-===//
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
/// \file FinalizeIntrinsicsPass.h
/// Defines the \c FinalizeIntrinsicsPass class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_IR_COMPILATION_FINALIZE_INTRINSICS_PASS_H
#define LUTHIER_TOOL_IR_COMPILATION_FINALIZE_INTRINSICS_PASS_H
#include <llvm/IR/PassManager.h>

namespace llvm {
class Module;
}

namespace luthier {

/// \brief A pass that, for each function carrying the Luthier intrinsic
/// function attribute:
/// - Deletes its body
/// - Sets external linkage
/// - Attaches the demangled intrinsic name as the value of the intrinsic
/// function attribute (to easily find its intrinsic processor)
/// - Renames it to a typed-suffix mangled form recognized by Luthier's MIR
/// lowering
class FinalizeIntrinsicsPass
    : public llvm::RequiredPassInfoMixin<FinalizeIntrinsicsPass> {
public:
  FinalizeIntrinsicsPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }

  static llvm::StringRef name() { return "luthier-finalize-intrinsics"; }
};

} // namespace luthier

#endif
