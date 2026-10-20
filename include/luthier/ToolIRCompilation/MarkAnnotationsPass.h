//===-- MarkAnnotationsPass.h ------------------------------------*- C++-*-===//
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
/// \file MarkAnnotationsPass.h
/// Defines \c MarkAnnotationsPass which reads <tt>llvm.global.annotations</tt>,
/// applies Luthier hook/intrinsic function attributes accordingly, and removes
/// the annotation array along with <tt>llvm.used</tt> and
/// <tt>llvm.compiler.used</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_IR_COMPILATION_MARK_ANNOTATIONS_PASS_H
#define LUTHIER_TOOL_IR_COMPILATION_MARK_ANNOTATIONS_PASS_H
#include <llvm/IR/PassManager.h>

namespace llvm {
class Module;
}

namespace luthier {

class MarkAnnotationsPass
    : public llvm::RequiredPassInfoMixin<MarkAnnotationsPass> {
public:
  MarkAnnotationsPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }

  static llvm::StringRef name() { return "luthier-mark-annotations"; }
};

} // namespace luthier

#endif
