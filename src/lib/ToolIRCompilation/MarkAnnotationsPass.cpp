//===-- MarkAnnotationsPass.cpp -------------------------------------------===//
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
/// \file MarkAnnotationsPass.cpp
/// Implements the \c MarkAnnotationsPass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolIRCompilation/MarkAnnotationsPass.h"

#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-mark-annotations"

namespace luthier {

llvm::PreservedAnalyses
MarkAnnotationsPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  llvm::Triple T(M.getTargetTriple());
  /// Only operate on device code
  if (T.getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();

  llvm::GlobalVariable *AnnotationGV =
      M.getGlobalVariable("llvm.global.annotations");
  if (AnnotationGV != nullptr) {
    auto *CA = llvm::cast<llvm::ConstantArray>(AnnotationGV->getOperand(0));
    for (llvm::Value *Op : CA->operands()) {
      auto *CS = llvm::cast<llvm::ConstantStruct>(Op);
      llvm::Value *AnnotatedVal = CS->getOperand(0)->stripPointerCasts();
      auto *F = llvm::dyn_cast<llvm::Function>(AnnotatedVal);
      if (F == nullptr)
        continue;
      auto *GV = llvm::cast<llvm::GlobalVariable>(
          CS->getOperand(1)->stripPointerCasts());
      llvm::StringRef Content;
      llvm::getConstantStringInfo(GV, Content);
      if (Content == IntrinsicAttribute) {
        F->addFnAttr(IntrinsicAttribute);
        LLVM_DEBUG(luthier::dbgs()
                   << "Marked intrinsic " << F->getName() << ".\n");
      } else if (Content == InjectedPayloadAttribute) {
        F->addFnAttr(InjectedPayloadAttribute);
        LLVM_DEBUG(luthier::dbgs()
                   << "Marked injected payload " << F->getName() << ".\n");
      } else if (Content == BuiltinAttribute) {
        F->addFnAttr(BuiltinAttribute);
        LLVM_DEBUG(luthier::dbgs()
           << "Marked builtin " << F->getName() << ".\n");
      }
    }
    AnnotationGV->dropAllReferences();
    AnnotationGV->eraseFromParent();
    return llvm::PreservedAnalyses::none();
  }

  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
