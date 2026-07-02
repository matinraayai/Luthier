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
  // Functions whose IR-level `llvm.global.annotations` entry names them
  // as a hook (`luthier.function.hook`). At the bottom we re-add these
  // to `@llvm.compiler.used` so the downstream loader's `-O3`
  // optimization pipeline doesn't DCE them — they're called only from
  // payload functions the loader synthesizes on the fly, so there's no
  // static caller to anchor them, and `@llvm.compiler.used` is the
  // only DCE-preservation marker that survives a fresh `Module` load.
  llvm::SmallVector<llvm::GlobalValue *, 4> HooksToPreserve;

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
      // Convert every Luthier-recognized annotation into a function
      // attribute so downstream passes (which run after this pass strips
      // `@llvm.global.annotations`) can still identify hooks /
      // intrinsics / payloads by querying `hasFnAttribute`. Previously
      // only intrinsics were preserved this way; hooks lost the
      // annotation entirely and showed up downstream as plain
      // internal-linkage device functions.
      if (Content == IntrinsicAttribute) {
        F->addFnAttr(IntrinsicAttribute);
        LLVM_DEBUG(luthier::dbgs()
                   << "Marked intrinsic " << F->getName() << ".\n");
      } else if (Content == InjectedPayloadAttribute) {
        F->addFnAttr(InjectedPayloadAttribute);
        LLVM_DEBUG(luthier::dbgs()
                   << "Marked injected payload " << F->getName() << ".\n");
      }
    }
    AnnotationGV->dropAllReferences();
    AnnotationGV->eraseFromParent();
  }

  // Drop the original `@llvm.compiler.used` / `@llvm.used` — they may
  // reference host-side helpers (the `__device_stub__*` HIP launch
  // stubs, the `_Z...` host shadows, etc.) that don't make sense in
  // the device-only embedded module and would otherwise pin those
  // declarations alive through DCE.
  for (const char *VarName : {"llvm.compiler.used", "llvm.used"}) {
    if (auto *GV = M.getGlobalVariable(VarName)) {
      GV->dropAllReferences();
      GV->eraseFromParent();
    }
  }

  // Reanchor hooks against `@llvm.compiler.used`. Each hook is called
  // only from runtime-synthesized payload functions (the loader builds
  // a payload-per-target-MI inside the IModule at runtime), so at IR
  // pipeline time inside the loader, no static caller exists to keep
  // the hook alive. Without `@llvm.compiler.used` membership, the
  // loader's `-O3` `GlobalDCE` strips every hook and the IModule ends
  // up empty by the time `IntrinsicMIRLoweringPass::setModuleSVASpec`
  // runs.
  if (!HooksToPreserve.empty())
    llvm::appendToCompilerUsed(M, HooksToPreserve);

  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
