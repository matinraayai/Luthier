//===-- SubstituteAMDGCNIntrinsicsPass.cpp --------------------------------===//
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
#include "luthier/ToolIRCompilation/SubstituteAMDGCNIntrinsicsPass.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/Intrinsic/IntrinsicCalls.h"
#include <initializer_list>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <tuple>

namespace luthier {

llvm::PreservedAnalyses
SubstituteAMDGCNIntrinsicsPass::run(llvm::Module &M,
                                    llvm::ModuleAnalysisManager &) {
  llvm::Triple T(M.getTargetTriple());
  /// Only operate on device code
  if (T.getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::PointerType *Int32PtrTy =
      llvm::PointerType::get(Ctx, llvm::AMDGPUAS::CONSTANT_ADDRESS);

  using Mapping = std::tuple<const char *, const char *, llvm::Type *>;
  for (llvm::Function &F : llvm::make_early_inc_range(M.functions())) {
    for (const auto &[LLVMName, LuthierName, ReturnType] :
         std::initializer_list<Mapping>{
             {"llvm.amdgcn.workgroup.id.x", "luthier::workgroupIdX", Int32Ty},
             {"llvm.amdgcn.workgroup.id.y", "luthier::workgroupIdY", Int32Ty},
             {"llvm.amdgcn.workgroup.id.z", "luthier::workgroupIdZ", Int32Ty},
             {"llvm.amdgcn.implicitarg.ptr", "luthier::implicitArgPtr",
              Int32PtrTy}}) {
      if (!F.getName().starts_with(LLVMName))
        continue;
      for (llvm::User *U : llvm::make_early_inc_range(F.users())) {
        auto *CI = llvm::dyn_cast<llvm::CallInst>(U);
        if (CI == nullptr)
          continue;
        llvm::IRBuilder<> Builder(CI);
        llvm::CallInst *NewCall =
            insertCallToIntrinsic(M, Builder, LuthierName, *ReturnType);
        CI->replaceAllUsesWith(NewCall);
        CI->eraseFromParent();
      }
      F.dropAllReferences();
      F.eraseFromParent();
      break;
    }
  }

  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
