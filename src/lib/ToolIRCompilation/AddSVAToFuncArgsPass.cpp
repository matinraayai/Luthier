//===-- AddSVAToFuncArgsPass.cpp ------------------------------------------===//
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
#include "luthier/ToolIRCompilation/AddSVAToFuncArgsPass.h"
#include "luthier/Intrinsic/IntrinsicCalls.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

namespace luthier {

namespace {

/// The pass must skip any function whose caller can't provide an SVA:
/// intrinsic / builtin declarations (they ARE the SVA plumbing, and
/// \c loadSVA needs no self-arg), and injected payloads (top-level entries
/// from the target application).
bool needsSVAsFirstArgReWrite(const llvm::Function &F) {
  if (F.hasFnAttribute(IntrinsicAttribute) ||
      F.hasFnAttribute(BuiltinAttribute))
    return false;
  if (F.hasFnAttribute(InjectedPayloadAttribute))
    return false;
  // Don't re-write a function that has already been re-written
  if (F.arg_size() > 0 && F.getArg(0)->hasAttribute(SVAParamAttribute))
    return false;
  return true;
}

/// Build an \c AttributeList for \p NewF by taking \p OldF's function &
/// return attributes verbatim, tagging index 0 with \c luthier.sva, and
/// shifting every parameter attribute right by one.
llvm::AttributeList buildShiftedAttributes(llvm::LLVMContext &Ctx,
                                           const llvm::Function &OldF) {
  llvm::AttributeList Old = OldF.getAttributes();
  llvm::AttributeList Out;
  Out = Out.addFnAttributes(Ctx, llvm::AttrBuilder(Ctx, Old.getFnAttrs()));
  Out = Out.addRetAttributes(Ctx, llvm::AttrBuilder(Ctx, Old.getRetAttrs()));
  llvm::AttrBuilder SVAAB(Ctx);
  SVAAB.addAttribute(SVAParamAttribute);
  Out = Out.addParamAttributes(Ctx, 0, SVAAB);
  for (unsigned I = 0, E = OldF.arg_size(); I < E; ++I)
    Out = Out.addParamAttributes(Ctx, I + 1,
                                 llvm::AttrBuilder(Ctx, Old.getParamAttrs(I)));
  return Out;
}

/// Rewrite \p OldF in-place-by-replacement: construct \p NewF with an i32
/// SVA arg prepended (tagged \c luthier.sva), migrate the body if any, and
/// return the replacement function. \p NewF still owns \p OldF's name — the
/// caller is responsible for erasing \p OldF from the module afterward.
llvm::Function *cloneWithSVAArg(llvm::Function &OldF, llvm::Type *Int32Ty) {
  llvm::LLVMContext &Ctx = OldF.getContext();
  llvm::FunctionType *OldFT = OldF.getFunctionType();
  llvm::SmallVector<llvm::Type *, 8> NewParams;
  NewParams.reserve(OldFT->getNumParams() + 1);
  NewParams.push_back(Int32Ty);
  for (llvm::Type *P : OldFT->params())
    NewParams.push_back(P);
  llvm::FunctionType *NewFT = llvm::FunctionType::get(
      OldFT->getReturnType(), NewParams, OldFT->isVarArg());

  llvm::Function *NewF =
      llvm::Function::Create(NewFT, OldF.getLinkage(), OldF.getAddressSpace(),
                             OldF.getName() + ".luthier.sva", OldF.getParent());
  NewF->copyAttributesFrom(&OldF);
  NewF->setAttributes(buildShiftedAttributes(Ctx, OldF));
  NewF->setCallingConv(OldF.getCallingConv());
  NewF->setSubprogram(OldF.getSubprogram());
  NewF->getArg(0)->setName("luthier.sva");

  if (!OldF.isDeclaration()) {
    llvm::ValueToValueMapTy VMap;
    auto NewA = NewF->arg_begin();
    ++NewA; // Skip the prepended SVA arg
    for (auto OldA = OldF.arg_begin(), OldE = OldF.arg_end(); OldA != OldE;
         ++OldA, ++NewA) {
      NewA->setName(OldA->getName());
      VMap[&*OldA] = &*NewA;
    }
    llvm::SmallVector<llvm::ReturnInst *, 4> Returns;
    llvm::CloneFunctionInto(NewF, &OldF, VMap,
                            llvm::CloneFunctionChangeType::LocalChangesOnly,
                            Returns);
  }
  return NewF;
}

/// Rewrite each direct call site \p CB (from \p OldF to \p NewF) so it calls
/// \p NewF with a fresh \c luthier::loadSVA() prepended to its arg list.
/// The new call inherits \p CB's calling convention and \p CB's callsite
/// attributes shifted right by one; the SVA arg gets the \c luthier.sva
/// callsite attribute so the pass's output shape matches the function
/// signature.
void rewriteCallSite(llvm::CallBase *CB, llvm::Function *NewF,
                     llvm::Type *Int32Ty, llvm::Module &M) {
  llvm::LLVMContext &Ctx = CB->getContext();
  llvm::IRBuilder<> Builder(CB);
  llvm::Value *SVAArg =
      insertCallToIntrinsic(M, Builder, "luthier::loadSVA", *Int32Ty);

  llvm::SmallVector<llvm::Value *, 8> NewArgs;
  NewArgs.reserve(CB->arg_size() + 1);
  NewArgs.push_back(SVAArg);
  for (llvm::Value *A : CB->args())
    NewArgs.push_back(A);

  llvm::AttributeList OldAL = CB->getAttributes();
  llvm::AttributeList NewAL;
  NewAL =
      NewAL.addFnAttributes(Ctx, llvm::AttrBuilder(Ctx, OldAL.getFnAttrs()));
  NewAL =
      NewAL.addRetAttributes(Ctx, llvm::AttrBuilder(Ctx, OldAL.getRetAttrs()));
  llvm::AttrBuilder SVAAB(Ctx);
  SVAAB.addAttribute(SVAParamAttribute);
  NewAL = NewAL.addParamAttributes(Ctx, 0, SVAAB);
  for (unsigned I = 0, E = CB->arg_size(); I < E; ++I)
    NewAL = NewAL.addParamAttributes(
        Ctx, I + 1, llvm::AttrBuilder(Ctx, OldAL.getParamAttrs(I)));

  llvm::CallInst *NewCI = Builder.CreateCall(NewF, NewArgs);
  NewCI->setCallingConv(CB->getCallingConv());
  NewCI->setAttributes(NewAL);
  NewCI->setDebugLoc(CB->getDebugLoc());
  NewCI->copyMetadata(*CB);
  NewCI->takeName(CB);
  CB->replaceAllUsesWith(NewCI);
  CB->eraseFromParent();
}

} // namespace

llvm::PreservedAnalyses
AddSVAToFuncArgsPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  /// Only operate on device code
  if (M.getTargetTriple().getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();

  llvm::LLVMContext &Ctx = M.getContext();
  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);

  // Snapshot the module's function list up front. Rewriting each eligible
  // function adds a replacement Function and, at every call site, a fresh
  // luthier::loadSVA declaration/call — walking M.functions() mid-mutation
  // would revisit our own additions.
  llvm::SmallVector<llvm::Function *, 32> Targets;
  for (llvm::Function &F : M.functions())
    if (needsSVAsFirstArgReWrite(F))
      Targets.push_back(&F);

  if (Targets.empty())
    return llvm::PreservedAnalyses::all();

  bool Changed = false;
  for (llvm::Function *OldF : Targets) {
    // Snapshot the direct-call sites before mutating any of them while
    // skipping intrinsics
    llvm::SmallVector<llvm::CallBase *, 16> Callers;
    for (llvm::User *U : OldF->users()) {
      auto *CB = llvm::dyn_cast<llvm::CallBase>(U);
      if (CB && CB->getCalledOperand() == OldF)
        Callers.push_back(CB);
    }

    llvm::Function *NewF = cloneWithSVAArg(*OldF, Int32Ty);

    for (llvm::CallBase *CB : Callers)
      rewriteCallSite(CB, NewF, Int32Ty, M);

    OldF->replaceAllUsesWith(NewF);

    llvm::StringRef OldName = OldF->getName();
    OldF->eraseFromParent();
    NewF->setName(OldName);
    Changed = true;
  }

  return Changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace luthier
