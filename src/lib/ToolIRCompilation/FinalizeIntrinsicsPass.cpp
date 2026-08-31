//===-- FinalizeIntrinsicsPass.cpp ----------------------------------------===//
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
#include "luthier/ToolIRCompilation/FinalizeIntrinsicsPass.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicCalls.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <cstdlib>
#include <cstring>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

namespace luthier {

namespace {

/// Returns the demangled function name with namespace prefix, e.g.
/// for an input mangling resolving to <tt>int a::b::c<int>()</tt>, returns
/// <tt>a::b::c</tt>. Frees the demangler-allocated buffers it uses.
std::string demangleWithNamespace(llvm::StringRef MangledFuncName) {
  llvm::ItaniumPartialDemangler Demangler;
  if (Demangler.partialDemangle(MangledFuncName.data()))
    llvm::report_fatal_error("Failed to demangle the intrinsic name " +
                             MangledFuncName + ".");

  std::string Out;
  llvm::raw_string_ostream OS(Out);

  size_t NSBufSize = 0;
  char *NS = Demangler.getFunctionDeclContextName(nullptr, &NSBufSize);
  if (NS != nullptr && std::strlen(NS) != 0) {
    OS << NS << "::";
  }
  std::free(NS);

  size_t BaseBufSize = 0;
  char *Base = Demangler.getFunctionBaseName(nullptr, &BaseBufSize);
  if (Base != nullptr)
    OS << Base;
  std::free(Base);

  return Out;
}

} // namespace

llvm::PreservedAnalyses
FinalizeIntrinsicsPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  llvm::Triple T(M.getTargetTriple());
  /// Only operate on device code
  if (T.getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();
  llvm::SmallVector<llvm::Function *, 8> Intrinsics;
  for (llvm::Function &F : M) {
    if (F.hasFnAttribute(IntrinsicAttribute))
      Intrinsics.push_back(&F);
  }

  for (llvm::Function *Intrinsic : Intrinsics) {
    if (!Intrinsic->isDeclaration()) {
      M.getContext().emitError(
          llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
              "Intrinsic {0} is not a declaration.", Intrinsic->getName()))));
    }

    llvm::StringRef MangledIntrinsicName = Intrinsic->getName();
    std::string DemangledIntrinsicName =
        demangleWithNamespace(MangledIntrinsicName);

    std::string FormattedIntrinsicName;
    llvm::raw_string_ostream FINOS(FormattedIntrinsicName);
    FINOS << DemangledIntrinsicName;

    llvm::Type *ReturnType = Intrinsic->getReturnType();
    if (!ReturnType->isVoidTy()) {
      FINOS << ".";
      ReturnType->print(FINOS);
    }
    for (const llvm::Argument &Arg : Intrinsic->args()) {
      FINOS << ".";
      Arg.getType()->print(FINOS);
    }

    Intrinsic->addFnAttr(IntrinsicAttribute, DemangledIntrinsicName);
    Intrinsic->setName(FormattedIntrinsicName);

    addAMDGCNAttrs(*Intrinsic);
  }

  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
