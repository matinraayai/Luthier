#include "embed_optimized_bitcode_pass.hpp"

#include "llvm/Passes/PassPlugin.h"
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-embed-optimized-bitcode-pass"

namespace luthier {

// TODO: Import these from Luthier proper once the separate compilation
// issue is resolved

static constexpr const char *ReservedManagedVar = "__luthier_reserved";

static constexpr const char *HookAttribute = "luthier_hook";

static constexpr const char *IntrinsicAttribute = "luthier_intrinsic";

static constexpr const char *HipCUIDPrefix = "__hip_cuid_";

/// Given a function's mangled name \p MangledFuncName,
/// partially demangles it and returns the base function name with its
/// namespace prefix \n
/// For example given a demangled function name int a::b::c<int>(), this
/// method returns a::b::c
/// \param MangledFuncName the mangled function name
/// \return the name of the function with its namespace prefix
static std::string
getDemangledFunctionNameWithNamespace(llvm::StringRef MangledFuncName) {
  // Get the name of the function, without its template arguments
  llvm::ItaniumPartialDemangler Demangler;
  // Ensure successful partial demangle operation
  if (Demangler.partialDemangle(MangledFuncName.data()))
    llvm::report_fatal_error("Failed to demangle the intrinsic name " +
                             MangledFuncName + ".");
  // Output string
  std::string Out;
  // Output string's ostream
  llvm::raw_string_ostream OS(Out);

  // TODO: check if this works when no namespace is present
  size_t BufferSize;
  char *FuncNamespaceBegin =
      Demangler.getFunctionDeclContextName(nullptr, &BufferSize);
  OS << FuncNamespaceBegin;
  OS << "::";
  char *FuncNameBase = Demangler.getFunctionBaseName(nullptr, &BufferSize);
  OS << FuncNameBase;
  return Out;
}

/// Groups the set of annotated values in \p M into instrumentation
/// hooks and intrinsics of instrumentation hooks \n
/// \note This function should get updated as Luthier's programming model
/// gets updated
/// \param [in] M Module to inspect
/// \param [out] Hooks a list of hook functions found in \p M
/// \param [out] Intrinsics a list of intrinsics found in \p M
/// \return any \c llvm::Error encountered during the process
static llvm::Error
getAnnotatedValues(const llvm::Module &M,
                   llvm::SmallVectorImpl<llvm::Function *> &Hooks,
                   llvm::SmallVectorImpl<llvm::Function *> &Intrinsics) {
  const llvm::GlobalVariable *V =
      M.getGlobalVariable("llvm.global.annotations");
  if (V == nullptr)
    return llvm::Error::success();
  const llvm::ConstantArray *CA = cast<llvm::ConstantArray>(V->getOperand(0));
  for (llvm::Value *Op : CA->operands()) {
    auto *CS = cast<llvm::ConstantStruct>(Op);
    // The first field of the struct contains a pointer to the annotated
    // variable.
    llvm::Value *AnnotatedVal = CS->getOperand(0)->stripPointerCasts();
    if (auto *Func = llvm::dyn_cast<llvm::Function>(AnnotatedVal)) {
      // The second field contains a pointer to a global annotation string.
      auto *GV =
          cast<llvm::GlobalVariable>(CS->getOperand(1)->stripPointerCasts());
      llvm::StringRef Content;
      llvm::getConstantStringInfo(GV, Content);
      if (Content == "luthier_hook") {
        Hooks.push_back(Func);
        LLVM_DEBUG(llvm::dbgs() << "Found hook " << Func->getName() << ".\n");
      } else if (Content == IntrinsicAttribute) {
        Intrinsics.push_back(Func);
        LLVM_DEBUG(llvm::dbgs()
                   << "Found intrinsic " << Func->getName() << ".\n");
      }
    };
  }
  return llvm::Error::success();
}

llvm::PreservedAnalyses
EmbedOptimizedBitcodePass::run(llvm::Module &M,
                               llvm::ModuleAnalysisManager &AM) {
  if (M.getGlobalVariable("llvm.embedded.module", /*AllowInternal=*/true))
    llvm::report_fatal_error(
        "Attempted to embed bitcode twice. Are you passing -fembed-bitcode?",
        /*gen_crash_diag=*/false);

  llvm::Triple T(M.getTargetTriple());
  // Only operate on the AMD GCN code objects
  if (T.getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();

  // Clone the module in order to preprocess it + not interfere with normal
  // HIP compilation
  auto ClonedModule = llvm::CloneModule(M);

  // Extract all the hooks and intrinsics
  llvm::SmallVector<llvm::Function *, 4> Hooks;
  llvm::SmallVector<llvm::Function *, 4> Intrinsics;
  if (auto Err = getAnnotatedValues(*ClonedModule, Hooks, Intrinsics))
    llvm::report_fatal_error(std::move(Err), true);

  // Remove the annotations variable from the Module now that it is processed
  auto AnnotationGV =
      ClonedModule->getGlobalVariable("llvm.global.annotations");
  if (AnnotationGV) {
    AnnotationGV->dropAllReferences();
    AnnotationGV->eraseFromParent();
  }

  // Remove the llvm.used and llvm.compiler.use variable list
  for (const auto &VarName : {"llvm.compiler.used", "llvm.used"}) {
    auto LLVMUsedVar = ClonedModule->getGlobalVariable(VarName);
    if (LLVMUsedVar != nullptr) {
      LLVMUsedVar->dropAllReferences();
      LLVMUsedVar->eraseFromParent();
    }
  }

  // Give each Hook function a "hook" attribute
  for (auto Hook : Hooks) {
    // TODO: remove the always inline attribute once Hooks support the anyreg
    // calling convention
    Hook->addFnAttr(HookAttribute);
    Hook->addFnAttr(llvm::Attribute::AlwaysInline);
  }
  // Remove the body of each intrinsic function and make them extern
  // Also demangle the name and format it similar to LLVM intrinsics
  for (auto Intrinsic : Intrinsics) {
    Intrinsic->deleteBody();
    llvm::StringRef MangledIntrinsicName = Intrinsic->getName();
    // Format the intrinsic name
    std::string FormattedIntrinsicName;
    llvm::raw_string_ostream FINOS(FormattedIntrinsicName);
    std::string DemangledIntrinsicName =
        getDemangledFunctionNameWithNamespace(MangledIntrinsicName);
    FINOS << DemangledIntrinsicName;
    llvm::outs() << FormattedIntrinsicName << "\n";
    // Add the output type if it's not void
    auto *ReturnType = Intrinsic->getReturnType();
    if (!ReturnType->isVoidTy()) {
      FINOS << ".";
      ReturnType->print(FINOS);
    }
    llvm::outs() << FormattedIntrinsicName << "\n";
    // Add the argument types
    for (const auto &Arg : Intrinsic->args()) {
      FINOS << ".";
      Arg.getType()->print(FINOS);
    }
    Intrinsic->addFnAttr("luthier_intrinsic", DemangledIntrinsicName);
    Intrinsic->setName(FormattedIntrinsicName);
  }

  // Remove all kernels that are meant to serve as a host handle
  for (auto &F : llvm::make_early_inc_range(ClonedModule->functions())) {

    if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
      F.dropAllReferences();
      F.eraseFromParent();
    }
  }

  // Convert all global variables to extern, remove any managed variable
  // initializers
  // Remove any unnecessary variables (e.g. "llvm.metadata")
  // Extract the CUID for identification
  for (auto &GV : llvm::make_early_inc_range(ClonedModule->globals())) {
    auto GVName = GV.getName();
    if (GVName.ends_with(".managed") || GVName == ReservedManagedVar ||
        GV.getSection() == "llvm.metadata") {
      GV.dropAllReferences();
      GV.eraseFromParent();
    } else if (!GVName.starts_with(HipCUIDPrefix)) {
      GV.setInitializer(nullptr);
      GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
      GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
      GV.setDSOLocal(false);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Embedded Module " << ClonedModule->getName()
                          << " dump: ");
  LLVM_DEBUG(ClonedModule->print(llvm::dbgs(), nullptr));

  llvm::SmallVector<char> Data;
  llvm::raw_svector_ostream OS(Data);
  auto PA = llvm::BitcodeWriterPass(OS).run(*ClonedModule, AM);

  llvm::embedBufferInModule(
      M, llvm::MemoryBufferRef(llvm::toStringRef(Data), "ModuleData"),
      ".llvmbc");

  return PA;
}
} // namespace luthier

llvm::PassPluginLibraryInfo getEmbedLuthierBitcodePassPluginInfo() {
  const auto Callback = [](llvm::PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback(
        [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel Opt) {
          MPM.addPass(luthier::EmbedOptimizedBitcodePass());
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "embed-luthier-bitcode", LLVM_VERSION_STRING,
          Callback};
};

#ifndef LLVM_LUTHIER_TOOL_COMPILE_PLUGIN_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getEmbedLuthierBitcodePassPluginInfo();
}
#endif