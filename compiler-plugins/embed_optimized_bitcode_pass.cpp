#include "embed_optimized_bitcode_pass.hpp"

#include "llvm/Passes/PassPlugin.h"
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-embed-optimized-bitcode-pass"

namespace luthier {
llvm::PreservedAnalyses
EmbedOptimizedBitcodePass::run(llvm::Module &M,
                               llvm::ModuleAnalysisManager &AM) {
  if (M.getGlobalVariable("llvm.embedded.module", /*AllowInternal=*/true))
    llvm::report_fatal_error(
        "Attempted to embed bitcode twice. Are you passing -fembed-bitcode?",
        /*gen_crash_diag=*/false);

  llvm::Triple T(M.getTargetTriple());
  if (T.getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();

  LLVM_DEBUG(llvm::dbgs() << "Embedded Module " << M.getName() << " dump: ");
  LLVM_DEBUG(M.print(llvm::dbgs(), nullptr));

  std::string Data;
  llvm::raw_string_ostream OS(Data);
  auto PA =
      llvm::BitcodeWriterPass(OS, /*ShouldPreserveUseListOrder=*/true, true)
          .run(M, AM);

  llvm::embedBufferInModule(M, llvm::MemoryBufferRef(Data, "ModuleData"),
                            ".llvmbc");

  return llvm::PreservedAnalyses::all();
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