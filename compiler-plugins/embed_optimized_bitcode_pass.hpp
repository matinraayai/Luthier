//===-- embed_optimized_bitcode_pass.hpp - Luthier tool compile plugin ----===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file creates a simple LLVM compile plugin that forces HIP to embedd
/// LLVM bitcode for Luthier tools.
//===----------------------------------------------------------------------===//

#ifndef COMPILE_PLUGIN_HPP
#define COMPILE_PLUGIN_HPP
#include <llvm/IR/PassManager.h>

namespace llvm {
class Module;
} // namespace llvm

namespace luthier {

/// Pass embeds a copy of the module optimized with the provided pass pipeline
/// into a global variable.
class EmbedOptimizedBitcodePass
    : public llvm::PassInfoMixin<EmbedOptimizedBitcodePass> {

public:
  EmbedOptimizedBitcodePass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);

  static bool isRequired() { return true; }
};

} // namespace luthier

#endif