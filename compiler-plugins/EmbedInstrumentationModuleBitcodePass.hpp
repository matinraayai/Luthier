//===-- EmbedInstrumentationModuleBitcodePass.hpp -------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes a LLVM compile plugin for embedding Luthier
/// instrumentation module bitcode inside tool device code.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_COMPILE_PLUGINS_EMBED_INSTRUMENTATION_MODULE_BITCODE_PASS_HPP
#define LUTHIER_COMPILE_PLUGINS_EMBED_INSTRUMENTATION_MODULE_BITCODE_PASS_HPP
#include <llvm/IR/PassManager.h>

namespace llvm {
class Module;
} // namespace llvm

namespace luthier {

/// \brief This pass intercepts the optimized HIP device module of a Luthier
/// tool, clones and pre-process it, and then embeds it inside the device
/// code's ELF
class EmbedInstrumentationModuleBitcodePass
    : public llvm::PassInfoMixin<EmbedInstrumentationModuleBitcodePass> {

public:
  EmbedInstrumentationModuleBitcodePass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);

  static bool isRequired() { return true; }
};

} // namespace luthier

#endif