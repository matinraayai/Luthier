//===-- CreateAndEmbedIModulePass.cpp -------------------------------------===//
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
/// \file CreateAndEmbedIModulePass.cpp
/// Implements the \c CreateAndEmbedIModulePass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolIRCompilation/CreateAndEmbedIModulePass.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolIRCompilation/ExternalizeGlobalsPass.h"
#include "luthier/ToolIRCompilation/FinalizeIntrinsicsPass.h"
#include "luthier/ToolIRCompilation/MarkAnnotationsPass.h"
#include "luthier/ToolIRCompilation/StripKernelsPass.h"
#include "luthier/ToolIRCompilation/SubstituteAMDGCNIntrinsicsPass.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-embed-imodule"

namespace luthier {

llvm::PreservedAnalyses
CreateAndEmbedIModulePass::run(llvm::Module &M,
                               llvm::ModuleAnalysisManager &AM) {
  if (M.getGlobalVariable("llvm.embedded.object", /*AllowInternal=*/true))
    llvm::report_fatal_error(
        "Attempted to embed bitcode twice. Are you passing -fembed-bitcode?",
        /*gen_crash_diag=*/false);

  llvm::Triple T(M.getTargetTriple());
  if (T.getArch() != llvm::Triple::ArchType::amdgcn)
    return llvm::PreservedAnalyses::all();

  std::unique_ptr<llvm::Module> Clone = llvm::CloneModule(M);

  // Pass an explicit null VFS: the defaulted vfs::getRealFileSystem() trips
  // clang's cc1 IO sandbox when this pass runs inside the device compile, and
  // none of the passes below touch the file system.
  llvm::ModuleAnalysisManager CloneMAM;
  llvm::PassBuilder PB(/*TM=*/nullptr, llvm::PipelineTuningOptions(),
                       /*PGOOpt=*/std::nullopt, /*PIC=*/nullptr,
                       /*FS=*/nullptr);
  PB.registerModuleAnalyses(CloneMAM);

  llvm::ModulePassManager InnerMPM;
  InnerMPM.addPass(MarkAnnotationsPass());
  InnerMPM.addPass(FinalizeIntrinsicsPass());
  InnerMPM.addPass(StripKernelsPass());
  InnerMPM.addPass(ExternalizeGlobalsPass());
  InnerMPM.addPass(SubstituteAMDGCNIntrinsicsPass());
  InnerMPM.run(*Clone, CloneMAM);

  LLVM_DEBUG(luthier::dbgs()
                 << "Embedded Module " << Clone->getName() << " dump:\n";
             Clone->print(luthier::dbgs(), nullptr));

  llvm::SmallVector<char> Data;
  llvm::raw_svector_ostream OS(Data);
  llvm::BitcodeWriterPass(OS).run(*Clone, CloneMAM);

  llvm::embedBufferInModule(
      M, llvm::MemoryBufferRef(llvm::toStringRef(Data), "ModuleData"),
      ".llvmbc");

  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
