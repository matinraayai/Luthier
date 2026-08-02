//===-- MockLoadAMDGPUCodeObjects.cpp -------------------------------------===//
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
/// \file MockLoadAMDGPUCodeObjects.cpp
/// Implements the \c MockLoadAMDGPUCodeObjects class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MockLoadAMDGPUCodeObjects.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/CodeObjectManagerAnalysis.h"
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-mock-load-amdgpu-code-objects"

namespace luthier {

bool MockAMDGPULoaderExternalVarParser::parse(
    llvm::cl::Option &O, llvm::StringRef ArgName, llvm::StringRef ArgValue,
    std::pair<std::string, uint64_t> &Val) {
  auto [ExternVarName, Addr] = ArgValue.split(':');
  Val.first = ExternVarName.str();
  if (Addr.getAsInteger(0, Val.second)) {
    return O.error("Failed to parse the address for variable " + Val.first +
                   ".");
  }
  return false;
}

MockLoadAMDGPUCodeObjects::MockLoadAMDGPUCodeObjects(
    MockAMDGPULoaderAnalysisOptions &Options)
    : Options(Options) {}

llvm::PreservedAnalyses
MockLoadAMDGPUCodeObjects::run(llvm::Module &M,
                               llvm::ModuleAnalysisManager &MAM) {
  llvm::LLVMContext &Ctx = M.getContext();
  /// Get the mock loader analysis and the code object analysis
  MockAMDGPULoader &Loader =
      MAM.getResult<MockAMDGPULoaderAnalysis>(M).getLoader();
  CodeObjectManagerAnalysis::Result CodeObjectManager =
      MAM.getResult<CodeObjectManagerAnalysis>(M);

  LLVM_DEBUG(luthier::dbgs()
             << "[MockLoadAMDGPUCodeObjects] "
             << Options.CodeObjectPathList.size() << " code object path(s), "
             << Options.ExternalVars.size()
             << " external variable definition(s)\n");

  /// Go over the code object paths and create buffers for each of them
  for (llvm::StringRef Path : Options.CodeObjectPathList) {
    LLVM_DEBUG(luthier::dbgs()
               << "[MockLoadAMDGPUCodeObjects] Reading code object from "
               << Path << "\n");
    llvm::Expected<llvm::MemoryBuffer &> CodeObjectBufferOrErr =
        CodeObjectManager.readCodeObjectFromFile(Path);
    if (auto Err = CodeObjectBufferOrErr.takeError()) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    auto LoadedCodeObjectOrErr = Loader.loadCodeObject(*CodeObjectBufferOrErr);
    if (auto Err = LoadedCodeObjectOrErr.takeError()) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
    LLVM_DEBUG(luthier::dbgs()
               << "[MockLoadAMDGPUCodeObjects] Loaded " << Path
               << " (load base "
               << llvm::format_hex(
                      reinterpret_cast<uint64_t>(
                          LoadedCodeObjectOrErr->getLoadedRegion().data()),
                      18)
               << ", size " << LoadedCodeObjectOrErr->getLoadedRegion().size()
               << ")\n");
  }

  /// Define the external variables
  for (auto &[SymName, SymAddr] : Options.ExternalVars) {
    LLVM_DEBUG(luthier::dbgs()
               << "[MockLoadAMDGPUCodeObjects] Defining external "
                  "symbol "
               << SymName << " at " << llvm::format_hex(SymAddr, 18) << "\n");
    if (auto Err = Loader.defineExternalSymbol(
            SymName, reinterpret_cast<void *>(SymAddr))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
  }

  /// Finalize the loader
  LLVM_DEBUG(luthier::dbgs()
             << "[MockLoadAMDGPUCodeObjects] Finalizing loader ("
             << Loader.loaded_code_objects_size() << " LCO(s), "
             << Loader.external_symbol_size() << " external symbol(s))\n");
  if (auto Err = Loader.finalize()) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return llvm::PreservedAnalyses::all();
  }

  return llvm::PreservedAnalyses::all();
};

} // namespace luthier