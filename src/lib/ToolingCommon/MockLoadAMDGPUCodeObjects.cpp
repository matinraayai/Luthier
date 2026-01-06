//===-- MockLoadAMDGPUCodeObjects.cpp -------------------------------------===//
// Copyright 2026 @ Northeastern University Computer Architecture Lab
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
/// \file
/// Implements the \c MockLoadAMDGPUCodeObjects class.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/MockLoadAMDGPUCodeObjects.h"
#include "luthier/Tooling/CodeObjectManagerAnalysis.h"
#include <llvm/IR/Module.h>

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

  /// Go over the code object paths and create buffers for each of them
  for (llvm::StringRef Path : Options.CodeObjectPathList) {
    llvm::Expected<llvm::MemoryBuffer &> CodeObjectBufferOrErr =
        CodeObjectManager.readCodeObjectFromFile(Path);
    LUTHIER_CTX_EMIT_ON_ERROR(Ctx, CodeObjectBufferOrErr.takeError());

    auto LoadedCodeObjectOrErr = Loader.loadCodeObject(*CodeObjectBufferOrErr);
    LUTHIER_CTX_EMIT_ON_ERROR(Ctx, LoadedCodeObjectOrErr.takeError());
  }

  /// Define the external variables
  for (auto &[SymName, SymAddr] : Options.ExternalVars) {
    LUTHIER_CTX_EMIT_ON_ERROR(
        Ctx, Loader.defineExternalSymbol(SymName,
                                         reinterpret_cast<void *>(SymAddr)));
  }
  /// Finalize the loader
  LUTHIER_CTX_EMIT_ON_ERROR(Ctx, Loader.finalize());

  return llvm::PreservedAnalyses::all();
};

} // namespace luthier