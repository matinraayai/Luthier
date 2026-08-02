//===-- AMDGPUMockLoaderPrinter.h -------------------------------*- C++ -*-===//
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
/// \file AMDGPUMockLoaderPrinter.h
/// Defines the \c AMDGPUMockLoaderPrinter pass which goes over all loaded
/// code objects in the loader, identifies any segment that has an executable
/// permission, disassembles their instructions, and prints them.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TESTING_AMDGPU_MOCK_LOADER_PRINTER_H
#define LUTHIER_TOOL_CODE_GEN_TESTING_AMDGPU_MOCK_LOADER_PRINTER_H
#include <llvm/IR/PassManager.h>

namespace luthier {

class AMDGPUMockLoaderPrinter
    : public llvm::PassInfoMixin<AMDGPUMockLoaderPrinter> {
  llvm::raw_ostream &OS;

public:
  explicit AMDGPUMockLoaderPrinter(llvm::raw_ostream &OS);

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};
} // namespace luthier

#endif