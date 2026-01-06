//===-- MockLoadAMDGPUCodeObjects.h -----------------------------*- C++ -*-===//
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
/// Defines the \c MockLoadAMDGPUCodeObjects which reads code objects from
/// specified files on the CLI and loads them into an instance of \c
/// MockAMDGPULoader class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_MOCK_AMDGPU_LOADER_ANALYSIS_H
#define LUTHIER_TOOLING_MOCK_AMDGPU_LOADER_ANALYSIS_H
#include "luthier/Tooling/MockAMDGPULoader.h"
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>

namespace luthier {

/// \brief Parser used to parse the external variable options passed to the
/// \c MockLoadAMDGPUCodeObjects pass
struct MockAMDGPULoaderExternalVarParser
    : public llvm::cl::parser<std::pair<std::string, uint64_t>> {

  MockAMDGPULoaderExternalVarParser(llvm::cl::Option &O)
      : llvm::cl::parser<std::pair<std::string, uint64_t>>(O) {}

  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef ArgValue, std::pair<std::string, uint64_t> &Val);
};

struct MockAMDGPULoaderAnalysisOptions {
  llvm::cl::OptionCategory MockLoaderOptions{
      "AMDGPU Mock Loader Options", "Options regarding how the AMDGPU mock "
                                    "loader loads the given device code"};

  llvm::cl::list<std::string> CodeObjectPathList{
      "code-object-paths",
      llvm::cl::desc("Path to the code objects to be loaded by the mock "
                     "loader; Must have have an extension of ./s/.so/.hsaco")};

  llvm::cl::list<std::pair<std::string, uint64_t>, bool,
                 MockAMDGPULoaderExternalVarParser>
      ExternalVars{
          "extern-var-defs",
          llvm::cl::desc(
              "A set of external variables to be defined by the loader. Must "
              "be formated as <var1>:<addr1> <var2>:<addr2> etc."),
          llvm::cl::NotHidden, llvm::cl::cat(MockLoaderOptions)};
};

class MockLoadAMDGPUCodeObjects
    : public llvm::PassInfoMixin<MockLoadAMDGPUCodeObjects> {
  MockAMDGPULoaderAnalysisOptions &Options;

public:
  explicit MockLoadAMDGPUCodeObjects(MockAMDGPULoaderAnalysisOptions &Options);

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};
} // namespace luthier

#endif