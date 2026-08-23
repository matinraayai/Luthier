//===-- MockLoadAMDGPUCodeObjects.h -----------------------------*- C++ -*-===//
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
/// \file MockLoadAMDGPUCodeObjects.h
/// Defines the \c MockLoadAMDGPUCodeObjects pass which reads code objects from
/// specified files on the CLI and loads them into the instance of \c
/// MockAMDGPULoader defined by the \c MockAMDGPULoaderAnalysis pass.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TESTING_MOCK_AMDGPU_LOADER_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_TESTING_MOCK_AMDGPU_LOADER_ANALYSIS_H
#include "luthier/ToolCodeGen/MockAMDGPULoader.h"
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <string>
#include <utility>
#include <variant>

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

/// \brief Spec of an entry point relative to the mock loader: a code object
/// index paired with either a symbol name or a load offset into that object.
using MockAMDGPULoaderEntryPointSpec =
    std::pair<uint64_t, std::variant<uint64_t, std::string>>;

/// \brief Parser for \c -initial-entrypoint, accepting
/// \c <code-object-index>:<mangled-symbol-name> or
/// \c <code-object-index>:<load-offset>
struct MockAMDGPULoaderInitialEntryPointParser
    : public llvm::cl::parser<MockAMDGPULoaderEntryPointSpec> {

  MockAMDGPULoaderInitialEntryPointParser(llvm::cl::Option &O)
      : llvm::cl::parser<MockAMDGPULoaderEntryPointSpec>(O) {}

  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef ArgValue, MockAMDGPULoaderEntryPointSpec &Val);
};

/// \brief Parser for \c -initial-execution-point, accepting
/// \c <code-object-index>:<mangled-symbol-name>
struct MockAMDGPULoaderInitialExecutionPointParser
    : public llvm::cl::parser<std::pair<uint64_t, std::string>> {

  MockAMDGPULoaderInitialExecutionPointParser(llvm::cl::Option &O)
      : llvm::cl::parser<std::pair<uint64_t, std::string>>(O) {}

  // parse - Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef ArgValue, std::pair<uint64_t, std::string> &Val);
};

struct MockAMDGPULoaderAnalysisOptions {
  llvm::cl::OptionCategory MockLoaderOptions{
      "AMDGPU Mock Loader Options",
      "Options regarding how the AMDGPU mock "
      "loader loads the given device code objects"};

  llvm::cl::list<std::string> CodeObjectPathList{
      "code-object-paths",
      llvm::cl::desc("Path to the code objects to be loaded by the mock "
                     "loader; Must have have an extension of ./s/.so/.hsaco"),
      llvm::cl::cat(MockLoaderOptions)};

  llvm::cl::list<std::pair<std::string, uint64_t>, bool,
                 MockAMDGPULoaderExternalVarParser>
      ExternalVars{
          "extern-var-defs",
          llvm::cl::desc(
              "A set of external variables to be defined by the loader. Must "
              "be formated as <var1>:<addr1> <var2>:<addr2> etc."),
          llvm::cl::NotHidden, llvm::cl::cat(MockLoaderOptions)};

  llvm::cl::opt<MockAMDGPULoaderEntryPointSpec, false,
                MockAMDGPULoaderInitialEntryPointParser>
      InitialEntryPoint{
          "initial-entrypoint",
          llvm::cl::desc(
              "The initial entry point of the lifting process. "
              "Formatted as <code-object-index>:<mangled-symbol-name> or "
              "<code-object-index>:<load-offset>. \n"
              "Code objects are zero indexed w.r.t the order they are "
              "specified to be loaded into the mock loader."),
          llvm::cl::NotHidden, llvm::cl::cat(MockLoaderOptions)};

  llvm::cl::opt<std::pair<uint64_t, std::string>, false,
                MockAMDGPULoaderInitialExecutionPointParser>
      InitialExecutionPoint{
          "initial-execution-point",
          llvm::cl::desc(
              "The initial execution point of the lifting process. "
              "Formatted as <code-object-index>:<mangled-symbol-name>. \n"
              "Code objects are zero indexed w.r.t the order they are "
              "specified to be loaded into the mock loader."),
          llvm::cl::NotHidden, llvm::cl::cat(MockLoaderOptions)};
};

/// \brief Loads the code objects named on the command line into the
/// \c MockAMDGPULoader, then records the initial entry and execution points on
/// the module.
///
/// \details Resolving \c -initial-entrypoint / \c -initial-execution-point is
/// this pass's job because their spelling is loader-relative — a code object
/// index plus a symbol or offset only means something to whoever performed the
/// load. The resolved addresses are written to the module as
/// \c luthier.initial_entry_point and \c luthier.initial_execution_point (see
/// \c InitialEntryPointAnalysis.h), so downstream analyses read them straight
/// off the module and never need to know a mock loader was involved.
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