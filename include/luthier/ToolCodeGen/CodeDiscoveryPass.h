//===-- CodeDiscoveryPass.h ---------------------------------------*-C++-*-===//
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
/// \file
/// Defines the \c CodeDiscoveryPass class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_CODE_DISCOVERY_PASS_H
#define LUTHIER_TOOL_CODE_GEN_CODE_DISCOVERY_PASS_H
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>

namespace luthier {

/// \brief Command-line options for \c CodeDiscoveryPass.
struct CodeDiscoveryPassOptions {
  llvm::cl::opt<bool> EagerDiscoverCallReturnEntryPoint{
      "eager-discover-call-ret-entry-point",
      llvm::cl::desc(
          "Eagerly enqueue the post-call return PC as an entry point to "
          "be added to the set of entry points for the code discovery pass."),
      llvm::cl::init(true)};
};

/// \brief Target module pass in charge of:
/// - Discovering all statically reachable code and entry points from an
/// initial entry point. The entry point can be any function
/// (entry or non-entry)
/// - Disassembling and creating equivalent machine functions for each entry
/// point
/// - Translating each recovered machine function to equivalent LLVM IR for
/// further semantics analysis
class CodeDiscoveryPass : public llvm::PassInfoMixin<CodeDiscoveryPass> {
  const CodeDiscoveryPassOptions &Opts;

public:
  explicit CodeDiscoveryPass(const CodeDiscoveryPassOptions &Opts)
      : Opts(Opts) {}

  llvm::PreservedAnalyses run(llvm::Module &TargetModule,
                              llvm::ModuleAnalysisManager &TargetMAM);
};

} // namespace luthier

#endif