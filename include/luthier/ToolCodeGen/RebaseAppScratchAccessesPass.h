//===-- RebaseAppScratchAccessesPass.h --------------------------*- C++ -*-===//
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
/// Declares \c RebaseAppScratchAccessesPass, a \c Prototype instrumentation 
/// pass that routes every private segment access in the target application 
/// to go right above where the instrumentation logic's private segment will 
/// sit in the final instrumented code.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_REBASE_APP_SCRATCH_ACCESSES_PASS_H
#define LUTHIER_TOOL_CODE_GEN_REBASE_APP_SCRATCH_ACCESSES_PASS_H
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h"
#include <llvm/IR/PassManager.h>
#include <memory>

namespace luthier {

class RebaseAppScratchAccessesPass
    : public llvm::RequiredPassInfoMixin<RebaseAppScratchAccessesPass> {
public:
  
  class Parser : public ToolDeviceCodeOffloadParserTrait<Parser> {
  public:
    using luthier::ToolDeviceCodeOffloadParserTrait<
        Parser>::ToolDeviceCodeOffloadParserTrait;
  };

  explicit RebaseAppScratchAccessesPass(llvm::Error &Err);

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);

  static llvm::StringRef name() {
    return "luthier-rebase-app-scratch-accesses";
  }

  static bool isRequired() { return true; }

private:
  std::unique_ptr<Parser> OffloadParser;
};

} // namespace luthier

#endif
