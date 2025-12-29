//===-- CodeDiscoveryPass.h -  ------------------------------------*-C++-*-===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
///
/// \file
/// Defines the \c CodeDiscoveryPass class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_CODE_DISCOVERY_PASS_H
#define LUTHIER_TOOLING_CODE_DISCOVERY_PASS_H
#include "luthier/Tooling/MachineFunctionEntryPoints.h"
#include <llvm/IR/PassManager.h>

namespace luthier {


struct CodeDiscoveryPassOptions {


};


/// \brief Target module pass in charge of:
/// - Discovering all statically reachable code and entry points from a starting
/// entry point
/// - Disassembling and creating equivalent machine functions for each entry
/// point
class CodeDiscoveryPass : public llvm::PassInfoMixin<CodeDiscoveryPass> {

  EntryPoint InitialEntryPoint;

public:
  explicit CodeDiscoveryPass(EntryPoint InitialEntryPoint)
      : InitialEntryPoint(InitialEntryPoint) {};

  llvm::PreservedAnalyses run(llvm::Module &TargetModule,
                              llvm::ModuleAnalysisManager &TargetMAM);
};

} // namespace luthier

#endif