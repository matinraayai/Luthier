//===-- CodeDiscoveryPass.h -  ------------------------------------*-C++-*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
#include <llvm/IR/PassManager.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>

namespace luthier {

/// \brief Pass in charge of discovering all statically reachable code and their
/// entry points, and creating their equivalent machine functions
class CodeDiscoveryPass : public llvm::PassInfoMixin<CodeDiscoveryPass> {

  using EntryPointType =
      std::variant<const llvm::amdhsa::kernel_descriptor_t *, uint64_t>;

  EntryPointType InitialEntryPoint;

public:
  explicit CodeDiscoveryPass(EntryPointType InitialEntryPoint)
      : InitialEntryPoint(InitialEntryPoint) {};

  llvm::PreservedAnalyses run(llvm::Module &TargetModule,
                              llvm::ModuleAnalysisManager &TargetMAM);
};

} // namespace luthier

#endif