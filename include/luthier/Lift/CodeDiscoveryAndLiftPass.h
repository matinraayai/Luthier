//===-- CodeDiscoveryAndLiftPass.h ------------------------------*- C++ -*-===//
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
/// \file
/// Implements the \c CodeDiscoveryAndLiftPass class, in charge of discovering
/// reachable code from the current execution point and lifting it to LLVM MIR.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LIFT_CODE_DISCOVERY_AND_LIFT_PASS_H
#define LUTHIER_LIFT_CODE_DISCOVERY_AND_LIFT_PASS_H
#include <llvm/IR/PassManager.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>

namespace luthier {

/// \brief A pass in charge of discovering statically reachable code from a
/// given starting execution point and lifting it to LLVM MIR
class CodeDiscoveryAndLiftPass
    : public llvm::PassInfoMixin<CodeDiscoveryAndLiftPass> {
private:
  using EntryPointType =
      std::variant<llvm::amdhsa::kernel_descriptor_t &, uint64_t>;

  const EntryPointType EntryPoint;

public:
  explicit CodeDiscoveryAndLiftPass(const EntryPointType EntryPoint)
      : EntryPoint(EntryPoint) {};

  llvm::PreservedAnalyses run(llvm::Module &TargetM,
                              llvm::ModuleAnalysisManager &TargetMAM);
};

} // namespace luthier

#endif