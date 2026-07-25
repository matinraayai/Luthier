//===-- ForwardISAStateToCalleesPass.h ------------------------*- C++ -*-===//
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
/// Declares the \c ForwardISAStateToCalleesPass which extends every callee of
/// an injected payload inside an instrumentation module with extra parameters
/// (one per SVA scalar-arg lane and per phys-reg channel it reads) and an
/// optional struct return (one slot per phys-reg channel it writes), then
/// rewrites all call sites to source/sink those values via fresh inline-asm
/// placeholders in the caller. Net effect: callee bodies leave the pass with
/// zero Luthier placeholders, and the MIR lowering stage only has to handle
/// payload entries.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_FORWARD_ISA_STATE_TO_CALLEES_PASS_H
#define LUTHIER_TOOL_CODE_GEN_FORWARD_ISA_STATE_TO_CALLEES_PASS_H
#include <GCNSubtarget.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class ForwardISAStateToCalleesPass
    : public llvm::PassInfoMixin<ForwardISAStateToCalleesPass> {

public:
  ForwardISAStateToCalleesPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &IModule,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif
