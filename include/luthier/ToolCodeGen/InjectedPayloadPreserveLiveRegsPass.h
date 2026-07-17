//===-- InjectedPayloadPreserveLiveRegsPass.h ------------------*- C++ -*-===//
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
/// \file InjectedPayloadPreserveLiveRegsPass.h
/// IModule transform: for each injected-payload function, snapshot the
/// physical registers that are live at the AppMI insertion point and not
/// already participating in the payload's declared read/write set, so the
/// payload restores them on exit and the target application's state is
/// preserved across the payload's execution.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_PRESERVE_LIVE_REGS_PASS_H
#define LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_PRESERVE_LIVE_REGS_PASS_H
#include "luthier/ToolCodeGen/LegacyPassSupport.h"
#include <llvm/Pass.h>

namespace luthier {

class InjectedPayloadPreserveLiveRegsPass;

LUTHIER_INITIALIZE_LEGACY_PASS_PROTOTYPE(InjectedPayloadPreserveLiveRegsPass);

/// \brief Legacy module pass: emits preservation copies in every injected-
/// payload function for the physical registers that are live at the
/// instrumentation point but neither read nor written by the payload.
///
/// For each payload \c F with insertion point \c AppMI:
///   - \c Preserve = Active(F-pre) \ (F.Reads U F.Writes)
///   - For each \c R in \c Preserve:
///     - In F's entry MBB: \c %vreg_R = COPY \c $R, mark \c $R as live-in.
///     - In every return MBB of F: \c $R = COPY \c %vreg_R before the
///       terminator; add \c implicit-use \c $R on the terminator so RA
///       treats it as live-out.
///
/// Depends on \c IModuleIPPredicatedLivenessAnalysis (for Active) and
/// \c InjectedPayloadSideEffectsAnalysis (for Reads/Writes).
class InjectedPayloadPreserveLiveRegsPass : public llvm::ModulePass {
public:
  static char ID;

  InjectedPayloadPreserveLiveRegsPass() : llvm::ModulePass(ID) {}

  [[nodiscard]] llvm::StringRef getPassName() const override {
    return "Luthier Injected Payload Preserve Live Regs Pass";
  }

  bool runOnModule(llvm::Module &IModule) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

} // namespace luthier

#endif
