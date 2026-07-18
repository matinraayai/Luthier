//===-- TargetModulePatcherPass.h -------------------------------*- C++ -*-===//
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
/// \file TargetModulePatcherPass.h
/// Prototype-level master pass that patches the IModule into the
/// target module to produce a fully-instrumented target code. Runs as the
/// final Prototype pass. This pass consists of two stages:
///
/// - **SVA Setup & Storage Code Emission**:
///   - For the initial-entry-point kernel: emit the SVA-setup sequence
///     that populates the SVA lanes from kernarg-preloaded SGPRs (see
///     \c emitCodeToSetupScratch \c emitCodeToStoreSGPRKernelArg)
///   - For each target MF: walk SVStorageAndLoadLocations'
///     StateValueStorageIntervals and emit
///     `currentSVS.emitCodeToSwitchSVS(MI, nextSVS)` at every interval
///     boundary, so the SVA migrates between storage schemes correctly
///     across the target's control flow.
///
/// **Phase B — Target Patching**
///   - Clone every non-payload Function + GlobalVariable + GlobalAlias
///     + GlobalIFunc from the IModule into the target module (per user:
///     "consented to having them present in the final binary").
///   - Strip stale `amdgpu-num-vgpr` / `amdgpu-num-sgpr` attributes from
///     target functions (CodeDiscoveryPass set them, and they're no
///     longer correct after instrumentation extends register usage).
///   - First iteration (minimal): every injected payload is outlined.
///     At each AppMI, replace with an `s_branch` to a per-payload label
///     emitted after the host function. Once all outlined payloads are
///     placed, walk the s_branches and relax any whose displacement
///     exceeds the s_branch limit to `s_setpc_b64`-via-scavenged-SGPRs,
///     using IModuleIPPredicatedLivenessAnalysis::getPMBBLiveIns and,
///     as a last resort, two free SVA lanes from
///     `StateValueArraySpecs::findLowestFreeLanes`.
///
/// Pipeline slot: very last Prototype-level pass, after
/// `injected-payload-pei` and `machine-passes` have finished lowering the
/// instrumentation module's MIR.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TARGET_MODULE_PATCHER_PASS_H
#define LUTHIER_TOOL_CODE_GEN_TARGET_MODULE_PATCHER_PASS_H
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/IR/PassManager.h>

namespace luthier {

class TargetModulePatcherPass
    : public llvm::PassInfoMixin<TargetModulePatcherPass> {
public:
  TargetModulePatcherPass() = default;

  llvm::PreservedAnalyses run(Prototype &IP,
                              PrototypeAnalysisManager &IPAM);
};

} // namespace luthier

#endif
