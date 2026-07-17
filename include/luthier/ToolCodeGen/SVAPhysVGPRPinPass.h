//===-- SVAPhysVGPRPinPass.h ------------------------------------*- C++ -*-===//
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
/// \file SVAPhysVGPRPinPass.h
/// InstrumentPrototype-level pass that runs pre-WWM-regalloc on injected-
/// payload MFs and pins the single MFInfo->SGPRSpillVGPRs[] entry (created
/// by IntrinsicMIRLoweringPass::materializeReadlanes) to the physical VGPR
/// that LRStateValueStorageAndLoadLocationsAnalysis selected as the SVA
/// load destination. Guarantees cross-MF consistency: every payload MF
/// and the kernel prolog use the SAME physical VGPR for the SVA without
/// any post-link patching.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_SVA_PHYS_VGPR_PIN_PASS_H
#define LUTHIER_TOOL_CODE_GEN_SVA_PHYS_VGPR_PIN_PASS_H
#include "luthier/ToolCodeGen/InstrumentPrototype.h"
#include <llvm/IR/PassManager.h>

namespace luthier {

class SVAPhysVGPRPinPass : public llvm::PassInfoMixin<SVAPhysVGPRPinPass> {
public:
  SVAPhysVGPRPinPass() = default;

  llvm::PreservedAnalyses run(InstrumentPrototype &IP,
                              InstrumentPrototypeAnalysisManager &IPAM);
};

} // namespace luthier

#endif
