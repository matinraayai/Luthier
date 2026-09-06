//===-- TargetModuleBranchRelaxation.h --------------------------*- C++ -*-===//
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
/// Target-module branch relaxer - A custom version of
/// \c llvm::BranchRelaxation. Tracks the same per-block size + offset
/// model as stock but performs the long-branch emission via a
/// Luthier-owned helper that delegates SGPR scavenging to
/// \c TargetModuleScavenger , which protects the SVA storage registers.
///
/// When the scavenger finds nothing free at the branch, the relaxer borrows
/// an \c SGPR_64 pair anyway and parks its application value in the SVA
/// across the jump via
/// \c StateValueArrayStorage::emitLongJumpSGPRSpill /
/// \c emitLongJumpSGPRRestore . That save is emitted cross-block by hand —
/// the spill goes in the trampoline block and the reload at the branch's
/// destination — rather than through \c TargetModuleScavenger 's own SVA
/// sink, which only handles a same-block reload.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TARGET_MODULE_BRANCH_RELAXATION_H
#define LUTHIER_TOOL_CODE_GEN_TARGET_MODULE_BRANCH_RELAXATION_H

#include "luthier/ToolCodeGen/TargetModuleScavenger.h"

namespace llvm {
class MachineFunction;
} // namespace llvm

namespace luthier {

class IPPredicatedCFG;
class IPPredicatedLiveness;
class SVStorageAndLoadLocations;
class StateValueArraySpecs;

class TargetModuleBranchRelaxation {
public:

  TargetModuleBranchRelaxation(const IPPredicatedCFG &IPCFG,
                               const IPPredicatedLiveness &IPLiveness,
                               const SVStorageAndLoadLocations &SVLoc,
                               const StateValueArraySpecs &Specs)
      : IPCFG(IPCFG), IPLiveness(IPLiveness), SVLoc(SVLoc), Specs(Specs) {}

  /// Run branch relaxation on \p MF. Returns true if any branch was
  /// relaxed. Mirrors \c llvm::BranchRelaxation::run.
  bool run(llvm::MachineFunction &MF);

private:
  const IPPredicatedCFG &IPCFG;
  const IPPredicatedLiveness &IPLiveness;
  const SVStorageAndLoadLocations &SVLoc;
  const StateValueArraySpecs &Specs;
};

} // namespace luthier

#endif
