//===-- MMISlotIndexesAnalysis.cpp ----------------------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file implements the <tt>MMISlotIndexesAnalysis</tt> pass.
//===----------------------------------------------------------------------===//
#include "tooling_common/MMISlotIndexesAnalysis.hpp"
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Module.h>

namespace luthier {

llvm::AnalysisKey MMISlotIndexesAnalysis::Key;

MMISlotIndexesAnalysis::Result
MMISlotIndexesAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  MMISlotIndexesAnalysis::Result Out;
  auto &MMI = MAM.getCachedResult<llvm::MachineModuleAnalysis>(M)->getMMI();

  for (const auto &F : M) {
    auto *MF = MMI.getMachineFunction(F);
    if (!MF)
      continue;
    Out.Res.insert({MF, llvm::SlotIndexes(*MF)});
  }
  return Out;
}
} // namespace luthier
