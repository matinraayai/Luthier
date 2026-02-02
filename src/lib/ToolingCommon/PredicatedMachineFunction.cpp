//===-- PredicatedMachineFunction.cpp -------------------------------------===//
// Copyright 2026 @ Northeastern University Computer Architecture Lab
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
/// \file PredicatedMachineFunction.cpp
/// Implements the \c PredicatedMachineFunction class and its builder.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/PredicatedMachineFunction.h"
#include "luthier/Tooling/FunctionAnnotations.h"
#include <SIMachineFunctionInfo.h>

namespace luthier {

void PredicatedMachineFunction::print(llvm::raw_ostream &OS) const {
  OS << "# Predicated Machine Function " << MF.getName() << ":\n";
  OS << "\n";
  for (const auto &SMBB : *this) {
    SMBB.print(OS, 2);
  }
  OS << "\n";
  OS << "\n# End Predicated Machine Function " << MF.getName() << ".\n\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void PredicatedMachineFunction::dump() const { print(llvm::dbgs()); }
#endif

std::unique_ptr<PredMFBuilder>
PredMFBuilder::createPredMF(IPPredicatedCFG &ParentCFG,
                            llvm::MachineFunction &MF) {
  std::unique_ptr<PredMFBuilder> Out(new PredMFBuilder(ParentCFG, MF));

  PredicatedMachineFunction &PredMF = Out->PredMF;

  LinearMBBBuilder *EntryFuncScalarMBB{nullptr};

  if (MF.getFunction().hasFnAttribute(InitialEntryPointAttr)) {
    EntryFuncScalarMBB =
        PredMF.LinearMBBs
            .emplace_back(LinearMBBBuilder::createEntryLinearMBB(PredMF))
            .get();
  }

  for (llvm::MachineBasicBlock &MBB : MF) {
    LinearMBBBuilder &CreatedLinearMBB = *PredMF.LinearMBBs.emplace_back(
        LinearMBBBuilder::createLinearMBB(MBB, PredMF));
    PredMF.MBBToLinearMBBs.insert({std::ref(MBB), std::ref(CreatedLinearMBB)});
  }

  // Link scalar MBBs and the entry/exit vector blocks
  for (auto &[MBB, SMBB] : PredMF.MBBToLinearMBBs) {
    if (EntryFuncScalarMBB && MBB.get().isEntryBlock()) {
      EntryFuncScalarMBB->addSuccessor(SMBB);
    }
    for (const auto *MBBSucc : MBB.get().successors()) {
      SMBB.get().addSuccessor(PredMF.MBBToLinearMBBs.at(*MBBSucc));
    }
    for (const auto *MBBPred : MBB.get().predecessors()) {
      SMBB.get().addPredecessor(PredMF.MBBToLinearMBBs.at(*MBBPred));
    }
  }
  return Out;
}

} // namespace luthier