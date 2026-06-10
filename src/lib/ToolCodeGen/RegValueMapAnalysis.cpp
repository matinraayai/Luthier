//===-- RegValueMapAnalysis.cpp -------------------------------------------===//
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
/// \file RegValueMapAnalysis.cpp
/// Implements the \c RegValueMapAnalysis and its \c RegValueMap result.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/RegValueMapAnalysis.h"
#include "luthier/LLVM/streams.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-reg-value-map"

namespace luthier {

bool RegValueMap::invalidate(llvm::Function &,
                             const llvm::PreservedAnalyses &PA,
                             llvm::FunctionAnalysisManager::Invalidator &) {
  /// The map holds raw Value pointers into the lifted body, so it is stale
  /// whenever the function changes; cheap to recompute, never pinned
  auto PAC = PA.getChecker<RegValueMapAnalysis>();
  return !PAC.preservedWhenStateless();
}

llvm::AnalysisKey RegValueMapAnalysis::Key;

RegValueMapAnalysis::Result
RegValueMapAnalysis::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  RegValueMap Result;

  /// Function-entry seeds: (Value, Name, Base, Off, Halves) tuples
  if (const llvm::MDNode *EntryMap = F.getMetadata(EntryRegMapMDKindName)) {
    for (const llvm::MDOperand &Op : EntryMap->operands()) {
      const auto *Entry = llvm::dyn_cast<llvm::MDNode>(Op.get());
      if (!Entry || Entry->getNumOperands() != 5)
        continue;
      const auto *VMD =
          llvm::dyn_cast<llvm::ValueAsMetadata>(Entry->getOperand(0));
      const auto *Base =
          llvm::mdconst::dyn_extract<llvm::ConstantInt>(Entry->getOperand(2));
      const auto *Off =
          llvm::mdconst::dyn_extract<llvm::ConstantInt>(Entry->getOperand(3));
      const auto *Halves =
          llvm::mdconst::dyn_extract<llvm::ConstantInt>(Entry->getOperand(4));
      if (!VMD || !Base || !Off || !Halves)
        continue;
      Result.EntrySeeds[{static_cast<unsigned>(Base->getZExtValue()),
                         static_cast<unsigned>(Off->getZExtValue()),
                         static_cast<unsigned>(Halves->getZExtValue())}] =
          VMD->getValue();
    }
  }

  /// Per-block exit values: forward walk; the last tagged definition of a
  /// slice wins
  llvm::SmallVector<RegValueDesc, 4> Descs;
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      Descs.clear();
      getRegValues(I, Descs);
      for (const RegValueDesc &D : Descs)
        Result.ExitValues[&BB][RegValueMap::getKey(D)] = &I;
    }
  }

  LLVM_DEBUG(luthier::dbgs()
                 << "[RegValueMap] " << F.getName() << ": "
                 << Result.EntrySeeds.size() << " entry seeds, "
                 << Result.ExitValues.size() << " blocks with exit values\n";);

  return Result;
}

void RegValueMap::print(llvm::raw_ostream &OS, const llvm::Function &F) const {
  OS << "RegValueMap for function '" << F.getName()
     << "': " << EntrySeeds.size() << " entry seeds\n";
  for (const llvm::BasicBlock &BB : F) {
    auto It = ExitValues.find(&BB);
    if (It == ExitValues.end())
      continue;
    OS << "  block ";
    BB.printAsOperand(OS, /*PrintType=*/false);
    OS << ": " << It->second.size() << " register slices\n";
    for (const auto &[Key, V] : It->second) {
      auto [Base, Off, Halves] = Key;
      OS << llvm::formatv("    reg {0} +h{1}:{2} = ", Base, Off, Halves);
      V->printAsOperand(OS, /*PrintType=*/true);
      OS << "\n";
    }
  }
}

llvm::PreservedAnalyses
RegValueMapPrinter::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  auto &FAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (llvm::Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    FAM.getResult<RegValueMapAnalysis>(F).print(OS, F);
  }
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
