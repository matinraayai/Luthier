//===-- TraceIRTranslatorAnalysis.cpp ------------------------------------===//
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
/// \file TraceIRTranslatorAnalysis.cpp
/// Implements the \c TraceIRTranslatorAnalysis and its \c TranslationState
/// result.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TraceIRTranslatorAnalysis.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/TraceIRTranslator.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Debug.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-trace-ir-translator"

namespace luthier {

TranslationState::TranslationState(llvm::MachineFunction &MF) : MF(MF) {
  scanDirtyMarks();
}

TranslationState::TranslationState(TranslationState &&) noexcept = default;

TranslationState::~TranslationState() = default;

void TranslationState::scanDirtyMarks() {
  /// The mark is written on the MBB's first MI, but other passes may have
  /// inserted instructions ahead of the carrier since it was marked, so the
  /// whole block is scanned
  for (const llvm::MachineBasicBlock &MBB : MF)
    for (const llvm::MachineInstr &MI : MBB)
      if (const auto *MD = TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
          MD && MD->needsRetranslation()) {
        DirtyMBBs.insert(&MBB);
        break;
      }
}

void TranslationState::clearDirtyMarks() {
  llvm::LLVMContext &Ctx = MF.getFunction().getContext();
  for (const llvm::MachineBasicBlock *MBB : DirtyMBBs)
    for (const llvm::MachineInstr &MI : *MBB)
      if (auto *MD = TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
          MD && MD->needsRetranslation())
        MD->setNeedsRetranslation(Ctx, false);
}

void TranslationState::markDirty(const llvm::MachineBasicBlock &MBB) {
  assert(MBB.getParent() == &MF && "MBB belongs to a different MF");
  DirtyMBBs.insert(&MBB);
  /// Persist the mark on the first instruction so the dirty state serializes
  /// with the MIR. Empty MBBs stay in-memory only
  if (MBB.empty())
    return;
  auto &FirstMI = const_cast<llvm::MachineInstr &>(MBB.front());
  TargetMachineInstrMDNode *MD =
      TargetMachineInstrMDNode::getInstrMDNodeIfExists(FirstMI);
  if (!MD) {
    auto MDOrErr = TargetMachineInstrMDNode::initializeMDNode(FirstMI);
    if (!MDOrErr) {
      LLVM_DEBUG(luthier::dbgs()
                     << "[TraceIRTranslator] Failed to persist dirty mark: "
                     << llvm::toString(MDOrErr.takeError()) << "\n";);
      return;
    }
    MD = &*MDOrErr;
  }
  MD->setNeedsRetranslation(MF.getFunction().getContext(), true);
}

void TranslationState::markErased(const llvm::MachineBasicBlock &MBB) {
  assert(MBB.getParent() == &MF && "MBB belongs to a different MF");
  DirtyMBBs.erase(&MBB);
  if (auto *BodyBB = const_cast<llvm::BasicBlock *>(MBB.getBasicBlock()))
    ErasedBodyBBs.insert(BodyBB);
}

bool TranslationState::isDirty() const {
  if (!DirtyMBBs.empty() || !ErasedBodyBBs.empty())
    return true;
  /// The function counts as dirty while never translated, and an MBB without
  /// an IR block is new and implicitly dirty
  if (MF.getFunction().empty() && !MF.empty())
    return true;
  return llvm::any_of(MF, [](const llvm::MachineBasicBlock &MBB) {
    return !MBB.getBasicBlock();
  });
}

bool TranslationState::canFlushIncrementally() const {
  /// No translator (first flush, or the analysis was recomputed after
  /// serialization) or untranslated function: full lift required
  if (!Translator || MF.getFunction().empty())
    return false;
  /// Erased or new MBBs change the block set; full re-translation
  if (!ErasedBodyBBs.empty())
    return false;
  if (llvm::any_of(MF, [](const llvm::MachineBasicBlock &MBB) {
        return !MBB.getBasicBlock();
      }))
    return false;
  /// CFG edge changes are beyond in-place body repair
  return llvm::all_of(DirtyMBBs, [&](const llvm::MachineBasicBlock *MBB) {
    return Translator->irSuccessorsMatchMIR(*MBB);
  });
}

llvm::Error TranslationState::flushFull() {
  llvm::Error Err = llvm::Error::success();
  Translator = std::make_unique<TraceIRTranslator>(MF, Err);
  if (Err) {
    Translator.reset();
    return Err;
  }
  Translator->translate();
  /// The full translate dropped every old IR block, so orphans from bailed
  /// incremental re-translations have no users left
  for (llvm::Instruction *I : PendingDeadInsts)
    I->dropAllReferences();
  for (llvm::Instruction *I : PendingDeadInsts) {
    assert(I->use_empty() && "pending dead instruction still has live users");
    I->deleteValue();
  }
  PendingDeadInsts.clear();
  return llvm::Error::success();
}

llvm::Error TranslationState::flush() {
  if (!isDirty())
    return llvm::Error::success();

  LLVM_DEBUG(luthier::dbgs()
                 << "[TraceIRTranslator] Flushing " << DirtyMBBs.size()
                 << " dirty MBBs of " << MF.getName() << "\n";);

  bool Incremental = canFlushIncrementally();

  /// Clear the marks before translating so the needsRetranslation flag never
  /// leaks into the lifted IR through the shared PC sections MDNodes
  clearDirtyMarks();
  llvm::SmallVector<const llvm::MachineBasicBlock *> ToRetranslate(
      DirtyMBBs.begin(), DirtyMBBs.end());
  DirtyMBBs.clear();
  ErasedBodyBBs.clear();

  if (!Incremental)
    return flushFull();

  LLVM_DEBUG(luthier::dbgs() << "[TraceIRTranslator] Incremental flush of "
                             << ToRetranslate.size() << " MBBs\n";);
  for (const llvm::MachineBasicBlock *MBB : ToRetranslate) {
    llvm::Expected<bool> NeedFull =
        Translator->retranslateMBB(*MBB, PendingDeadInsts);
    if (!NeedFull)
      return NeedFull.takeError();
    /// An unkeyed value escaped the old body: in-place repair is impossible
    if (*NeedFull)
      return flushFull();
  }
  Translator->runPostTranslateCleanup();
  return llvm::Error::success();
}

llvm::AnalysisKey TraceIRTranslatorAnalysis::Key;

TraceIRTranslatorAnalysis::Result
TraceIRTranslatorAnalysis::run(llvm::MachineFunction &MF,
                                llvm::MachineFunctionAnalysisManager &) {
  return TranslationState{MF};
}

} // namespace luthier
