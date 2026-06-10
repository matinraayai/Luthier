//===-- TraceIRTranslatorAnalysis.h ------------------------------*-C++-*-===//
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
/// \file TraceIRTranslatorAnalysis.h
/// Describes the \c TraceIRTranslatorAnalysis, a pinned \c
/// llvm::MachineFunction analysis owning the lifted IR of a machine function
/// and keeping it up to date with the MIR. Passes that mutate the MIR mark
/// the affected \c llvm::MachineBasicBlock s dirty on the \c TranslationState
/// result instead of invalidating the analysis; consumers of the lifted IR
/// call \c TranslationState::flush() before reading, which re-translates the
/// dirty blocks.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TRACE_IR_TRANSLATOR_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_TRACE_IR_TRANSLATOR_ANALYSIS_H
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Error.h>

namespace llvm {
class BasicBlock;
class Instruction;
class MachineBasicBlock;
class MachineFunction;
} // namespace llvm

namespace luthier {

class TraceIRTranslator;

/// \brief Result of \c TraceIRTranslatorAnalysis: owns the lifted IR of a
/// machine function and keeps it in sync with the MIR via re-translation of
/// dirty MBBs.
///
/// The result is pinned: \c invalidate always returns \c false, following the
/// \c llvm::LazyCallGraph / \c llvm::MemorySSA "update, don't invalidate"
/// model. Passes that mutate the MIR must obtain the cached result and mark
/// the affected MBBs dirty; the next consumer that calls \c flush triggers
/// re-translation, so no client ever invokes the \c TraceIRTranslator
/// directly. The only escape hatch is clearing the analysis manager (which
/// also happens when the MF itself is destroyed).
///
/// Dirty marking rules for mutating passes:
/// - Inserted, erased, or modified instructions: \c markDirty on the MBB.
/// - Added/removed control-flow edges: \c markDirty on the predecessor.
/// - Newly created MBBs need no marking: any MBB with no IR block
///   (\c MBB.getBasicBlock() returning \c nullptr) is implicitly dirty.
/// - Erased MBBs: \c markErased before erasure.
///
/// Dirty marks are persisted in the representation itself: \c markDirty sets
/// the \c needsRetranslation flag on the \c TargetMachineInstrMDNode of the
/// MBB's first instruction, so the dirty state survives serialization and is
/// rebuilt from a metadata scan when the analysis is recomputed. Empty MBBs
/// are tracked in memory only — they have no instruction to carry the mark
/// and translate to an empty BodyBB anyway.
class TranslationState {
  friend class TraceIRTranslatorAnalysis;

  llvm::MachineFunction &MF;

  /// Persistent translator carrying the per-MBB boundary register-value
  /// state that seeds incremental re-translation. Created on the first
  /// flush; recreated whenever a structural change forces a full
  /// re-translation
  std::unique_ptr<TraceIRTranslator> Translator;

  /// MBBs marked dirty since the last flush. Mirrors the per-MI
  /// needsRetranslation metadata; rebuilt from it on analysis (re)computation
  llvm::SmallPtrSet<const llvm::MachineBasicBlock *, 16> DirtyMBBs;

  /// BodyBBs of erased MBBs awaiting removal at the next flush
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> ErasedBodyBBs;

  /// Detached old-body instructions that still had external users when an
  /// incremental re-translation bailed out; deleted right after the full
  /// re-translation drops their users
  llvm::SmallVector<llvm::Instruction *> PendingDeadInsts;

  explicit TranslationState(llvm::MachineFunction &MF);

  /// Rebuilds \c DirtyMBBs from the needsRetranslation metadata marks
  void scanDirtyMarks();

  /// Clears the needsRetranslation marks of all dirty MBBs
  void clearDirtyMarks();

  /// \returns true if the pending changes can be re-translated in place by
  /// \c TraceIRTranslator::retranslateMBB; false when a structural change
  /// (new/erased MBBs, CFG edge changes, untranslated function, or no
  /// persistent translator) requires a full re-translation
  bool canFlushIncrementally() const;

  /// Re-translates the whole function from scratch with a fresh translator
  llvm::Error flushFull();

public:
  TranslationState(TranslationState &&) noexcept;

  ~TranslationState();

  /// Marks \p MBB for re-translation; also use on the predecessor for edge
  /// additions/removals. Cheap; never triggers translation by itself
  void markDirty(const llvm::MachineBasicBlock &MBB);

  /// Marks \p MBB as about to be erased from the MIR; its BodyBB is removed
  /// at the next \c flush. Call before erasing the MBB
  void markErased(const llvm::MachineBasicBlock &MBB);

  /// Brings the lifted IR up to date with the MIR by re-translating dirty
  /// MBBs and removing the BodyBBs of erased ones. Performs the initial full
  /// translation if the function has not been translated yet. Idempotent
  /// when clean. Consumers of the lifted IR must call this before reading
  llvm::Error flush();

  /// \returns true if any pending work would make the lifted IR stale;
  /// readers may assert on this instead of calling \c flush
  [[nodiscard]] bool isDirty() const;

  /// Pinned: the translation is never invalidated by the pass manager; it is
  /// kept up to date through \c markDirty + \c flush instead
  bool invalidate(llvm::MachineFunction &, const llvm::PreservedAnalyses &,
                  llvm::MachineFunctionAnalysisManager::Invalidator &) {
    return false;
  }
};

/// \brief Pinned \c llvm::MachineFunction analysis owning the incrementally
/// maintained MIR-to-IR translation of a lifted machine function
class TraceIRTranslatorAnalysis
    : public llvm::AnalysisInfoMixin<TraceIRTranslatorAnalysis> {
  friend llvm::AnalysisInfoMixin<TraceIRTranslatorAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = TranslationState;

  Result run(llvm::MachineFunction &MF,
             llvm::MachineFunctionAnalysisManager &MFAM);
};

} // namespace luthier

#endif
