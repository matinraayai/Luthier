//===-- IPReachingRefAnalysis.h ----------------------------------*- C++-*-===//
// Copyright 2022-2026 @ Northeastern University Computer Architecture Lab
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
/// \file IPReachingRefAnalysis.h
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_TOOLING_IP_REACHING_DEF_ANALYSIS_H
#define LUTHIER_TOOLING_IP_REACHING_DEF_ANALYSIS_H
#include "IPLoopTraversal.h"
#include "luthier/Tooling/IPVectorCFG.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TinyPtrVector.h>
#include <llvm/CodeGen/LoopTraversal.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachinePassManager.h>

namespace llvm {
class MachineBasicBlock;
class MachineInstr;
} // namespace llvm

namespace luthier {
class IPVectorRegLiveness;
/// Thin wrapper around "int" used to store reaching definitions,
/// using an encoding that makes it compatible with TinyPtrVector.
/// The 0th LSB is forced zero (and will be used for pointer union tagging),
/// The 1st LSB is forced one (to make sure the value is non-zero).
class ReachingDef {
  uintptr_t Encoded;
  friend struct llvm::PointerLikeTypeTraits<ReachingDef>;
  explicit ReachingDef(uintptr_t Encoded) : Encoded(Encoded) {}

public:
  ReachingDef(std::nullptr_t) : Encoded(0) {}
  ReachingDef(int Instr) : Encoded(((uintptr_t)Instr << 2) | 2) {}
  operator int() const { return ((int)Encoded) >> 2; }
};
} // namespace luthier

namespace llvm {
template <> struct PointerLikeTypeTraits<luthier::ReachingDef> {
  static constexpr int NumLowBitsAvailable = 1;

  static inline void *getAsVoidPointer(const luthier::ReachingDef &RD) {
    return reinterpret_cast<void *>(RD.Encoded);
  }

  static inline luthier::ReachingDef getFromVoidPointer(void *P) {
    return luthier::ReachingDef(reinterpret_cast<uintptr_t>(P));
  }

  static inline luthier::ReachingDef getFromVoidPointer(const void *P) {
    return luthier::ReachingDef(reinterpret_cast<uintptr_t>(P));
  }
};
} // namespace llvm

namespace luthier {

// The storage for all reaching definitions.
class MBBReachingDefsInfo {
public:
  void init(unsigned NumBlockIDs) { AllReachingDefs.resize(NumBlockIDs); }

  unsigned numBlockIDs() const { return AllReachingDefs.size(); }

  void startBasicBlock(unsigned MBBNumber, unsigned NumRegUnits) {
    AllReachingDefs[MBBNumber].resize(NumRegUnits);
  }

  void append(unsigned MBBNumber, llvm::MCRegUnit Unit, int Def) {
    AllReachingDefs[MBBNumber][static_cast<unsigned>(Unit)].push_back(Def);
  }

  void prepend(unsigned MBBNumber, llvm::MCRegUnit Unit, int Def) {
    auto &Defs = AllReachingDefs[MBBNumber][static_cast<unsigned>(Unit)];
    Defs.insert(Defs.begin(), Def);
  }

  void replaceFront(unsigned MBBNumber, llvm::MCRegUnit Unit, int Def) {
    assert(!AllReachingDefs[MBBNumber][static_cast<unsigned>(Unit)].empty());
    *AllReachingDefs[MBBNumber][static_cast<unsigned>(Unit)].begin() = Def;
  }

  void clear() { AllReachingDefs.clear(); }

  llvm::ArrayRef<ReachingDef> defs(unsigned MBBNumber,
                                   llvm::MCRegUnit Unit) const {
    if (AllReachingDefs[MBBNumber].empty())
      // Block IDs are not necessarily dense.
      return llvm::ArrayRef<ReachingDef>();
    return AllReachingDefs[MBBNumber][static_cast<unsigned>(Unit)];
  }

private:
  /// All reaching defs of a given RegUnit for a given MBB.
  using MBBRegUnitDefs = llvm::TinyPtrVector<ReachingDef>;
  /// All reaching defs of all reg units for a given MBB
  using MBBDefsInfo = std::vector<MBBRegUnitDefs>;

  /// All reaching defs of all reg units for all MBBs
  llvm::SmallVector<MBBDefsInfo, 4> AllReachingDefs;
};

/// This class provides the reaching def analysis.
class ReachingDefInfo {
private:
  luthier::LoopTraversal::TraversalOrder TraversedMBBOrder;
  const IPVectorCFG *IPVecCFG;
  unsigned NumRegUnits = 0;
  int ObjectIndexBegin = 0;
  /// Instruction that defined each register, relative to the beginning of the
  /// current basic block.  When a LiveRegsDefInfo is used to represent a
  /// live-out register, this value is relative to the end of the basic block,
  /// so it will be a negative number.
  using LiveRegsDefInfo = std::vector<int>;
  LiveRegsDefInfo LiveRegs;

  /// Keeps clearance information for all registers. Note that this
  /// is different from the usual definition notion of liveness. The CPU
  /// doesn't care whether or not we consider a register killed.
  using OutRegsInfoMap = llvm::SmallVector<LiveRegsDefInfo, 4>;
  OutRegsInfoMap MBBOutRegsInfos;

  /// Current instruction number.
  /// The first instruction in each basic block is 0.
  int CurInstr = -1;

  /// Maps instructions to their instruction Ids, relative to the beginning of
  /// their basic blocks.
  llvm::DenseMap<const llvm::MachineInstr *, int> InstIds;

  MBBReachingDefsInfo MBBReachingDefs;

  /// MBBFrameObjsReachingDefs[{i, j}] is a list of instruction indices
  /// (relative to begining of MBB i) that define frame index j in MBB i. This
  /// is used in answering reaching definition queries.
  using MBBFrameObjsReachingDefsInfo =
      llvm::DenseMap<std::pair<unsigned, int>, llvm::SmallVector<int>>;
  MBBFrameObjsReachingDefsInfo MBBFrameObjsReachingDefs;

  /// Default values are 'nothing happened a long time ago'.
  const int ReachingDefDefaultVal = -(1 << 21);

  using InstSet = llvm::SmallPtrSetImpl<const llvm::MachineInstr *>;
  using BlockSet = llvm::SmallPtrSetImpl<const llvm::MachineBasicBlock *>;

public:
  ReachingDefInfo();
  ReachingDefInfo(ReachingDefInfo &&) noexcept;
  ~ReachingDefInfo();
  /// Handle invalidation explicitly.
  bool invalidate(llvm::Module &Module, const llvm::PreservedAnalyses &PA,
                  llvm::ModuleAnalysisManager::Invalidator &);

  void run(llvm::Module &TargetModule, llvm::ModuleAnalysisManager &TargetMAM);

  void print(llvm::raw_ostream &OS);

  void releaseMemory();

  /// Initialize data structures.
  void init(const IPVectorCFG &IPVecCFG,
            const IPVectorRegLiveness &IPRegLiveness);

  /// Traverse the machine function, mapping definitions.
  void traverse(const IPVectorCFG &IPVecCFG,
                const IPVectorRegLiveness &IPRegLiveness);

  /// Provides the instruction id of the closest reaching def instruction of
  /// Reg that reaches MI, relative to the begining of MI's basic block.
  /// Note that Reg may represent a stack slot.
  int getReachingDef(const llvm::MachineInstr *MI, llvm::Register Reg) const;

  /// Return whether A and B use the same def of Reg.
  bool hasSameReachingDef(const llvm::MachineInstr *A,
                          const llvm::MachineInstr *B,
                          llvm::Register Reg) const;

  /// Return whether the reaching def for MI also is live out of its parent
  /// block.
  bool isReachingDefLiveOut(const llvm::MachineInstr *MI,
                            llvm::Register Reg) const;

  /// Return the local MI that produces the live out value for Reg, or
  /// nullptr for a non-live out or non-local def.
  const llvm::MachineInstr *
  getLocalLiveOutMIDef(const llvm::MachineBasicBlock *MBB,
                       llvm::Register Reg) const;

  /// If a single MachineInstr creates the reaching definition, then return it.
  /// Otherwise return null.
  const llvm::MachineInstr *getUniqueReachingMIDef(const llvm::MachineInstr *MI,
                                                   llvm::Register Reg) const;

  /// If a single MachineInstr creates the reaching definition, for MIs operand
  /// at Idx, then return it. Otherwise return null.
  const llvm::MachineInstr *getMIOperand(const llvm::MachineInstr *MI,
                                         unsigned Idx) const;

  /// If a single MachineInstr creates the reaching definition, for MIs MO,
  /// then return it. Otherwise return null.
  const llvm::MachineInstr *getMIOperand(const llvm::MachineInstr *MI,
                                         const llvm::MachineOperand &MO) const;

  /// Provide whether the register has been defined in the same basic block as,
  /// and before, MI.
  bool hasLocalDefBefore(const llvm::MachineInstr *MI,
                         llvm::Register Reg) const;

  /// Return whether the given register is used after MI, whether it's a local
  /// use or a live out.
  bool isRegUsedAfter(const llvm::MachineInstr *MI, llvm::Register Reg) const;

  /// Return whether the given register is defined after MI.
  bool isRegDefinedAfter(const llvm::MachineInstr *MI,
                         llvm::Register Reg) const;

  /// Provides the clearance - the number of instructions since the closest
  /// reaching def instuction of Reg that reaches MI.
  int getClearance(const llvm::MachineInstr *MI, llvm::Register Reg) const;

  /// Provides the uses, in the same block as MI, of register that MI defines.
  /// This does not consider live-outs.
  void getReachingLocalUses(const llvm::MachineInstr *MI, llvm::Register Reg,
                            InstSet &Uses) const;

  /// Search MBB for a definition of Reg and insert it into Defs. If no
  /// definition is found, recursively search the predecessor blocks for them.
  void getLiveOuts(const llvm::MachineBasicBlock *MBB, llvm::Register Reg,
                   InstSet &Defs, BlockSet &VisitedBBs) const;
  void getLiveOuts(const llvm::MachineBasicBlock *MBB, llvm::Register Reg,
                   InstSet &Defs) const;

  /// For the given block, collect the instructions that use the live-in
  /// value of the provided register. Return whether the value is still
  /// live on exit.
  bool getLiveInUses(const llvm::MachineBasicBlock *MBB, llvm::Register Reg,
                     InstSet &Uses) const;

  /// Collect the users of the value stored in Reg, which is defined
  /// by MI.
  void getGlobalUses(const llvm::MachineInstr *MI, llvm::Register Reg,
                     InstSet &Uses) const;

  /// Collect all possible definitions of the value stored in Reg, which is
  /// used by MI.
  void getGlobalReachingDefs(const llvm::MachineInstr *MI, llvm::Register Reg,
                             InstSet &Defs) const;

  /// Return whether From can be moved forwards to just before To.
  bool isSafeToMoveForwards(const llvm::MachineInstr *From,
                            const llvm::MachineInstr *To) const;

  /// Return whether From can be moved backwards to just after To.
  bool isSafeToMoveBackwards(const llvm::MachineInstr *From,
                             const llvm::MachineInstr *To) const;

  /// Assuming MI is dead, recursively search the incoming operands which are
  /// killed by MI and collect those that would become dead.
  void collectKilledOperands(const llvm::MachineInstr *MI, InstSet &Dead) const;

  /// Return whether removing this instruction will have no effect on the
  /// program, returning the redundant use-def chain.
  bool isSafeToRemove(const llvm::MachineInstr *MI, InstSet &ToRemove) const;

  /// Return whether removing this instruction will have no effect on the
  /// program, ignoring the possible effects on some instructions, returning
  /// the redundant use-def chain.
  bool isSafeToRemove(const llvm::MachineInstr *MI, InstSet &ToRemove,
                      InstSet &Ignore) const;

  /// Return whether a MachineInstr could be inserted at MI and safely define
  /// the given register without affecting the program.
  bool isSafeToDefRegAt(const llvm::MachineInstr *MI, llvm::Register Reg) const;

  /// Return whether a MachineInstr could be inserted at MI and safely define
  /// the given register without affecting the program, ignoring any effects
  /// on the provided instructions.
  bool isSafeToDefRegAt(const llvm::MachineInstr *MI, llvm::Register Reg,
                        InstSet &Ignore) const;

private:
  /// Set up LiveRegs by merging predecessor live-out values.
  void enterBasicBlock(
      const VectorMBB *MBB,
      llvm::ArrayRef<llvm::MachineBasicBlock::RegisterMaskPair> LiveIns);

  /// Update live-out values.
  void leaveBasicBlock(const VectorMBB *MBB);

  /// Process he given basic block.
  void processBasicBlock(
      const LoopTraversal::TraversedMBBInfo &TraversedMBB,
      llvm::ArrayRef<llvm::MachineBasicBlock::RegisterMaskPair> LiveIns);

  /// Process block that is part of a loop again.
  void reprocessBasicBlock(const VectorMBB *MBB);

  /// Update def-ages for registers defined by MI.
  /// Also break dependencies on partial defs and undef uses.
  void processDefs(const llvm::MachineInstr &MI, unsigned int VectorMBBIdx);

  /// Utility function for isSafeToMoveForwards/Backwards.
  template <typename Iterator>
  bool isSafeToMove(const llvm::MachineInstr *From,
                    const llvm::MachineInstr *To) const;

  /// Return whether removing this instruction will have no effect on the
  /// program, ignoring the possible effects on some instructions, returning
  /// the redundant use-def chain.
  bool isSafeToRemove(const llvm::MachineInstr *MI, InstSet &Visited,
                      InstSet &ToRemove, InstSet &Ignore) const;

  /// Provides the MI, from the given block, corresponding to the Id or a
  /// nullptr if the id does not refer to the block.
  const llvm::MachineInstr *getInstFromId(const llvm::MachineBasicBlock *MBB,
                                          int InstId) const;

  /// Provides the instruction of the closest reaching def instruction of
  /// Reg that reaches MI, relative to the begining of MI's basic block.
  /// Note that Reg may represent a stack slot.
  const llvm::MachineInstr *getReachingLocalMIDef(const llvm::MachineInstr *MI,
                                                  llvm::Register Reg) const;
};

class ReachingDefAnalysis
    : public llvm::AnalysisInfoMixin<ReachingDefAnalysis> {
  friend AnalysisInfoMixin<ReachingDefAnalysis>;
  static llvm::AnalysisKey Key;

public:
  using Result = ReachingDefInfo;

  Result run(llvm::Module &TargetModule,
             llvm::ModuleAnalysisManager &TargetMAM);
};

/// Printer pass for the \c ReachingDefInfo results.
class ReachingDefPrinterPass
    : public llvm::PassInfoMixin<ReachingDefPrinterPass> {
  llvm::raw_ostream &OS;

public:
  explicit ReachingDefPrinterPass(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(llvm::MachineFunction &MF,
                              llvm::MachineFunctionAnalysisManager &MFAM);

  static bool isRequired() { return true; }
};

} // namespace luthier

#endif
