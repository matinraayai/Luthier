//===-- IPPredicatedLoopTraversal.h ------------------------------*- C++-*-===//
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
/// \file IPPredicatedLoopTraversal.h
/// Describes the \c IPPredicatedLoopTraversal class.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_TOOLING_IP_LOOP_TRAVERSAL_H
#define LUTHIER_TOOLING_IP_LOOP_TRAVERSAL_H
#include "luthier/Tooling/IPPredicatedCFG.h"
#include <llvm/ADT/SmallVector.h>

namespace luthier {

/// \brief The inter-procedural version of \c llvm::LoopTraversal re-implemented
/// to work with the \c IPPredicatedCFG
class IPPredicatedLoopTraversal {
private:
  struct PredicatedMBBInfo {
    /// Whether we have gotten to this block in primary processing yet.
    bool PrimaryCompleted = false;

    /// The number of predecessors for which primary processing has completed
    unsigned IncomingProcessed = 0;

    /// The value of `IncomingProcessed` at the start of primary processing
    unsigned PrimaryIncoming = 0;

    /// The number of predecessors for which all processing steps are done.
    unsigned IncomingCompleted = 0;

    PredicatedMBBInfo() = default;
  };
  using PredMBBInfoMap = llvm::SmallVector<PredicatedMBBInfo, 4>;
  /// Helps keep track if we processed this block and all its predecessors.
  PredMBBInfoMap MBBInfos;

public:
  struct TraversedPredMBBInfo {
    /// The basic block.
    const PredicatedMachineBasicBlock *MBB = nullptr;

    /// True if this is the first time we process the basic block.
    bool PrimaryPass = true;

    /// True if the block that is ready for its final round of processing.
    bool IsDone = true;

    TraversedPredMBBInfo(const PredicatedMachineBasicBlock *BB = nullptr,
                         bool Primary = true, bool Done = true)
        : MBB(BB), PrimaryPass(Primary), IsDone(Done) {}
  };
  IPPredicatedLoopTraversal() = default;

  /// Identifies basic blocks that are part of loops and should to be
  ///  visited twice and returns efficient traversal order for all the blocks.
  typedef llvm::SmallVector<TraversedPredMBBInfo, 4> IPTraversalOrder;
  IPTraversalOrder traverse(const IPPredicatedCFG &IPVecCFG);

private:
  /// \return \c true if the block is ready for its final round of processing.
  bool isBlockDone(const PredicatedMachineBasicBlock *MBB);
};

} // namespace luthier

#endif // LLVM_CODEGEN_LOOPTRAVERSAL_H
