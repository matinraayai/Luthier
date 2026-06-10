//===-- MIToIRMappingAnalysis.h ----------------------------------*-C++-*-===//
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
/// \file MIToIRMappingAnalysis.h
/// Describes the \c MIToIRMappingAnalysis, a \c llvm::MachineFunction analysis
/// that provides a bidirectional mapping between the \c llvm::MachineInstr s of
/// a lifted machine function and the \c llvm::Instruction s they were
/// translated into by the \c MIRToIRTranslator. The mapping is reconstructed by
/// matching the \c TargetMachineInstrMDNode (PC sections) metadata that the
/// translator copies, by pointer, from each source MI onto every IR instruction
/// it emits for that MI.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_MI_TO_IR_MAPPING_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_MI_TO_IR_MAPPING_ANALYSIS_H
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/PassManager.h>

namespace llvm {
class Instruction;
class MachineInstr;
} // namespace llvm

namespace luthier {

/// \brief Result of \c MIToIRMappingAnalysis: a bidirectional mapping between
/// the machine instructions of a single lifted machine function and the IR
/// instructions they were translated into.
///
/// A single \c llvm::MachineInstr can lower to one or more \c llvm::Instruction
/// s (e.g. an inline-asm call followed by \c llvm::ExtractElementInst s for its
/// outputs); every IR instruction maps back to exactly one source MI. IR
/// instructions synthesized by the translator that do not correspond to a
/// single source MI (e.g. EXEC-predicate checks or \c PHI nodes inserted for
/// vector blocks) are intentionally left unmapped.
class MIToIRMapping {
private:
  friend class MIToIRMappingAnalysis;

  /// Maps each source MI to the ordered (program-order) list of IR
  /// instructions it was translated into.
  llvm::DenseMap<const llvm::MachineInstr *,
                 llvm::SmallVector<llvm::Instruction *, 4>>
      MIToIRInsts;

  /// Inverse map: each translated IR instruction to its single source MI.
  llvm::DenseMap<const llvm::Instruction *, llvm::MachineInstr *> IRInstToMI;

  void addEntry(llvm::MachineInstr &MI, llvm::Instruction &I) {
    MIToIRInsts[&MI].push_back(&I);
    IRInstToMI.insert({&I, &MI});
  }

public:
  MIToIRMapping() = default;

  /// \returns the IR instructions \p MI was translated into, in program order,
  /// or an empty range if \p MI has no translated IR instructions.
  [[nodiscard]] llvm::ArrayRef<llvm::Instruction *>
  getIRInstructions(const llvm::MachineInstr &MI) const {
    auto It = MIToIRInsts.find(&MI);
    if (It == MIToIRInsts.end())
      return {};
    return It->second;
  }

  /// \returns the source \c llvm::MachineInstr that \p I was translated from,
  /// or \c nullptr if \p I is not a translation of a single MI.
  [[nodiscard]] llvm::MachineInstr *
  getMachineInstr(const llvm::Instruction &I) const {
    auto It = IRInstToMI.find(&I);
    return It == IRInstToMI.end() ? nullptr : It->second;
  }

  [[nodiscard]] bool contains(const llvm::MachineInstr &MI) const {
    return MIToIRInsts.contains(&MI);
  }

  [[nodiscard]] bool contains(const llvm::Instruction &I) const {
    return IRInstToMI.contains(&I);
  }

  /// \returns the number of mapped IR instructions.
  [[nodiscard]] unsigned size() const { return IRInstToMI.size(); }

  using mi_to_ir_const_iterator =
      llvm::DenseMap<const llvm::MachineInstr *,
                     llvm::SmallVector<llvm::Instruction *, 4>>::const_iterator;

  [[nodiscard]] mi_to_ir_const_iterator mi_to_ir_begin() const {
    return MIToIRInsts.begin();
  }

  [[nodiscard]] mi_to_ir_const_iterator mi_to_ir_end() const {
    return MIToIRInsts.end();
  }

  [[nodiscard]] llvm::iterator_range<mi_to_ir_const_iterator> mi_to_ir() const {
    return llvm::make_range(mi_to_ir_begin(), mi_to_ir_end());
  }

  using ir_to_mi_const_iterator =
      llvm::DenseMap<const llvm::Instruction *,
                     llvm::MachineInstr *>::const_iterator;

  [[nodiscard]] ir_to_mi_const_iterator ir_to_mi_begin() const {
    return IRInstToMI.begin();
  }

  [[nodiscard]] ir_to_mi_const_iterator ir_to_mi_end() const {
    return IRInstToMI.end();
  }

  [[nodiscard]] llvm::iterator_range<ir_to_mi_const_iterator> ir_to_mi() const {
    return llvm::make_range(ir_to_mi_begin(), ir_to_mi_end());
  }

  bool invalidate(llvm::MachineFunction &MF, const llvm::PreservedAnalyses &PA,
                  llvm::MachineFunctionAnalysisManager::Invalidator &Inv);
};

/// \brief A \c llvm::MachineFunction analysis that provides a bidirectional
/// mapping between a lifted machine function's \c llvm::MachineInstr s and the
/// \c llvm::Instruction s they were translated into.
class MIToIRMappingAnalysis
    : public llvm::AnalysisInfoMixin<MIToIRMappingAnalysis> {
  friend llvm::AnalysisInfoMixin<MIToIRMappingAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = MIToIRMapping;

  MIToIRMappingAnalysis() = default;

  Result run(llvm::MachineFunction &MF,
             llvm::MachineFunctionAnalysisManager &MFAM);
};

} // namespace luthier

#endif
