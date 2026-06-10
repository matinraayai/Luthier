//===-- RegValueMapAnalysis.h -------------------------------------*-C++-*-===//
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
/// \file RegValueMapAnalysis.h
/// Describes the \c RegValueMapAnalysis, a \c llvm::Function analysis that
/// reconstructs, for every basic block of a lifted function, the mapping from
/// physical-register slices to the IR values holding their content at the
/// block's exit. The map is rebuilt purely from the \c luthier.reg
/// per-instruction metadata and the function-level \c luthier.entry_reg_map
/// emitted by the \c TraceIRTranslator, so it is cheap to recompute, follows
/// normal pass-manager invalidation, and survives serialization round trips.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_REG_VALUE_MAP_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_REG_VALUE_MAP_ANALYSIS_H
#include "luthier/ToolCodeGen/RegValueMetadata.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/PassManager.h>

namespace llvm {
class BasicBlock;
class Function;
class Value;
class raw_ostream;
} // namespace llvm

namespace luthier {

/// \brief Result of \c RegValueMapAnalysis: per-basic-block exit values of
/// physical-register slices plus the function-entry register seeds.
class RegValueMap {
  friend class RegValueMapAnalysis;

  /// Hashable form of \c RegValueDesc: (base register id, half-word offset,
  /// number of halves)
  using DescKey = std::tuple<unsigned, unsigned, unsigned>;

  static DescKey getKey(const RegValueDesc &D) {
    return {D.BaseReg.id(), D.HalfWordOffset, D.NumHalves};
  }

  using BlockMap = llvm::DenseMap<DescKey, llvm::Value *>;

  /// Last tagged definition of each register slice per block
  llvm::DenseMap<const llvm::BasicBlock *, BlockMap> ExitValues;

  /// Register seeds at function entry (arguments / constants tagged in the
  /// luthier.entry_reg_map function metadata)
  BlockMap EntrySeeds;

public:
  RegValueMap() = default;

  /// \returns the IR value holding register slice \p D at the end of \p BB,
  /// or \c nullptr if \p BB does not (re)define it
  [[nodiscard]] llvm::Value *getExitValue(const llvm::BasicBlock &BB,
                                          const RegValueDesc &D) const {
    auto It = ExitValues.find(&BB);
    if (It == ExitValues.end())
      return nullptr;
    return It->second.lookup(getKey(D));
  }

  /// \returns the seed value of register slice \p D at function entry, or
  /// \c nullptr if it is not seeded
  [[nodiscard]] llvm::Value *getEntrySeed(const RegValueDesc &D) const {
    return EntrySeeds.lookup(getKey(D));
  }

  /// \returns the number of register slices (re)defined in \p BB
  [[nodiscard]] unsigned getNumExitValues(const llvm::BasicBlock &BB) const {
    auto It = ExitValues.find(&BB);
    return It == ExitValues.end() ? 0 : It->second.size();
  }

  [[nodiscard]] unsigned getNumEntrySeeds() const { return EntrySeeds.size(); }

  void print(llvm::raw_ostream &OS, const llvm::Function &F) const;

  bool invalidate(llvm::Function &F, const llvm::PreservedAnalyses &PA,
                  llvm::FunctionAnalysisManager::Invalidator &);
};

/// \brief \c llvm::Function analysis reconstructing per-block register-slice
/// exit values from the translator's \c luthier.reg metadata
class RegValueMapAnalysis
    : public llvm::AnalysisInfoMixin<RegValueMapAnalysis> {
  friend llvm::AnalysisInfoMixin<RegValueMapAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = RegValueMap;

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

/// Prints the \c RegValueMapAnalysis result of every function in the module
class RegValueMapPrinter : public llvm::PassInfoMixin<RegValueMapPrinter> {
  llvm::raw_ostream &OS;

public:
  explicit RegValueMapPrinter(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace luthier

#endif
