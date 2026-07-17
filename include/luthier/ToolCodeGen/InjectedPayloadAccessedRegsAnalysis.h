//===-- InjectedPayloadAccessedRegsAnalysis.h -------------------*- C++ -*-===//
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
/// Describes \c InjectedPayloadAccessedRegsAnalysis, a function-level analysis
/// that, for each injected-payload \c llvm::Function in the instrumentation
/// module, returns the set of physical registers the payload reads from and
/// writes at their injection site in the target application.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_ACCESSED_REGS_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_ACCESSED_REGS_ANALYSIS_H
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCRegister.h>

namespace llvm {
class Function;
class raw_ostream;
} // namespace llvm

namespace luthier {

class InjectedPayloadAccessedRegsAnalysis;

/// \brief Per-payload phys-reg read/write sets.
class InjectedPayloadAccessedRegs {
public:
  using PhysRegSetT = llvm::DenseSet<llvm::MCRegister>;
  using iterator = PhysRegSetT::const_iterator;

private:
  friend class InjectedPayloadAccessedRegsAnalysis;

  PhysRegSetT Reads;
  PhysRegSetT Writes;

public:
  /// == Reads ================================================================
  iterator reads_begin() const { return Reads.begin(); }
  iterator reads_end() const { return Reads.end(); }
  llvm::iterator_range<iterator> reads() const {
    return {reads_begin(), reads_end()};
  }
  size_t reads_size() const { return Reads.size(); }
  bool reads_empty() const { return Reads.empty(); }
  bool reads_contains(llvm::MCRegister R) const { return Reads.contains(R); }

  /// == Writes ===============================================================
  iterator writes_begin() const { return Writes.begin(); }
  iterator writes_end() const { return Writes.end(); }
  llvm::iterator_range<iterator> writes() const {
    return {writes_begin(), writes_end()};
  }
  size_t writes_size() const { return Writes.size(); }
  bool writes_empty() const { return Writes.empty(); }
  bool writes_contains(llvm::MCRegister R) const { return Writes.contains(R); }

  /// Invalidated whenever the enclosing \c Function's IR changes. The
  /// result is derived by walking the function's call sites (both un-lowered
  /// Luthier-intrinsic calls and inline-asm placeholder calls emitted by
  /// \c ProcessIntrinsicsAtIRLevelPass), so any pass that fails to preserve
  /// this analysis must invalidate the cached result.
  bool invalidate(llvm::Function &F, const llvm::PreservedAnalyses &PA,
                  llvm::FunctionAnalysisManager::Invalidator &Inv);
};

/// \brief Function-level analysis: per-payload accessed phys-reg sets.
///
/// Returns an empty \c InjectedPayloadAccessedRegs (no reads, no writes) for
/// any function that is not marked with \c InjectedPayloadAttribute.
class InjectedPayloadAccessedRegsAnalysis
    : public llvm::AnalysisInfoMixin<InjectedPayloadAccessedRegsAnalysis> {
  friend llvm::AnalysisInfoMixin<InjectedPayloadAccessedRegsAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = InjectedPayloadAccessedRegs;

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

/// \brief Printer pass for \c InjectedPayloadAccessedRegsAnalysis used for
/// testing.
class InjectedPayloadAccessedRegsPrinterPass
    : public llvm::PassInfoMixin<InjectedPayloadAccessedRegsPrinterPass> {
  llvm::raw_ostream &OS;

public:
  explicit InjectedPayloadAccessedRegsPrinterPass(llvm::raw_ostream &OS)
      : OS(OS) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

} // namespace luthier

#endif
