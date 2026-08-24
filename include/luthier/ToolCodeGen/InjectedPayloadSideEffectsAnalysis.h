//===-- InjectedPayloadSideEffectsAnalysis.h -------------------*- C++ -*-===//
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
/// Describes \c InjectedPayloadSideEffectsAnalysis, a function-level analysis
/// that, for each injected-payload \c llvm::Function in the instrumentation
/// module, returns the set of physical registers the payload reads from and
/// writes at their injection site in the target application, as well as
/// the SVA lanes and implicit args it touches upon.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_SIDE_EFFECTS_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_SIDE_EFFECTS_ANALYSIS_H
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCRegister.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>

namespace llvm {
class Function;
class raw_ostream;
} // namespace llvm

namespace luthier {

class InjectedPayloadSideEffectsAnalysis;

/// \brief Per-payload phys-reg read/write sets.
class InjectedPayloadSideEffects {
public:
  using PhysRegSetT = llvm::DenseSet<llvm::MCRegister>;
  using SVASetT = llvm::SmallDenseSet<ScalarValueArgument, 4>;
  using ImplicitArgSetT = llvm::SmallDenseSet<llvm::StringRef, 32>;
  using iterator = PhysRegSetT::const_iterator;
  using sva_iterator = SVASetT::const_iterator;
  using implicit_arg_iterator = ImplicitArgSetT::const_iterator;

private:
  friend class InjectedPayloadSideEffectsAnalysis;

  PhysRegSetT Reads;
  PhysRegSetT Writes;
  SVASetT SVAs;
  ImplicitArgSetT ImplicitArgs;

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

  /// == Scalar-value arguments ===============================================
  sva_iterator svas_begin() const { return SVAs.begin(); }
  sva_iterator svas_end() const { return SVAs.end(); }
  llvm::iterator_range<sva_iterator> svas() const {
    return {svas_begin(), svas_end()};
  }
  size_t svas_size() const { return SVAs.size(); }
  bool svas_empty() const { return SVAs.empty(); }
  bool svas_contains(ScalarValueArgument SA) const { return SVAs.contains(SA); }

  /// == Implicit arguments ===================================================
  /// AMDGPU \c amdgpu-no-<foo> attr names for each implicit-arg-buffer
  /// entry the payload transitively uses.
  implicit_arg_iterator implicit_args_begin() const {
    return ImplicitArgs.begin();
  }
  implicit_arg_iterator implicit_args_end() const { return ImplicitArgs.end(); }
  llvm::iterator_range<implicit_arg_iterator> implicit_args() const {
    return {implicit_args_begin(), implicit_args_end()};
  }
  size_t implicit_args_size() const { return ImplicitArgs.size(); }
  bool implicit_args_empty() const { return ImplicitArgs.empty(); }
  bool implicit_args_contains(llvm::StringRef A) const {
    return ImplicitArgs.contains(A);
  }

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
/// Returns an empty \c InjectedPayloadSideEffects (no reads, no writes) for
/// any function that is not marked with \c InjectedPayloadAttribute.
class InjectedPayloadSideEffectsAnalysis
    : public llvm::AnalysisInfoMixin<InjectedPayloadSideEffectsAnalysis> {
  friend llvm::AnalysisInfoMixin<InjectedPayloadSideEffectsAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = InjectedPayloadSideEffects;

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  /// The \c amdgpu-no-<foo> attr names the analysis inspects on
  /// payload functions.
  static llvm::ArrayRef<llvm::StringRef> getAllImplicitArgOptOutAttrs();
};

/// \brief Printer pass for \c InjectedPayloadSideEffectsAnalysis used for
/// testing.
class InjectedPayloadSideEffectsPrinterPass
    : public llvm::PassInfoMixin<InjectedPayloadSideEffectsPrinterPass> {
  llvm::raw_ostream &OS;

public:
  explicit InjectedPayloadSideEffectsPrinterPass(llvm::raw_ostream &OS)
      : OS(OS) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

} // namespace luthier

#endif
