//===-- IntrinsicIRLoweringInfoAnalysis.h -----------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
///
/// \file
/// Describes the \c IntrinsicIRLoweringInfoAnalysis which provides the
/// \c IRLoweringInfo associated with each Luthier intrinsic use inside the
/// instrumentation module.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_IR_LOWERING_INFO_ANALYSIS_H
#define LUTHIER_INTRINSIC_IR_LOWERING_INFO_ANALYSIS_H
#include <llvm/IR/PassManager.h>
#include <luthier/Instrumentation/IntrinsicProcessor.h>

namespace luthier {

/// \brief Provides the \c IntrinsicIRLoweringInfo of each Luthier intrinsic use
/// inside each function of the instrumentation module
/// \details After the \c ProcessIntrinsicsAtIRLevelPass runs the IR processor
/// on a Luthier intrinsic use, it stores its \c IntrinsicIRLoweringInfo in
/// this analysis pass so that the \c IntrinsicMIRLoweringPass can later
/// retrieve it to lower the intrinsic use to a sequence of Machine
/// instructions in SSA form. \n
/// After the intrinsic use is processed by \c ProcessIntrinsicsAtIRLevelPass
/// it is replaced with a call to a dummy inline assembly instruction. The
/// dummy assembly instruction contains a unique unsigned integer identifier
/// used to identify the intrinsic use after instruction selection at the MIR
/// level. The same index can be used with this analysis to query the IR
/// lowering info of the desired intrinsic use.
class IntrinsicIRLoweringInfoAnalysis
    : public llvm::AnalysisInfoMixin<IntrinsicIRLoweringInfoAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<IntrinsicIRLoweringInfoAnalysis>;

  static llvm::AnalysisKey Key;

public:
  class Result {
  private:
    friend class IntrinsicIRLoweringInfoAnalysis;

    /// Where the IR lowering info for each intrinsic use inside the
    /// instrumentation function is stored. The index of each entry in this
    /// vector corresponds to its inline assembly index used inside the IR/MIR
    std::vector<IntrinsicIRLoweringInfo> LoweringInfoVector{};

    /// Constructor; Only usable by the \c IntrinsicIRLoweringInfoAnalysis
    Result() = default;

  public:
    /// \returns Const iterator to the beginning of the IR lowering info
    auto begin() const { return llvm::enumerate(LoweringInfoVector).begin(); }

    /// \returns Const iterator to the end of the IR lowering info
    auto end() const { return llvm::enumerate(LoweringInfoVector).end(); }

    /// \returns \c true if there are no intrinsic lowering info is stored,
    /// \c false otherwise
    bool empty() const { return LoweringInfoVector.empty(); }

    /// \returns the number of IR lowering info records stored
    size_t size() const { return LoweringInfoVector.size(); }

    /// \returns the IR lowering info associated with the intrinsic use with
    /// index \p IntrinsicIdx
    /// \warning Does not do any bounds checking
    const IntrinsicIRLoweringInfo &at(unsigned int IntrinsicIdx) const {
      return LoweringInfoVector.at(IntrinsicIdx);
    }

    /// Creates and stores a new entry index for the passed \p LoweringInfo
    /// \returns the index associated with the newly added IR lowering info
    unsigned int
    addIntrinsicIRLoweringInfo(IntrinsicIRLoweringInfo &&LoweringInfo) {
      (void)LoweringInfoVector.emplace_back(LoweringInfo);
      return LoweringInfoVector.size() - 1;
    }

    /// Prevents the invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Function &, const llvm::PreservedAnalyses &,
               llvm::FunctionAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// Default constructor
  IntrinsicIRLoweringInfoAnalysis() = default;


  Result run(llvm::Function &, llvm::FunctionAnalysisManager &) { return {}; }
};

} // namespace luthier

#endif