//===-- StateValueArraySpecs.h ----------------------------------*- C++ -*-===//
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
/// Defines the \c StateValueArraySpecs class describing the SVA lane layout
/// used across all functions of the instrumentation module, and the
/// \c StateValueArraySpecsAnalysis that computes it by walking the IR of the
/// instrumentation module for uses of \c luthier::readSVA .
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_STATE_VALUE_ARRAY_SPECS_H
#define LUTHIER_TOOL_CODE_GEN_STATE_VALUE_ARRAY_SPECS_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/IR/PassManager.h>

namespace llvm {
class GCNSubtarget;
} // namespace llvm

namespace luthier {

class StateValueArraySpecsAnalysis;

class StateValueArraySpecs {
  static constexpr uint8_t StackPointerRegSpillLane{0};

  static constexpr uint8_t FramePointerRegSSpillLane{1};

  static constexpr uint8_t StackPointerStoreLane{2};

  std::optional<uint8_t> BufferRsrcSpillLane{std::nullopt};

  std::optional<uint8_t> ScratchSpillLane{std::nullopt};

  llvm::DenseMap<ScalarValueArgument, uint8_t> ScalarArguments{};

  friend class StateValueArraySpecsAnalysis;

public:
  StateValueArraySpecs() = default;

  [[nodiscard]] constexpr uint8_t getStackPointerRegSpillLane() const {
    return StackPointerRegSpillLane;
  }

  [[nodiscard]] constexpr uint8_t getFramePointerRegSpillLane() const {
    return FramePointerRegSSpillLane;
  }

  [[nodiscard]] constexpr uint8_t getStackPointerStoreLane() const {
    return StackPointerStoreLane;
  }

  [[nodiscard]] std::optional<uint8_t> getRsrcBufferSpillLane() const {
    return BufferRsrcSpillLane;
  }

  [[nodiscard]] std::optional<uint8_t> getScratchSpillLane() const {
    return ScratchSpillLane;
  }

  using const_argument_lane_iterator =
      decltype(ScalarArguments)::const_iterator;

  [[nodiscard]] const_argument_lane_iterator argument_lane_begin() const {
    return ScalarArguments.begin();
  }

  [[nodiscard]] const_argument_lane_iterator argument_lane_end() const {
    return ScalarArguments.end();
  }

  [[nodiscard]] unsigned argument_lane_size() const {
    return ScalarArguments.size();
  }

  [[nodiscard]] bool argument_lane_contains(ScalarValueArgument SA) const {
    return ScalarArguments.contains(SA);
  }

  [[nodiscard]] const_argument_lane_iterator
  findArgumentLane(ScalarValueArgument SA) const {
    return ScalarArguments.find(SA);
  }

  static unsigned getArgumentLaneSize(ScalarValueArgument SA);

  /// Return up to \p NumLanes lowest-numbered SVA lanes that are not
  /// claimed by any of the fixed kernel-prolog slots (lanes 0-2 plus the
  /// FS / buffer-rsrc region) and not allocated to a scalar-value
  /// argument. Lanes range over <tt>0 .. WaveSize-1</tt>. Returns fewer
  /// than \p NumLanes if the SVA is saturated.
  ///
  /// Used by \c TargetModulePatcherPass's branch-relaxation fallback: when
  /// the per-MBB live-in set has no two dead SGPRs available at the
  /// branch, we spill two app SGPRs into the lowest free SVA lanes around
  /// the relaxed jump.
  [[nodiscard]] llvm::SmallVector<uint8_t, 4>
  findLowestFreeLanes(unsigned NumLanes, unsigned WaveSize) const;

  bool invalidate(Prototype &, const llvm::PreservedAnalyses &PA,
                  PrototypeAnalysisManager::Invalidator &);
};

/// \brief Prototype-level analysis producing the \c StateValueArraySpecs
/// for the instrumentation module.
///
/// \details The analysis walks every IR \c Function of the instrumentation
/// module, finds \c luthier::readSVA call sites or inline assembly place
/// holders, aggregates the requested \c ScalarValueArguments, and lays out the
/// SVA lanes accordingly. When the target module's initial entry point is not a
/// kernel, every SA is treated as used — the target is being instrumented from
/// within an already-instrumented kernel, and the SVA has already been set up
/// to preserve every SA.
class StateValueArraySpecsAnalysis
    : public llvm::AnalysisInfoMixin<StateValueArraySpecsAnalysis> {
  friend llvm::AnalysisInfoMixin<StateValueArraySpecsAnalysis>;

  static llvm::AnalysisKey Key;

public:
  StateValueArraySpecsAnalysis() = default;

  using Result = StateValueArraySpecs;

  Result run(Prototype &IP, PrototypeAnalysisManager &IPAM);
};

} // namespace luthier

#endif
