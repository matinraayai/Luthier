//===-- StateValueArraySpecs.h ----------------------------------*- C++ -*-===//
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
///
/// \file StateValueArraySpecs.h
/// Defines the \c StateValueArraySpecs class used to set up and read named
/// metadata used in a \c llvm::Module to express the specifications of the
/// state value array used across all functions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_STATE_VALUE_ARRAY_SPECS_H
#define LUTHIER_TOOLING_STATE_VALUE_ARRAY_SPECS_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/CodeGen/Register.h>

namespace llvm {
class GCNSubtarget;
}

namespace luthier {

class StateValueArraySpecs {
  llvm::DenseMap<llvm::Register, uint8_t> FrameSpillLanes{};

  llvm::DenseMap<llvm::Register, uint8_t> InstrumentationFrameLanes{};

  llvm::DenseMap<ScalarValueArgument, uint8_t> ScalarArguments{};

  StateValueArraySpecs() = default;

public:
  using const_frame_spill_iterator = decltype(FrameSpillLanes)::const_iterator;

  [[nodiscard]] const_frame_spill_iterator frame_spill_begin() const {
    return FrameSpillLanes.begin();
  }

  [[nodiscard]] const_frame_spill_iterator frame_spill_end() const {
    return FrameSpillLanes.end();
  }

  [[nodiscard]] unsigned frame_spill_size() const {
    return FrameSpillLanes.size();
  }

  [[nodiscard]] bool frame_spill_contains(llvm::Register Reg) const {
    return FrameSpillLanes.contains(Reg);
  }

  [[nodiscard]] const_frame_spill_iterator
  findFrameSpillLane(llvm::Register Reg) const {
    return FrameSpillLanes.find(Reg);
  }

  using const_instrumentation_frame_iterator =
      decltype(InstrumentationFrameLanes)::const_iterator;

  [[nodiscard]] const_instrumentation_frame_iterator
  instrumentation_frame_begin() const {
    return InstrumentationFrameLanes.begin();
  }

  [[nodiscard]] const_instrumentation_frame_iterator
  instrumentation_frame_end() const {
    return InstrumentationFrameLanes.end();
  }

  [[nodiscard]] unsigned instrumentation_frame_size() const {
    return InstrumentationFrameLanes.size();
  }

  [[nodiscard]] bool instrumentation_frame_contains(llvm::Register Reg) const {
    return InstrumentationFrameLanes.contains(Reg);
  }

  [[nodiscard]] const_instrumentation_frame_iterator
  findInstrumentationFrameLane(llvm::Register Reg) const {
    return InstrumentationFrameLanes.find(Reg);
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

  static std::unique_ptr<StateValueArraySpecs>
  getSVASpecs(const llvm::Module &M, const llvm::GCNSubtarget &STI);

  static std::unique_ptr<StateValueArraySpecs> setModuleSVASpec(
      llvm::Module &M, const llvm::GCNSubtarget &STI,
      const llvm::SmallDenseSet<ScalarValueArgument> &RequestedSVArgs);
};

} // namespace luthier

#endif