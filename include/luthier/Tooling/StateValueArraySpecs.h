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
#include <SIRegisterInfo.h>

namespace llvm {
class GCNSubtarget;
}

namespace luthier {

class StateValueArraySpecs {
  static constexpr uint8_t StackPointerRegSpillLane{0};

  static constexpr uint8_t FramePointerRegSSpillLane{1};

  static constexpr uint8_t StackPointerStoreLane{2};

  llvm::MCRegister BufferRsrcOrScratchSpillLane = llvm::MCRegister::NoRegister;

  llvm::DenseMap<ScalarValueArgument, uint8_t> ScalarArguments{};

  StateValueArraySpecs() = default;

public:
  [[nodiscard]] constexpr uint8_t getStackPointerRegSpillLane() const {
    return StackPointerRegSpillLane;
  }

  [[nodiscard]] constexpr uint8_t getFramePointerRegSpillLane() const {
    return FramePointerRegSSpillLane;
  }

  [[nodiscard]] constexpr uint8_t getStackPointerStoreLane() const {
    return StackPointerStoreLane;
  }

  [[nodiscard]] std::optional<uint8_t>
  getFrameRsrcOrScratchStoreLaneIfExists() const {
    return BufferRsrcOrScratchSpillLane != llvm::MCRegister::NoRegister
               ? std::optional{3}
               : std::nullopt;
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
  getSVASpecs(const llvm::Module &M, const llvm::TargetMachine &TM);

  static std::unique_ptr<StateValueArraySpecs> setModuleSVASpec(
      llvm::Module &M, const llvm::TargetMachine &TM,
      const llvm::SmallDenseSet<ScalarValueArgument> &RequestedSVArgs);
};

} // namespace luthier

#endif