//===-- StateValueArraySpecs.cpp ------------------------------------------===//
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
///
/// \file StateValueArraySpecs.cpp
/// Implements functions used to query the state value array specs.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"

#include <AMDGPU.h>
#include <AMDGPUTargetMachine.h>
#include <GCNSubtarget.h>
#include <MCTargetDesc/AMDGPUMCTargetDesc.h>

namespace luthier {

llvm::SmallVector<uint8_t, 4>
StateValueArraySpecs::findLowestFreeLanes(unsigned NumLanes,
                                          unsigned WaveSize) const {
  // Lane occupancy:
  //   0       — StackPointerRegSpillLane (SGPR0 of PRIVATE_SEGMENT_BUFFER)
  //   1       — FramePointerRegSSpillLane (SGPR1)
  //   2       — StackPointerStoreLane (instrumentation SGPR32)
  //   3..N-1  — BufferRsrcOrScratchSpillLane region (FS = 2 lanes, buffer
  //             rsrc = 4 lanes, architected-FS = 0 lanes)
  //   …       — Each ScalarArguments[SA] entry holds 1, 2, or 4 contiguous
  //             lanes starting at the stored base.
  llvm::BitVector Occupied(WaveSize, false);

  auto markRange = [&](unsigned Base, unsigned Count) {
    for (unsigned i = 0; i < Count; ++i) {
      unsigned L = Base + i;
      if (L < WaveSize)
        Occupied.set(L);
    }
  };

  markRange(StackPointerRegSpillLane, 1);
  markRange(FramePointerRegSSpillLane, 1);
  markRange(StackPointerStoreLane, 1);

  if (BufferRsrcOrScratchSpillLane != llvm::MCRegister::NoRegister) {
    unsigned BufferSize =
        (BufferRsrcOrScratchSpillLane == llvm::AMDGPU::FLAT_SCR) ? 2 : 4;
    markRange(/*Base=*/3, BufferSize);
  }
  for (const auto &[SA, Base] : ScalarArguments)
    markRange(Base, getArgumentLaneSize(SA));

  llvm::SmallVector<uint8_t, 4> Out;
  for (unsigned L = 0; L < WaveSize && Out.size() < NumLanes; ++L)
    if (!Occupied.test(L))
      Out.push_back(static_cast<uint8_t>(L));
  return Out;
}

unsigned StateValueArraySpecs::getArgumentLaneSize(ScalarValueArgument SA) {
  switch (SA) {
  case WAVEFRONT_PRIVATE_SEGMENT_BUFFER:
    return ScalarValueArgumentInfo<WAVEFRONT_PRIVATE_SEGMENT_BUFFER>::NumLanes;
  case KERNEL_ARG_PTR:
    return ScalarValueArgumentInfo<KERNEL_ARG_PTR>::NumLanes;
  case DISPATCH_ID:
    return ScalarValueArgumentInfo<DISPATCH_ID>::NumLanes;
  case FLAT_SCRATCH:
    return ScalarValueArgumentInfo<FLAT_SCRATCH>::NumLanes;
  case PRIVATE_SEGMENT_WAVE_BYTE_OFFSET:
    return ScalarValueArgumentInfo<PRIVATE_SEGMENT_WAVE_BYTE_OFFSET>::NumLanes;
  case DISPATCH_PTR:
    return ScalarValueArgumentInfo<DISPATCH_PTR>::NumLanes;
  case QUEUE_PTR:
    return ScalarValueArgumentInfo<QUEUE_PTR>::NumLanes;
  case WORK_ITEM_PRIVATE_SEGMENT_SIZE:
    return ScalarValueArgumentInfo<WORK_ITEM_PRIVATE_SEGMENT_SIZE>::NumLanes;
  case USER_ARG_PTR:
    return ScalarValueArgumentInfo<USER_ARG_PTR>::NumLanes;
  case IMPLICIT_ARG_OFFSET:
    return ScalarValueArgumentInfo<IMPLICIT_ARG_OFFSET>::NumLanes;
  default:
    llvm_unreachable("Invalid scalar value argument");
  }
}

std::unique_ptr<StateValueArraySpecs>
StateValueArraySpecs::getSVASpecs(const llvm::Module &M,
                                  const llvm::TargetMachine &TM) {
  std::unique_ptr<StateValueArraySpecs> Out{new StateValueArraySpecs()};
  uint8_t NextLane = StackPointerStoreLane;

  if (!M.empty()) {
    const auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(*M.begin());
    bool IsArchitectedFS = ST.hasArchitectedFlatScratch();
    bool HasFS = ST.enableFlatScratch();
#ifdef _DEBUG
    /// Check if all functions have the same scratch accessing instructions
    /// enabled in their subtargets. This is an assertion because this operation
    /// is more of a sanity check than something that can happen in Luthier
    for (const llvm::Function &F : M) {
      const auto &FuncST = TM.getSubtarget<llvm::GCNSubtarget>(*M.begin());
      bool FuncHasArchitectedFS = FuncST.hasArchitectedFlatScratch();
      bool FuncHasFS = FuncST.enableFlatScratch();
      assert(FuncHasArchitectedFS == IsArchitectedFS && FuncHasFS == HasFS &&
             "Functions has different scratch access requirements");
    }
#endif

    if (HasFS && !IsArchitectedFS) {
      Out->BufferRsrcOrScratchSpillLane = llvm::AMDGPU::FLAT_SCR;
      NextLane += 2;
    } else if (!IsArchitectedFS) {
      Out->BufferRsrcOrScratchSpillLane = llvm::AMDGPU::PRIVATE_RSRC_REG;
      NextLane += 4;
    }
    /// Next determine the scalar values used via the named MD in the module
    using SVArgUnderlyingType = std::underlying_type_t<ScalarValueArgument>;

    auto GetSingleArgIfPresent = [&]<SVArgUnderlyingType SVArg>() {
      if (constexpr auto CastedSVArg = static_cast<ScalarValueArgument>(SVArg);
          M.getNamedMetadata(ScalarValueArgumentInfo<CastedSVArg>::NamedMD)) {
        Out->ScalarArguments.insert({CastedSVArg, NextLane});
        NextLane += ScalarValueArgumentInfo<CastedSVArg>::NumLanes;
      }
    };

    // std::make_integer_sequence<T, N> produces [0, N-1]; the inclusive
    // SCALAR_VALUE_ARGUMENT_LAST sentinel is the highest valid enumerator,
    // so the count is LAST+1 to cover every SA in [FIRST, LAST].
    constexpr auto SVArgSequence =
        std::make_integer_sequence<SVArgUnderlyingType,
                                   SCALAR_VALUE_ARGUMENT_LAST + 1>{};

    [&]<SVArgUnderlyingType... SVArgs>(
        std::integer_sequence<SVArgUnderlyingType, SVArgs...>) {
      (GetSingleArgIfPresent.operator()<SVArgs>(), ...);
    }(SVArgSequence);
    return std::move(Out);
  }
  // Empty IModule: no SVA layout to derive. Callers
  // (TargetModulePatcherPass etc.) treat nullptr as "spec unavailable"
  // and either skip SVA-dependent work or surface an error.
  return nullptr;
}

std::unique_ptr<StateValueArraySpecs> StateValueArraySpecs::setModuleSVASpec(
    llvm::Module &M, const llvm::TargetMachine &TM,
    const llvm::SmallDenseSet<ScalarValueArgument> &RequestedSVArgs) {

  using SVArgUnderlyingType = std::underlying_type_t<ScalarValueArgument>;

  auto InsertSingleArgIfPresent = [&]<SVArgUnderlyingType SVArg>() {
    if (constexpr auto CastedSVArg = static_cast<ScalarValueArgument>(SVArg);
        RequestedSVArgs.contains(CastedSVArg)) {
      (void)M.getOrInsertNamedMetadata(
          ScalarValueArgumentInfo<CastedSVArg>::NamedMD);
    }
  };

  constexpr auto SVArgSequence =
      std::make_integer_sequence<SVArgUnderlyingType,
                                 SCALAR_VALUE_ARGUMENT_LAST + 1>{};

  [&]<SVArgUnderlyingType... SVArgs>(
      std::integer_sequence<SVArgUnderlyingType, SVArgs...>) {
    (InsertSingleArgIfPresent.operator()<SVArgs>(), ...);
  }(SVArgSequence);

  return getSVASpecs(M, TM);
}
} // namespace luthier