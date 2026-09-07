//===-- ScalarValueArgument.h -----------------------------------*- C++ -*-===//
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
/// Defines \c luthier::ScalarValueArgument , the set of wave-uniform values
/// Luthier's kernel prologue can preserve in the state value array on behalf of
/// instrumentation code, and the per-argument lane counts that describe the
/// SVA's layout.
///
/// This header is deliberately dependency-free: device code includes it
/// through \c luthier/Intrinsic/Intrinsics.h to name an argument in a
/// \c luthier::readSVA call, and nothing in LLVM's headers survives being
/// compiled for \c amdgcn in HIP device mode.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_SCALAR_VALUE_ARGUMENT_H
#define LUTHIER_INTRINSIC_SCALAR_VALUE_ARGUMENT_H
#include <cstdint>

namespace luthier {

/// \brief A set of scalar value arguments Luthier's intrinsic lowering
/// mechanism can ensure access to
/// \details These values are only available to the kernel as "arguments"
/// as they come preloaded in SGPRs on the kernel's start. These values can
/// be overwritten the moment they are unused by the original kernel; Which
/// is why to ensure access to these values in instrumentation routines,
/// Luthier must emit a prologue on top of the kernel's original code to
/// save these values in the state value array VGPR to preserve them
enum ScalarValueArgument : uint8_t {
  /// Wavefront's private segment buffer; Only applies to targets with
  /// absolute flat scratch or offset flat scratch
  WAVEFRONT_PRIVATE_SEGMENT_BUFFER = 0,
  /// Marks the first defined scalar value argument
  SCALAR_VALUE_ARGUMENT_FIRST = WAVEFRONT_PRIVATE_SEGMENT_BUFFER,
  /// 64-bit Dispatch ID of the kernel
  DISPATCH_ID = 1,
  /// 64-bit flat scratch base address of the wavefront
  FLAT_SCRATCH = 2,
  /// 64-bit address of the dispatch packet of the kernel being executed
  /// TODO: If the original kernel wants the dispatch pointer, we need to
  /// make it point to the **"original"** packet not the instrumented one
  DISPATCH_PTR = 3,
  /// 64-bit address of the HSA queue used to launch the kernel
  QUEUE_PTR = 4,
  /// Number of bytes at the very bottom of each work-item's private segment
  /// that Luthier reserves for its own use. The instrumentation's stack starts
  /// at offset zero of the wavefront's private segment and grows up to this
  /// value; the application's own frame is displaced upward by exactly this
  /// much (see \c RebaseAppScratchAccessesPass ). Unlike every other entry
  /// here this is not a hardware-preloaded value — the target module patcher
  /// computes it and writes it into the SVA lane as an immediate.
  WORK_ITEM_INSTRUMENTATION_PRIVATE_SEGMENT_SIZE = 5,
  /// 64-bit address of the instrumentation implicit argument buffer
  IMPLICIT_ARG_BUFFER = 6,
  /// 32-bit X component of the workgroup ID (preloaded system SGPR)
  WORKGROUP_ID_X = 7,
  /// 32-bit Y component of the workgroup ID (preloaded system SGPR)
  WORKGROUP_ID_Y = 8,
  /// 32-bit Z component of the workgroup ID (preloaded system SGPR)
  WORKGROUP_ID_Z = 9,
  /// 32-bit X component of lane 0's workitem ID at kernel entry
  WORKITEM_ID_X = 10,
  /// 32-bit Y component of lane 0's workitem ID at kernel entry
  WORKITEM_ID_Y = 11,
  /// 32-bit Z component of lane 0's workitem ID at kernel entry
  WORKITEM_ID_Z = 12,
  /// Marks the last defined scalar value argument
  SCALAR_VALUE_ARGUMENT_LAST = WORKITEM_ID_Z
  // NOTE: The SVA is exactly saturated on GFX10 (wave32 = 32 lanes), so any
  // new entry here has to be paid for by dropping an existing one.
  // /// 64-bit address of the instrumentation routine's argument buffer
  // USER_ARG_PTR = 15,
  //   /// 32-bit private segment wave offset
  // PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 4,
};

template <ScalarValueArgument SA> struct ScalarValueArgumentInfo;

template <> struct ScalarValueArgumentInfo<WAVEFRONT_PRIVATE_SEGMENT_BUFFER> {
  static constexpr uint8_t NumLanes = 4;
};

template <> struct ScalarValueArgumentInfo<DISPATCH_ID> {
  static constexpr uint8_t NumLanes = 2;
};

template <> struct ScalarValueArgumentInfo<FLAT_SCRATCH> {
  static constexpr uint8_t NumLanes = 2;
};

// template <> struct ScalarValueArgumentInfo<PRIVATE_SEGMENT_WAVE_BYTE_OFFSET>
// {
//   static constexpr uint8_t NumLanes = 1;
// };

template <> struct ScalarValueArgumentInfo<QUEUE_PTR> {
  static constexpr uint8_t NumLanes = 2;
};

template <> struct ScalarValueArgumentInfo<DISPATCH_PTR> {
  static constexpr uint8_t NumLanes = 2;
};

template <>
struct ScalarValueArgumentInfo<WORK_ITEM_INSTRUMENTATION_PRIVATE_SEGMENT_SIZE> {
  static constexpr uint8_t NumLanes = 1;
};

template <> struct ScalarValueArgumentInfo<IMPLICIT_ARG_BUFFER> {
  static constexpr uint8_t NumLanes = 2;
};

template <> struct ScalarValueArgumentInfo<WORKGROUP_ID_X> {
  static constexpr uint8_t NumLanes = 1;
};

template <> struct ScalarValueArgumentInfo<WORKGROUP_ID_Y> {
  static constexpr uint8_t NumLanes = 1;
};

template <> struct ScalarValueArgumentInfo<WORKGROUP_ID_Z> {
  static constexpr uint8_t NumLanes = 1;
};

template <> struct ScalarValueArgumentInfo<WORKITEM_ID_X> {
  static constexpr uint8_t NumLanes = 1;
};

template <> struct ScalarValueArgumentInfo<WORKITEM_ID_Y> {
  static constexpr uint8_t NumLanes = 1;
};

template <> struct ScalarValueArgumentInfo<WORKITEM_ID_Z> {
  static constexpr uint8_t NumLanes = 1;
};

// template <> struct ScalarValueArgumentInfo<USER_ARG_PTR> {
//   static constexpr uint8_t NumLanes = 2;
// };

} // namespace luthier

#endif
