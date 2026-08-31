//===-- AMDGPUPreloadValueMapping.h ------------------------------*- C++-*-===//
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
/// Bidirectional mapping between Luthier's \c ScalarValueArgument enum and
/// the AMDGPU backend's \c AMDGPUFunctionArgInfo::PreloadedValue enum, plus
/// the \c amdgpu-no-* negated-usage attribute for each SV. Used by the
/// StateValueArraySpecs analysis and the IntrinsicMIRLoweringPass preloaded
/// arg rewriter to translate between the two representations.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_AMDGPU_PRELOAD_VALUE_MAPPING_H
#define LUTHIER_TOOL_CODE_GEN_AMDGPU_PRELOAD_VALUE_MAPPING_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <SIMachineFunctionInfo.h>
#include <llvm/ADT/StringRef.h>
#include <optional>

namespace luthier {

/// Map a Luthier \c ScalarValueArgument to the AMDGPU preloaded-value slot
/// it corresponds to. Returns \c nullopt for SVs with no direct AMDGPU
/// preload counterpart (should not occur for the current enum).
inline std::optional<llvm::AMDGPUFunctionArgInfo::PreloadedValue>
amdgpuPreloadToStateValueArg(ScalarValueArgument SA) {
  using PV = llvm::AMDGPUFunctionArgInfo::PreloadedValue;
  switch (SA) {
  case WAVEFRONT_PRIVATE_SEGMENT_BUFFER:
    return PV::PRIVATE_SEGMENT_BUFFER;
  case KERNEL_ARG_PTR:
    return PV::KERNARG_SEGMENT_PTR;
  case DISPATCH_ID:
    return PV::DISPATCH_ID;
  case FLAT_SCRATCH:
    return PV::FLAT_SCRATCH_INIT;
  case DISPATCH_PTR:
    return PV::DISPATCH_PTR;
  case QUEUE_PTR:
    return PV::QUEUE_PTR;
  case WORK_ITEM_PRIVATE_SEGMENT_SIZE:
    return PV::PRIVATE_SEGMENT_SIZE;
  case IMPLICIT_ARG_BUFFER:
    return PV::IMPLICIT_ARG_PTR;
  case WORKGROUP_ID_X:
    return PV::WORKGROUP_ID_X;
  case WORKGROUP_ID_Y:
    return PV::WORKGROUP_ID_Y;
  case WORKGROUP_ID_Z:
    return PV::WORKGROUP_ID_Z;
  case WORKITEM_ID_X:
    return PV::WORKITEM_ID_X;
  case WORKITEM_ID_Y:
    return PV::WORKITEM_ID_Y;
  case WORKITEM_ID_Z:
    return PV::WORKITEM_ID_Z;
  }
  static_assert(
      SCALAR_VALUE_ARGUMENT_LAST == WORKITEM_ID_Z,
      "extend amdgpuPreloadToStateValueArg for new ScalarValueArgument");
  return std::nullopt;
}

/// Reverse of \c amdgpuPreloadToStateValueArg. Only kinds that the AMDGPU
/// backend allocates in a way Luthier's SVA can preserve are returned;
/// unknown kinds map to \c nullopt.
inline std::optional<ScalarValueArgument>
mapSVArgToAMDGPUPreload(llvm::AMDGPUFunctionArgInfo::PreloadedValue PV) {
  using PVE = llvm::AMDGPUFunctionArgInfo::PreloadedValue;
  switch (PV) {
  case PVE::PRIVATE_SEGMENT_BUFFER:
    return WAVEFRONT_PRIVATE_SEGMENT_BUFFER;
  case PVE::KERNARG_SEGMENT_PTR:
    return KERNEL_ARG_PTR;
  case PVE::DISPATCH_ID:
    return DISPATCH_ID;
  case PVE::FLAT_SCRATCH_INIT:
    return FLAT_SCRATCH;
  case PVE::DISPATCH_PTR:
    return DISPATCH_PTR;
  case PVE::QUEUE_PTR:
    return QUEUE_PTR;
  case PVE::PRIVATE_SEGMENT_SIZE:
    return WORK_ITEM_PRIVATE_SEGMENT_SIZE;
  case PVE::IMPLICIT_ARG_PTR:
    return IMPLICIT_ARG_BUFFER;
  case PVE::WORKGROUP_ID_X:
    return WORKGROUP_ID_X;
  case PVE::WORKGROUP_ID_Y:
    return WORKGROUP_ID_Y;
  case PVE::WORKGROUP_ID_Z:
    return WORKGROUP_ID_Z;
  case PVE::WORKITEM_ID_X:
    return WORKITEM_ID_X;
  case PVE::WORKITEM_ID_Y:
    return WORKITEM_ID_Y;
  case PVE::WORKITEM_ID_Z:
    return WORKITEM_ID_Z;
  default:
    return std::nullopt;
  }
}

/// The AMDGPU attribute name whose *absence* on a function indicates the
/// function may use \p SA — the AMDGPUAttributor uses these
/// \c "amdgpu-no-*" attributes to mark not-used values. If \c StringRef is
/// empty, no attribute maps to this SA and the caller must fall back to
/// something else (SIMFI, whole-payload conservatively enabling it, etc).
inline llvm::StringRef amdgpuNoUsageAttrForSA(ScalarValueArgument SA) {
  switch (SA) {
  case DISPATCH_ID:
    return "amdgpu-no-dispatch-id";
  case DISPATCH_PTR:
    return "amdgpu-no-dispatch-ptr";
  case QUEUE_PTR:
    return "amdgpu-no-queue-ptr";
  case IMPLICIT_ARG_BUFFER:
    return "amdgpu-no-implicitarg-ptr";
  case FLAT_SCRATCH:
    return "amdgpu-no-flat-scratch-init";
  case WORKGROUP_ID_X:
    return "amdgpu-no-workgroup-id-x";
  case WORKGROUP_ID_Y:
    return "amdgpu-no-workgroup-id-y";
  case WORKGROUP_ID_Z:
    return "amdgpu-no-workgroup-id-z";
  case WORKITEM_ID_X:
    return "amdgpu-no-workitem-id-x";
  case WORKITEM_ID_Y:
    return "amdgpu-no-workitem-id-y";
  case WORKITEM_ID_Z:
    return "amdgpu-no-workitem-id-z";
  case KERNEL_ARG_PTR:
  case WAVEFRONT_PRIVATE_SEGMENT_BUFFER:
  case WORK_ITEM_PRIVATE_SEGMENT_SIZE:
    // No dedicated amdgpu-no-* attribute — the backend infers these from
    // the caller-side ABI / target features. Callers should keep them
    // conservatively marked used when a fallback is needed.
    return llvm::StringRef();
  }
  static_assert(SCALAR_VALUE_ARGUMENT_LAST == WORKITEM_ID_Z,
                "extend amdgpuNoUsageAttrForSA for new ScalarValueArgument");
  return llvm::StringRef();
}

} // namespace luthier

#endif
