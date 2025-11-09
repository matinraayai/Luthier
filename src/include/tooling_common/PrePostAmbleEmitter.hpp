//===-- PrePostAmbleEmitter.hpp -------------------------------------------===//
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
/// This file describes the Pre and post amble emitter,
/// which will emits code before and after
/// using the information gathered from code gen passes when generating
/// the hooks. It also describes the \c FunctionPreambleDescriptor and its
/// analysis pass, which describes the preamble specs for each function
/// inside the target application.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_PRE_POST_AMBLE_EMITTER_HPP
#define LUTHIER_TOOLING_COMMON_PRE_POST_AMBLE_EMITTER_HPP
#include "luthier/tooling/LiftedRepresentation.h"
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/Support/Error.h>
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>
#include <luthier/intrinsic/IntrinsicProcessor.h>

namespace luthier {

class SVStorageAndLoadLocations;

class LiftedRepresentation;

/// \brief Holds information regarding the specifications of the state
/// value array register
struct SVADescriptor {

  struct LaneDescriptor {
    unsigned StartLane;
    unsigned EndLane;
  };

  /// \brief Enum for each state value entry saved inside the state value array
  enum StateValue {
    /// Holds the flat scratch register pair used to access the instrumentation
    /// stack; Used for targets that don't support architected flat scratch
    INSTRUMENTATION_FLAT_SCRATCH = 0,

    /// Slot for spilling the flat scratch register pair in order to set up the
    /// instrumentation frame; Used for targets that don't support architected
    /// flat scratch
    APPLICATION_FLAT_SCRATCH_SPILL_SLOT = 1,

    /// Holds the instrumentation stack pointer
    INSTRUMENTATION_STACK_PTR = 2,

    /// Slot for spilling the s32 register of the target kernel in order to
    /// set up the instrumentation frame
    APPLICATION_STACK_PTR_SPILL_SLOT = 3,

    /// The private segment buffer (V#) of the kernel's queue
    /// Can also be accessed via the
    /// \c &amd_queue_v2_t::scratch_resource_descriptor field of the kernel's
    /// queue
    /// This Value is not used by the LLVM compiler on sub-targets with
    /// architected flat scratch
    QUEUE_PRIVATE_SEGMENT_BUFFER = 4,

    /// The 64-bit address of the AQL dispatch packet used to launch this kernel
    AQL_DISPATCH_PACKET_PTR = 5,

    /// The 64-bit address of the \c amd_queue_v2_t object used to launch the
    /// kernel
    QUEUE_PTR = 6,

    /// The 64-bit address of the target application's kernel argument segment
    TARGET_KERN_ARG_SEGMENT_PTR = 7,

    /// The 64-bit address of the user (instrumentation)'s kernel argument
    /// segment
    /// The user kernel arguments are used to pass additional explicit arguments
    /// as well as hidden arguments to the instrumentation routine during each
    /// kernel launch
    TARGET_USER_ARG_SEGMENT_PTR = 8,

    /// 64-bit Dispatch ID of the kernel
    DISPATCH_ID = 9,

    /// The flat scratch init user SGPR value
    /// Depending on the sub-target, this field can have a different meaning
    /// It is not supported on targets with architected flat scratch
    FLAT_SCRATCH_INIT = 10,

    /// Size of a work-item's private segment found in the
    /// \c &::hsa_kernel_dispatch_packet_t::private_segment_size field rounded
    /// up by the CP to a multiple of DWORD
    /// Note that this is the private segment size of the instrumented kernel,
    /// not the target application's private segment size
    WORK_ITEM_PRIVATE_SEGMENT_SIZE = 11,

    /// 32-bit offset of the wavefront's private segment from the queue's
    /// private segment base
    PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 12,

  };

private:
  /// Set of lanes in the SVA reserved for setting up/tearing down the
  /// instrumentation stack
  /// These lanes are allocated regardless of whether the instrumentation
  /// logic makes use of them or not
  llvm::SmallDenseMap<StateValue, LaneDescriptor> InstrumentationFrameLanes{};

  /// Set of lanes in SVA for anything other than the instrumentation frame
  /// These lanes are allocated on request by the instrumentation logic
  llvm::SmallDenseMap<StateValue, std::pair<unsigned int, unsigned int>>
      NonIFrameLanes{};

  /// Whether the prologue requires setting up the always allocated
  /// instrumentation frame lanes
  bool RequiresScratchAndStackSetup{false};

  /// Total number of bytes of scratch space requested by the instrumentation
  /// logic on top of the application stack
  /// This value is hard coded in the prologue code
  unsigned int RequestedAdditionalStackSizeInBytes{0};

  /// Next free lane in the state value array
  unsigned int NextFreeLane{0};

  /// Whether the layout of the SVA is finalized or not; If \c true then the
  /// layout of the lanes cannot be changed further by any passes. The passes
  /// can still decide if they want to set up the instrumentation frame or not
  /// in the prologue
  bool IsFrozen{false};

public:
  /// \returns the number of lanes in the state value array the \p SV occupies
  static unsigned int getStateValueSize(StateValue SV);

  /// \returns \c true if \p SV is part of the values used to construct the
  /// instrumentation frame, \c false otherwise
  static bool isInstrumentationFrameValue(StateValue SV) {
    switch (SV) {
    case INSTRUMENTATION_FLAT_SCRATCH:
    case APPLICATION_FLAT_SCRATCH_SPILL_SLOT:
    case INSTRUMENTATION_STACK_PTR:
    case APPLICATION_STACK_PTR_SPILL_SLOT:
      return true;
    default:
      return false;
    }
  }

  LaneDescriptor getOrAllocateSVALane(StateValue SV);

  [[nodiscard]] bool usesSVA() const {
    return RequiresScratchAndStackSetup ||
           RequestedAdditionalStackSizeInBytes || !NonIFrameLanes.empty();
  }

  SVADescriptor(const llvm::GCNSubtarget &ST, bool IsInitialEntryPointAKernel);

  /// Never invalidate the results
  bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

class FunctionPreambleDescriptorAnalysis
    : public llvm::AnalysisInfoMixin<FunctionPreambleDescriptorAnalysis> {
  friend llvm::AnalysisInfoMixin<FunctionPreambleDescriptorAnalysis>;

  static llvm::AnalysisKey Key;

public:
  FunctionPreambleDescriptorAnalysis() = default;

  using Result = SVADescriptor;

  Result run(llvm::Module &TargetModule,
             llvm::ModuleAnalysisManager &TargetMAM);
};

class PrePostAmbleEmitter : public llvm::PassInfoMixin<PrePostAmbleEmitter> {

public:
  explicit PrePostAmbleEmitter() = default;

  llvm::PreservedAnalyses run(llvm::Module &TargetModule,
                              llvm::ModuleAnalysisManager &TargetMAM);
};

} // namespace luthier

#endif