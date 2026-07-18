//===-- PrePostAmbleEmitter.h -----------------------------------*- C++ -*-===//
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
/// \file
/// This file describes the Pre and post amble emitter,
/// which will emits code before and after
/// using the information gathered from code gen passes when generating
/// the hooks. It also describes the \c FunctionPreambleDescriptor and its
/// analysis pass, which describes the preamble specs for each function
/// inside the target application.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_PRE_POST_AMBLE_EMITTER_H
#define LUTHIER_TOOL_CODE_GEN_PRE_POST_AMBLE_EMITTER_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/Support/Error.h>

namespace luthier {

class SVStorageAndLoadLocations;

/// \brief a struct which aggregates information about the preamble code
/// required to be emitted for each function inside a \c LiftedRepresentation
struct FunctionPreambleDescriptor {
  /// \brief struct describing the specifications of the preamble code for
  /// each kernel inside the \c LR
  typedef struct KernelPreambleSpecs {
    [[nodiscard]] bool usesSVA() const {
      return RequiresScratchAndStackSetup ||
             RequestedAdditionalStackSizeInBytes ||
             !RequestedKernelArguments.empty();
    }
    /// Whether the preamble requires setting up scratch and an instrumentation
    /// stack
    bool RequiresScratchAndStackSetup{false};
    /// Number of bytes of scratch space requested on top of the application
    /// stack; This value is hard coded in the preamble assembly code
    unsigned int RequestedAdditionalStackSizeInBytes{0};
    /// A set of kernel arguments that are accessed by the injected payload
    /// functions
    llvm::SmallDenseSet<ScalarValueArgument, 8> RequestedKernelArguments{};
  } KernelPreambleSpecs;

  /// \brief struct describing the specifications of the preamble code for
  /// each kernel inside the \c LR
  typedef struct DeviceFunctionPreambleSpecs {
    /// Whether or not any hooks inside the device function access the
    /// state value array
    bool UsesStateValueArray{false};
    /// Indicates if the device function requires additional code before and
    /// and after it to pop/push the state value array off of the application
    /// stack
    bool RequiresPreAndPostAmble{false};
    /// Whether the device function makes use of stack/scratch
    bool RequiresScratchAndStackSetup{false};
    /// A set of kernel arguments accessed by the device function injected
    /// payloads
    llvm::SmallDenseSet<ScalarValueArgument, 8> RequestedKernelArguments{};
  } DeviceFunctionPreambleSpecs;

  /// preamble specs for each kernel inside the \c LR
  llvm::SmallDenseMap<const llvm::MachineFunction *, KernelPreambleSpecs, 4>
      Kernels{};

  /// pre/post amble specs for each device function inside the \c LR
  llvm::SmallDenseMap<const llvm::MachineFunction *,
                      DeviceFunctionPreambleSpecs, 4>
      DeviceFunctions{};

  /// Never invalidate the results
  bool invalidate(Prototype &, const llvm::PreservedAnalyses &,
                  PrototypeAnalysisManager::Invalidator &) {
    return false;
  }
};

/// \brief \c Prototype-level analysis that computes, for each
/// target \c MachineFunction, the preamble spec required to correctly host
/// its injected payloads.
///
/// The analysis walks every target \c MachineFunction and, at each
/// \c MachineInstr registered as an instrumentation point by
/// \c InjectedPayloadAndInstPointAnalysis, unions the SVA scalar arguments
/// requested by every payload attached to that MI (obtained from
/// \c InjectedPayloadSideEffectsAnalysis on the instrumentation module) into
/// the descriptor entry for the containing target MF.
class FunctionPreambleDescriptorAnalysis
    : public llvm::AnalysisInfoMixin<FunctionPreambleDescriptorAnalysis> {
  friend llvm::AnalysisInfoMixin<FunctionPreambleDescriptorAnalysis>;

  static llvm::AnalysisKey Key;

public:
  FunctionPreambleDescriptorAnalysis() = default;

  using Result = FunctionPreambleDescriptor;

  Result run(Prototype &IP,
             PrototypeAnalysisManager &IPAM);
};

} // namespace luthier

#endif