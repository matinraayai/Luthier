//===-- InstrumentationStateValueManager.hpp ------------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file describes the <tt>InstrumentationStateValueManager</tt>,
/// which is in charge of managing the instrumented kernel's state value
/// across the instrumented app (via either reserving a static space or a
/// dynamic space to store it), and ensuring access to it by hooks via
/// inserting prologue/epilogues across hooks.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_INSTRUMENTATION_STATE_VALUE_MANAGER
#define LUTHIER_INSTRUMENTATION_STATE_VALUE_MANAGER
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/hsa/LoadedCodeObjectDeviceFunction.h"
#include "luthier/hsa/LoadedCodeObjectKernel.h"
#include "luthier/hsa/Metadata.h"
#include <AMDGPUArgumentUsageInfo.h>
#include <llvm/CodeGen/LiveInterval.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/SlotIndexes.h>

namespace llvm {

class MachineFunction;

} // namespace llvm

namespace luthier {

namespace hsa {

class LoadedCodeObjectSymbol;

class LoadedCodeObjectKernel;

} // namespace hsa

class LiftedRepresentation;

class InstrumentationStateValueManager {

private:
  /// The lifted representation being worked on
  LiftedRepresentation &LR;

  /// Mapping between the MIs of the target app getting instrumented and their
  /// hooks
  llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap;

  /// Instrumentation Module
  llvm::Module &Module;

  /// Instrumentation MachineModuleInfo
  llvm::MachineModuleInfo &MMI;

  /// The kernel of the lifted representation
  /// TODO make this work with LRs with multiple kernels
  std::pair<const hsa::LoadedCodeObjectKernel *, llvm::MachineFunction *>
      Kernel;

  /// The map of the hidden arguments of the kernel to ensure access for
  /// intrinsics at the MIR lowering stage
  llvm::SmallDenseMap<KernelArgumentType, uint32_t, 16>
      HiddenKernArgsToOffsetMap;

  /// Contains generated code to be inserted before each instruction inside the
  /// app; These take precedence over hook code
  llvm::DenseMap<llvm::MachineInstr *, llvm::MachineFunction *>
      PrologueAndRegStateMoveCode{};

  /// The epilogue code for all app device functions involved in the LR;
  /// These come after hooks
  llvm::SmallDenseMap<llvm::MachineInstr *, llvm::MachineFunction *, 4>
      EpilogueCode{};

  /// Holds the information for a kernel arguments
  typedef llvm::SmallDenseSet<KernelArgumentType, HIDDEN_END> KernelArguments;

  /// A set of args that needs to be ensured for the argument manager
  /// for the intrinsics
  llvm::SmallDenseSet<KernelArgumentType, 4> EnsuredArgs{};

  /// Whether the prologue insertion has been completed or not
  bool IsPrologueEmitted{false};

public:
  InstrumentationStateValueManager(
      LiftedRepresentation &LR,
      llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap,
      llvm::Module &InstrumentationModule,
      llvm::MachineModuleInfo &InstrumentationMMI);

  /// Queries if the kernel has access to the argument without being modified
  [[nodiscard]] bool hasAccessToArg(KernelArgumentType Arg) const;

  /// Tells the argument manager to ensure access to the arg after ISEL
  void ensureAccessToArg(KernelArgumentType Arg);

  llvm::MCRegister getArgReg(KernelArgumentType Arg);

public:
  llvm::Error insertPrologue();
};

} // namespace luthier

#endif