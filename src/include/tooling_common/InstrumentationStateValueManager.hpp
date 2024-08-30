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
  struct InstrumentationStateValueSegment {
  private:
    /// Start point of the interval (inclusive)
    llvm::SlotIndex Start;
    /// End point of the interval (exclusive)
    llvm::SlotIndex End;
    /// Where the kernel arguments are stored; If zero, then it is not
    /// stored in register
    llvm::MCRegister KernelArgumentsRegisterLocation;
    /// Where the flat scratch pointing to the bottom of the instrumentation
    /// stack
    llvm::MCRegister FlatScratchInstrumentationStackReg;

  public:
    InstrumentationStateValueSegment(llvm::SlotIndex S, llvm::SlotIndex E,
                                     llvm::MCRegister KAReg,
                                     llvm::MCRegister FSReg)
        : Start(S), End(E), KernelArgumentsRegisterLocation(KAReg),
          FlatScratchInstrumentationStackReg(FSReg) {
      if (S < E)
        llvm::report_fatal_error("Cannot create empty or backwards segment");
    }

    [[nodiscard]] llvm::MCRegister getValueRegisterLocation() const {
      return KernelArgumentsRegisterLocation;
    }

    [[nodiscard]] llvm::Register
    getInstrumentationStackFlatScratchLocation() const {
      return FlatScratchInstrumentationStackReg;
    }

    /// Return true if the index is covered by this segment.
    [[nodiscard]] bool contains(llvm::SlotIndex I) const {
      return Start <= I && I < End;
    }

    /// Return true if the given interval, [S, E), is covered by this segment.
    [[nodiscard]] bool containsInterval(llvm::SlotIndex S,
                                        llvm::SlotIndex E) const {
      if (S < E)
        llvm::report_fatal_error("Backwards interval");
      return (Start <= S && S < End) && (Start < E && E <= End);
    }

    bool operator<(const InstrumentationStateValueSegment &Other) const {
      return std::tie(Start, End) < std::tie(Other.Start, Other.End);
    }

    bool operator==(const InstrumentationStateValueSegment &Other) const {
      return Start == Other.Start && End == Other.End;
    }

    bool operator!=(const InstrumentationStateValueSegment &Other) const {
      return !(*this == Other);
    }
  };

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

  /// Slot index tracking for each instruction in the \c LR
  llvm::SmallDenseMap<llvm::MachineFunction *,
                      std::unique_ptr<llvm::SlotIndexes>>
      FunctionsSlotIndexes;

  /// Holds the information for a kernel arguments
  typedef llvm::SmallDenseSet<KernelArgumentType, HIDDEN_END> KernelArguments;

  /// A set of args that needs to be ensured for the argument manager
  /// for the intrinsics
  llvm::SmallDenseSet<KernelArgumentType, 4> EnsuredArgs{};

  /// Whether the prologue insertion has been completed or not
  bool IsPrologueEmitted{false};

  /// Keeps track of where the value state and flat scratch registers will
  /// be stored in each function involved in the \c LiftedRepresentation
  llvm::SmallDenseMap<llvm::MachineFunction *,
                      llvm::SmallVector<InstrumentationStateValueSegment>>
      ValueStateRegAndFlatScratchIntervals{};

  /// \brief Tries to find a fixed location to store the instrumentation value
  /// register in a VPGR or an AGPR, as well as an SGPR pair to store the flat
  /// address pointing to the beginning of the instrumentation private segment
  /// \details The order of searching for these fixed locations is as follows:
  /// 1. Find an unused VGPR. This is the ideal scenario, as an unused VGPR has
  /// enough space to accommodate the prologue/epilogue of an instrumentation
  /// hook. No additional code needs to be inserted inside the kernel itself
  /// for managing the value register, and the flat scratch register will be
  /// 0\n
  /// 2. If no unused VGPRs are found, then this routine will find the next
  /// unused AGPR. This usually comes at no cost to the occupancy, as the app
  /// will get the same amount of AGPRs as it gets VGPRs. In gfx90A-, since
  /// AGPRs cannot be used directly by vector instructions and have to be moved
  /// to a VGPR, it is likely that a single VGPR must be spilled to accommodate
  /// this issue. For this scenario, an unused SGPR pair must also be found to
  /// hold the address of the instrumentation's private segment. No other
  /// action is necessary by the value manager except generating
  /// prologue/epilogue code.\n
  /// 3. If no unused V/AGPRs are found in the kernel, then as a last resort,
  /// this function tries to find an unused SGPR pair for the instrumentation
  /// scratch address to be stored. In this scenario, besides emitting prologue
  /// /epiloge code, the value manager must also emit code in the kernel that
  /// moves the state value register around or spill it
  ///
  /// TODO: This function must take an argument indicating whether the tool
  /// writer wants to respect the original kernel's granulated register usage
  /// or not.
  [[nodiscard]] std::pair<llvm::MCRegister, llvm::MCRegister>
  findFixedRegisterLocationsToStoreInstrValueVGPRAndInstrFlatScratchReg() const;

  const InstrumentationStateValueSegment *
  getValueSegmentForInstr(llvm::MachineFunction &MF,
                          llvm::MachineInstr &MI) const;

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