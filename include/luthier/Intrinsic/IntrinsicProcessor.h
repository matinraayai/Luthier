//===-- IntrinsicProcessor.h ------------------------------------*- C++ -*-===//
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
/// This file describes Luthier's Intrinsic Processor structs and functions,
/// required to define custom Luthier intrinsics by a tool.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_INTRINSIC_PROCESSOR_H
#define LUTHIER_INTRINSIC_INTRINSIC_PROCESSOR_H

#include <functional>
#include <llvm/ADT/Any.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/Error.h>
#include <string>
#include <llvm/MC/MCRegister.h>

namespace llvm {

class MachineFunction;

class TargetRegisterInfo;

class TargetInstrInfo;

class TargetRegisterClass;

class Register;

class Value;

class Function;

class CallInst;

class GCNTargetMachine;

class MachineInstrBuilder;
} // namespace llvm

namespace luthier {

/// \brief a set of kernel arguments Luthier's intrinsic lowering mechanism
/// can ensure access to
/// \details these values are only available to the kernel as "arguments"
/// as they come either preloaded in S/VGPRs or they are passed as "hidden"
/// arguments in the kernel argument buffer. As these values (or the way to
/// access them) are stored in GPRs they can be overwritten the moment they
/// are unused by the instrumented app. To ensure access to these values
/// in instrumentation routines, Luthier must emit a prologue on top of the
/// kernel's original prologue to save these values in an unused register,
/// or spill them to the top of the instrumentation stack's buffer to be
/// loaded when necessary
enum KernelArgumentType {
  PRIVATE_SEGMENT_BUFFER = 0,
  ALWAYS_IN_SGPR_BEGIN = PRIVATE_SEGMENT_BUFFER,
  KERNARG_SEGMENT_PTR = 1,
  DISPATCH_ID = 2,
  FLAT_SCRATCH_INIT = 3,
  PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 4,
  ALWAYS_IN_SGPR_END = PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
  DISPATCH_PTR = 5,
  EITHER_IN_SGPR_OR_HIDDEN_BEGIN = DISPATCH_PTR,
  QUEUE_PTR = 6,
  PRIVATE_SEGMENT_SIZE = 7,
  EITHER_IN_SGPR_OR_HIDDEN_END = PRIVATE_SEGMENT_SIZE,
  GLOBAL_OFFSET_X = 8,
  HIDDEN_BEGIN = GLOBAL_OFFSET_X,
  GLOBAL_OFFSET_Y = 9,
  GLOBAL_OFFSET_Z = 10,
  PRINT_BUFFER = 11,
  HOSTCALL_BUFFER = 12,
  DEFAULT_QUEUE = 12,
  COMPLETION_ACTION = 13,
  MULTIGRID_SYNC = 14,
  BLOCK_COUNT_X = 15,
  BLOCK_COUNT_Y = 16,
  BLOCK_COUNT_Z = 17,
  GROUP_SIZE_X = 18,
  GROUP_SIZE_Y = 19,
  GROUP_SIZE_Z = 20,
  REMAINDER_X = 21,
  REMAINDER_Y = 22,
  REMAINDER_Z = 23,
  GRID_DIMS = 24,
  HEAP_V1 = 25,
  DYNAMIC_LDS_SIZE = 26,
  PRIVATE_BASE = 27,
  SHARED_BASE = 28,
  HIDDEN_END = SHARED_BASE,
  WORK_ITEM_X = 29,
  WORK_ITEM_Y = 30,
  WORK_ITEM_Z = 31
};

/// \brief Contains information about the values used/defined by
/// a \c llvm::CallInst to a Luthier Intrinsic, and its inline assembly
/// constraint (e.g. 'v', 's', etc)
/// \details This struct is used to keep track of how an LLVM IR value
/// used/defined by a \c llvm::CallInst to a Luthier Intrinsic should be mapped
/// to a \c llvm::Register; For example,
/// if value <tt>%1</tt> used by the IR call instruction
/// \code
/// %1 = tail call i32 @"luthier::myIntrinsic.i32"(i32 %0)
/// \endcode
/// needs to become an SGPR after ISEL passes are finished, <tt>%1</tt> will
/// have an <tt>'s'</tt> \c Constraint \n
struct IntrinsicValueLoweringInfo {
  const llvm::Value *Val; ///< The IR value to be lowered
  std::string Constraint; ///< The inline asm constraint describing how \c Val
                          /// should be lowered
};

/// \brief Holds information about the output of the IR processing stage of an
/// intrinsic IR call instruction,
/// including how all values used/defined by a Luthier
/// intrinsic use (i.e. its output and input arguments) must be
/// lowered to registers
/// \details This struct is the return value for the
/// <tt>IntrinsicIRProcessorFunc</tt>. Internally, \c luthier::CodeGenerator
/// stores the results of all IR processor function calls, and then passes
/// them to the <tt>IntrinsicMIRProcessorFunc</tt> after ISEL passes are
/// complete to generate <tt>llvm::MachineInstr</tt>s in its place
struct IntrinsicIRLoweringInfo {
private:
  /// Name of the intrinsic; Used by \c luthier::CodeGenerator for keeping
  /// track of the lowering operation at the MIR stage
  std::string IntrinsicName;
  /// How the output value (if present) must be lowered to a
  /// \c llvm::Register
  IntrinsicValueLoweringInfo OutValue{nullptr, ""};
  /// How the argument values (if present) must be lowered to a
  /// \c llvm::Register
  llvm::SmallVector<IntrinsicValueLoweringInfo, 4> Args{};
  /// An arbitrary data (if needed) to be passed from the IR processing stage to
  /// the MIR processing stage
  llvm::Any Data{};

  /// A set of physical registers that needs to be accessed by this intrinsic
  llvm::SmallDenseSet<llvm::MCRegister, 1> AccessedPhysicalRegisters{};

  /// A set of kernel arguments that needs to be accessed by this intrinsic
  llvm::SmallDenseSet<KernelArgumentType, 4> AccessedKernelArguments{};

public:
  /// \param Name the name of the intrinsic being lowered
  /// \note this function is called internally by Luthier on the result of
  /// \c IntrinsicIRProcessorFunc is returned; Hence setting the name of the
  /// intrinsic inside the IR processor has no effect
  void setIntrinsicName(llvm::StringRef Name) { this->IntrinsicName = Name; }

  /// \returns the name of the intrinsic being lowered
  [[nodiscard]] llvm::StringRef getIntrinsicName() const {
    return IntrinsicName;
  }

  /// Sets the inline asm constraint to \p Constraint for the given
  /// \p Val
  void setReturnValueInfo(const llvm::Value *Val, llvm::StringRef Constraint) {
    OutValue.Val = Val;
    OutValue.Constraint = Constraint;
  }

  /// \returns the return value's \c IntrinsicValueLoweringInfo
  [[nodiscard]] const IntrinsicValueLoweringInfo &getReturnValueInfo() const {
    return OutValue;
  }

  /// Adds a new argument, with \p Val and \p Constraint describing its
  /// \c IntrinsicValueLoweringInfo
  void addArgInfo(const llvm::Value *Val, llvm::StringRef Constraint) {
    Args.emplace_back(Val, std::string(Constraint));
  }

  /// \returns All arguments' \c IntrinsicValueLoweringInfo
  llvm::ArrayRef<IntrinsicValueLoweringInfo> getArgsInfo() const {
    return Args;
  }

  /// Sets the lowering data to \p D
  /// The lowering data is made available to the \c IntrinsicMIRProcessorFunc
  template <typename T> void setLoweringData(T D) { Data = D; }

  /// \returns the lowering data which will be made available to the
  /// \c IntrinsicMIRProcessorFunc when emitting Machine Instructions
  template <typename T> const T &getLoweringData() const {
    return *llvm::any_cast<T>(&Data);
  }

  /// Asks the code generator to ensure access to the \p PhysReg during
  /// the MIR lowering stage
  void requestAccessToPhysicalRegister(llvm::MCRegister PhysReg) {
    AccessedPhysicalRegisters.insert(PhysReg);
  }

  [[nodiscard]] const llvm::SmallDenseSet<llvm::MCRegister, 1> &
  getAccessedPhysicalRegisters() const {
    return AccessedPhysicalRegisters;
  }

  const llvm::SmallDenseSet<KernelArgumentType, 4> &
  getRequestedKernelArgument() const {
    return AccessedKernelArguments;
  }

  /// Asks the code generator to ensure access to the \p KernArg during
  /// the MIR lowering stage
  void requestAccessToKernelArgument(KernelArgumentType KernArg) {
    AccessedKernelArguments.insert(KernArg);
  }
};

/// \brief describes a function type used by each Luthier intrinsic to process
/// its uses in LLVM IR, and return a \c IntrinsicIRLoweringInfo which will
/// describe how its use/def values will be lowered to
/// <tt>llvm::MachineOperand</tt>s, as well as any arbitrary information
/// required to be passed down from the IR processing stage to the MIR
/// processing stage
typedef std::function<llvm::Expected<IntrinsicIRLoweringInfo>(
    const llvm::Function &, const llvm::CallInst &,
    const llvm::GCNTargetMachine &)>
    IntrinsicIRProcessorFunc;

/// \brief describes a function type used for each intrinsic to generate
/// <tt>llvm::MachineInstr</tt>s in place of its IR calls.
/// The MIR processor takes in the
/// \c IntrinsicIRLoweringInfo generated by its \c IntrinsicIRProcessorFunc as
/// well as the lowered registers and their inline assembly flags for
/// its used/defined values. A lambda which will create an
/// \c llvm::MachineInstr at the place of emission given an instruction opcode
/// is also passed to this function
typedef std::function<llvm::Error(
    const IntrinsicIRLoweringInfo &,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>>,
    const std::function<llvm::MachineInstrBuilder(int)> &,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &,
    const llvm::MachineFunction &,
    const std::function<llvm::Register(llvm::MCRegister)> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &)>
    IntrinsicMIRProcessorFunc;

/// \brief Used internally by \c luthier::CodeGenerator to keep track of
/// registered intrinsics and how to process them
struct IntrinsicProcessor {
  IntrinsicIRProcessorFunc IRProcessor{};
  IntrinsicMIRProcessorFunc MIRProcessor{};
};

} // namespace luthier

#endif