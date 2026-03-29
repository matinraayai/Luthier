//===-- IntrinsicProcessor.h ------------------------------------*- C++ -*-===//
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
/// \file IntrinsicProcessor.h
/// Describes Luthier's Intrinsic Processor structs and functions, required to
/// define both internal and custom Luthier intrinsics.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_INTRINSIC_PROCESSOR_H
#define LUTHIER_INTRINSIC_INTRINSIC_PROCESSOR_H
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>
#include <string>

namespace llvm {

class MachineFunction;

class MachineInstr;

class TargetRegisterClass;

class Register;

class Value;

class Function;

class CallInst;

class GCNTargetMachine;

class MachineInstrBuilder;
} // namespace llvm

namespace luthier {

/// The header of the intrinsics's extra info in the PC sections of an
/// instruction
static constexpr auto IntrinsicExtraInfoHeader = "luthier.intrinsic.extra_info";

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
  /// 64-bit address of the kernel's argument buffer
  KERNEL_ARG_PTR = 1,
  /// 64-bit Dispatch ID of the kernel
  DISPATCH_ID = 2,
  /// 64-bit flat scratch base address of the wavefront
  FLAT_SCRATCH = 3,
  /// 32-bit private segment wave offset
  PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 4,
  /// 64-bit address of the dispatch packet of the kernel being executed
  DISPATCH_PTR = 5,
  /// 64-bit address of the HSA queue used to launch the kernel
  QUEUE_PTR = 6,
  /// Size of a work-item's private segment
  WORK_ITEM_PRIVATE_SEGMENT_SIZE = 7,
  /// 64-bit address of the instrumentation routine's argument buffer
  USER_ARG_PTR = 8,
  /// 32-bit offset of the instrumentation implicit argument buffer from the
  /// \c USER_ARG_PTR
  IMPLICIT_ARG_OFFSET = 9,
  /// Marks the last defined scalar value argument
  SCALAR_VALUE_ARGUMENT_LAST = IMPLICIT_ARG_OFFSET
};

template <ScalarValueArgument SA> struct ScalarValueArgumentInfo;

template <> struct ScalarValueArgumentInfo<WAVEFRONT_PRIVATE_SEGMENT_BUFFER> {
  static constexpr uint8_t NumLanes = 4;
  static constexpr auto NamedMD =
      "luthier.sva.wavefront_private_segment_buffer";
};

template <> struct ScalarValueArgumentInfo<KERNEL_ARG_PTR> {
  static constexpr uint8_t NumLanes = 2;
  static constexpr auto NamedMD = "luthier.sva.kernel_arg_ptr";
};

template <> struct ScalarValueArgumentInfo<DISPATCH_ID> {
  static constexpr uint8_t NumLanes = 2;
  static constexpr auto NamedMD = "luthier.sva.dispatch_id";
};

template <> struct ScalarValueArgumentInfo<FLAT_SCRATCH> {
  static constexpr uint8_t NumLanes = 2;
  static constexpr auto NamedMD = "luthier.sva.flat_scratch";
};

template <> struct ScalarValueArgumentInfo<PRIVATE_SEGMENT_WAVE_BYTE_OFFSET> {
  static constexpr uint8_t NumLanes = 1;
  static constexpr auto NamedMD =
      "luthier.sva.private_segment_wave_byte_offset";
};

template <> struct ScalarValueArgumentInfo<QUEUE_PTR> {
  static constexpr uint8_t NumLanes = 2;
  static constexpr auto NamedMD = "luthier.sva.queue_ptr";
};

template <> struct ScalarValueArgumentInfo<DISPATCH_PTR> {
  static constexpr uint8_t NumLanes = 1;
  static constexpr auto NamedMD = "luthier.sva.dispatch_ptr";
};

template <> struct ScalarValueArgumentInfo<WORK_ITEM_PRIVATE_SEGMENT_SIZE> {
  static constexpr uint8_t NumLanes = 1;
  static constexpr auto NamedMD = "luthier.sva.workitem_private_segment_size";
};

template <> struct ScalarValueArgumentInfo<USER_ARG_PTR> {
  static constexpr uint8_t NumLanes = 2;
  static constexpr auto NamedMD = "luthier.sva.user_arg_ptr";
};

template <> struct ScalarValueArgumentInfo<IMPLICIT_ARG_OFFSET> {
  static constexpr uint8_t NumLanes = 1;
  static constexpr auto NamedMD = "luthier.sva.implicit_arg_offset";
};

/// \brief Holds the result of the IR processing stage of an intrinsic IR call
/// instruction, including how all non-constant values used/defined by a Luthier
/// intrinsic use (i.e. its output and input arguments) must be lowered to
/// registers
struct IntrinsicIRLoweringInfo {
  /// \brief Contains information about the non-constant values used/defined by
  /// a \c llvm::CallInst to a Luthier Intrinsic, and its inline assembly
  /// constraint (e.g. 'v' for VGPR, 's' for SGPR, 'a' for AGPR)
  /// \details This struct is used to keep track of how an LLVM IR non-constant
  /// value used/defined by a \c llvm::CallInst to a Luthier Intrinsic should be
  /// mapped to a \c llvm::Register; For example, if value <tt>%1</tt> used by
  /// the IR call instruction
  /// \code
  /// %1 = tail call i32 @"luthier::myIntrinsic.i32"(i32 %0)
  /// \endcode
  /// needs to become an SGPR after ISEL passes are finished, <tt>%1</tt> will
  /// have an <tt>'s'</tt> \c Constraint \n
  struct ValueLoweringInfo {
    /// The IR value to be lowered; Must be non-const and can be null if there
    /// is no value to be lowered
    const llvm::Value *Val;
    /// The inline asm constraint describing how \c Val should be lowered
    std::string Constraint;
  };

private:
  /// How the output value (if present) must be lowered to a
  /// \c llvm::Register
  ValueLoweringInfo OutValue{nullptr, ""};
  /// How the argument values (if present) must be lowered to a
  /// \c llvm::Register
  llvm::SmallVector<ValueLoweringInfo, 4> Args{};
  /// A list of <tt>llvm::Constant</tt>s passed as extra information to the MIR
  /// lowering stage
  llvm::SmallVector<llvm::Constant *> ExtraInfoValues{};

public:
  /// Sets the inline asm constraint to \p Constraint for the given
  /// \p Val
  void setReturnValueInfo(const llvm::Value &Val, llvm::StringRef Constraint) {
    OutValue.Val = &Val;
    OutValue.Constraint = Constraint;
  }

  /// \returns the return value's \c IntrinsicValueLoweringInfo
  [[nodiscard]] const ValueLoweringInfo &getReturnValueInfo() const {
    return OutValue;
  }

  /// Adds a new argument, with \p Val and \p Constraint describing its
  /// \c IntrinsicValueLoweringInfo
  void addArgInfo(const llvm::Value &Val, llvm::StringRef Constraint) {
    Args.emplace_back(&Val, std::string(Constraint));
  }

  /// \returns All arguments' \c IntrinsicValueLoweringInfo
  llvm::ArrayRef<ValueLoweringInfo> getArgsInfo() const { return Args; }

  /// Adds \p Val as an extra value to be passed to the MIR lowering stage
  void addExtraLoweringValue(llvm::Constant &Val) {
    ExtraInfoValues.emplace_back(&Val);
  }

  /// \returns The list of all extra lowering constants
  llvm::ArrayRef<llvm::Constant *> getExtraLoweringValues() const {
    return ExtraInfoValues;
  }
};

/// \brief describes a function used by each Luthier intrinsic to process
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
/// <tt>llvm::MachineInstr</tt>s in place of its IR calls
/// \details The MIR processor takes in the extra constant info generated
/// by its \c IntrinsicIRProcessorFunc as well as the lowered registers and
/// their inline assembly flags for its used/defined values. Convenience Lambdas
/// for creating instructions at the place of emission, creating virtual
/// registers from register classes, accessing scalar values and physical
/// registers are also provided. A mapping between the physical registers from
/// the original kernel that need to be overwritten and their new virtual
/// register values can also be returned by the intrinsic MIR processor
typedef std::function<llvm::Error(
    const llvm::MachineFunction &,
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>>,
    llvm::ArrayRef<llvm::Constant *>,
    const std::function<llvm::MachineInstrBuilder(int)> &,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &,
    const std::function<llvm::Register(ScalarValueArgument)> &,
    const std::function<llvm::Register(llvm::MCRegister)> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &)>
    IntrinsicMIRProcessorFunc;

/// \brief Used internally to store the intrinsic processors
struct IntrinsicProcessor {
  IntrinsicIRProcessorFunc IRProcessor{};
  IntrinsicMIRProcessorFunc MIRProcessor{};
};

} // namespace luthier

#endif