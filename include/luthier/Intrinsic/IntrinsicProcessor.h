//===-- IntrinsicProcessor.h ------------------------------------*- C++ -*-===//
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
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Metadata.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>
#include <string>

namespace llvm {

class MachineFunction;

class MachineInstr;

class MachineOperand;

class TargetRegisterClass;

class Register;

class Value;

class Function;

class CallInst;

class GCNTargetMachine;

class MachineInstrBuilder;

class Metadata;

class MDNode;
} // namespace llvm

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
  /// 64-bit address of the kernel's argument buffer
  KERNEL_ARG_PTR = 1,
  /// 64-bit Dispatch ID of the kernel
  DISPATCH_ID = 2,
  /// 64-bit flat scratch base address of the wavefront
  FLAT_SCRATCH = 3,
  /// 64-bit address of the dispatch packet of the kernel being executed
  /// TODO: If the original kernel wants the dispatch pointer, we need to
  /// make it point to the **"original"** packet not the instrumented one
  DISPATCH_PTR = 4,
  /// 64-bit address of the HSA queue used to launch the kernel
  QUEUE_PTR = 5,
  /// Size of a work-item's private segment
  WORK_ITEM_PRIVATE_SEGMENT_SIZE = 6,
  /// 64-bit address of the instrumentation implicit argument buffer
  IMPLICIT_ARG_BUFFER = 7,
  /// 32-bit X component of the workgroup ID (preloaded system SGPR)
  WORKGROUP_ID_X = 8,
  /// 32-bit Y component of the workgroup ID (preloaded system SGPR)
  WORKGROUP_ID_Y = 9,
  /// 32-bit Z component of the workgroup ID (preloaded system SGPR)
  WORKGROUP_ID_Z = 10,
  /// 32-bit X component of lane 0's workitem ID at kernel entry
  WORKITEM_ID_X = 11,
  /// 32-bit Y component of lane 0's workitem ID at kernel entry
  WORKITEM_ID_Y = 12,
  /// 32-bit Z component of lane 0's workitem ID at kernel entry
  WORKITEM_ID_Z = 13,
  /// Marks the last defined scalar value argument
  SCALAR_VALUE_ARGUMENT_LAST = WORKITEM_ID_Z
  // TODO: Optimize usage of these lanes for GFX10; We are at the limit
  // /// 64-bit address of the instrumentation routine's argument buffer
  // USER_ARG_PTR = 15,
  //   /// 32-bit private segment wave offset
  // PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 4,
};

template <ScalarValueArgument SA> struct ScalarValueArgumentInfo;

template <> struct ScalarValueArgumentInfo<WAVEFRONT_PRIVATE_SEGMENT_BUFFER> {
  static constexpr uint8_t NumLanes = 4;
};

template <> struct ScalarValueArgumentInfo<KERNEL_ARG_PTR> {
  static constexpr uint8_t NumLanes = 2;
};

template <> struct ScalarValueArgumentInfo<DISPATCH_ID> {
  static constexpr uint8_t NumLanes = 2;
};

template <> struct ScalarValueArgumentInfo<FLAT_SCRATCH> {
  static constexpr uint8_t NumLanes = 2;
};

template <> struct ScalarValueArgumentInfo<PRIVATE_SEGMENT_WAVE_BYTE_OFFSET> {
  static constexpr uint8_t NumLanes = 1;
};

template <> struct ScalarValueArgumentInfo<QUEUE_PTR> {
  static constexpr uint8_t NumLanes = 2;
};

template <> struct ScalarValueArgumentInfo<DISPATCH_PTR> {
  static constexpr uint8_t NumLanes = 2;
};

template <> struct ScalarValueArgumentInfo<WORK_ITEM_PRIVATE_SEGMENT_SIZE> {
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
/// <tt>llvm::MachineInstr</tt>s in place of its IR calls.
///
/// \details Because each intrinsic declares its ISA-state effects up front
/// (via \c IntrinsicIRLoweringInfo::Effects ), the driver pre-stages every
/// scalar-arg and phys-reg value the processor might need and hands them in
/// as ready-to-use virtual registers — no per-intrinsic callback dispatch
/// required.
///
/// Parameters, in order:
///  - \c MF : the enclosing MachineFunction (for queries on TRI/MRI/subtarget).
///  - \c Args : the placeholder's inline-asm operands (flag + vreg pairs).
///  - \c Aux : the placeholder's aux MDNode (\c SA enum, MCRegister id, etc.,
///    interpreted per-intrinsic).
///  - \c MIBuilder : creates a \c MachineInstrBuilder at the placeholder's
///    program point. The processor uses it to emit the MIs that replace the
///    placeholder.
///  - \c VirtRegBuilder : allocates a fresh virtual register of a requested
///    \c TargetRegisterClass for processor-internal intermediates.
///  - \c SVAVRegs : map from each \c ScalarValueArgument the IR processor
///    declared in \c Effects.ReadSVAs to a virtual register holding that
///    SA's value at the placeholder's program point. The register is the
///    SA's natural width (single SGPR for 1-lane SAs; REG_SEQUENCE'd wide
///    SGPR for multi-lane SAs).
///  - \c ReadPhysRegVRegs : map from each 32-bit channel that the IR
///    processor's \c Effects.ReadPhysRegs decomposes into, to a virtual
///    register tracking that channel's current value at the placeholder's
///    program point (sourced via the driver's per-channel SSAUpdater).
///  - \c WritePhysRegSlots : output map. For each 32-bit channel declared
///    in \c Effects.WrittenPhysRegs the processor inserts an entry
///    \c {channel, vreg-holding-new-value}. The driver records the new
///    value with the per-channel SSAUpdater after the processor returns,
///    so subsequent reads of that channel see it and the return-block
///    restore COPYs back the right value.
typedef std::function<llvm::Error(
    const llvm::MachineFunction &,
    llvm::ArrayRef<
        std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>,
    const std::function<llvm::MachineInstrBuilder(int)> &,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &)>
    IntrinsicMIRProcessorFunc;

/// \brief Used internally to store the intrinsic processors
struct IntrinsicProcessor {
  IntrinsicIRProcessorFunc IRProcessor{};
  IntrinsicMIRProcessorFunc MIRProcessor{};
};

} // namespace luthier

#endif