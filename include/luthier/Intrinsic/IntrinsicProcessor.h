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

/// Module-level named-MDNode holding the lookup table from a Luthier
/// intrinsic placeholder's opaque key (used as the InlineAsm template
/// string so it survives SelectionDAG ISel as the first
/// \c MachineOperand of the resulting \c INLINEASM \c MachineInstr) to its
/// intrinsic name and forwarded aux metadata. Each operand is a 3-element
/// \c MDNode of the form
/// \code !{!"<key>", !"<intrinsic_name>", <aux MDNode or empty MDNode>}
/// \endcode
static constexpr llvm::StringLiteral LuthierIntrinsicNamedMDName{
    "luthier.intrinsic.placeholders"};

/// Opaque-key prefix used as the InlineAsm template string of every Luthier
/// intrinsic placeholder; the suffix is a monotonic per-call-signature
/// counter assigned by \c ProcessIntrinsicsAtIRLevelPass. Two calls whose
/// (intrinsic name, return type, argument types, constant argument values,
/// aux MDNode contents) match share a key.
static constexpr llvm::StringLiteral LuthierIntrinsicPlaceholderKeyPrefix{
    "luthier.placeholder."};

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
  /// 32-bit workgroup id (blockIdx) X/Y/Z. On non-architected-SGPR targets
  /// these
  /// are the preloaded workgroup-id system SGPRs captured into the SVA at entry
  /// (the app may clobber them); on architected-SGPR targets they come from
  /// TTMP registers instead and are read directly (no SVA capture).
  WORKGROUP_ID_X = 10,
  WORKGROUP_ID_Y = 11,
  WORKGROUP_ID_Z = 12,
  /// The wave's lane-0 work-item id, packed as x|y<<10|z<<20. Captured at entry
  /// (via v_readlane of the preloaded work-item-id VGPR) so the per-lane
  /// work-item id (threadIdx) can be recomputed at any injection point as
  /// <tt>decompose(packed_lane0_flat + mbcnt)</tt>.
  WORKITEM_ID_PACKED_LANE0 = 13,
  /// Marks the last defined scalar value argument
  SCALAR_VALUE_ARGUMENT_LAST = WORKITEM_ID_PACKED_LANE0
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

template <> struct ScalarValueArgumentInfo<WORKGROUP_ID_X> {
  static constexpr uint8_t NumLanes = 1;
  static constexpr auto NamedMD = "luthier.sva.workgroup_id_x";
};

template <> struct ScalarValueArgumentInfo<WORKGROUP_ID_Y> {
  static constexpr uint8_t NumLanes = 1;
  static constexpr auto NamedMD = "luthier.sva.workgroup_id_y";
};

template <> struct ScalarValueArgumentInfo<WORKGROUP_ID_Z> {
  static constexpr uint8_t NumLanes = 1;
  static constexpr auto NamedMD = "luthier.sva.workgroup_id_z";
};

template <> struct ScalarValueArgumentInfo<WORKITEM_ID_PACKED_LANE0> {
  static constexpr uint8_t NumLanes = 1;
  static constexpr auto NamedMD = "luthier.sva.workitem_id_packed_lane0";
};

/// \brief Describes the ISA-state effects of a single Luthier intrinsic at
/// the placeholder layer (after the IR processing stage has replaced each
/// intrinsic call with an inline-asm placeholder). Populated by each
/// intrinsic's \c IRProcessor and serialized by
/// \c ProcessIntrinsicsAtIRLevelPass into the
/// \c !luthier.intrinsic.placeholders named-MD side channel
///
/// \c ForwardISAStateToCalleesPass uses this information to compute, per
/// callee Function, the union of SVA scalar args read, phys-regs read, and
/// phys-regs written transitively, and extends the callee's signature
/// accordingly
struct IntrinsicISAStateEffects {
  /// Scalar value arguments this intrinsic reads.
  llvm::SmallVector<ScalarValueArgument, 1> ReadSVAs;
  /// Physical registers this intrinsic reads. Wide registers are allowed;
  /// downstream consumers decompose into 32-bit channels via TRI.
  llvm::SmallVector<llvm::MCRegister, 1> ReadPhysRegs;
  /// Physical registers this intrinsic writes. Same channel-decomposition
  /// note as above.
  llvm::SmallVector<llvm::MCRegister, 1> WrittenPhysRegs;
};

/// Decode an effects MDNode produced by \c ProcessIntrinsicsAtIRLevelPass
/// (shape: empty MDNode or 3-operand
/// \c !{!{sva-i32s}, !{read-physreg-i32s}, !{written-physreg-i32s}}).
/// An empty / malformed node decodes to a record with all three vectors
/// empty (i.e. "no callee-visible ISA-state effects").
IntrinsicISAStateEffects
decodeIntrinsicISAStateEffects(const llvm::MDNode *EffNode);

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
  /// Metadata values forwarded to the MIR lowering stage as a payload MDNode
  llvm::SmallVector<llvm::Metadata *> ExtraInfoValues{};
  /// ISA-state effects produced by this intrinsic. Populated by the IR
  /// processor; consumed by \c ForwardISAStateToCalleesPass via the
  /// serialized form in \c !luthier.intrinsic.placeholders .
  IntrinsicISAStateEffects Effects{};

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

  /// Adds \p Val as an extra metadata value to be forwarded to the MIR
  /// lowering stage
  void addExtraLoweringValue(llvm::Metadata &Val) {
    ExtraInfoValues.emplace_back(&Val);
  }

  /// Convenience overload: wraps \p Val in \c ConstantAsMetadata before
  /// forwarding it to the MIR lowering stage
  void addExtraLoweringValue(llvm::Constant &Val) {
    ExtraInfoValues.emplace_back(llvm::ConstantAsMetadata::get(&Val));
  }

  /// \returns The list of all extra lowering metadata values
  llvm::ArrayRef<llvm::Metadata *> getExtraLoweringValues() const {
    return ExtraInfoValues;
  }

  /// Direct access to the effects record so processors can populate it
  /// inline with their existing \c setReturnValueInfo / \c addArgInfo /
  /// \c addExtraLoweringValue calls.
  IntrinsicISAStateEffects &getEffects() { return Effects; }

  [[nodiscard]] const IntrinsicISAStateEffects &getEffects() const {
    return Effects;
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
    llvm::ArrayRef<std::pair<llvm::InlineAsm::Flag, llvm::Register>>,
    llvm::MDNode *, const std::function<llvm::MachineInstrBuilder(int)> &,
    const std::function<llvm::Register(const llvm::TargetRegisterClass *)> &,
    const llvm::DenseMap<ScalarValueArgument, llvm::Register> &,
    const llvm::DenseMap<llvm::MCRegister, llvm::Register> &,
    llvm::DenseMap<llvm::MCRegister, llvm::Register> &)>
    IntrinsicMIRProcessorFunc;

/// \brief Used internally to store the intrinsic processors
struct IntrinsicProcessor {
  IntrinsicIRProcessorFunc IRProcessor{};
  IntrinsicMIRProcessorFunc MIRProcessor{};
};

} // namespace luthier

#endif