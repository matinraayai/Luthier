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
#include "luthier/Intrinsic/ScalarValueArgument.h"
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