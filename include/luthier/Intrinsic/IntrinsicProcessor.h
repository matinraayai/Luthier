//===-- IntrinsicProcessor.h ------------------------------------*- C++ -*-===//
//
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
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/Error.h>
#include <string>

namespace llvm {
class Register;

class Value;

class Function;

class CallInst;

class GCNTargetMachine;

class MachineInstrBuilder;
} // namespace llvm

namespace luthier {

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

public:

  /// \param Name the name of the intrinsic being lowered
  /// \note this function is called internally by Luthier on the result of
  /// \c IntrinsicIRProcessorFunc is returned; Hence setting the name of the
  /// intrinsic inside the IR processor has no effect
  void setIntrinsicName(llvm::StringRef Name) {
    this->IntrinsicName = Name;
  }

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
};

/// \brief describes a function type used by each Luthier intrinsic to process
/// its uses in LLVM IR, and return a \c IntrinsicIRLoweringInfo which will describe
/// how its use/def values will be lowered to <tt>llvm::MachineOperand</tt>s,
/// as well as any arbitrary information required to be passed down from the
/// IR processing stage to the MIR processing stage
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
    const std::function<llvm::MachineInstrBuilder(int)> &)>
    IntrinsicMIRProcessorFunc;

/// \brief Used internally by \c luthier::CodeGenerator to keep track of
/// registered intrinsics and how to process them
struct IntrinsicProcessor {
  IntrinsicIRProcessorFunc IRProcessor{};
  IntrinsicMIRProcessorFunc MIRProcessor{};
};

} // namespace luthier

#endif