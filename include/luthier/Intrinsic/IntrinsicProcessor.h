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

#include "luthier/types.h"
#include <functional>
#include <llvm/ADT/Any.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>
#include <string>

namespace llvm {

class MachineFunction;

class MachineInstr;

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
  /// Kernel's private segment buffer
  KERNEL_PRIVATE_SEGMENT_BUFFER = 0,
  /// Enum marking the beginning of kernel arguments always passed on SGPRs
  ALWAYS_IN_SGPR_BEGIN = KERNEL_PRIVATE_SEGMENT_BUFFER,
  /// 64-bit address of the kernel's argument buffer
  KERNARG_SEGMENT_PTR = 1,
  /// 32-bit offset from the beginning of the kernel's argument buffer where
  /// the kernel's hidden arguments starts
  HIDDEN_KERNARG_OFFSET = 2,
  /// 32-bit offset from the beginning of the kernel's argument buffer where
  /// the instrumentation-passed (i.e. user) argument buffer starts
  USER_KERNARG_OFFSET = 3,
  /// 64-bit Dispatch ID of the kernel
  DISPATCH_ID = 4,
  /// 64-bit flat scratch base address of the wavefront
  FLAT_SCRATCH = 5,
  /// 32-bit private segment wave offset
  PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 6,
  /// Enum marking the end of kernel arguments always passed on SGPRs
  ALWAYS_IN_SGPR_END = PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
  /// 64-bit address of the dispatch packet of the kernel being executed
  DISPATCH_PTR = 7,
  /// Enum marking the beginning of kernel arguments that can either be passed
  /// on SGPRs or hidden kernel arguments
  EITHER_IN_SGPR_OR_HIDDEN_BEGIN = DISPATCH_PTR,
  /// 64-bit address of the HSA queue used to launch the kernel
  QUEUE_PTR = 8,
  ///
  PRIVATE_SEGMENT_SIZE = 9,
  EITHER_IN_SGPR_OR_HIDDEN_END = PRIVATE_SEGMENT_SIZE,
  GLOBAL_OFFSET_X = 10,
  HIDDEN_BEGIN = GLOBAL_OFFSET_X,
  GLOBAL_OFFSET_Y = 11,
  GLOBAL_OFFSET_Z = 12,
  PRINT_BUFFER = 13,
  HOSTCALL_BUFFER = 14,
  DEFAULT_QUEUE = 15,
  COMPLETION_ACTION = 16,
  MULTIGRID_SYNC = 17,
  BLOCK_COUNT_X = 18,
  BLOCK_COUNT_Y = 18,
  BLOCK_COUNT_Z = 19,
  GROUP_SIZE_X = 20,
  GROUP_SIZE_Y = 21,
  GROUP_SIZE_Z = 22,
  REMAINDER_X = 23,
  REMAINDER_Y = 24,
  REMAINDER_Z = 25,
  GRID_DIMS = 26,
  HEAP_V1 = 27,
  DYNAMIC_LDS_SIZE = 28,
  PRIVATE_BASE = 29,
  SHARED_BASE = 30,
  HIDDEN_END = SHARED_BASE,
  WORK_ITEM_X = 31,
  WORK_ITEM_Y = 32,
  WORK_ITEM_Z = 33
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
  std::string IntrinsicName{};
  /// The inline assembly that serves as a place holder for the intrinsic
  /// until after instruction selection; Used by \c luthier::CodeGenerator
  const llvm::InlineAsm *PlaceHolderInlineAsm{nullptr};
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
  llvm::SmallDenseSet<llvm::MCRegister, 4> AccessedPhysicalRegisters{};

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

  /// Sets the inline assembly placeholder instruction
  void setPlaceHolderInlineAsm(llvm::InlineAsm &IA) {
    this->PlaceHolderInlineAsm = &IA;
  }

  /// Gets the inline assembly placeholder instruction
  [[nodiscard]] const llvm::InlineAsm &getPlaceHolderInlineAsm() const {
    return *this->PlaceHolderInlineAsm;
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

  /// Iterators/Query functions for the physical registers accessed by the
  /// intrinsic

  using const_accessed_phys_regs_iterator =
      decltype(AccessedPhysicalRegisters)::ConstIterator;

  [[nodiscard]] llvm::iterator_range<const_accessed_phys_regs_iterator>
  accessed_phys_regs() const {
    return llvm::make_range(accessed_phys_regs_begin(),
                            accessed_phys_regs_end());
  }

  [[nodiscard]] const_accessed_phys_regs_iterator
  accessed_phys_regs_begin() const {
    return AccessedPhysicalRegisters.begin();
  }

  [[nodiscard]] const_accessed_phys_regs_iterator
  accessed_phys_regs_end() const {
    return AccessedPhysicalRegisters.end();
  }

  [[nodiscard]] bool accessed_phys_regs_empty() const {
    return AccessedPhysicalRegisters.empty();
  }

  [[nodiscard]] size_t accessed_phys_regs_size() const {
    return AccessedPhysicalRegisters.size();
  }

  /// Asks the code generator to ensure access to the \p KernArg during
  /// the MIR lowering stage
  void requestAccessToKernelArgument(KernelArgumentType KernArg) {
    AccessedKernelArguments.insert(KernArg);
  }

  /// Iterators/Query functions for the kernel arguments accessed by the
  /// intrinsic

  using const_accessed_kernargs_iterator =
      decltype(AccessedKernelArguments)::ConstIterator;

  [[nodiscard]] const_accessed_kernargs_iterator
  accessed_kernargs_begin() const {
    return AccessedKernelArguments.begin();
  }

  [[nodiscard]] const_accessed_kernargs_iterator accessed_kernargs_end() const {
    return AccessedKernelArguments.end();
  }

  [[nodiscard]] bool accessed_kernargs_empty() const {
    return AccessedKernelArguments.empty();
  }

  [[nodiscard]] size_t accessed_kernargs_size() const {
    return AccessedKernelArguments.size();
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

/// If the passed MI is an inline assembly instruction and a place holder
/// for a Luthier intrinsic, returns the unique index associated with it
/// \param MI the \c llvm::MachineInstr being inspected
/// \return the unique index in the MI's inline assembly string, -1 if
/// \p MI is not an inline assembly or its inline assembly string is empty,
/// or an \c llvm::Error if its assembly string fails to convert to an
/// unsigned int
llvm::Expected<unsigned int>
getIntrinsicInlineAsmPlaceHolderIdx(const llvm::MachineInstr &MI);

/// Builds a \c llvm::CallInst invoking the intrinsic indicated by
/// \p IntrinsicName at the instruction position indicated by the \p Builder
/// with the given \p ReturnType and \p Args
/// \tparam IArgs Arguments passed to the intrinsic; Can be either a scalar
/// or a reference to a \c llvm::Value
/// \param M the instrumentation module where the intrinsic will be inserted to
/// \param Builder the instruction builder used to build the call instruction
/// \param IntrinsicName the name of the intrinsic
/// \param ReturnType the return type of the intrinsic call instruction
/// \param Args the arguments to the intrinsic function
/// \return a \c llvm::CallInst to the intrinsic function
template <typename... IArgs>
llvm::CallInst *insertCallToIntrinsic(llvm::Module &M,
                                      llvm::IRBuilderBase &Builder,
                                      llvm::StringRef IntrinsicName,
                                      llvm::Type &ReturnType, IArgs... Args) {
  auto &LLVMContext = Builder.getContext();
  /// Construct the intrinsic's LLVM function type and its argument value
  /// list
  llvm::SmallVector<llvm::Type *> IntrinsicArgTypes;
  llvm::SmallVector<llvm::Value *> IntrinsicArgValues;
  for (auto Arg : {Args...}) {
    // If Arg is a scalar, create the appropriate LLVM Constant value for it
    // and add it to the argument list
    if constexpr (std::is_scalar_v<decltype(Arg)>) {
      auto *ArgType = llvm::Type::getScalarTy<decltype(Arg)>(LLVMContext);
      IntrinsicArgTypes.push_back(ArgType);
      if constexpr (std::is_integral_v<decltype(Arg)>) {
        IntrinsicArgValues.push_back(llvm::ConstantInt::get(
            ArgType, Arg, std::is_signed_v<decltype(Arg)>));
      } else {
        IntrinsicArgValues.push_back(llvm::ConstantFP::get(ArgType, Arg));
      }
      // Otherwise if it's a value, then get its type and add it to the
      // argument list
    } else {
      IntrinsicArgTypes.push_back(Arg.getType());
      IntrinsicArgValues.push_back(&Arg);
    }
  }
  auto *IntrinsicFuncType =
      llvm::FunctionType::get(&ReturnType, IntrinsicArgTypes, false);
  // Format the readReg intrinsic function name
  std::string FormattedIntrinsicName{IntrinsicName};
  llvm::raw_string_ostream IntrinsicNameOS(FormattedIntrinsicName);
  // Format the intrinsic function name
  IntrinsicNameOS << ".";
  IntrinsicFuncType->getReturnType()->print(IntrinsicNameOS);
  // Format the input types
  IntrinsicNameOS << ".";
  for (const auto &[i, InputType] : llvm::enumerate(IntrinsicArgTypes)) {
    InputType->print(IntrinsicNameOS);
    if (i != IntrinsicArgTypes.size() - 1) {
      IntrinsicNameOS << ".";
    }
  }
  // Create the intrinsic function in the module, or get it if it already
  // exists
  auto ReadRegFunc = M.getOrInsertFunction(
      FormattedIntrinsicName, IntrinsicFuncType,
      llvm::AttributeList().addFnAttribute(
          LLVMContext, LUTHIER_INTRINSIC_ATTRIBUTE, IntrinsicName));

  return Builder.CreateCall(ReadRegFunc, IntrinsicArgValues);
}

} // namespace luthier

#endif