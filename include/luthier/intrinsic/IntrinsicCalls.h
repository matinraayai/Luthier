//===-- IntrinsicCalls.h ----------------------------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// This file defines methods used to insert calls to Luthier intrinsics.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INTRINSIC_CALLS_H
#define LUTHIER_INTRINSIC_CALLS_H
#include "luthier/consts.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace luthier {

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
llvm::CallInst *insertCallToIntrinsic(llvm::Module &M,
                                      llvm::IRBuilderBase &Builder,
                                      llvm::StringRef IntrinsicName,
                                      llvm::Type &ReturnType) {
  auto &LLVMContext = Builder.getContext();
  /// Construct the intrinsic's LLVM function type and its argument value
  /// list
  auto *IntrinsicFuncType = llvm::FunctionType::get(&ReturnType, false);
  // Format the readReg intrinsic function name
  std::string FormattedIntrinsicName{IntrinsicName};
  llvm::raw_string_ostream IntrinsicNameOS(FormattedIntrinsicName);
  // Format the intrinsic function name
  IntrinsicNameOS << ".";
  IntrinsicFuncType->getReturnType()->print(IntrinsicNameOS);
  // Create the intrinsic function in the module, or get it if it already
  // exists
  auto ReadRegFunc = M.getOrInsertFunction(
      FormattedIntrinsicName, IntrinsicFuncType,
      llvm::AttributeList().addFnAttribute(LLVMContext, IntrinsicAttribute,
                                           IntrinsicName));

  return Builder.CreateCall(ReadRegFunc);
}

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
      llvm::AttributeList().addFnAttribute(LLVMContext, IntrinsicAttribute,
                                           IntrinsicName));

  return Builder.CreateCall(ReadRegFunc, IntrinsicArgValues);
}

} // namespace luthier

#endif