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
#ifndef LUTHIER_COMPILER_PLUGINS_INTRINSIC_CALLS_H
#define LUTHIER_COMPILER_PLUGINS_INTRINSIC_CALLS_H
#include "LuthierConsts.h"
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

} // namespace luthier

#endif