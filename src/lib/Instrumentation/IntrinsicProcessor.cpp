//===-- IntrinsicProcessor.cpp --------------------------------------------===//
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
/// Implements part of Luthier's Intrinsic Processor functions not already
/// defined by its header.
//===----------------------------------------------------------------------===//
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/Support/Error.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/Instrumentation/IntrinsicProcessor.h>

namespace luthier {

llvm::Expected<unsigned int>
getIntrinsicInlineAsmPlaceHolderIdx(const llvm::MachineInstr &MI) {
  if (MI.isInlineAsm()) {
    auto IntrinsicIdxAsString =
        MI.getOperand(llvm::InlineAsm::MIOp_AsmString).getSymbolName();
    // Empty inline assembly instructions can be skipped
    if (llvm::StringRef(IntrinsicIdxAsString).empty())
      return -1;
    try {
      // std::stoul throws exceptions and needs to be converted
      unsigned int IntrinsicIdx = std::stoul(IntrinsicIdxAsString);
      return IntrinsicIdx;
    } catch (const std::exception &Exception) {
      return llvm::make_error<GenericLuthierError>(
          "Caught an exception when getting the intrinsic index of the "
          "inline assembly instruction "
          "{0}. The exception: {1}.",
          IntrinsicIdxAsString, Exception.what());
    }
  } else {
    return -1;
  }
}

} // namespace luthier