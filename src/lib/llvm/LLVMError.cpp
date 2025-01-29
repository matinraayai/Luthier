//===-- LLVMError.cpp - Luthier LLVM Error Type ---------------------------===//
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
/// This file implements the \c luthier::LLVMError class.
//===----------------------------------------------------------------------===//
#include <llvm/Support/Signals.h>
#include <luthier/llvm/LLVMError.h>

namespace luthier {

char LLVMError::ID = 0;

void LLVMError::log(llvm::raw_ostream &OS) const {
  OS << "Call to LLVM library in file " << File << ", line: " << LineNumber
     << "encountered an error : " << OriginalErrorMsg << "\n";
  OS << "Stacktrace: \n" << StackTrace << "\n";
}

llvm::Error LLVMError::llvmErrorCheck(llvm::Error Expr, llvm::StringRef File,
                                      int LineNumber) {
  if (Expr) {
    std::string StackTrace;
    llvm::raw_string_ostream STStream(StackTrace);
    llvm::sys::PrintStackTrace(STStream);
    std::string OriginalErrorMessage = llvm::toStringWithoutConsuming(Expr);
    llvm::consumeError(std::move(Expr));
    return llvm::make_error<LLVMError>(File, LineNumber, StackTrace,
                                       OriginalErrorMessage);
  }
  return llvm::Error::success();
}

} // namespace luthier
