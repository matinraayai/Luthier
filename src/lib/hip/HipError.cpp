//===-- HipError.cpp - Luthier Comgr Error Type ---------------------------===//
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
/// This file implements the \c luthier::HipError class.
//===----------------------------------------------------------------------===//
#include "hip/HipRuntimeApiInterceptor.hpp"
#include <llvm/Support/Signals.h>
#include <luthier/hip/HipError.h>

namespace luthier {

char HipError::ID = 0;

llvm::Error HipError::HipErrorCheck(llvm::StringRef FileName, int LineNumber,
                                    hipError_t Expr, llvm::StringRef ExprStr,
                                    hipError_t Expected) {
  if (Expr != Expected) {
    std::string StackTrace;
    llvm::raw_string_ostream STStream(StackTrace);
    llvm::sys::PrintStackTrace(STStream);
    return llvm::make_error<HipError>(FileName, LineNumber, StackTrace, Expr,
                                      ExprStr);
  }
  return llvm::Error::success();
}

void luthier::HipError::log(llvm::raw_ostream &OS) const {
  OS << "HIP error encountered in file " << File << ", line: " << LineNumber
     << ": ";
  OS << "HIP call in expression " << Expression << " failed with error code ";
  OS << Error;
  if (hip::HipRuntimeApiInterceptor::isInitialized()) {
    // Try to get the Error name if the hip runtime interceptor is
    // initialized.
    const auto &DispatchAPITable =
        hip::HipRuntimeApiInterceptor::instance().getSavedApiTableContainer();
    OS << ", ";
    const char *ErrorName = DispatchAPITable.hipGetErrorName_fn(Error);
    if (ErrorName != nullptr) {
      OS << ErrorName << ".\n";
    } else {
      OS << "Unknown Error.\n";
    }
  } else {
    OS << ".\n";
  }
  OS << "Stacktrace: \n" << StackTrace << "\n";
}

} // namespace luthier