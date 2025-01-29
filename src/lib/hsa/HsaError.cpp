//===-- HsaError.cpp - Luthier HSA Error Type -----------------------------===//
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
/// This file implements the \c luthier::HsaError class.
//===----------------------------------------------------------------------===//
#include "hsa/HsaRuntimeInterceptor.hpp"
#include <llvm/Support/Signals.h>
#include <luthier/hsa/HsaError.h>

namespace luthier {

char HsaError::ID = 0;

llvm::Error HsaError::HsaErrorCheck(llvm::StringRef FileName, int LineNumber,
                                    hsa_status_t Expr, llvm::StringRef ExprStr,
                                    hsa_status_t Expected) {
  if (Expr != Expected) {
    std::string StackTrace;
    llvm::raw_string_ostream STStream(StackTrace);
    llvm::sys::PrintStackTrace(STStream);
    return llvm::make_error<HsaError>(FileName, LineNumber, StackTrace, Expr,
                                      ExprStr);
  }
  return llvm::Error::success();
}

void luthier::HsaError::log(llvm::raw_ostream &OS) const {
  OS << "HSA error encountered in file " << File << ", line: " << LineNumber
     << ": ";
  OS << "HSA call in expression " << Expression << " failed with error code ";
  OS << Error;
  if (hsa::HsaRuntimeInterceptor::isInitialized()) {
    // Try to get the Error name if the hsa runtime interceptor is
    // initialized.
    const auto &DispatchAPITable =
        hsa::HsaRuntimeInterceptor::instance().getSavedApiTableContainer();
    OS << ", ";
    const char *ErrorName;
    hsa_status_t Status =
        DispatchAPITable.core.hsa_status_string_fn(Error, &ErrorName);
    if (Status == HSA_STATUS_SUCCESS) {
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
