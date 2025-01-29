//===-- ComgrError.cpp - Luthier Comgr Error Type -------------------------===//
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
/// This file implements the \c luthier::ComgrError class.
//===----------------------------------------------------------------------===//
#include <llvm/Support/Signals.h>
#include <luthier/comgr/ComgrError.h>

namespace luthier {

char ComgrError::ID = 0;

llvm::Error ComgrError::ComgrErrorCheck(llvm::StringRef FileName,
                                        int LineNumber, amd_comgr_status_t Expr,
                                        llvm::StringRef ExprStr,
                                        const amd_comgr_status_t Expected) {
  if (Expr != Expected) {
    std::string StackTrace;
    llvm::raw_string_ostream STStream(StackTrace);
    llvm::sys::PrintStackTrace(STStream);
    return llvm::make_error<ComgrError>(FileName, LineNumber, StackTrace, Expr,
                                        ExprStr);
  }
  return llvm::Error::success();
}

void ComgrError::log(llvm::raw_ostream &OS) const {
  const char *ErrorMsg;
  amd_comgr_status_t Status = amd_comgr_status_string(Error, &ErrorMsg);
  OS << "COMGR error encountered in file " << File << ", line: " << LineNumber
     << ": ";
  OS << "COMGR call in expression " << Expression << " failed with error code "
     << Error << ". ";
  if (Status == AMD_COMGR_STATUS_SUCCESS) {
    OS << "More info about the error according to COMGR: ";
    OS << ErrorMsg << ".\n";
  } else {
    OS << "Failed to get additional info regarding the error from comgr.\n";
  }
  OS << "Stacktrace: \n" << StackTrace << "\n";
}

} // namespace luthier