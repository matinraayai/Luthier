//===-- Error.cpp - Luthier LLVM Error Types ------------------------------===//
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
/// This file implements different types of \c llvm::Error used by Luthier.
//===----------------------------------------------------------------------===//
#include <common/Error.hpp>

char luthier::LuthierError::ID = 0;

llvm::Error luthier::LuthierError::luthierErrorCheck(const bool Expr,
                                                     llvm::StringRef File,
                                                     int LineNumber,
                                                     llvm::StringRef ErrorMsg) {
  if (!Expr) {
    std::string StackTrace;
    llvm::raw_string_ostream STStream(StackTrace);
    llvm::sys::PrintStackTrace(STStream);
    return llvm::make_error<LuthierError>(File, LineNumber, StackTrace,
                                          ErrorMsg);
  }
  return llvm::Error::success();
}

char luthier::ComgrError::ID = 0;

llvm::Error luthier::ComgrError::comgrErrorCheck(
    llvm::StringRef FileName, int LineNumber, amd_comgr_status_t Expr,
    llvm::StringRef ExprStr, const amd_comgr_status_t Expected) {
  if (Expr != Expected) {
    std::string StackTrace;
    llvm::raw_string_ostream STStream(StackTrace);
    llvm::sys::PrintStackTrace(STStream);
    return llvm::make_error<ComgrError>(FileName, LineNumber, StackTrace, Expr,
                                        ExprStr);
  }
  return llvm::Error::success();
}

char luthier::HsaError::ID = 0;

llvm::Error luthier::HsaError::hsaErrorCheck(llvm::StringRef FileName,
                                             int LineNumber, hsa_status_t Expr,
                                             llvm::StringRef ExprStr,
                                             hsa_status_t Expected) {
  std::string StackTrace;
  llvm::raw_string_ostream STStream(StackTrace);
  llvm::sys::PrintStackTrace(STStream);
  if (Expr != Expected) {
    return llvm::make_error<HsaError>(FileName, LineNumber, StackTrace, Expr,
                                      ExprStr);
  }
  return llvm::Error::success();
}
