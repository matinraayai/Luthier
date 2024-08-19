//===-- Error.cpp - Luthier LLVM Error Types ------------------------------===//
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
/// This file implements different types of \c llvm::Error used by Luthier.
//===----------------------------------------------------------------------===//
#include <common/Error.hpp>

char luthier::InvalidArgument::ID = 0;

llvm::Error luthier::InvalidArgument::invalidArgumentCheck(
    llvm::StringRef FileName, int LineNumber, llvm::StringRef FunctionName,
    bool Expr, llvm::StringRef ExprStr) {
  return (!Expr) ? llvm::make_error<InvalidArgument>(FileName, LineNumber,
                                                     FunctionName, ExprStr)
                 : llvm::Error::success();
}

char luthier::AssertionError::ID = 0;

llvm::Error luthier::AssertionError::assertionCheck(llvm::StringRef FileName,
                                                    int LineNumber, bool Expr,
                                                    llvm::StringRef ExprStr) {
  return (!Expr)
             ? llvm::make_error<AssertionError>(FileName, LineNumber, ExprStr)
             : llvm::Error::success();
}

char luthier::HsaError::ID = 0;

llvm::Error luthier::HsaError::hsaErrorCheck(llvm::StringRef FileName,
                                             int LineNumber, hsa_status_t Expr,
                                             llvm::StringRef ExprStr,
                                             hsa_status_t Expected) {
  return (Expr != Expected)
             ? llvm::make_error<HsaError>(FileName, LineNumber, Expr, ExprStr)
             : llvm::Error::success();
}

char luthier::ComgrError::ID = 0;

llvm::Error luthier::ComgrError::comgrErrorCheck(llvm::StringRef FileName,
                                                 int LineNumber,
                                                 amd_comgr_status_t Expr,
                                                 llvm::StringRef ExprStr,
                                                 amd_comgr_status_t Expected) {
  return (Expr != Expected)
             ? llvm::make_error<ComgrError>(FileName, LineNumber, Expr, ExprStr)
             : llvm::Error::success();
}