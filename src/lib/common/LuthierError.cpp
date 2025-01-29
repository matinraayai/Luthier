//===-- LuthierError.cpp - Luthier Error Type -----------------------------===//
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
/// This file implements the \c luthier::LuthierError class.
//===----------------------------------------------------------------------===//
#include <luthier/common/LuthierError.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace luthier {

char LuthierError::ID = 0;

llvm::Error LuthierError::luthierErrorCheck(const bool Expr,
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

void LuthierError::log(llvm::raw_ostream &OS) const {
  OS << "Luthier error encountered in file " << File << ", line: " << LineNumber
     << ": ";
  OS << ErrorMsg << "\n";
  OS << "Stacktrace: \n" << StackTrace << "\n";
}

} // namespace luthier
