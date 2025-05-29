//===-- GenericLuthierError.cpp -------------------------------------------===//
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
/// Implements the GenericLuthierError class.
//===----------------------------------------------------------------------===//
#include <luthier/common/GenericLuthierError.h>

namespace luthier {

char GenericLuthierError::ID = 0;

llvm::Error GenericLuthierError::luthierErrorCheck(const bool Expr,
                                                   std::string File,
                                                   int LineNumber,
                                                   std::string ErrorMsg) {
  if (!Expr) {
    std::string StackTrace;
    llvm::raw_string_ostream STStream(StackTrace);
    llvm::sys::PrintStackTrace(STStream);
    return llvm::make_error<GenericLuthierError>(
        std::move(File), LineNumber, StackTrace, std::move(ErrorMsg));
  }
  return llvm::Error::success();
}

void GenericLuthierError::log(llvm::raw_ostream &OS) const {
  OS << "Error encountered in file " << File << ", line: " << LineNumber << ": "
     << ErrorMsg << ".\n";
  OS << "Stacktrace: \n" << StackTrace << "\n";
}

} // namespace luthier
