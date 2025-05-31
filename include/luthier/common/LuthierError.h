//===-- LuthierError.h ------------------------------------------*- C++ -*-===//
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
/// Defines <tt>LuthierError</tt>, containing the common part among all
/// \c llvm::ErrorInfo classes defined by Luthier, as well as RTTI mechanism
/// for checking whether a given \c llvm::Error originated from Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_LUTHIER_ERROR_H
#define LUTHIER_COMMON_LUTHIER_ERROR_H
#include <llvm/Support/Error.h>

namespace luthier {

class LuthierError : public llvm::ErrorInfo<LuthierError> {

protected:
  /// Path to the file the error was encountered
  const std::string File;
  /// Line number of the file the error was encountered
  const int LineNumber;
  /// Stack trace of where the error occurred
  const std::string StackTrace;
  /// Expression that failed the error checking
  const std::string Expression;

  LuthierError(std::string File, int LineNumber, std::string StackTrace,
               std::string Expression)
      : File(std::move(File)), LineNumber(LineNumber),
        StackTrace(std::move(StackTrace)), Expression(std::move(Expression)) {}

public:
  static char ID;

  [[nodiscard]] std::error_code convertToErrorCode() const override {
    llvm_unreachable("Not implemented");
  }
};

} // namespace luthier

#endif
