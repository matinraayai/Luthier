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
/// This file implements an \c llvm::Error encountered inside both the Luthier
/// tooling library and Luthier tools.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ERROR_LUTHIER_ERROR_H
#define LUTHIER_ERROR_LUTHIER_ERROR_H
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Signals.h>

namespace luthier {

/// \brief Error used to indicate issues encountered in Luthier code not
/// related to any other library
class LuthierError final : public llvm::ErrorInfo<LuthierError> {
public:
  static char ID;               ///< ID of the Error
  const std::string File;       ///< Path to the file the error was encountered
  const int LineNumber;         ///< Line number of the file the error was
                                ///< encountered
  const std::string StackTrace; ///< Stack trace of where the error occurred
  const std::string ErrorMsg;   ///< Message describing the error

  LuthierError(const llvm::StringRef File, const int LineNumber,
               const llvm::StringRef StackTrace, const llvm::StringRef ErrorMsg)
      : File(File), LineNumber(LineNumber), StackTrace(StackTrace),
        ErrorMsg(ErrorMsg) {}

  template <typename... Ts>
  LuthierError(const llvm::StringRef File, const int LineNumber,
               const llvm::StringRef StackTrace, char const *Fmt,
               const Ts &...Vals)
      : File(File), LineNumber(LineNumber), StackTrace(StackTrace),
        ErrorMsg(llvm::formatv(Fmt, Vals...).str()) {}

  static llvm::Error luthierErrorCheck(bool Expr, llvm::StringRef File,
                                       int LineNumber,
                                       llvm::StringRef ErrorMsg);

  template <typename... Ts>
  static llvm::Error luthierErrorCheck(const bool Expr, llvm::StringRef File,
                                       int LineNumber, char const *Fmt,
                                       const Ts &...Vals) {
    if (!Expr) {
      std::string StackTrace;
      llvm::raw_string_ostream STStream(StackTrace);
      llvm::sys::PrintStackTrace(STStream);
      return llvm::make_error<LuthierError>(File, LineNumber, StackTrace, Fmt,
                                            Vals...);
    }
    return llvm::Error::success();
  }

  void log(llvm::raw_ostream &OS) const override;

  [[nodiscard]] std::error_code convertToErrorCode() const override {
    llvm_unreachable("Not implemented");
  }
};

#define LUTHIER_CREATE_ERROR(...)                                              \
  llvm::make_error<luthier::LuthierError>(                                     \
      __FILE__, __LINE__,                                                      \
      []() {                                                                   \
        std::string Out;                                                       \
        llvm::raw_string_ostream OutStream(Out);                               \
        llvm::sys::PrintStackTrace(OutStream);                                 \
        return Out;                                                            \
      }(),                                                                     \
      __VA_ARGS__)

#define LUTHIER_ERROR_CHECK(Expr, ...)                                         \
  luthier::LuthierError::luthierErrorCheck(Expr, __FILE__, __LINE__,           \
                                           __VA_ARGS__)

} // namespace luthier

#endif