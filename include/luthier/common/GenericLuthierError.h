//===-- GenericLuthierError.h -----------------------------------*- C++ -*-===//
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
/// Describes <tt>GenericLuthierError</tt>, which represents generic Luthier
/// errors not related third-party libraries used by Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_GENERIC_LUTHIER_ERROR_H
#define LUTHIER_COMMON_GENERIC_LUTHIER_ERROR_H
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Signals.h>
#include <luthier/common/LuthierError.h>

namespace luthier {

/// \brief Error used to indicate generic issues encountered in Luthier code not
/// related to any other library
class GenericLuthierError final : public LuthierError {
private:
  /// Message describing the error; Should not end with full stop "."
  const std::string ErrorMsg;

public:
  static char ID;

  GenericLuthierError(std::string File, int LineNumber, std::string StackTrace,
                      std::string ErrorMsg)
      : LuthierError(std::move(File), LineNumber, std::move(StackTrace)),
        ErrorMsg(std::move(ErrorMsg)) {};

  template <typename... Ts>
  GenericLuthierError(std::string File, const int LineNumber,
                      std::string StackTrace, char const *Fmt,
                      const Ts &...Vals)
      : LuthierError(std::move(File), LineNumber, std::move(StackTrace)),
        ErrorMsg(llvm::formatv(Fmt, Vals...).str()) {}

  static llvm::Error luthierErrorCheck(bool Expr, std::string File,
                                       int LineNumber, std::string ErrorMsg);

  template <typename... Ts>
  static llvm::Error luthierErrorCheck(const bool Expr, std::string File,
                                       int LineNumber, char const *Fmt,
                                       const Ts &...Vals) {
    if (!Expr) {
      std::string StackTrace;
      llvm::raw_string_ostream STStream(StackTrace);
      llvm::sys::PrintStackTrace(STStream);
      return llvm::make_error<GenericLuthierError>(std::move(File), LineNumber,
                                                   StackTrace, Fmt, Vals...);
    }
    return llvm::Error::success();
  }

  void log(llvm::raw_ostream &OS) const override;
};

/// \brief Macro used to create generic Luthier errors
#define LUTHIER_CREATE_ERROR(...)                                              \
  llvm::make_error<luthier::GenericLuthierError>(                              \
      __FILE__, __LINE__,                                                      \
      []() {                                                                   \
        std::string Out;                                                       \
        llvm::raw_string_ostream OutStream(Out);                               \
        llvm::sys::PrintStackTrace(OutStream);                                 \
        return Out;                                                            \
      }(),                                                                     \
      , __VA_ARGS__)

/// \brief Macro for checking for generic Luthier issues
#define LUTHIER_ERROR_CHECK(Expr, ...)                                         \
  luthier::LuthierError::luthierErrorCheck(Expr, __FILE__, __LINE__, #Expr,    \
                                           __VA_ARGS__)

} // namespace luthier

#endif