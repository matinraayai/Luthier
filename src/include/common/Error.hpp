//===-- Error.hpp - Luthier LLVM Error Types ------------------------------===//
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
#ifndef LUTHIER_COMMON_ERROR_HPP
#define LUTHIER_COMMON_ERROR_HPP

#include <amd_comgr/amd_comgr.h>
#include <hsa/hsa.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>

#include <luthier/ErrorCheck.h>
#include <luthier/types.h>

/// Workaround for GCC or other compilers that don't have this macro built-in
/// Source:
/// https://stackoverflow.com/questions/8487986/file-macro-shows-full-path
#if !defined(__FILE_NAME__)

#define __FILE_NAME__                                                          \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

namespace luthier {

/// \brief Error used to indicate issues encountered in Luthier code not
/// related to any other ROCm library
class LuthierError : public llvm::ErrorInfo<LuthierError> {
public:
  static char ID;             ///< ID of the Error
  const std::string FileName; ///< Name of the file the error was encountered
  const int LineNumber;       ///< Line number of the file the error was
                              ///< encountered
  const std::string ErrorMsg; ///< Message describing the error

  LuthierError(llvm::StringRef FileName, int LineNumber,
               llvm::StringRef ErrorMsg)
      : FileName(FileName), LineNumber(LineNumber), ErrorMsg(ErrorMsg) {}

  template <typename... Ts>
  LuthierError(llvm::StringRef FileName, int LineNumber, char const *Fmt,
               const Ts &...Vals)
      : FileName(FileName), LineNumber(LineNumber),
        ErrorMsg(llvm::formatv(Fmt, Vals...).str()) {}

  static llvm::Error luthierErrorCheck(llvm::StringRef FileName, int LineNumber,
                                       bool Expr, llvm::StringRef ErrorMsg);

  template <typename... Ts>
  static llvm::Error luthierErrorCheck(llvm::StringRef FileName, int LineNumber,
                                       bool Expr, char const *Fmt,
                                       const Ts &...Vals) {
    return (!Expr) ? llvm::make_error<LuthierError>(FileName, LineNumber, Fmt,
                                                    Vals...)
                   : llvm::Error::success();
  }

  void log(llvm::raw_ostream &OS) const override {
    OS << "File " << FileName << ", line: " << LineNumber << ": ";
    OS << ErrorMsg << "\n";
  }

  [[nodiscard]] std::error_code convertToErrorCode() const override {
    llvm_unreachable("Not implemented");
  }
};

#define LUTHIER_CREATE_ERROR(...)                                              \
  llvm::make_error<luthier::LuthierError>(__FILE_NAME__, __LINE__, __VA_ARGS__)

#define LUTHIER_ERROR_CHECK(Expr, ...)                                         \
  luthier::LuthierError::luthierErrorCheck(__FILE_NAME__, __LINE__, Expr,      \
                                           __VA_ARGS__)

/// \brief Describes errors caused by calls to the COMGR library
class ComgrError : public llvm::ErrorInfo<ComgrError> {
public:
  static char ID;             ///< ID of the Error
  const std::string FileName; ///< Name of the file the error was encountered
  const int LineNumber; ///< Line number of the file the error was encountered
  const std::string Expression;   ///< Expression that caused the error
  const amd_comgr_status_t Error; ///< Encapsulated COMGR error

  /// Public constructor for COMGR Error
  /// \note This constructor is not meant to be called directly; Use the
  /// error checking macros \c LUTHIER_COMGR_ERROR_CHECK and
  /// \c LUTHIER_COMGR_SUCCESS_CHECK instead;
  /// \param FileName name of the file the error occurred; Meant to be populated
  /// with the \c __FILE_NAME__ macro
  /// \param LineNumber LineNumber line number of the code; Meant to be
  /// populated with the \c __LINE_NUMBER__ macro
  /// \param Error the COMGR error code that occurred
  /// \param Expression the expression which failed; Meant to be the
  /// stringify-ed version of the input expression
  ComgrError(llvm::StringRef FileName, int LineNumber, amd_comgr_status_t Error,
             llvm::StringRef Expression)
      : FileName(FileName), LineNumber(LineNumber), Expression(Expression),
        Error(Error) {}

  /// Factory function used by macros to check for COMGR errors
  /// If \p Expr result is the same as the \p Expected status, then a
  /// \c llvm::ErrorSuccess is returned; Otherwise, a \c luthier::ComgrError
  /// is returned. The output of this function can be used with the
  /// \c LUTHIER_RETURN_ON_ERROR macro for convenient error checking
  /// \note This function is not meant to be called directly; Use the
  /// error checking macros \c LUTHIER_COMGR_ERROR_CHECK and
  /// \c LUTHIER_COMGR_SUCCESS_CHECK instead
  /// \param FileName name of the file the error occurred; Meant to be populated
  /// with the \p __FILE_NAME__ macro
  /// \param LineNumber line number of the code; Meant to be populated with the
  /// \c __LINE_NUMBER__ macro
  /// \param Expr the expression which is meant to be checked
  /// \param ExprStr the stringify-ed version of \p Expr
  /// \param Expected the COMGR status code that is expected; Defaults to
  /// \c AMD_COMGR_STATUS_SUCCESS
  /// \return \c llvm::ErrorSuccess if the check is successful, or a
  /// \c luthier::ComgrError otherwise
  static llvm::Error
  comgrErrorCheck(llvm::StringRef FileName, int LineNumber,
                  amd_comgr_status_t Expr, llvm::StringRef ExprStr,
                  amd_comgr_status_t Expected = AMD_COMGR_STATUS_SUCCESS);

  void log(llvm::raw_ostream &OS) const override {
    const char *ErrorMsg;
    amd_comgr_status_string(Error, &ErrorMsg);
    OS << "File " << FileName << ", line: " << LineNumber << ": ";
    OS << "COMGR call in expression " << Expression
       << " failed with error code ";
    OS << Error << ". info about the error according to COMGR: ";
    OS << ErrorMsg << "\n";
  }

  [[nodiscard]] std::error_code convertToErrorCode() const override {
    llvm_unreachable("Not implemented");
  }
};

#define LUTHIER_COMGR_ERROR_CHECK(Expr, Expected)                              \
  luthier::ComgrError::comgrErrorCheck(__FILE_NAME__, __LINE__, Expr, #Expr,   \
                                       Expected)

#define LUTHIER_COMGR_SUCCESS_CHECK(Expr)                                      \
  luthier::ComgrError::comgrErrorCheck(__FILE_NAME__, __LINE__, Expr, #Expr)

/// \brief Describes errors caused by the HSA library; Mostly used by
/// the \c hsa namespace objects
class HsaError : public llvm::ErrorInfo<HsaError> {
public:
  static char ID;             ///< ID of the Error
  const std::string FileName; ///< Name of the file the error was encountered
  const int LineNumber; ///< Line number of the file the error was encountered
  const std::string Expression; ///< Call that caused the error
  const hsa_status_t Error;     ///< Encapsulated HSA error

  /// Public constructor for HSA Error
  /// \note This constructor is not meant to be called directly; Use the
  /// error checking macros \c LUTHIER_HSA_ERROR_CHECK and
  /// \c LUTHIER_HSA_SUCCESS_CHECK instead
  /// \param FileName name of the file the error occurred; Meant to be populated
  /// with the \c __FILE_NAME__ macro
  /// \param LineNumber line number of the code; Meant to be populated with the
  /// \c __LINE_NUMBER__ macro
  /// \param Error the HSA error code that occurred
  /// \param Expression the expression which failed; Meant to be the
  /// stringify-ed version of the input expression
  HsaError(llvm::StringRef FileName, int LineNumber, hsa_status_t Error,
           llvm::StringRef Expression)
      : FileName(FileName), LineNumber(LineNumber), Expression(Expression),
        Error(Error) {}

  /// Factory function used by macros to check for HSA errors
  /// If \p Expr result is the same as the \p Expected status, then a
  /// \c llvm::ErrorSuccess is returned; Otherwise, a \c luthier::HsaError
  /// is returned. The output of this function can be used with the
  /// \c LUTHIER_RETURN_ON_ERROR macro for convenient error checking
  /// \note This function is not meant to be called directly; Use the
  /// error checking macros \p LUTHIER_HSA_ERROR_CHECK and
  /// \c LUTHIER_HSA_SUCCESS_CHECK instead
  /// \param FileName  name of the file the error occurred; Meant to be
  /// populated with the \c __FILE_NAME__ macro
  /// \param LineNumber line number of the code; Meant to be populated with the
  /// \c __LINE_NUMBER__ macro
  /// \param Expr the expression which is meant to be checked
  /// \param ExprStr the stringify-ed version of \p Expr
  /// \param Expected the HSA status code that is expected; Defaults to
  /// \p HSA_STATUS_SUCCESS
  /// \return \c llvm::ErrorSuccess if the check is successful, or a
  /// \c luthier::HsaError otherwise
  static llvm::Error hsaErrorCheck(llvm::StringRef FileName, int LineNumber,
                                   hsa_status_t Expr, llvm::StringRef ExprStr,
                                   hsa_status_t Expected = HSA_STATUS_SUCCESS);

  void log(llvm::raw_ostream &OS) const override {
    const char *ErrorMsg;
    hsa_status_string(Error, &ErrorMsg);
    OS << "File " << FileName << ", line: " << LineNumber << ": ";
    OS << "HSA call in expression " << Expression << " failed with error code ";
    OS << Error << ". Additional info about the error according to HSA: ";
    OS << ErrorMsg << "\n";
  }

  [[nodiscard]] std::error_code convertToErrorCode() const override {
    llvm_unreachable("Not implemented");
  }
};

#define LUTHIER_HSA_ERROR_CHECK(Expr, Expected)                                \
  luthier::HsaError::hsaErrorCheck(__FILE_NAME__, __LINE__, Expr, #Expr,       \
                                   Expected)

#define LUTHIER_HSA_SUCCESS_CHECK(Expr)                                        \
  luthier::HsaError::hsaErrorCheck(__FILE_NAME__, __LINE__, Expr, #Expr)

} // namespace luthier

#endif
