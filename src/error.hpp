#ifndef ERROR_HPP
#define ERROR_HPP

#include <amd_comgr/amd_comgr.h>
#include <hsa/hsa.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/FormatVariadicDetails.h>

// Workaround for GCC or other compilers that don't have this macro built-in
// Source:
// https://stackoverflow.com/questions/8487986/file-macro-shows-full-path
#if !defined(__FILE_NAME__)

#define __FILE_NAME__                                                          \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define LUTHIER_CHECK_WITH_MSG(pred, msg)                                      \
  if (!pred) {                                                                 \
    llvm::report_fatal_error(                                                  \
        llvm::formatv("Luthier check on file {0}, line {1} failed: {2}.",      \
                      __FILE_NAME__, __LINE__, msg)                            \
            .str()                                                             \
            .c_str());                                                         \
  }

#define LUTHIER_CHECK(pred)                                                    \
  if (!pred) {                                                                 \
    llvm::report_fatal_error(                                                  \
        llvm::formatv(                                                         \
            "Luthier check for expression {0} on file {1}, line {2} failed.",  \
            #pred, __FILE_NAME__, __LINE__)                                    \
            .str()                                                             \
            .c_str());                                                         \
  }
/**
 * \brief returns from the function if the given \p llvm::Error argument
 * is not llvm::Error::success()
 */
#define LUTHIER_RETURN_ON_ERROR(Error)                                         \
  do {                                                                         \
    if (llvm::errorToBool(Error)) {                                            \
      return (Error);                                                          \
    }                                                                          \
  } while (0)

namespace luthier {

class InvalidArgument : public llvm::ErrorInfo<InvalidArgument> {
public:
  static char ID;       //< ID of the Error
  std::string FileName; //< Name of the file the error was encountered
  const int LineNumber; //< Line number of the file the error was encountered
  std::string FunctionName; //< Name of the function its argument was invalid
  std::string Expr;         //< Expression that failed

  InvalidArgument(llvm::StringRef FileName, int LineNumber,
                  llvm::StringRef FunctionName, llvm::StringRef Expr)
      : FileName(FileName), LineNumber(LineNumber), FunctionName(FunctionName),
        Expr(Expr) {}

  static llvm::Error invalidArgumentCheck(llvm::StringRef FileName,
                                          int LineNumber,
                                          llvm::StringRef FunctionName,
                                          bool Expr, llvm::StringRef ExprStr);

  void log(llvm::raw_ostream &OS) const override {
    OS << "File " << FileName << ", "
       << "line: " << LineNumber << ": ";
    OS << "Invalid argument passed to function " << FunctionName << "; ";
    OS << "Failed argument check: " << Expr;
  }

  std::error_code convertToErrorCode() const override {
    llvm_unreachable("Not implemented");
  }
};

#define LUTHIER_ARGUMENT_ERROR_CHECK(Expr)                                     \
  luthier::InvalidArgument::invalidArgumentCheck(                              \
      __FILE_NAME__, __LINE__, __PRETTY_FUNCTION__, Expr, #Expr)

class AssertionError : public llvm::ErrorInfo<AssertionError> {
public:
  static char ID;       //< ID of the Error
  std::string FileName; //< Name of the file the error was encountered
  const int LineNumber; //< Line number of the file the error was encountered
  std::string Expr;     //< Expression that failed

  AssertionError(llvm::StringRef FileName, int LineNumber, llvm::StringRef Expr)
      : FileName(FileName), LineNumber(LineNumber), Expr(Expr) {}

  static llvm::Error assertionCheck(llvm::StringRef FileName, int LineNumber,
                                    bool Expr, llvm::StringRef ExprStr);

  void log(llvm::raw_ostream &OS) const override {
    OS << "File " << FileName << ", line: " << LineNumber << ": ";
    OS << "Failed assertion: " << Expr;
  }

  std::error_code convertToErrorCode() const override {
    llvm_unreachable("Not implemented");
  }
};

#define LUTHIER_ASSERTION(Expr)                                                \
  luthier::AssertionError::assertionCheck(__FILE_NAME__, __LINE__, Expr, #Expr)

class DisassemblerError : public llvm::ErrorInfo<DisassemblerError> {};

/**
 * \brief Errors caused by the COMGR library
 */
class ComgrError : public llvm::ErrorInfo<ComgrError> {
public:
  static char ID;         //< ID of the Error
  std::string FileName;   //< Name of the file the error was encountered
  const int LineNumber;   //< Line number of the file the error was encountered
  std::string Expression; //< Expression that caused the error
  amd_comgr_status_t Error; //< Encapsulated COMGR error

  /**
   * Public constructor for \p COMGR Error
   * This constructor is not meant to be called directly; Use the
   * error checking macros \p LUTHIER_COMGR_ERROR_CHECK and
   * \p LUTHIER_COMGR_SUCCESS_CHECK instead;
   * \param FileName name of the file the error occurred; Meant to be populated
   * with the \p __FILE_NAME__ macro
   * \param LineNumber line number of the code; Meant to be populated with the
   * \p __LINE_NUMBER__ macro
   * \param Error the COMGR error code that occurred
   * \param Expression the expression which failed; Meant to be the stringify-ed
   * version of the input expression
   */
  ComgrError(llvm::StringRef FileName, int LineNumber, amd_comgr_status_t Error,
             llvm::StringRef Expression)
      : FileName(FileName), LineNumber(LineNumber), Expression(Expression),
        Error(Error) {}

  /**
   * Factory function used by macros to check for COMGR errors
   * If \p Expr result is the same as the \p Expected status, then a
   * \p llvm::Error::success() is returned; Otherwise, a \p luthier::ComgrError
   * is returned. The output of this function can be used with the
   * \p LUTHIER_RETURN_ON_ERROR macro for convenient error checking
   * This function is not meant to be called directly; Use the
   * error checking macros \p LUTHIER_COMGR_ERROR_CHECK and
   * \p LUTHIER_COMGR_SUCCESS_CHECK instead;
   * \param FileName name of the file the error occurred; Meant to be populated
   * with the \p __FILE_NAME__ macro
   * \param LineNumber line number of the code; Meant to be populated with the
   * \p __LINE_NUMBER__ macro
   * \param Expr the expression which is meant to be checked
   * \param Expression the stringify-ed version of \p Expr
   * \param Expected the COMGR status code that is expected; Defaults to
   * \p AMD_COMGR_STATUS_SUCCESS
   */
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
    OS << Error << ". Additional info about the error according to COMGR: ";
    OS << ErrorMsg;
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

/**
 * \brief Errors caused by the HSA library
 * It must only be used by objects/functions inside the \p luthier::hsa
 * namespace
 */
class HsaError : public llvm::ErrorInfo<HsaError> {
public:
  static char ID;         //< ID of the Error
  std::string FileName;   //< Name of the file the error was encountered
  const int LineNumber;   //< Line number of the file the error was encountered
  std::string Expression; //< Call that caused the error
  hsa_status_t Error;     //< Encapsulated HSA error

  /**
   * Public constructor for \p Hsa Error
   * This constructor is not meant to be called directly; Use the
   * error checking macros \p LUTHIER_HSA_ERROR_CHECK and
   * \p LUTHIER_HSA_SUCCESS_CHECK instead;
   * \param FileName name of the file the error occurred; Meant to be populated
   * with the \p __FILE_NAME__ macro
   * \param LineNumber line number of the code; Meant to be populated with the
   * \p __LINE_NUMBER__ macro
   * \param Error the HSA error code that occurred
   * \param Expression the expression which failed; Meant to be the stringify-ed
   * version of the input expression
   */
  HsaError(llvm::StringRef FileName, int LineNumber, hsa_status_t Error,
           llvm::StringRef Expression)
      : FileName(FileName), LineNumber(LineNumber), Expression(Expression),
        Error(Error) {}

  /**
   * Factory function used by macros to check for HSA errors
   * If \p Expr result is the same as the \p Expected status, then a
   * \p llvm::Error::success() is returned; Otherwise, a \p luthier::HsaError
   * is returned. The output of this function can be used with the
   * \p LUTHIER_RETURN_ON_ERROR macro for convenient error checking
   * This function is not meant to be called directly; Use the
   * error checking macros \p LUTHIER_HSA_ERROR_CHECK and
   * \p LUTHIER_HSA_SUCCESS_CHECK instead;
   * \param FileName name of the file the error occurred; Meant to be populated
   * with the \p __FILE_NAME__ macro
   * \param LineNumber line number of the code; Meant to be populated with the
   * \p __LINE_NUMBER__ macro
   * \param Expr the expression which is meant to be checked
   * \param Expression the stringify-ed version of \p Expr
   * \param Expected the HSA status code that is expected; Defaults to
   * \p HSA_STATUS_SUCCESS
   */
  static llvm::Error hsaErrorCheck(llvm::StringRef FileName, int LineNumber,
                                   hsa_status_t Expr, llvm::StringRef ExprStr,
                                   hsa_status_t Expected = HSA_STATUS_SUCCESS);

  void log(llvm::raw_ostream &OS) const override {
    const char *ErrorMsg;
    hsa_status_string(Error, &ErrorMsg);
    OS << "File " << FileName << ", line: " << LineNumber << ": ";
    OS << "HSA call in expression " << Expression << " failed with error code ";
    OS << Error << ". Additional info about the error according to HSA: ";
    OS << ErrorMsg;
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
