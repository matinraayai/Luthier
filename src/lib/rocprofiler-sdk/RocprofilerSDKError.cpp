#include <llvm/Support/Signals.h>
#include <luthier/rocprofiler-sdk/RocprofilerSDKError.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace luthier {

char RocprofilerSDKError::ID;

llvm::Error RocprofilerSDKError::RocprofilerSDKErrorCheck(
    llvm::StringRef FileName, int LineNumber, rocprofiler_status_t Expr,
    llvm::StringRef ExprStr, rocprofiler_status_t Expected) {
  if (Expr != Expected) {
    std::string StackTrace;
    llvm::raw_string_ostream STStream(StackTrace);
    llvm::sys::PrintStackTrace(STStream);
    return llvm::make_error<RocprofilerSDKError>(FileName, LineNumber,
                                                 StackTrace, Expr, ExprStr);
  }
  return llvm::Error::success();
}

void RocprofilerSDKError::log(llvm::raw_ostream &OS) const {
  const char *ErrorMsg = rocprofiler_get_status_name(Error);
  OS << "Rocprofiler error encountered in file " << File
     << ", line: " << LineNumber << ": ";
  OS << "Rocprofiler-sdk call in expression " << Expression
     << " failed with error code " << Error << ", ";
  if (ErrorMsg) {
    OS << ErrorMsg << ".\n";
  } else {
    OS << "Unknown Error.\n";
  }
  OS << "Stacktrace: \n" << StackTrace << "\n";
}

} // namespace luthier