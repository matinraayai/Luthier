#include <common/error.hpp>

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