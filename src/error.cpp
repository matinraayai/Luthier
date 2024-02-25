#include "error.hpp"

char luthier::InvalidArgument::ID = 0;

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