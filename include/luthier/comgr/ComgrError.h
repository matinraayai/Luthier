#ifndef LUTHIER_COMGR_ERROR_H
#define LUTHIER_COMGR_ERROR_H
#include "luthier/common/ROCmLibraryErrorDefine.h"
#include <amd_comgr/amd_comgr.h>
#include <llvm/Support/Error.h>

namespace luthier {

LUTHIER_DEFINE_ROCM_LIBRARY_ERROR(Comgr, amd_comgr_status_t,
                                  AMD_COMGR_STATUS_SUCCESS);

/// Macro to check for an expected value of Comgr status
#define LUTHIER_COMGR_ERROR_CHECK(Expr, Expected)                              \
  luthier::ComgrError::ComgrErrorCheck(__FILE__, __LINE__, Expr, #Expr,        \
                                       Expected)

/// Macro to check for success of a Comgr operation
#define LUTHIER_COMGR_SUCCESS_CHECK(Expr)                                      \
  luthier::ComgrError::ComgrErrorCheck(__FILE__, __LINE__, Expr, #Expr)

} // namespace luthier

#endif