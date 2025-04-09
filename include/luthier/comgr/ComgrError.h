//===-- ComgrError.h --------------------------------------------*- C++ -*-===//
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
/// This file describes the \c ComgrError class which encapsulates errors
/// encountered when making calls to the AMD CoMGR library. It also defines
/// the \c LUTHIER_COMGR_ERROR_CHECK and \c LUTHIER_COMGR_SUCCESS_CHECK macros,
/// which can be used to check the results of a CoMGR operation.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMGR_COMGR_ERROR_H
#define LUTHIER_COMGR_COMGR_ERROR_H
#include <amd_comgr/amd_comgr.h>
#include <llvm/Support/Error.h>

namespace luthier {

#include "luthier/common/ROCmLibraryErrorDefine.h"
LUTHIER_DEFINE_ROCM_LIBRARY_ERROR(Comgr, amd_comgr_status_t,
                                  AMD_COMGR_STATUS_SUCCESS);

/// Macro to check for an expected value of Comgr status
#define LUTHIER_COMGR_ERROR_CHECK(Expr, Expected)                              \
  luthier::ComgrError::ComgrErrorCheck(__FILE__, __LINE__, Expr, #Expr,        \
                                       Expected)

/// Macro to check for success of a Comgr operation
#define LUTHIER_COMGR_SUCCESS_CHECK(Expr)                                      \
  luthier::ComgrError::ComgrErrorCheck(__FILE__, __LINE__, Expr, #Expr)

#undef LUTHIER_DEFINE_ROCM_LIBRARY_ERROR

} // namespace luthier

#endif