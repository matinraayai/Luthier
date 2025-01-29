//===-- RocprofilerSDKError.h -----------------------------------*- C++ -*-===//
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
/// This file implements an \c llvm::Error for issues encountered when calling
///// rocprofiler-sdk APIs.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ERROR_ROCPROFILER_SDK_ERROR_H
#define LUTHIER_ERROR_ROCPROFILER_SDK_ERROR_H
#include <llvm/Support/Error.h>
#include <rocprofiler-sdk/fwd.h>

namespace luthier {

#include "luthier/common/ROCmLibraryErrorDefine.h"
LUTHIER_DEFINE_ROCM_LIBRARY_ERROR(RocprofilerSDK, rocprofiler_status_t,
                                  ROCPROFILER_STATUS_SUCCESS);

/// Macro to check for an expected value of a rocprofiler-sdk status
#define LUTHIER_ROCPROFILER_ERROR_CHECK(Expr, Expected)                        \
  luthier::RocprofilerSDKError::RocprofilerSDKErrorCheck(                      \
      __FILE__, __LINE__, Expr, #Expr, Expected)

/// Macro to check for the success of a rocprofiler-sdk operation
#define LUTHIER_ROCPROFILER_SUCCESS_CHECK(Expr)                                \
  luthier::RocprofilerSDKError::RocprofilerSDKErrorCheck(__FILE__, __LINE__,   \
                                                         Expr, #Expr)

} // namespace luthier

#undef LUTHIER_DEFINE_ROCM_LIBRARY_ERROR

#endif