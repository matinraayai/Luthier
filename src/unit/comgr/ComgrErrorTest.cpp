//===-- ComgrErrorTest.cpp ------------------------------------------------===//
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
/// This file includes tests for the \c ComgrError class and its associated
/// error checking macros.
//===----------------------------------------------------------------------===//
#include <amd_comgr/amd_comgr.h>
#include <gtest/gtest.h>
#include <llvm/Support/Error.h>
#include <luthier/comgr/ComgrError.h>

TEST(LuthierComgrTests, SuccessCheck) {
  /// Check if AMD_COMGR_STATUS_SUCCESS will not create a COMGR error when
  /// checked
  llvm::Error Err = LUTHIER_COMGR_SUCCESS_CHECK(AMD_COMGR_STATUS_SUCCESS);
  EXPECT_EQ(Err.operator bool(), false);
  /// Check if comgr error check will not create a COMGR error if the
  /// expected comgr status matches the actual status
  for (amd_comgr_status_t StatusCode :
       {AMD_COMGR_STATUS_SUCCESS, AMD_COMGR_STATUS_ERROR,
        AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
        AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES}) {
    Err = LUTHIER_COMGR_ERROR_CHECK(StatusCode, StatusCode);
    EXPECT_EQ(Err.operator bool(), false);
  }
}

TEST(LuthierComgrTests, FailureCheck) {
  llvm::Error Err = llvm::Error::success();
  for (amd_comgr_status_t StatusCode :
       {AMD_COMGR_STATUS_SUCCESS, AMD_COMGR_STATUS_ERROR,
        AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
        AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES}) {
    /// Check if anything other than AMD_COMGR_STATUS_SUCCESS will create a
    /// COMGR error when checked
    if (StatusCode != AMD_COMGR_STATUS_SUCCESS) {
      Err = LUTHIER_COMGR_SUCCESS_CHECK(StatusCode);
      EXPECT_EQ(Err.operator bool(), true);
    }
    /// Check if comgr error check will create a COMGR error if the
    /// expected comgr status matches the actual status
    for (amd_comgr_status_t StatusCodeInner :
       {AMD_COMGR_STATUS_SUCCESS, AMD_COMGR_STATUS_ERROR,
        AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
        AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES}) {
      if (StatusCode != StatusCodeInner) {
        Err = LUTHIER_COMGR_ERROR_CHECK(StatusCode,
                          StatusCodeInner);
        EXPECT_EQ(Err.operator bool(), true);
      }
    }
  }
}
