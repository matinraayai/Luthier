//===-- HsaApiTableInterceptor.cpp ----------------------------------------===//
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
/// This file implements the <tt>luthier::hsa::HsaApiTableInterceptor</tt>
/// class.
//===----------------------------------------------------------------------===//
#include "hsa/HsaApiTableInterceptor.hpp"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

llvm::Error HsaApiTableInterceptor::checkApiTableCopySuccess() const {
  const auto &[version, core_, amd_ext_, finalizer_ext_, image_ext_, tools_,
               pc_sampling_ext_] = SavedApiTable.root;
  /// Check if the copy was initiated in the first place
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      version.major_id, "Failed to copy the HSA API table"));
  /// Check if the core API table was copied successfully
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      core_->version.major_id, "Failed to copy the Core HSA API table"));
  /// Check if the AMD ext API table was copied successfully
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      amd_ext_->version.major_id, "Failed to copy the AMD Ext HSA API table"));
  /// Check if the finalizer ext API table was copied successfully
#if defined(HSA_FINALIZER_API_TABLE_MAJOR_VERSION)
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(finalizer_ext_->version.major_id,
                          "Failed to copy the Finalizer Ext HSA API table"));
#endif
  /// Check if the Image ext API table was copied successfully
#if defined(HSA_IMAGE_API_TABLE_MAJOR_VERSION)
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(image_ext_->version.major_id,
                          "Failed to copy the Image Ext HSA API table"));
#endif

  /// Check if the Tool ext API table was copied successfully
#if defined(HSA_TOOLS_API_TABLE_MAJOR_VERSION)
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      tools_->version.major_id, "Failed to copy the Tools Ext HSA API table"));
#endif

  /// Check if the Tool ext API table was copied successfully
#if defined(HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION)
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(pc_sampling_ext_->version.major_id,
                          "Failed to copy the PC Sampling Ext HSA API table"));
#endif
  return llvm::Error::success();
}

} // namespace luthier::hsa
