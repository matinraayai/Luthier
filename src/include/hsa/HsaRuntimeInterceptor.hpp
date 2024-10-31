//===-- HsaRuntimeInterceptor.hpp - HSA Runtime API Interceptor -----------===//
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
/// This file implements the <tt>luthier::hsa::HsaRuntimeInterceptor</tt>,
/// in charge of capturing HSA API functions as well as packet dispatch events.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_RUNTIME_INTERCEPT_HPP
#define LUTHIER_HSA_HSA_RUNTIME_INTERCEPT_HPP

#include <hsa/hsa_ven_amd_loader.h>

#include <functional>

#include <llvm/ADT/DenseSet.h>

#include "common/Error.hpp"
#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/Singleton.hpp"
#include <luthier/hsa/TraceApi.h>
#include <luthier/types.h>

namespace luthier::hsa {

class HsaRuntimeInterceptor
    : public Singleton<HsaRuntimeInterceptor>,
      public ROCmLibraryApiInterceptor<ApiEvtID, ApiEvtArgs, HsaApiTable,
                                       HsaApiTableContainer> {
private:
  /// Holds function pointers to the AMD's loader API \n
  /// The loader API does not get intercepted in this class
  hsa_ven_amd_loader_1_03_pfn_s AmdTable{};

public:
  HsaRuntimeInterceptor() = default;

  ~HsaRuntimeInterceptor() {
    //  TODO: Should we even uninstall the wrappers for clean up?
    //  Commented out for now since sometimes entries of the api table end up
    //  being null pointers
    //    uninstallApiTables();
    AmdTable = {};
    Singleton<HsaRuntimeInterceptor>::~Singleton();
  }

  HsaRuntimeInterceptor(const HsaRuntimeInterceptor &) = delete;
  HsaRuntimeInterceptor &operator=(const HsaRuntimeInterceptor &) = delete;

  [[nodiscard]] const hsa_ven_amd_loader_1_03_pfn_t &
  getHsaVenAmdLoaderTable() const {
    return AmdTable;
  }

  void uninstallApiTables() {
    *RuntimeApiTable->core_ = SavedRuntimeApiTable.core;
    *RuntimeApiTable->amd_ext_ = SavedRuntimeApiTable.amd_ext;
    *RuntimeApiTable->finalizer_ext_ = SavedRuntimeApiTable.finalizer_ext;
    *RuntimeApiTable->image_ext_ = SavedRuntimeApiTable.image_ext;
  }

  llvm::Error enableUserCallback(ApiEvtID Op);

  llvm::Error disableUserCallback(ApiEvtID Op);

  llvm::Error enableInternalCallback(ApiEvtID Op);

  llvm::Error disableInternalCallback(ApiEvtID Op);

  llvm::Error captureApiTable(HsaApiTable *Table) {
    RuntimeApiTable = Table;
    SavedRuntimeApiTable.core = *Table->core_;
    SavedRuntimeApiTable.amd_ext = *Table->amd_ext_;
    SavedRuntimeApiTable.image_ext = *Table->image_ext_;
    SavedRuntimeApiTable.finalizer_ext = *Table->finalizer_ext_;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        Table->core_->hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_03_pfn_t),
            &AmdTable)));
    Status = API_TABLE_CAPTURED;
    // Run the disable API function to install the queue interceptor function
    return disableInternalCallback(HSA_API_EVT_ID_hsa_queue_create);
  }
};
} // namespace luthier::hsa

#endif
