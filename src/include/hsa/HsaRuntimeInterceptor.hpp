//===-- HsaRuntimeInterceptor.hpp - HSA Runtime API Interceptor -----------===//
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
/// This file implements the <tt>luthier::hsa::HsaRuntimeInterceptor</tt>,
/// in charge of capturing HSA API functions as well as packet dispatch events.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_RUNTIME_INTERCEPT_HPP
#define LUTHIER_HSA_HSA_RUNTIME_INTERCEPT_HPP

#include "common/Error.hpp"
#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/Singleton.hpp"
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/ADT/DenseSet.h>
#include <luthier/ErrorCheck.h>
#include <luthier/hsa/TraceApi.h>
#include <luthier/types.h>

namespace luthier::hsa {

class HsaRuntimeInterceptor final
    : public Singleton<HsaRuntimeInterceptor>,
      public ROCmLibraryApiInterceptor<ApiEvtID, ApiEvtArgs, HsaApiTable,
                                       HsaApiTableContainer> {
private:
  /// Holds function pointers to the AMD's loader API \n
  /// The loader API does not get intercepted in this class
  hsa_ven_amd_loader_1_03_pfn_s AmdTable{};

  void uninstallApiTables() {
    if (RuntimeApiTable) {
      if (RuntimeApiTable->core_)
        *RuntimeApiTable->core_ = SavedRuntimeApiTable.core;
      if (RuntimeApiTable->amd_ext_)
        *RuntimeApiTable->amd_ext_ = SavedRuntimeApiTable.amd_ext;
      if (RuntimeApiTable->finalizer_ext_)
        *RuntimeApiTable->finalizer_ext_ = SavedRuntimeApiTable.finalizer_ext;
      if (RuntimeApiTable->image_ext_)
        *RuntimeApiTable->image_ext_ = SavedRuntimeApiTable.image_ext;
    }
  }

public:
  HsaRuntimeInterceptor() = default;

  ~HsaRuntimeInterceptor() override {
    //  TODO: Check the timing correctness of uninstalling API tables
    uninstallApiTables();
    AmdTable = {};
    Singleton::~Singleton();
  }

  HsaRuntimeInterceptor(const HsaRuntimeInterceptor &) = delete;
  HsaRuntimeInterceptor &operator=(const HsaRuntimeInterceptor &) = delete;

  [[nodiscard]] const hsa_ven_amd_loader_1_03_pfn_t &
  getHsaVenAmdLoaderTable() const {
    return AmdTable;
  }

  llvm::Error enableUserCallback(ApiEvtID Op) override;

  llvm::Error disableUserCallback(ApiEvtID Op) override;

  llvm::Error enableInternalCallback(ApiEvtID Op) override;

  llvm::Error disableInternalCallback(ApiEvtID Op) override;

  llvm::Error captureApiTable(HsaApiTable *Table) override {
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
