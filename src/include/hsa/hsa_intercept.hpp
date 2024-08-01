#ifndef HSA_INTERCEPT_HPP
#define HSA_INTERCEPT_HPP

#include <hsa/hsa_ven_amd_loader.h>

#include <functional>

#include <llvm/ADT/DenseSet.h>

#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/error.hpp"
#include "common/singleton.hpp"
#include <luthier/hsa_trace_api.h>
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
    ROCmLibraryApiInterceptor<
        ApiEvtID, ApiEvtArgs, HsaApiTable,
        HsaApiTableContainer>::~ROCmLibraryApiInterceptor();
    uninstallApiTables();
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

  bool enableUserCallback(ApiEvtID Op);

  void disableUserCallback(ApiEvtID Op);

  bool enableInternalCallback(ApiEvtID Op);

  void disableInternalCallback(ApiEvtID Op);

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
    // Run the disable API function to install the queue interceptor function
    disableInternalCallback(HSA_API_EVT_ID_hsa_queue_create);
    return llvm::Error::success();
  }
};
} // namespace luthier::hsa

#endif
