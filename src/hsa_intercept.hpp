#ifndef HSA_INTERCEPT_HPP
#define HSA_INTERCEPT_HPP

#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <functional>
#include <unordered_set>

#include "error.hpp"
#include "luthier_types.h"

namespace luthier::hsa {
class Interceptor {
private:
  HsaApiTable *InternalHsaApiTable;
  HsaApiTableContainer SavedTables;
  hsa_ven_amd_loader_1_03_pfn_s AmdTable;
  std::unordered_set<hsa_api_evt_id_t> EnabledUserOps;
  std::unordered_set<hsa_api_evt_id_t> EnabledInternalOps;

  std::function<void(hsa_api_evt_args_t *, const luthier::ApiEvtPhase,
                     const hsa_api_evt_id_t)>
      userCallback_{};
  std::function<void(hsa_api_evt_args_t *, const luthier::ApiEvtPhase,
                     const hsa_api_evt_id_t, bool *)>
      internalCallback_{};

  void installCoreApiTableWrappers(CoreApiTable *table);

  void installAmdExtTableWrappers(AmdExtTable *table);

  void installImageExtTableWrappers(ImageExtTable *table);

  void installFinalizerExtTableWrappers(FinalizerExtTable *Table);

  Interceptor() {}
  ~Interceptor() {
    uninstallApiTables();
    SavedTables = {};
    AmdTable = {};
  }

public:
  Interceptor(const Interceptor &) = delete;
  Interceptor &operator=(const Interceptor &) = delete;

  [[nodiscard]] const HsaApiTableContainer &getSavedHsaTables() const {
    return SavedTables;
  }

  [[nodiscard]] const hsa_ven_amd_loader_1_03_pfn_t &
  getHsaVenAmdLoaderTable() const {
    return AmdTable;
  }

  void uninstallApiTables() {
    *InternalHsaApiTable->core_ = SavedTables.core;
    *InternalHsaApiTable->amd_ext_ = SavedTables.amd_ext;
    *InternalHsaApiTable->finalizer_ext_ = SavedTables.finalizer_ext;
    *InternalHsaApiTable->image_ext_ = SavedTables.image_ext;
  }

  void setUserCallback(
      const std::function<void(hsa_api_evt_args_t *, const luthier::ApiEvtPhase,
                               const hsa_api_evt_id_t)> &callback) {
    userCallback_ = callback;
  }

  void setInternalCallback(
      const std::function<void(hsa_api_evt_args_t *, const luthier::ApiEvtPhase,
                               const hsa_api_evt_id_t, bool *)> &callback) {
    internalCallback_ = callback;
  }

  [[nodiscard]] const inline std::function<void(hsa_api_evt_args_t *,
                                                const luthier::ApiEvtPhase,
                                                const hsa_api_evt_id_t)> &
  getUserCallback() const {
    return userCallback_;
  }

  [[nodiscard]] const inline std::function<
      void(hsa_api_evt_args_t *, const luthier::ApiEvtPhase,
           const hsa_api_evt_id_t, bool *)> &
  getInternalCallback() const {
    return internalCallback_;
  }

  [[nodiscard]] bool isUserCallbackEnabled(hsa_api_evt_id_t op) const {
    return EnabledUserOps.contains(op);
  }

  [[nodiscard]] bool isInternalCallbackEnabled(hsa_api_evt_id_t op) const {
    return EnabledInternalOps.contains(op);
  }

  void enableUserCallback(hsa_api_evt_id_t op) { EnabledUserOps.insert(op); }

  void disableUserCallback(hsa_api_evt_id_t op) { EnabledUserOps.erase(op); }

  void enableInternalCallback(hsa_api_evt_id_t op) {
    EnabledInternalOps.insert(op);
  }

  void disableInternalCallback(hsa_api_evt_id_t op) {
    EnabledInternalOps.erase(op);
  }

  void enableAllUserCallbacks() {
    for (auto i = HSA_API_EVT_ID_FIRST;
         i <= HSA_API_EVT_ID_LAST; ++i) {
      enableUserCallback(static_cast<hsa_api_evt_id_t>(i));
    }
    for (auto i = static_cast<unsigned int>(HSA_API_EVT_ID_FIRST);
         i <= static_cast<unsigned int>(HSA_API_EVT_ID_LAST); ++i) {
      enableUserCallback(static_cast<hsa_api_evt_id_t>(i));
    }
  }
  void disableAllUserCallbacks() { EnabledUserOps.clear(); }

  void enableAllInternalCallbacks() {
    for (auto i = static_cast<unsigned int>(HSA_API_ID_FIRST);
         i <= static_cast<unsigned int>(HSA_API_ID_LAST); ++i) {
      enableInternalCallback(static_cast<hsa_api_evt_id_t>(i));
    }
    for (auto i = static_cast<unsigned int>(HSA_EVT_ID_FIRST);
         i <= static_cast<unsigned int>(HSA_EVT_ID_LAST); ++i) {
      enableInternalCallback(static_cast<hsa_api_evt_id_t>(i));
    }
  }

  void disableAllInternalCallbacks() { EnabledInternalOps.clear(); }

  bool captureHsaApiTable(HsaApiTable *table) {
    InternalHsaApiTable = table;
    installCoreApiTableWrappers(table->core_);
    installAmdExtTableWrappers(table->amd_ext_);
    installImageExtTableWrappers(table->image_ext_);
    installFinalizerExtTableWrappers(table->finalizer_ext_);

    return (table->core_->hsa_system_get_major_extension_table_fn(
                HSA_EXTENSION_AMD_LOADER, 1,
                sizeof(hsa_ven_amd_loader_1_03_pfn_t),
                &AmdTable) == HSA_STATUS_SUCCESS);
  }

  static inline hsa::Interceptor &instance() {
    static hsa::Interceptor instance;
    return instance;
  }
};
} // namespace luthier::hsa

#endif
