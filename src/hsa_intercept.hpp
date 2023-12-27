#ifndef HSA_INTERCEPT_HPP
#define HSA_INTERCEPT_HPP

#include "error.h"
#include "luthier_types.h"

#include <functional>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <unordered_set>

namespace luthier {
class HsaInterceptor {
 private:
    HsaApiTable* internalHsaApiTable_;
    HsaApiTableContainer savedTables_;
    HsaApiTableContainer interceptTables_;
    hsa_ven_amd_loader_1_03_pfn_s amdTable_;
    std::unordered_set<hsa_api_evt_id_t> enabledOps_;

    std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t, const hsa_api_evt_id_t)> userCallback_{};
    std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t, const hsa_api_evt_id_t, bool*)> internalCallback_{};

    void installCoreApiWrappers(CoreApiTable *table);

    void installAmdExtWrappers(AmdExtTable *table);

    void installImageExtWrappers(ImageExtTable *table);

    HsaInterceptor() {}
    ~HsaInterceptor() {
        savedTables_ = {};
        interceptTables_ = {};
        amdTable_ = {};
    }

 public:
    HsaInterceptor(const HsaInterceptor &) = delete;
    HsaInterceptor &operator=(const HsaInterceptor &) = delete;

    [[nodiscard]] const HsaApiTableContainer &getSavedHsaTables() const {
        return savedTables_;
    }

    [[nodiscard]] const hsa_ven_amd_loader_1_03_pfn_t &getHsaVenAmdLoaderTable() const {
        return amdTable_;
    }

    [[nodiscard]] const std::unordered_set<hsa_api_evt_id_t> &getEnabledOps() const {
        return enabledOps_;
    }

    void setUserCallback(const std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t, const hsa_api_evt_id_t)> &callback) {
        userCallback_ = callback;
    }

    void setInternalCallback(const std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t, const hsa_api_evt_id_t, bool*)> &callback) {
        internalCallback_ = callback;
    }

    void enableCallback(hsa_api_evt_id_t op) {
        enabledOps_.insert(op);
    }

    void disableCallback(hsa_api_evt_id_t op) {
        enabledOps_.erase(op);
    }

    [[nodiscard]] const inline std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t, const hsa_api_evt_id_t)> &getUserCallback() const {
        return userCallback_;
    }

    [[nodiscard]] const inline std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t, const hsa_api_evt_id_t, bool*)> &getInternalCallback() const {
        return internalCallback_;
    }

    bool captureHsaApiTable(HsaApiTable *table) {
        internalHsaApiTable_ = table;
        installCoreApiWrappers(table->core_);
        installAmdExtWrappers(table->amd_ext_);
        installImageExtWrappers(table->image_ext_);
        LUTHIER_HSA_CHECK(table->core_->hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_03_pfn_t),
            &amdTable_));
        return true;
    }

    static inline HsaInterceptor &instance() {
        static HsaInterceptor instance;
        return instance;
    }


};
}

#endif
