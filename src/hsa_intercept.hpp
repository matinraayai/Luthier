#ifndef HSA_INTERCEPT_HPP
#define HSA_INTERCEPT_HPP

#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <functional>
#include <unordered_set>

#include "error.hpp"
#include "luthier_types.h"

#include <thread>

namespace luthier {
class HsaInterceptor {
 private:
    HsaApiTable *internalHsaApiTable_;
    HsaApiTableContainer savedTables_;
    HsaApiTableContainer interceptTables_;
    hsa_ven_amd_loader_1_03_pfn_s amdTable_;
    std::unordered_set<hsa_api_evt_id_t> enabledUserOps_;
    std::unordered_set<hsa_api_evt_id_t> enabledInternalOps_;
    // create thread local variable here to switch between using/not using 20 and 21
    // needs to be static to be accepted by the class
    static thread_local bool enable_temp_callback;

    void luthier_temp_disable_hsa_callback() {
        enable_temp_callback = false;
    }

    void luthier_temp_enable_hsa_callback() {
        enable_temp_callback = true;
    }

    /* Task Steps:
     * 1. Define thread_local variable (bool) in hsa_interceptor.hpp & cpp
         *  If enable_temp_callback is false (disable) --> temporarily ignore enabledUserOps_ & enabledInternalOps_
         *  If enable_temp_callback is true (enable) --> leave enabledUserOps_ & enabledInternalOps_ alone
         *  Add #include<thread> to use thread-local (?)
     * 2. Create function luthier_temp_disable_hsa_callback()
         * set enable_temp_callback to false
     * 3. Create function luthier_temp_enable_hsa_callback()
         * set enable_temp_callback to true
     * 4. Create function isCallbackTempEnabled()
         * return enable_temp_callback
     * 5. Add isCallbackTempEnabled to shouldCallback variable in hsa_intercept.cpp
         * Q: Do we need to add it to every instance of shouldCallback within the hsa_intercept.cpp file?
     * */

    std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t, const hsa_api_evt_id_t)> userCallback_{};
    std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t, const hsa_api_evt_id_t, bool *)>
        internalCallback_{};

    void installCoreApiWrappers(CoreApiTable *table);

    void installAmdExtWrappers(AmdExtTable *table);

    void installImageExtWrappers(ImageExtTable *table);

    HsaInterceptor() {}
    ~HsaInterceptor() {
        uninstallApiTables();
        savedTables_ = {};
        interceptTables_ = {};
        amdTable_ = {};
    }

 public:
    HsaInterceptor(const HsaInterceptor &) = delete;
    HsaInterceptor &operator=(const HsaInterceptor &) = delete;

    [[nodiscard]] const HsaApiTableContainer &getSavedHsaTables() const { return savedTables_; }

    [[nodiscard]] const hsa_ven_amd_loader_1_03_pfn_t &getHsaVenAmdLoaderTable() const { return amdTable_; }

    void uninstallApiTables() {
        *internalHsaApiTable_->core_ = savedTables_.core;
        *internalHsaApiTable_->amd_ext_ = savedTables_.amd_ext;
        *internalHsaApiTable_->finalizer_ext_ = savedTables_.finalizer_ext;
        *internalHsaApiTable_->image_ext_ = savedTables_.image_ext;
    }

    void setUserCallback(const std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t,
                                                  const hsa_api_evt_id_t)> &callback) {
        userCallback_ = callback;
    }

    void setInternalCallback(const std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t,
                                                      const hsa_api_evt_id_t, bool *)> &callback) {
        internalCallback_ = callback;
    }

    [[nodiscard]] const inline std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t,
                                                  const hsa_api_evt_id_t)> &
    getUserCallback() const {
        return userCallback_;
    }

    [[nodiscard]] const inline std::function<void(hsa_api_evt_args_t *, const luthier_api_evt_phase_t,
                                                  const hsa_api_evt_id_t, bool *)> &
    getInternalCallback() const {
        return internalCallback_;
    }

    [[nodiscard]] bool isUserCallbackEnabled(hsa_api_evt_id_t op) const { return enabledUserOps_.contains(op); }

    [[nodiscard]] bool isInternalCallbackEnabled(hsa_api_evt_id_t op) const { return enabledInternalOps_.contains(op); }

    // Step 4.
    bool isCallbackTempEnabled() { return enable_temp_callback; }

    void enableUserCallback(hsa_api_evt_id_t op) { enabledUserOps_.insert(op); }

    void disableUserCallback(hsa_api_evt_id_t op) { enabledUserOps_.erase(op); }

    void enableInternalCallback(hsa_api_evt_id_t op) { enabledInternalOps_.insert(op); }

    void disableInternalCallback(hsa_api_evt_id_t op) { enabledInternalOps_.erase(op); }

    void enableAllUserCallbacks() {
        for (auto i = static_cast<unsigned int>(HSA_API_ID_FIRST); i <= static_cast<unsigned int>(HSA_API_ID_LAST);
             ++i) {
            enableUserCallback(static_cast<hsa_api_evt_id_t>(i));
        }
        for (auto i = static_cast<unsigned int>(HSA_EVT_ID_FIRST); i <= static_cast<unsigned int>(HSA_EVT_ID_LAST);
             ++i) {
            enableUserCallback(static_cast<hsa_api_evt_id_t>(i));
        }
    }
    void disableAllUserCallbacks() { enabledUserOps_.clear(); }

    void enableAllInternalCallbacks() {
        for (auto i = static_cast<unsigned int>(HSA_API_ID_FIRST); i <= static_cast<unsigned int>(HSA_API_ID_LAST);
             ++i) {
            enableInternalCallback(static_cast<hsa_api_evt_id_t>(i));
        }
        for (auto i = static_cast<unsigned int>(HSA_EVT_ID_FIRST); i <= static_cast<unsigned int>(HSA_EVT_ID_LAST);
             ++i) {
            enableInternalCallback(static_cast<hsa_api_evt_id_t>(i));
        }
    }

    void disableAllInternalCallbacks() { enabledInternalOps_.clear(); }

    bool captureHsaApiTable(HsaApiTable *table) {
        internalHsaApiTable_ = table;
        installCoreApiWrappers(table->core_);
        installAmdExtWrappers(table->amd_ext_);
        installImageExtWrappers(table->image_ext_);
        LUTHIER_HSA_CHECK(table->core_->hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_03_pfn_t), &amdTable_));
        return true;
    }

    static inline HsaInterceptor &instance() {
        static HsaInterceptor instance;
        return instance;
    }
};
}// namespace luthier

#endif
