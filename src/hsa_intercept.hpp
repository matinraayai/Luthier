#ifndef HSA_INTERCEPT_HPP
#define HSA_INTERCEPT_HPP

#include "error_check.hpp"
#include "sibir_types.hpp"
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hsa.h>

#include <functional>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

namespace sibir {
class HsaInterceptor {
 private:
    HsaApiTableContainer savedTables_;
    HsaApiTableContainer interceptTables_;
    hsa_ven_amd_loader_1_03_pfn_s amdTable_;

    std::function<void(hsa_api_args_t *, const sibir_api_phase_t, const hsa_api_id_t)> callback_;

    void installCoreApiWrappers(CoreApiTable *table);

    void installAmdExtWrappers(AmdExtTable *table);

    void installImageExtWrappers(ImageExtTable *table);

    HsaInterceptor() {}
    ~HsaInterceptor() {
        memset(&savedTables_, 0, sizeof(HsaApiTableContainer));
        memset(&interceptTables_, 0, sizeof(HsaApiTableContainer));
        memset(&amdTable_, 0, sizeof(hsa_ven_amd_loader_1_03_pfn_t));
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

    void SetCallback(const std::function<void(hsa_api_args_t *, const sibir_api_phase_t, const hsa_api_id_t)> &callback) {
        callback_ = callback;
    }

    bool captureHsaApiTable(HsaApiTable *table) {
        installCoreApiWrappers(table->core_);
        installAmdExtWrappers(table->amd_ext_);
        installImageExtWrappers(table->image_ext_);
        SIBIR_HSA_CHECK(table->core_->hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_03_pfn_t),
            &amdTable_));
        return true;
    }

    static inline HsaInterceptor &Instance() {
        static HsaInterceptor instance;
        return instance;
    }

    [[nodiscard]] const inline std::function<void(hsa_api_args_t *, const sibir_api_phase_t, const hsa_api_id_t)> &GetCallback() const {
        return callback_;
    }
};
}

#endif
