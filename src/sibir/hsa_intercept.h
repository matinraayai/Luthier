#ifndef HSA_INTERCEPT_H_
#define HSA_INTERCEPT_H_

#include "sibir_types.h"
#include "error_check.h"
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hsa.h>

#include <functional>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>


class SibirHsaInterceptor {
 private:
    HsaApiTableContainer savedTables_;
    HsaApiTableContainer interceptTables_;
    hsa_ven_amd_loader_1_03_pfn_s amdTable_;

    std::function<void(hsa_api_args_t*, const sibir_api_phase_t, const hsa_api_id_t)> callback_;

    void installCoreApiWrappers(CoreApiTable* table);

    void installAmdExtWrappers(AmdExtTable* table);

    void installImageExtWrappers(ImageExtTable* table);

    SibirHsaInterceptor() {}
    ~SibirHsaInterceptor() {
        memset(&savedTables_, 0, sizeof(HsaApiTableContainer));
        memset(&interceptTables_, 0, sizeof(HsaApiTableContainer));
        memset(&amdTable_, 0, sizeof(hsa_ven_amd_loader_1_03_pfn_t));
    }

 public:

    SibirHsaInterceptor(const SibirHsaInterceptor &) = delete;
    SibirHsaInterceptor & operator=(const SibirHsaInterceptor &) = delete;

    [[nodiscard]] const HsaApiTableContainer &getSavedHsaTables() const {
        return savedTables_;
    }

    [[nodiscard]] const hsa_ven_amd_loader_1_03_pfn_t &getHsaVenAmdLoaderTable() const {
        return amdTable_;
    }

    void SetCallback(const std::function<void(hsa_api_args_t *, const sibir_api_phase_t, const hsa_api_id_t)> &callback) {
        callback_ = callback;
    }

    bool captureHsaApiTable(HsaApiTable* table) {
        installCoreApiWrappers(table->core_);
        installAmdExtWrappers(table->amd_ext_);
        installImageExtWrappers(table->image_ext_);
        SIBIR_HSA_CHECK(table->core_->hsa_system_get_major_extension_table_fn(
                HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_03_pfn_t),
                &amdTable_));
        return true;
    }



    static inline SibirHsaInterceptor& Instance() {
        static SibirHsaInterceptor instance;
        return instance;
    }

    [[nodiscard]] const inline std::function<void(hsa_api_args_t *, const sibir_api_phase_t, const hsa_api_id_t)> &GetCallback() const {
        return callback_;
    }
};

#endif
