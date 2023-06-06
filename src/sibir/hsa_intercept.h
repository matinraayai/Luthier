#ifndef HSA_INTERCEPT_H_
#define HSA_INTERCEPT_H_

#include "sibir_types.h"
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hsa.h>

#include <functional>
#include <hsa/hsa_api_trace.h>


class SibirHsaInterceptor {
 private:
    HsaApiTableContainer savedTables_;
    HsaApiTableContainer interceptTables_;
    std::function<void(hsa_api_args_t*, const sibir_api_phase_t, const hsa_api_id_t)> callback_;

    void installCoreApiWrappers(CoreApiTable* table);

    void installAmdExtWrappers(AmdExtTable* table);

    void installImageExtWrappers(ImageExtTable* table);

    SibirHsaInterceptor() {}
    ~SibirHsaInterceptor() {}

 public:

    SibirHsaInterceptor(const SibirHsaInterceptor &) = delete;
    SibirHsaInterceptor & operator=(const SibirHsaInterceptor &) = delete;

    [[nodiscard]] const HsaApiTableContainer &getSavedHsaTables() const {
        return savedTables_;
    }

    void SetCallback(const std::function<void(hsa_api_args_t *, const sibir_api_phase_t, const hsa_api_id_t)> &callback) {
        callback_ = callback;
    }

    bool captureHsaApiTable(HsaApiTable* table) {
        installCoreApiWrappers(table->core_);
        installAmdExtWrappers(table->amd_ext_);
        installImageExtWrappers(table->image_ext_);
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
