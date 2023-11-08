#ifndef HSA_INTERCEPT_HPP
#define HSA_INTERCEPT_HPP

#include "error.h"
#include "luthier_types.hpp"
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hsa.h>

#include <functional>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <unordered_set>

namespace luthier {
class HsaInterceptor {
 private:
    HsaApiTableContainer savedTables_;
    HsaApiTableContainer interceptTables_;
    hsa_ven_amd_loader_1_03_pfn_s amdTable_;
    std::unordered_set<uint32_t> op_filters_;

    std::function<void(hsa_api_args_t *, const luthier_api_phase_t, const hsa_api_id_t)> callback_;

    void installCoreApiWrappers(CoreApiTable *table);

    void installAmdExtWrappers(AmdExtTable *table);

    void installImageExtWrappers(ImageExtTable* table);

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

    [[nodiscard]] const std::unordered_set<uint32_t> &getOpFiltersSet() const {
        return op_filters_;
    }

    void SetCallback(const std::function<void(hsa_api_args_t *, const luthier_api_phase_t, const hsa_api_id_t)> &callback) {
        callback_ = callback;
    }

    void enable_callback_impl(uint32_t op) {
        if (op < 0 || op > 192) throw std::invalid_argument("Op not in range [0, 192]");
        op_filters_.insert(op);
    }

    void disable_callback_impl(uint32_t op) {
        if (op < 0 || op > 192) throw std::invalid_argument("Op not in range [0, 192]");
        op_filters_.erase(op);
    }

    bool captureHsaApiTable(HsaApiTable *table) {
        installCoreApiWrappers(table->core_);
        installAmdExtWrappers(table->amd_ext_);
        installImageExtWrappers(table->image_ext_);
        LUTHIER_HSA_CHECK(table->core_->hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_03_pfn_t),
            &amdTable_));
        return true;
    }

    static inline HsaInterceptor &Instance() {
        static HsaInterceptor instance;
        return instance;
    }

    [[nodiscard]] const inline std::function<void(hsa_api_args_t *, const luthier_api_phase_t, const hsa_api_id_t)> &GetCallback() const {
        return callback_;
    }
};
}

#endif
