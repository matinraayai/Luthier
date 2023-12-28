#ifndef HIP_INTERCEPT_HPP
#define HIP_INTERCEPT_HPP

#include "luthier_types.h"
#include <dlfcn.h>
#include <experimental/filesystem>
#include <functional>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <fmt/core.h>
#include <any>
#include <optional>
#include <unordered_set>

namespace fs = std::experimental::filesystem;

// Borrowed from RocTracer's BaseLoader
namespace luthier {
class HipInterceptor {
 private:
    void *handle_{nullptr};
    std::function<void(void *, const luthier_api_evt_phase_t, const int)> userCallback_{};
    std::function<void(void *, const luthier_api_evt_phase_t, const int, bool*, std::optional<std::any>*)> internalCallback_{};
    std::unordered_set<uint32_t> enabledOps_;

    HipInterceptor();

    ~HipInterceptor() {
        if (handle_ != nullptr) ::dlclose(handle_);
    }

 public:
    HipInterceptor(const HipInterceptor &) = delete;
    HipInterceptor &operator=(const HipInterceptor &) = delete;

    [[nodiscard]] bool IsEnabled() const { return handle_ != nullptr; }

    [[nodiscard]] const std::function<void(void *, const luthier_api_evt_phase_t, const int)> &getUserCallback() const {
        return userCallback_;
    }

    [[nodiscard]] const std::unordered_set<uint32_t> &getEnabledOps() const {
        return enabledOps_;
    }

    void SetUserCallback(const std::function<void(void *, const luthier_api_evt_phase_t, const int)> &callback) {
        userCallback_ = callback;
    }

    [[nodiscard]] const std::function<void(void *, const luthier_api_evt_phase_t, const int, bool*, std::optional<std::any>*)> &getInternalCallback() const {
        return internalCallback_;
    }
    void SetInternalCallback(const std::function<void(void *, const luthier_api_evt_phase_t, const int, bool*, std::optional<std::any>*)> &internal_callback) {
        internalCallback_ = internal_callback;
    }

    void enableCallback(uint32_t op) {
        //if (op < 0 || op > 192) throw std::invalid_argument("Op not in range [0, 192]");
        enabledOps_.insert(op);
    }

    void disableCallback(uint32_t op) {
        //if (op < 0 || op > 192) throw std::invalid_argument("Op not in range [0, 192]");
        enabledOps_.erase(op);
    }

    void enableAllCallback() {
        for (int i = static_cast<int>(HIP_API_ID_FIRST); i <= static_cast<int>(HIP_API_ID_LAST); ++i) {
            enableCallback(i);
        }
        for (int i = static_cast<int>(HIP_PRIVATE_API_ID_FIRST); i <= static_cast<int>(HIP_PRIVATE_API_ID_LAST); ++i) {
            enableCallback(i);
        }
    }

    void disableAllCallback() {
        enabledOps_.clear();
    }

    void *GetHipFunction(const char *symbol) const {
        assert(IsEnabled());

        void *function_ptr = ::dlsym(handle_, symbol);
        if (function_ptr == nullptr)
            throw std::runtime_error(fmt::format("symbol lookup '{:s}' failed: {:s}", std::string(symbol), std::string(::dlerror())));
        return function_ptr;
    }

    template<typename FunctionPtr>
    FunctionPtr GetHipFunction(const char *symbol) const {
        assert(IsEnabled());
        return reinterpret_cast<FunctionPtr>(GetHipFunction(symbol));
    }

    static inline HipInterceptor &Instance() {
        static HipInterceptor instance;
        return instance;
    }
};
};// namespace luthier

#endif
