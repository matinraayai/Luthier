#ifndef HIP_INTERCEPT_HPP
#define HIP_INTERCEPT_HPP

#include <dlfcn.h>
#include <fmt/core.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <any>
#include <experimental/filesystem>
#include <functional>
#include <optional>

#include "luthier_types.h"

namespace fs = std::experimental::filesystem;

// Borrowed from RocTracer's BaseLoader
namespace luthier {
class HipInterceptor {
 private:
    void *handle_{nullptr};
    std::function<void(void *, const luthier_api_evt_phase_t, const int)> userCallback_{
        [](void *, const luthier_api_evt_phase_t, const int) {}};
    std::function<void(void *, const luthier_api_evt_phase_t, const int, bool *, std::optional<std::any> *)>
        internalCallback_{[](void *, const luthier_api_evt_phase_t, const int, bool *, std::optional<std::any> *) {}};

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

    void SetUserCallback(const std::function<void(void *, const luthier_api_evt_phase_t, const int)> &callback) {
        userCallback_ = callback;
    }

    [[nodiscard]] const std::function<void(void *, const luthier_api_evt_phase_t, const int, bool *,
                                           std::optional<std::any> *)> &
    getInternalCallback() const {
        return internalCallback_;
    }
    void SetInternalCallback(const std::function<void(void *, const luthier_api_evt_phase_t, const int, bool *,
                                                      std::optional<std::any> *)> &internal_callback) {
        internalCallback_ = internal_callback;
    }

    void *GetHipFunction(const char *symbol) const {
        assert(IsEnabled());

        void *function_ptr = ::dlsym(handle_, symbol);
        if (function_ptr == nullptr)
            throw std::runtime_error(
                fmt::format("symbol lookup '{:s}' failed: {:s}", std::string(symbol), std::string(::dlerror())));
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
