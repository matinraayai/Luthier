#ifndef HIP_INTERCEPT_HPP
#define HIP_INTERCEPT_HPP

#include "luthier_types.hpp"
#include <dlfcn.h>
#include <experimental/filesystem>
#include <functional>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <link.h>
#include <fmt/core.h>

namespace fs = std::experimental::filesystem;

// Borrowed from RocTracer's BaseLoader
namespace luthier {
class HipInterceptor {
 private:
    void *handle_{nullptr};
    std::function<void(void *, const luthier_api_phase_t, const int)> callback_;

    HipInterceptor() {
        // Iterate through the process' loaded shared objects and try to dlopen the first entry with a
        // file name starting with the given 'pattern'. This allows the loader to acquire a handle
        // to the target library iff it is already loaded. The handle is used to query symbols
        // exported by that library.

        auto callback = [this](dl_phdr_info *info) {
            if (handle_ == nullptr && fs::path(info->dlpi_name).filename().string().rfind("libamdhip64.so", 0) == 0)
                handle_ = ::dlopen(info->dlpi_name, RTLD_LAZY);
        };
        dl_iterate_phdr(
            [](dl_phdr_info *info, size_t size, void *data) {
                (*reinterpret_cast<decltype(callback) *>(data))(info);
                return 0;
            },
            &callback);
    }

    ~HipInterceptor() {
        if (handle_ != nullptr) ::dlclose(handle_);
    }

 public:
    HipInterceptor(const HipInterceptor &) = delete;
    HipInterceptor &operator=(const HipInterceptor &) = delete;

    [[nodiscard]] bool IsEnabled() const { return handle_ != nullptr; }

    [[nodiscard]] const std::function<void(void *, const luthier_api_phase_t, const int)> &getCallback() const {
        return callback_;
    }

    void SetCallback(const std::function<void(void *, const luthier_api_phase_t, const int)> &callback) {
        callback_ = callback;
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
