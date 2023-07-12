#ifndef HIP_INTERCEPT_H_
#define HIP_INTERCEPT_H_

#include <hip/hip_runtime.h>
#include "sibir_types.h"
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <functional>
#include <experimental/filesystem>
#include <link.h>

namespace fs = std::experimental::filesystem;

// Borrowed from RocTracer's BaseLoader
class SibirHipInterceptor {
 private:
    void* handle_{nullptr};
    std::function<void(void*, const sibir_api_phase_t, const int)> callback_;

    SibirHipInterceptor() {
        // Iterate through the process' loaded shared objects and try to dlopen the first entry with a
        // file name starting with the given 'pattern'. This allows the loader to acquire a handle
        // to the target library iff it is already loaded. The handle is used to query symbols
        // exported by that library.

        auto callback = [this](dl_phdr_info* info) {
            if (handle_ == nullptr &&
                fs::path(info->dlpi_name).filename().string().rfind("libamdhip64.so", 0) == 0)
                handle_ = ::dlopen(info->dlpi_name, RTLD_LAZY);
        };
        dl_iterate_phdr(
            [](dl_phdr_info* info, size_t size, void* data) {
                (*reinterpret_cast<decltype(callback)*>(data))(info);
                return 0;
            },
            &callback);
    }

    ~SibirHipInterceptor() {
        if (handle_ != nullptr) ::dlclose(handle_);
    }

 public:
    SibirHipInterceptor(const SibirHipInterceptor &) = delete;
    SibirHipInterceptor & operator=(const SibirHipInterceptor &) = delete;

    [[nodiscard]] bool IsEnabled() const { return handle_ != nullptr; }

    [[nodiscard]] const std::function<void(void*, const sibir_api_phase_t, const int)>& getCallback() const {
        return callback_;
    }

    void SetCallback(const std::function<void(void*, const sibir_api_phase_t, const int)> &callback) {
        callback_ = callback;
    }

    void* GetHipFunction(const char* symbol) const {
        assert(IsEnabled());

        void* function_ptr = ::dlsym(handle_, symbol);
        if (function_ptr == nullptr)
            throw std::runtime_error("symbol lookup'" + std::string(symbol) + "' failed: " + std::string(::dlerror()));
        return function_ptr;
    }

    template <typename FunctionPtr> FunctionPtr GetHipFunction(const char* symbol) const {
        assert(IsEnabled());

        auto function_ptr = reinterpret_cast<FunctionPtr>(GetHipFunction(symbol));
        if (function_ptr == nullptr)
            throw std::runtime_error("symbol lookup'" + std::string(symbol) + "' failed: " + std::string(::dlerror()));
        return function_ptr;
    }

    static inline SibirHipInterceptor & Instance() {
        static SibirHipInterceptor instance;
        return instance;
    }
};


#endif
