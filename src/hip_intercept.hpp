#ifndef HIP_INTERCEPT_HPP
#define HIP_INTERCEPT_HPP

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <any>
#include <experimental/filesystem>
#include <functional>
#include <optional>
#include <unordered_set>

#include "error.hpp"
#include <luthier/types.h>

namespace fs = std::experimental::filesystem;

// Borrowed from RocTracer's BaseLoader
namespace luthier {
class HipInterceptor {
private:
  void *Handle{nullptr};
  std::function<void(void *, const ApiEvtPhase, const int)> UserCallback{};
  std::function<void(void *, const ApiEvtPhase, const int, bool *,
                     std::optional<std::any> *)>
      InternalCallback{};
  std::unordered_set<unsigned int> enabledUserCallbacks_;
  std::unordered_set<unsigned int> enabledInternalCallbacks_;

  HipInterceptor();

  ~HipInterceptor() {
    if (Handle != nullptr)
      ::dlclose(Handle);
  }

public:
  HipInterceptor(const HipInterceptor &) = delete;
  HipInterceptor &operator=(const HipInterceptor &) = delete;

  [[nodiscard]] bool isEnabled() const { return Handle != nullptr; }

  [[nodiscard]] const std::function<void(void *, const ApiEvtPhase,
                                         const int)> &
  getUserCallback() const {
    return UserCallback;
  }

  [[nodiscard]] const std::function<void(void *, const ApiEvtPhase,
                                         const int, bool *,
                                         std::optional<std::any> *)> &
  getInternalCallback() const {
    return InternalCallback;
  }

  [[nodiscard]] bool isUserCallbackEnabled(uint32_t op) const {
    return enabledUserCallbacks_.contains(op);
  }

  [[nodiscard]] bool isInternalCallbackEnabled(uint32_t op) const {
    return enabledInternalCallbacks_.contains(op);
  }

  void setUserCallback(
      const std::function<void(void *, const ApiEvtPhase,
                               const int)> &callback) {
    UserCallback = callback;
  }

  void setInternalCallback(
      const std::function<void(void *, const luthier::Api, const int, bool *,
                               std::optional<std::any> *)> &internalCallback) {
    InternalCallback = internalCallback;
  }

  void enableUserCallback(uint32_t op) {
    if (!(op >= HIP_API_ID_FIRST && op <= HIP_API_ID_LAST ||
          op >= HIP_PRIVATE_API_ID_FIRST && op <= HIP_PRIVATE_API_ID_LAST))
      llvm::report_fatal_error(
          llvm::formatv(
              "Op ID {0} not in hip_api_id_t or hip_private_api_id_t."),
          op);
    enabledUserCallbacks_.insert(op);
  }

  void enableInternalCallback(uint32_t op) {
    if (!(op >= HIP_API_ID_FIRST && op <= HIP_API_ID_LAST ||
          op >= HIP_PRIVATE_API_ID_FIRST && op <= HIP_PRIVATE_API_ID_LAST))
      llvm::report_fatal_error(
          llvm::formatv(
              "Op ID {0} not in hip_api_id_t or hip_private_api_id_t."),
          op);
    enabledInternalCallbacks_.insert(op);
  }

  void disableUserCallback(uint32_t op) {
    if (!(op >= HIP_API_ID_FIRST && op <= HIP_API_ID_LAST ||
          op >= HIP_PRIVATE_API_ID_FIRST && op <= HIP_PRIVATE_API_ID_LAST))
      llvm::report_fatal_error(
          llvm::formatv(
              "Op ID {0} not in hip_api_id_t or hip_private_api_id_t."),
          op);
    enabledUserCallbacks_.erase(op);
  }

  void disableInternalCallback(uint32_t op) {
    if (!(op >= HIP_API_ID_FIRST && op <= HIP_API_ID_LAST ||
          op >= HIP_PRIVATE_API_ID_FIRST && op <= HIP_PRIVATE_API_ID_LAST))
      llvm::report_fatal_error(
          llvm::formatv(
              "Op ID {0} not in hip_api_id_t or hip_private_api_id_t."),
          op);
    enabledInternalCallbacks_.erase(op);
  }

  void enableAllUserCallbacks() {
    for (int i = static_cast<int>(HIP_API_ID_FIRST);
         i <= static_cast<int>(HIP_API_ID_LAST); ++i) {
      enableUserCallback(i);
    }
    for (int i = static_cast<int>(HIP_PRIVATE_API_ID_FIRST);
         i <= static_cast<int>(HIP_PRIVATE_API_ID_LAST); ++i) {
      enableUserCallback(i);
    }
  }

  void enableAllInternalCallbacks() {
    for (int i = static_cast<int>(HIP_API_ID_FIRST);
         i <= static_cast<int>(HIP_API_ID_LAST); ++i) {
      enableInternalCallback(i);
    }
    for (int i = static_cast<int>(HIP_PRIVATE_API_ID_FIRST);
         i <= static_cast<int>(HIP_PRIVATE_API_ID_LAST); ++i) {
      enableInternalCallback(i);
    }
  }

  void disableAllUserCallbacks() { enabledUserCallbacks_.clear(); }

  void disableAllInternalCallbacks() { enabledInternalCallbacks_.clear(); }

  void *getHipFunction(const char *symbol) const {
    LUTHIER_CHECK(isEnabled());

    void *functionPtr = ::dlsym(Handle, symbol);
    if (functionPtr == nullptr)
      llvm::report_fatal_error(
          llvm::formatv("symbol lookup '{0:s}' failed: {1:s}",
                        std::string(symbol), std::string(::dlerror())));
    return functionPtr;
  }

  template <typename FunctionPtr>
  FunctionPtr getHipFunction(const char *symbol) const {
    LUTHIER_CHECK(isEnabled());
    return reinterpret_cast<FunctionPtr>(getHipFunction(symbol));
  }

  static inline HipInterceptor &instance() {
    static HipInterceptor instance;
    return instance;
  }
};
}; // namespace luthier

#endif
