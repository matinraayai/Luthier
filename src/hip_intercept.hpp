#ifndef HIP_INTERCEPT_HPP
#define HIP_INTERCEPT_HPP

#include <dlfcn.h>

#include <experimental/filesystem>
#include <functional>
#include <llvm/ADT/DenseSet.h>

#include "error.hpp"
#include <luthier/hip_trace_api.h>
#include <luthier/types.h>

namespace fs = std::experimental::filesystem;

// Helper to store ApiID in llvm::DenseSets
namespace llvm {

template <> struct DenseMapInfo<luthier::hip::ApiID> {
  static inline luthier::hip::ApiID getEmptyKey() {
    return luthier::hip::ApiID(
        DenseMapInfo<
            std::underlying_type_t<luthier::hip::ApiID>>::getEmptyKey());
  }

  static inline luthier::hip::ApiID getTombstoneKey() {
    return luthier::hip::ApiID(
        DenseMapInfo<
            std::underlying_type_t<luthier::hip::ApiID>>::getTombstoneKey());
  }

  static unsigned getHashValue(const luthier::hip::ApiID &ApiID) {
    return DenseMapInfo<std::underlying_type_t<luthier::hip::ApiID>>::
        getHashValue(
            static_cast<std::underlying_type_t<luthier::hip::ApiID>>(ApiID));
  }

  static bool isEqual(const luthier::hip::ApiID &LHS,
                      const luthier::hip::ApiID &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

// Borrowed from RocTracer's BaseLoader
namespace luthier::hip {

typedef std::function<void(ApiArgs &, ApiReturn *, const ApiEvtPhase,
                           const int)>
    internal_callback_t;

class Interceptor {
private:
  void *Handle{nullptr};
  //  std::function<void(void *, const ApiEvtPhase, const int)> UserCallback{};
  //  llvm::DenseSet<unsigned int> EnabledUserCallbacks{};

  internal_callback_t InternalCallback{};
  llvm::DenseSet<ApiID> EnabledInternalCallbacks{};

  Interceptor();

  ~Interceptor() {
    if (Handle != nullptr)
      ::dlclose(Handle);
  }

public:
  Interceptor(const Interceptor &) = delete;
  Interceptor &operator=(const Interceptor &) = delete;

  [[nodiscard]] bool isEnabled() const { return Handle != nullptr; }

  //  [[nodiscard]] const std::function<void(void *, const ApiEvtPhase, const
  //  int)>
  //      &
  //  getUserCallback() const {
  //    return UserCallback;
  //  }
  //
  //  [[nodiscard]] bool isUserCallbackEnabled(uint32_t op) const {
  //    return EnabledUserCallbacks.contains(op);
  //  }

  //  void setUserCallback(const std::function<void(void *, const ApiEvtPhase,
  //                                                const int)> &callback) {
  //    UserCallback = callback;
  //  }

  //  void enableUserCallback(uint32_t op) {
  //    if (!(op >= HIP_API_ID_FIRST && op <= HIP_API_ID_LAST ||
  //          op >= HIP_PRIVATE_API_ID_FIRST && op <= HIP_PRIVATE_API_ID_LAST))
  //      llvm::report_fatal_error(
  //          llvm::formatv(
  //              "Op ID {0} not in hip_api_id_t or hip_private_api_id_t."),
  //          op);
  //    EnabledUserCallbacks.insert(op);
  //  }

  //  void disableUserCallback(uint32_t op) {
  //    if (!(op >= HIP_API_ID_FIRST && op <= HIP_API_ID_LAST ||
  //          op >= HIP_PRIVATE_API_ID_FIRST && op <= HIP_PRIVATE_API_ID_LAST))
  //      llvm::report_fatal_error(
  //          llvm::formatv(
  //              "Op ID {0} not in hip_api_id_t or hip_private_api_id_t."),
  //          op);
  //    EnabledUserCallbacks.erase(op);
  //  }

  //  void enableAllUserCallbacks() {
  //    for (int i = static_cast<int>(HIP_API_ID_FIRST);
  //         i <= static_cast<int>(HIP_API_ID_LAST); ++i) {
  //      enableUserCallback(i);
  //    }
  //    for (int i = static_cast<int>(HIP_PRIVATE_API_ID_FIRST);
  //         i <= static_cast<int>(HIP_PRIVATE_API_ID_LAST); ++i) {
  //      enableUserCallback(i);
  //    }
  //  }
  //  void disableAllUserCallbacks() { EnabledUserCallbacks.clear(); }

  [[nodiscard]] const internal_callback_t &getInternalCallback() const {
    return InternalCallback;
  }

  [[nodiscard]] bool isInternalCallbackEnabled(ApiID Op) const {
    return EnabledInternalCallbacks.contains(Op);
  }

  void setInternalCallback(const internal_callback_t &CB) {
    InternalCallback = CB;
  }

  void enableInternalCallback(ApiID Op) { EnabledInternalCallbacks.insert(Op); }

  void disableInternalCallback(ApiID Op) { EnabledInternalCallbacks.erase(Op); }

  void enableAllInternalCallbacks() {
    for (std::underlying_type<ApiID>::type I = HIP_API_ID_FIRST;
         I <= HIP_API_ID_LAST; I++) {
      enableInternalCallback(ApiID(I));
    }
  }

  void disableAllInternalCallbacks() { EnabledInternalCallbacks.clear(); }

  void *getHipFunction(llvm::StringRef Symbol) const {
    LUTHIER_CHECK(isEnabled());

    void *functionPtr = ::dlsym(Handle, Symbol.data());
    if (functionPtr == nullptr)
      llvm::report_fatal_error(
          llvm::formatv("symbol lookup '{0:s}' failed: {1:s}",
                        std::string(Symbol), std::string(::dlerror())));
    return functionPtr;
  }

  template <typename FunctionPtr>
  FunctionPtr getHipFunction(const char *Symbol) const {
    LUTHIER_CHECK(isEnabled());
    return reinterpret_cast<FunctionPtr>(getHipFunction(Symbol));
  }

  static inline Interceptor &instance() {
    static Interceptor Instance;
    return Instance;
  }
};

} // namespace luthier::hip

#endif
