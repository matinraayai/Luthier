#ifndef HSA_INTERCEPT_HPP
#define HSA_INTERCEPT_HPP

#include <hsa/hsa_ven_amd_loader.h>

#include <functional>

#include <llvm/ADT/DenseSet.h>

#include "error.hpp"
#include <luthier/hsa_trace_api.h>
#include <luthier/types.h>

#include <thread>

// Helper to store ApiEvtID in llvm::DenseSets
namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::ApiEvtID> {
  static inline luthier::hsa::ApiEvtID getEmptyKey() {
    return luthier::hsa::ApiEvtID(
        DenseMapInfo<
            std::underlying_type_t<luthier::hsa::ApiEvtID>>::getEmptyKey());
  }

  static inline luthier::hsa::ApiEvtID getTombstoneKey() {
    return luthier::hsa::ApiEvtID(
        DenseMapInfo<
            std::underlying_type_t<luthier::hsa::ApiEvtID>>::getTombstoneKey());
  }

  static unsigned getHashValue(const luthier::hsa::ApiEvtID &ApiID) {
    return DenseMapInfo<std::underlying_type_t<luthier::hsa::ApiEvtID>>::
        getHashValue(
            static_cast<std::underlying_type_t<luthier::hsa::ApiEvtID>>(ApiID));
  }

  static bool isEqual(const luthier::hsa::ApiEvtID &LHS,
                      const luthier::hsa::ApiEvtID &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

namespace luthier::hsa {

typedef std::function<void(ApiEvtArgs *, const luthier::ApiEvtPhase,
                           const ApiEvtID)>
    user_callback_t;

typedef std::function<void(ApiEvtArgs *, const luthier::ApiEvtPhase,
                           const ApiEvtID, bool *)>
    internal_callback_t;

class Interceptor {
private:
  HsaApiTable *InternalHsaApiTable{};
  HsaApiTableContainer SavedTables{};
  hsa_ven_amd_loader_1_03_pfn_s AmdTable{};
  llvm::DenseSet<ApiEvtID> EnabledUserOps{};
  llvm::DenseSet<ApiEvtID> EnabledInternalOps{};

  // create thread local variable here to switch between using/not using 20 and 21
  // needs to be static to be accepted by the class
  static thread_local bool EnableTempCallback;

  void tempDisableCallback() {
    EnableTempCallback = false;
  }

  void tempEnableCallback() {
    EnableTempCallback = true;
  }

  /* Task Steps:
     * 1. Define thread_local variable (bool) in hsa_interceptor.hpp & cpp
         *  If enable_temp_callback is false (disable) --> temporarily ignore enabledUserOps_ & enabledInternalOps_
         *  If enable_temp_callback is true (enable) --> leave enabledUserOps_ & enabledInternalOps_ alone
         *  Add #include<thread> to use thread-local (?)
     * 2. Create function luthier_temp_disable_hsa_callback()
         * set enable_temp_callback to false
     * 3. Create function luthier_temp_enable_hsa_callback()
         * set enable_temp_callback to true
     * 4. Create function isCallbackTempEnabled()
         * return enable_temp_callback
     * 5. Add isCallbackTempEnabled to shouldCallback variable in hsa_intercept.cpp
         * Q: Do we need to add it to every instance of shouldCallback within the hsa_intercept.cpp file?
         * // Step 5. don't touch this update python script
            bool isCallbackTempEnabled = hsaInterceptor.isCallbackTempEnabled();
            bool shouldCallback = (isUserCallbackEnabled || isInternalCallbackEnabled) && isCallbackTempEnabled;
     * */

  user_callback_t UserCallback{};
  internal_callback_t InternalCallback{};

  void installCoreApiTableWrappers(CoreApiTable *Table);

  void installAmdExtTableWrappers(AmdExtTable *Table);

  void installImageExtTableWrappers(ImageExtTable *Table);

  void installFinalizerExtTableWrappers(FinalizerExtTable *Table);

  Interceptor() = default;
  ~Interceptor() {
    uninstallApiTables();
    SavedTables = {};
    AmdTable = {};
  }

public:
  Interceptor(const Interceptor &) = delete;
  Interceptor &operator=(const Interceptor &) = delete;

  [[nodiscard]] const HsaApiTableContainer &getSavedHsaTables() const {
    return SavedTables;
  }

  [[nodiscard]] const hsa_ven_amd_loader_1_03_pfn_t &
  getHsaVenAmdLoaderTable() const {
    return AmdTable;
  }

  void uninstallApiTables() {
    *InternalHsaApiTable->core_ = SavedTables.core;
    *InternalHsaApiTable->amd_ext_ = SavedTables.amd_ext;
    *InternalHsaApiTable->finalizer_ext_ = SavedTables.finalizer_ext;
    *InternalHsaApiTable->image_ext_ = SavedTables.image_ext;
  }

  void setUserCallback(const user_callback_t &CB) { UserCallback = CB; }

  void setInternalCallback(const internal_callback_t &Callback) {
    InternalCallback = Callback;
  }

  [[nodiscard]] const inline user_callback_t &getUserCallback() const {
    return UserCallback;
  }

  [[nodiscard]] const inline internal_callback_t &getInternalCallback() const {
    return InternalCallback;
  }

  [[nodiscard]] bool isUserCallbackEnabled(ApiEvtID Op) const {
    return EnabledUserOps.contains(Op);
  }

  [[nodiscard]] bool isInternalCallbackEnabled(ApiEvtID Op) const {
    return EnabledInternalOps.contains(Op);
  }

  void enableUserCallback(ApiEvtID Op) { EnabledUserOps.insert(Op); }

  void disableUserCallback(ApiEvtID Op) { EnabledUserOps.erase(Op); }

  // Step 4.
  bool isCallbackTempEnabled() { return EnableTempCallback; }

  void enableInternalCallback(ApiEvtID Op) { EnabledInternalOps.insert(Op); }

  void disableInternalCallback(ApiEvtID Op) { EnabledInternalOps.erase(Op); }

  void enableAllUserCallbacks() {
    for (std::underlying_type<ApiEvtID>::type I = HSA_API_EVT_ID_FIRST;
         I <= HSA_API_EVT_ID_LAST; ++I) {
      enableUserCallback(ApiEvtID(I));
    }
  }
  void disableAllUserCallbacks() { EnabledUserOps.clear(); }

  void enableAllInternalCallbacks() {
    for (std::underlying_type<ApiEvtID>::type I = HSA_API_EVT_ID_FIRST;
         I <= HSA_API_EVT_ID_LAST; ++I) {
      enableInternalCallback(ApiEvtID(I));
    }
  }

  void disableAllInternalCallbacks() { EnabledInternalOps.clear(); }

  bool captureHsaApiTable(HsaApiTable *Table) {
    InternalHsaApiTable = Table;
    installCoreApiTableWrappers(Table->core_);
    installAmdExtTableWrappers(Table->amd_ext_);
    installImageExtTableWrappers(Table->image_ext_);
    installFinalizerExtTableWrappers(Table->finalizer_ext_);

    return (Table->core_->hsa_system_get_major_extension_table_fn(
                HSA_EXTENSION_AMD_LOADER, 1,
                sizeof(hsa_ven_amd_loader_1_03_pfn_t),
                &AmdTable) == HSA_STATUS_SUCCESS);
  }

  static inline hsa::Interceptor &instance() {
    static hsa::Interceptor Instance;
    return Instance;
  }
};
} // namespace luthier::hsa

#endif
