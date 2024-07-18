#ifndef HSA_INTERCEPT_HPP
#define HSA_INTERCEPT_HPP

#include <hsa/hsa_ven_amd_loader.h>

#include <functional>

#include <llvm/ADT/DenseSet.h>

#include "common/error.hpp"
#include <luthier/hsa_trace_api.h>
#include <luthier/types.h>
#include "singleton.hpp"

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

class Interceptor : public Singleton<Interceptor> {
private:
  HsaApiTable *InternalHsaApiTable{};
  HsaApiTableContainer SavedTables{};
  hsa_ven_amd_loader_1_03_pfn_s AmdTable{};
  llvm::DenseSet<ApiEvtID> EnabledUserOps{};
  llvm::DenseSet<ApiEvtID> EnabledInternalOps{};

  user_callback_t UserCallback{
      [](ApiEvtArgs *, const luthier::ApiEvtPhase, const ApiEvtID) {}};
  internal_callback_t InternalCallback{};

  void installCoreApiTableWrappers(CoreApiTable *Table);

  void installAmdExtTableWrappers(AmdExtTable *Table);

  void installImageExtTableWrappers(ImageExtTable *Table);

  void installFinalizerExtTableWrappers(FinalizerExtTable *Table);

public:

  Interceptor() = default;
  ~Interceptor() {
    uninstallApiTables();
    SavedTables = {};
    AmdTable = {};
    Singleton<Interceptor>::~Singleton();
  }

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
};
} // namespace luthier::hsa

#endif
