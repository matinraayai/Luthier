//===-- hip_intercept.hpp - Luthier's HIP API Interceptor -----------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains Luthier's HIP API Interceptor Singleton, implemented
/// using the rocprofiler-sdk API for capturing HIP compiler and runtime API
/// tables.
//===----------------------------------------------------------------------===//

#ifndef HIP_INTERCEPT_HPP
#define HIP_INTERCEPT_HPP

#include <functional>
#include <hip/amd_detail/hip_api_trace.hpp>
#include <llvm/ADT/DenseSet.h>

#include "error.hpp"
#include "singleton.hpp"
#include <luthier/hip_trace_api.h>
#include <luthier/types.h>

// TODO: 1. Overhaul Python generation script to use the new profiler API
//  enums + Generate Wrappers at CMake time by running the intercept generation
//  Python script + Enable HIP API runtime callbacks again
//  2. Move setting the callbacks to constructor arguments
//  3. Make a table of (APIID -> (Orig Func, Wrapper Func) in the auto generated
//  Python file to allow for complete uninstallation of wrapper functions when
//  No callback is requested from both the user and the tool
//  4. Update the Header of hip_intercept.cpp once the Python script is
//  overhauled

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

namespace luthier::hip {

typedef std::function<void(ApiArgs &, ApiReturn *, const ApiEvtPhase,
                           const int)>
    internal_callback_t;

class Interceptor : public Singleton<Interceptor> {
private:
  // Memory location of the Compiler Dispatch Table in use by the HIP runtime
  HipCompilerDispatchTable *LoadedCompilerDispatchTable{};
  // Memory location of the Runtime Dispatch Table in use by the HIP runtime
  HipDispatchTable *LoadedRuntimeDispatchTable{};

  // A saved copy of the original Compiler functions in HIP
  // (e.g. __hipRegisterFatBinary)
  HipCompilerDispatchTable SavedCompilerDispatchTable{};
  // A saved copy of the original runtime functions in HIP
  HipDispatchTable SavedDispatchTable{};

  //  std::function<void(void *, const ApiEvtPhase, const int)> UserCallback{};
  //  llvm::DenseSet<unsigned int> EnabledUserCallbacks{};

  internal_callback_t InternalCallback{};
  llvm::DenseSet<ApiID> EnabledInternalCallbacks{};

public:
  Interceptor() = default;

  Interceptor(const Interceptor &) = delete;
  Interceptor &operator=(const Interceptor &) = delete;

  [[nodiscard]] const HipCompilerDispatchTable &getSavedCompilerTable() const {
    return SavedCompilerDispatchTable;
  }

  [[nodiscard]] const HipDispatchTable &getSavedRuntimeTable() const {
    return SavedDispatchTable;
  }

  void captureCompilerDispatchTable(HipCompilerDispatchTable *CompilerTable);

  void captureRuntimeTable(HipDispatchTable *RuntimeTable);

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
};

} // namespace luthier::hip

#endif
