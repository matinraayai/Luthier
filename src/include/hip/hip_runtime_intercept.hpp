//===-- hip_runtime_intercept.hpp - Luthier's HIP API Interceptor
//-----------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains Luthier's HIP API Interceptor Singleton, implemented
/// using the rocprofiler-sdk API for capturing HIP compiler and runtime API
/// tables.
//===----------------------------------------------------------------------===//

#ifndef HIP_RUNTIME_INTERCEPT_HPP
#define HIP_RUNTIME_INTERCEPT_HPP

#include <functional>
#include <hip/amd_detail/hip_api_trace.hpp>
#include <llvm/ADT/DenseSet.h>

#include "common/error.hpp"
#include "common/singleton.hpp"
#include <luthier/hip_trace_api.h>
#include <luthier/types.h>
#include <rocprofiler-sdk/hip/api_args.h>
#include <rocprofiler-sdk/hip/api_id.h>

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
template <> struct DenseMapInfo<rocprofiler_hip_runtime_api_id_t> {
  static inline rocprofiler_hip_runtime_api_id_t getEmptyKey() {
    return rocprofiler_hip_runtime_api_id_t(
        DenseMapInfo<std::underlying_type_t<rocprofiler_hip_runtime_api_id_t>>::
            getEmptyKey());
  }

  static inline rocprofiler_hip_runtime_api_id_t getTombstoneKey() {
    return rocprofiler_hip_runtime_api_id_t(
        DenseMapInfo<std::underlying_type_t<rocprofiler_hip_runtime_api_id_t>>::
            getTombstoneKey());
  }

  static unsigned getHashValue(const rocprofiler_hip_runtime_api_id_t &ApiID) {
    return DenseMapInfo<
        std::underlying_type_t<rocprofiler_hip_runtime_api_id_t>>::
        getHashValue(static_cast<
                     std::underlying_type_t<rocprofiler_hip_runtime_api_id_t>>(
            ApiID));
  }

  static bool isEqual(const rocprofiler_hip_runtime_api_id_t &LHS,
                      const rocprofiler_hip_runtime_api_id_t &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

namespace luthier::hip {

typedef std::function<void(rocprofiler_hip_api_args_t *, const ApiEvtPhase,
                           const rocprofiler_hip_runtime_api_id_t, bool *)>
    runtime_internal_callback_t;

typedef std::function<void(rocprofiler_hip_api_args_t *, const ApiEvtPhase,
                           const rocprofiler_hip_runtime_api_id_t)>
    runtime_user_callback_t;

class RuntimeInterceptor : public Singleton<RuntimeInterceptor> {
private:
  HipDispatchTable SavedDispatchTable{};
  llvm::DenseSet<rocprofiler_hip_runtime_api_id_t> EnabledUserOps{};
  llvm::DenseSet<rocprofiler_hip_runtime_api_id_t> EnabledInternalOps{};

  runtime_user_callback_t UserCallback{};
  runtime_internal_callback_t InternalCallback{};

public:
  RuntimeInterceptor() = default;
  ~RuntimeInterceptor() {
    SavedDispatchTable = {};
    Singleton<RuntimeInterceptor>::~Singleton();
  }

  RuntimeInterceptor(const RuntimeInterceptor &) = delete;
  RuntimeInterceptor &operator=(const RuntimeInterceptor &) = delete;

  [[nodiscard]] const HipDispatchTable &getSavedRuntimeTable() const {
    return SavedDispatchTable;
  }

  void captureRuntimeTable(HipDispatchTable *RuntimeTable);

  [[nodiscard]] const inline runtime_user_callback_t &getUserCallback() const {
    return UserCallback;
  }

  [[nodiscard]] bool
  isUserCallbackEnabled(rocprofiler_hip_runtime_api_id_t op) const {
    return EnabledUserOps.contains(op);
  }

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

  [[nodiscard]] const runtime_internal_callback_t &getInternalCallback() const {
    return InternalCallback;
  }

  [[nodiscard]] bool
  isInternalCallbackEnabled(rocprofiler_hip_runtime_api_id_t Op) const {
    return EnabledInternalOps.contains(Op);
  }

  void setInternalCallback(const runtime_internal_callback_t &CB) {
    InternalCallback = CB;
  }

  void enableInternalCallback(rocprofiler_hip_runtime_api_id_t Op) {
    EnabledInternalOps.insert(Op);
  }

  void disableInternalCallback(rocprofiler_hip_runtime_api_id_t Op) {
    EnabledInternalOps.erase(Op);
  }

  void enableAllInternalCallbacks() {
    for (std::underlying_type<rocprofiler_hip_runtime_api_id_t>::type I =
             HIP_API_ID_FIRST;
         I <= HIP_API_ID_LAST; I++) {
      enableInternalCallback(rocprofiler_hip_runtime_api_id_t(I));
    }
  }

  void disableAllInternalCallbacks() { EnabledInternalOps.clear(); }
};

} // namespace luthier::hip

#endif
