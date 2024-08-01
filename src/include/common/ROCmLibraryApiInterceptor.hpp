//===-- ROCmLibraryApiInterceptor.hpp - ROCMm Interceptor Interface -------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains Luthier's ROCm library API table interceptor interface,
/// which provides the basics of all other API table interceptor singletons
/// in Luthier.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_COMMON_ROCM_LIBRARY_API_INTERCEPT_HPP
#define LUTHIER_COMMON_ROCM_LIBRARY_API_INTERCEPT_HPP

#include <functional>
#include <llvm/ADT/DenseSet.h>
#include <shared_mutex>

#include "common/error.hpp"
#include <luthier/llvm_dense_map_info.h>

namespace luthier {

template <typename ApiIDEnumType, typename ApiArgsType, typename ApiTableType,
          typename ApiTableContainerType>
class ROCmLibraryApiInterceptor {
public:
  /// typedef for the callback functions used by the interceptor
  typedef std::function<void(ApiArgsType *, const ApiEvtPhase,
                             const ApiIDEnumType)>
      callback_t;

protected:
  /// Pointer to where the intercepted runtime stores the API table;
  /// Modifying fields of this struct will cause the intercepted runtime to call
  /// different functions
  ApiTableType *RuntimeApiTable{};
  /// A table container with all its fields pointing to the "actual" API
  /// functions; Depending on the runtime, its type can be the same as the
  /// \c ApiTableType
  ApiTableContainerType SavedRuntimeApiTable{};
  /// Mutex to protect \c EnabledUserOps, \c EnabledInternalOps, and enabling
  /// disabling capturing of each Op
  mutable std::shared_mutex EnabledOpsMutex;
  /// Set of API functions set to be intercepted by the user of the tool
  llvm::DenseSet<ApiIDEnumType> EnabledUserOps{};
  /// Set of API functions set to be intercepted by Luthier internally
  llvm::DenseSet<ApiIDEnumType> EnabledInternalOps{};
  /// Mutex to protect Callback functions
  mutable std::shared_mutex CallbackMutex;
  /// Callback requested by the user to be performed on interception of each
  /// enabled API
  callback_t UserCallback{
      [](ApiArgsType *, const luthier::ApiEvtPhase, const ApiIDEnumType) {}};
  /// Callback requested by Luthier internally to be performed on interception
  /// of each enabled API
  callback_t InternalCallback{
      [](ApiArgsType *, const luthier::ApiEvtPhase, const ApiIDEnumType) {}};
  /// Ensures the \c freezeRuntimeApiTable will be performed only once
  std::once_flag FreezeRuntimeApiTableFlag{};
  /// Keeps track of whether the runtime API table has been frozen or not
  bool IsRuntimeApiTableFrozen{false};

public:
  ROCmLibraryApiInterceptor() = default;
  ~ROCmLibraryApiInterceptor() = default;

  ROCmLibraryApiInterceptor(const ROCmLibraryApiInterceptor &) = delete;
  ROCmLibraryApiInterceptor &
  operator=(const ROCmLibraryApiInterceptor &) = delete;

  /// \return a const reference to the saved API table container with
  /// the "actual" API functions
  [[nodiscard]] const ApiTableContainerType &getSavedApiTableContainer() const {
    return SavedRuntimeApiTable;
  }

  /// Sets the callback function \p CB to be performed on each captured API
  /// for the user of the Luthier tool
  /// \param CB callback to be performed
  /// \note this function acquires a unique lock over the callback
  /// mutex; Hence any read locks over the callbacks must be released first
  void setUserCallback(const callback_t &CB) {
    std::unique_lock Lock(CallbackMutex);
    UserCallback = CB;
  }

  /// Sets the callback function \p CB to be performed on each captured API
  /// internally by Luthier
  /// \param CB callback to be performed
  /// \note this function acquires a unique lock over the callback
  /// mutex; Hence any read locks over the callbacks must be released first
  void setInternalCallback(const callback_t &CB) {
    std::unique_lock Lock(CallbackMutex);
    InternalCallback = CB;
  }

  /// \return a reference to the user callback
  /// \note this function is not thread-safe; Call
  [[nodiscard]] inline std::pair<const callback_t *,
                                 std::shared_lock<std::shared_mutex>>
  getUserCallback() const {
    std::shared_lock Lock(CallbackMutex);
    return std::make_pair(&UserCallback, std::move(Lock));
  }

  /// \return a reference to the internal Luthier callback
  [[nodiscard]] inline std::pair<const callback_t *,
                                 std::shared_lock<std::shared_mutex>>
  getInternalCallback() const {
    std::shared_lock Lock(CallbackMutex);
    return std::make_pair(&InternalCallback, std::move(Lock));
  }

  /// Checks if the Luthier tool user will get a callback every time a function
  /// for API \p Op is captured
  /// \param Op the API enum queried
  /// \return true if the Luthier tool user will get a callback, false otherwise
  [[nodiscard]] bool isUserCallbackEnabled(ApiIDEnumType Op) const {
    std::shared_lock Lock(EnabledOpsMutex);
    return EnabledUserOps.contains(Op);
  }

  /// Checks if the Luthier tool internally will get a callback every time a
  /// function for API \p Op is captured
  /// \param Op the API enum queried
  /// \return true if the Luthier tool will get a callback, false otherwise
  [[nodiscard]] bool isInternalCallbackEnabled(ApiIDEnumType Op) const {
    std::shared_lock Lock(EnabledOpsMutex);
    return EnabledInternalOps.contains(Op);
  }

  /// \brief Freezes the runtime API table this interceptor is in charge of
  /// \details Before this function is called, enabling and disabling
  /// callbacks is done by directly installing/uninstalling callback functions
  /// in the runtime's API table. This is safe because at capture time, the
  /// target runtime has not started up yet and therefore, is not used. \n
  /// Calling this function indicates that it is no longer safe to make
  /// changes directly to the target runtime's API tables. Although
  /// enable/disable API functions can be used after this point, they no longer
  /// modify the API tables, and only modify the set of APIs the interceptor has
  /// to perform callbacks for. Wrapper functions then check for the
  /// enabled/disabled APIs manually. \n
  /// This function should be called everytime a wrapper function of this
  /// runtime has been invoked, to indicate the runtime has started up. It can
  /// also be invoked manually. \n
  /// Subsequent calls to this function does not do anything
  void freezeRuntimeApiTable() {
    std::call_once(FreezeRuntimeApiTableFlag,
                   [&]() { IsRuntimeApiTableFrozen = true; });
  }

  /// If successful, enables callbacks for the Luthier tool user every time the
  /// API of type \p Op is captured by the interceptor
  /// \note This function is almost certain to be successful if called before the
  /// runtime api table is frozen; After freezing the api table however, if the
  /// \p Op has not been set to be captured before (either internally by
  /// Luthier or externally by the tool user), it will fail. This is because
  /// at that point, it will be too late to install a wrapper function for the
  /// \p Op
  /// \param Op the API enum to be captured
  /// \returns true if the callback has been successfully enabled; false if
  /// the API table has been frozen and a wrapper cannot be installed
  virtual bool enableUserCallback(ApiIDEnumType Op) = 0;

  /// Disables callbacks for the Luthier tool user every time the API of
  /// type \p Op is captured by the interceptor
  /// \param Op the API enum to be captured
  virtual void disableUserCallback(ApiIDEnumType Op) = 0;

  /// If successful, enables callbacks for the Luthier tool internally every time
  /// the API of type \p Op is captured by the interceptor
  /// \note This function is almost certain to be successful if called before the
  /// runtime api table is frozen; After freezing the api table however, if the
  /// \p Op has not been set to be captured before (either internally by
  /// Luthier or externally by the tool user), it will fail. This is because
  /// at that point, it will be too late to install a wrapper function for the
  /// \p Op
  /// \param Op the API enum to be captured
  /// \returns true if the callback has been successfully enabled; false if
  /// the API table has been frozen and a wrapper cannot be installed
  virtual bool enableInternalCallback(ApiIDEnumType Op) = 0;

  /// Disables callbacks for the Luthier tool internally every time the API of
  /// type \p Op is captured by the interceptor
  /// \param Op the API enum to be captured
  virtual void disableInternalCallback(ApiIDEnumType Op) = 0;

  /// Called by rocprofiler when the API table of choice has been captured and
  /// passed to Luthier
  /// \param Table the pointer to the API table the target runtime uses to
  /// dispatch functions
  /// \returns an \c llvm::Error describing if the operation was successful or
  /// not
  virtual llvm::Error captureApiTable(ApiTableType *Table) = 0;
};

} // namespace luthier

#endif
