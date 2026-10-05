//===-- ApiTableRegistrationCallbackProvider.h -------------------*- C++-*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
/// \file
/// Defines the \c ApiTableRegistrationCallbackProvider class which provide
/// a callback to its user or sub-classes when an API table is registered with
/// rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_API_TABLE_REGISTRATION_CALLBACK_PROVIDER_H
#define LUTHIER_ROCPROFILER_API_TABLE_REGISTRATION_CALLBACK_PROVIDER_H
#include "luthier/Common/DynamicLibrary.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/Rocprofiler/ApiTable.h"
#include "luthier/Rocprofiler/ApiTableEnumInfo.h"
#include "luthier/Rocprofiler/RocprofilerError.h"
#include <atomic>
#include <exception>
#include <mutex>
#include <type_traits>

namespace luthier::rocprofiler {

/// \c ApiTableEnumInfo is the library-specific customization point declared in
/// ApiTableEnumInfo.h. Its specializations live in HsaApiTableEnumInfo.h and
/// HipApiTableEnumInfo.h; this header depends only on the primary template, so
/// it pulls in neither HSA nor HIP headers.

/// \brief a generic class used to safely request a callback to be invoked when
/// an Api table is registered with rocprofiler-sdk. Can be either used as is
/// or inherited by a sub-class
template <rocprofiler_intercept_table_t TableType>
class ApiTableRegistrationCallbackProvider {
private:
  /// Set to \c true the first time \c std::terminate fires after
  /// \c installHostAbortedSentinelOnce runs. Read by the destructor to
  /// distinguish "the host aborted before rocprofiler could call our
  /// callback" (warning-worthy) from "rocprofiler is up but never called
  /// us" (programming bug). We need a side channel because by the time
  /// the destructor runs, the C++ runtime has cleared the in-flight
  /// exception (LLVM's signal handler intercepts \c SIGABRT and exits
  /// cleanly, triggering normal static-destructor unwind — so
  /// \c std::uncaught_exceptions() returns 0).
  ///
  /// Each template instantiation has its own copy; that's fine because
  /// the chained terminate handlers below set every instantiation's
  /// flag in order, so all destructors observe the abort regardless of
  /// which one's handler was the outermost.
  inline static std::atomic<bool> HostAbortedDuringExecution{false};
  /// Previous terminate handler captured at install time so this
  /// instantiation's handler can chain to it. \c std::set_terminate
  /// takes a plain function pointer (not a \c std::function), so the
  /// handler can't close over this — it has to be a static field the
  /// handler reads directly.
  inline static std::atomic<std::terminate_handler> PrevTerminateHandler{
      nullptr};

  /// Both sentinels have static storage duration and are read/written by the
  /// chained terminate handler, which can fire arbitrarily late during program
  /// shutdown. Keeping them trivially destructible guarantees no static
  /// destructor is registered for them, so the handler can never touch a
  /// destroyed object during exit.
  static_assert(
      std::is_trivially_destructible_v<decltype(HostAbortedDuringExecution)>,
      "HostAbortedDuringExecution must remain trivially destructible");
  static_assert(
      std::is_trivially_destructible_v<decltype(PrevTerminateHandler)>,
      "PrevTerminateHandler must remain trivially destructible");

  static void hostAbortedTerminateHandler() noexcept {
    HostAbortedDuringExecution.store(true);
    if (auto Prev = PrevTerminateHandler.load())
      Prev();
    std::abort();
  }

  static void installHostAbortedSentinelOnce() {
    static std::once_flag Installed;
    std::call_once(Installed, [] {
      /// Chain to whatever terminate handler is currently installed so
      /// we don't suppress LLVM's stack-trace dump or another
      /// instantiation's sentinel — we just observe and pass through.
      PrevTerminateHandler.store(
          std::set_terminate(hostAbortedTerminateHandler));
    });
  }

  /// Whether \c rocprofiler_at_intercept_table_registration succeeded for this
  /// object. If registration failed, rocprofiler never stored our \c this
  /// pointer, hence it is safe to destroy the object
  bool SuccessfullyRegistered = false;

protected:
  /// The rocprofiler-sdk library used to request the registration callback.
  const DynamicLibrary &RocprofilerLib;

  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk
  std::atomic<bool> WasRegistrationInvoked{false};

  using CallbackType = std::function<void(
      llvm::ArrayRef<typename ApiTableEnumInfo<TableType>::ApiTableType *>
          Tables,
      uint64_t LibVersion, uint64_t LibInstance)>;

  /// Callback invoked inside the registration callback
  const CallbackType Callback;

  /// API table registration callback for used by rocprofiler-sdk
  /// \note Declared \c noexcept: rocprofiler-sdk invokes this from C code
  /// (\c execute_intercepts) between a raw \c mutex.lock()/unlock() with no
  /// RAII, so an exception escaping here would both leave that mutex locked
  /// (deadlocking later registrations) and unwind across the C ABI (undefined
  /// behavior). \c noexcept turns any unexpected escape (e.g. a \c
  /// std::bad_alloc from \c llvm::formatv under OOM) into a clean
  /// \c std::terminate instead.
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data) noexcept {
    /// Check for errors
    if (NumTables != ApiTableEnumInfo<TableType>::NumApiTables) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
          llvm::formatv("Expected rocprofiler to register {0} API table(s), "
                        "instead got {1}",
                        ApiTableEnumInfo<TableType>::NumApiTables, NumTables)));
    }
    if (Type != TableType) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          LUTHIER_MAKE_ROCPROFILER_ERROR(llvm::formatv(
              "Expected to get API table of type {0}, but instead got "
              "{1}",
              TableType, Type)));
    }
    if (Tables == nullptr) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
          "API tables passed by rocprofiler is nullptr"));
    }

    auto &RegProvider =
        *static_cast<ApiTableRegistrationCallbackProvider *>(Data);

    llvm::ArrayRef TablesAsArrayRef(
        reinterpret_cast<typename ApiTableEnumInfo<TableType>::ApiTableType **>(
            Tables),
        NumTables);

    for (const auto *Table : TablesAsArrayRef) {
      if (!Table) {
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
            "API table passed by rocprofiler is nullptr"));
      }
    }

    RegProvider.Callback(TablesAsArrayRef, LibVersion, LibInstance);
    RegProvider.WasRegistrationInvoked.store(true);
  }

public:
  /// Constructor
  /// \note Must only be invoked before rocprofiler-sdk has been
  /// fully configured (i.e, inside the \c rocprofiler_configure function)
  /// \param CB The callback to be invoked once rocprofiler-sdk has reported
  /// back with the requested API table. The callback provides the following
  /// arguments: a) A list of pointers of the API tables passed by
  /// rocprofiler-sdk; These pointers are checked to not be \c nullptr before
  /// they are passed to the client. b)\c uint64_t version of the library as
  /// passed by rocprofiler-sdk; and c) \c uint64_t indicating the number of
  /// times this library has registered itself with rocprofiler-sdk before. The
  /// class also checks if rocprofiler-sdk has passed the correct number of
  /// tables expected for the library of choice, as described in the \c
  /// ApiTableEnumInfo of the \c TableType before invoking the \p CB
  /// \param RocprofilerLib the rocprofiler-sdk library of the target
  /// application used to request the callback; Must outlive this object
  /// \param Err an externally initialized \c llvm::Error that will report
  /// back any errors encountered by this constructor
  ApiTableRegistrationCallbackProvider(CallbackType CB,
                                       const DynamicLibrary &RocprofilerLib,
                                       llvm::Error &Err)
      : RocprofilerLib(RocprofilerLib), Callback(std::move(CB)) {
    llvm::ErrorAsOutParameter EAO(Err);
    /// Install once on first construction so the destructor can later
    /// distinguish a host-side abort from a real "rocprofiler never
    /// called us" bug.
    installHostAbortedSentinelOnce();
    Err = std::move(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        RocprofilerLib
            .callFunction<rocprofiler_at_intercept_table_registration>(
                ApiTableRegistrationCallbackProvider::apiRegistrationCallback,
                TableType, this),
        llvm::formatv("Failed to request a callback on {0} API table "
                      "initialization from "
                      "rocprofiler-sdk",
                      ApiTableEnumInfo<TableType>::ApiTableName)));
    SuccessfullyRegistered = !Err.operator bool();
  };

  /// Destructor
  /// \note The object must only be destroyed only if: a) rocprofiler is
  /// finalizing or finalized - the registration callback can no longer fire,
  /// since \c rocprofiler_set_api_table is ignored once finalization has begun;
  /// or b) the destructor is running while the host is aborting (stack
  /// unwinding from an uncaught exception / \c std::terminate) — the program is
  /// on its way out, so an unfulfilled callback is a symptom of the abort, not
  /// a Luthier bug.
  ///
  /// Destroying outside these cases is a use-after-free hazard: rocprofiler-sdk
  /// cannot de-register the callback (it stores this object's \c this pointer
  /// in an append-only list) and will invoke it with the dangling pointer the
  /// next time the runtime registers its API table.
  /// \note Having already received the callback is NOT a safe-to-destroy
  /// condition: rocprofiler re-invokes it on every subsequent registration
  /// (e.g. an HSA finalize→re-init bumps the library instance and re-fires),
  /// so finalization is the only state in which it can no longer fire.
  /// \note If registration with rocprofiler-sdk failed in the constructor, none
  /// of the above applies — rocprofiler never stored our \c this, so the object
  /// is unconditionally safe to destroy.
  virtual ~ApiTableRegistrationCallbackProvider() {
    /// Registration never took effect, so rocprofiler-sdk holds no pointer to
    /// this object — there is nothing to reconcile.
    if (!SuccessfullyRegistered)
      return;

    int RocprofilerFiniStatus = 0;
    /// Abort rather than exit: this runs in a destructor that may execute
    /// during program shutdown, where exit() would re-enter the exit sequence.
    LUTHIER_ABORT_ON_FATAL_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        RocprofilerLib.callFunction<rocprofiler_is_finalized>(
            &RocprofilerFiniStatus),
        "Failed to check rocprofiler's finalization status."));
    /// Case (a): rocprofiler is finalizing or finalized — expected
    /// shutdown path, regardless of whether our callback fired.
    if (RocprofilerFiniStatus != 0)
      return;

    /// Case (b): the host called \c std::terminate (typically via an
    /// uncaught exception, a noexcept violation, or an unhandled
    /// signal that the runtime translated into a terminate). The
    /// program is on its way out via abort; an unfulfilled callback in
    /// that path is a consequence of the host crashing, not a Luthier
    /// bug. We can't rely on \c std::uncaught_exceptions() here
    /// because by the time our destructor runs (via static-destructor
    /// unwind from LLVM's signal handler's exit path) the exception
    /// state has already been cleared — so we use a side-channel flag
    /// recorded by a chained \c std::set_terminate handler.
    if (HostAbortedDuringExecution.load()) {
      luthier::errs() << "[ApiTableRegistrationCallbackProvider] warning: "
                      << "destroyed after host-side std::terminate; "
                      << "rocprofiler-sdk had not invoked the registration "
                      << "callback yet\n";
      return;
    }

    /// Rocprofiler is not finalizing and the host is not aborting, but the
    /// callback was never invoked — destroying now leaves a dangling
    /// registration that rocprofiler will invoke on the next API table
    /// registration. Abort (not exit) since we may be inside shutdown.
    LUTHIER_ABORT_ON_FATAL_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
        "ApiTableRegistrationCallbackProvider destroyed while "
        "rocprofiler-sdk is not finalizing, but the "
        "registration callback was never invoked."));
  }

  /// Checks whether rocprofiler-sdk has invoked the registration callback and
  /// the requested user callback has been invoked
  [[nodiscard]] bool wasRegistrationCallbackInvoked() const {
    return WasRegistrationInvoked.load();
  }
};

} // namespace luthier::rocprofiler

#endif