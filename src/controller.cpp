#include "controller.hpp"
#include "code_generator.hpp"
#include "code_lifter.hpp"
#include "hip_intercept.hpp"
#include "log.hpp"
#include "luthier/hip_trace_api.h"
#include "target_manager.hpp"
#include "tool_executable_manager.hpp"

#include "luthier/luthier.h"
#include "luthier/types.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/PrettyStackTrace.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-controller"

namespace luthier {

template <> Controller *Singleton<Controller>::Instance{nullptr};

Controller *Controller::C{nullptr};

namespace hip {
static void internalApiCallback(ApiArgs &Args, ApiReturn *Out,
                                ApiEvtPhase Phase, int ApiId) {
  LUTHIER_LOG_FUNCTION_CALL_START
  if (Phase == API_EVT_PHASE_BEFORE) {
    if (ApiId == hip::HIP_API_ID___hipRegisterFunction) {
      auto &COM = ToolExecutableManager::instance();
      auto &LastRFuncArgs = Args.__hipRegisterFunction;
      // If the function doesn't have __luthier_wrap__ in its name then it
      // belongs to the instrumented application or HIP can manage it on its own
      // since no device function is present to strip from it
      if (llvm::StringRef(LastRFuncArgs.deviceFunction)
              .find(luthier::HookHandlePrefix) != llvm::StringRef::npos) {
        COM.registerInstrumentationHookWrapper(LastRFuncArgs.hostFunction,
                                               LastRFuncArgs.deviceFunction);
      }
    }
  }
  LUTHIER_LOG_FUNCTION_CALL_END
}
} // namespace hip

namespace hsa {

void internalApiCallback(hsa::ApiEvtArgs *CBData, ApiEvtPhase Phase,
                         hsa::ApiEvtID ApiId, bool *SkipFunction) {
  LUTHIER_LOG_FUNCTION_CALL_START
  if (Phase == API_EVT_PHASE_AFTER &&
      ApiId == HSA_API_EVT_ID_hsa_executable_freeze) {
    hsa::Executable Exec(CBData->hsa_executable_destroy.executable);
    // Cache the executable and its items
    if (auto Err = Platform::instance().cacheExecutableOnExecutableFreeze(Exec))
      llvm::report_fatal_error("Tool executable register failed");
    // Check if the executable belongs to the tool and not the app
    if (auto Err =
            ToolExecutableManager::instance().registerIfLuthierToolExecutable(
                Exec)) {
      llvm::report_fatal_error("Tool executable check failed");
    }
  }
  if (Phase == API_EVT_PHASE_AFTER &&
      ApiId == HSA_API_EVT_ID_hsa_executable_load_agent_code_object) {
    // because the output of hsa_executable_load_agent_code_object can be set to
    // nullptr by the app, we have to access it by iterating over the LCOs of
    // the Exec it was created for
    hsa::Executable Exec(
        CBData->hsa_executable_load_agent_code_object.executable);
    if (auto Err =
            Platform::instance().cacheExecutableOnLoadedCodeObjectCreation(
                Exec)) {
      llvm::report_fatal_error("Caching of Loaded Code Object failed!");
    }
  }
  if (Phase == API_EVT_PHASE_BEFORE &&
      ApiId == HSA_API_EVT_ID_hsa_executable_destroy) {
    hsa::Executable Exec(CBData->hsa_executable_destroy.executable);
    if (auto Err =
            CodeLifter::instance().invalidateCachedExecutableItems(Exec)) {
      llvm::report_fatal_error("Executable cache invalidation failed");
    }

    if (auto Err = Platform::instance().invalidateExecutableOnExecutableDestroy(
            Exec)) {
      llvm::report_fatal_error("Executable cache invalidation failed");
    }
  }
  LUTHIER_LOG_FUNCTION_CALL_END
}
} // namespace hsa

static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                    uint64_t LibVersion, uint64_t LibInstance,
                                    void **Tables, uint64_t NumTables,
                                    void *Data) {
  if (Type == ROCPROFILER_HSA_TABLE) {
    LLVM_DEBUG(llvm::dbgs() << "Capturing the HSA API Tables.\n");
    auto &HsaInterceptor = luthier::hsa::Interceptor::instance();
    auto &HsaApiTableCaptureCallback =
        Controller::instance().getAtHSAApiTableCaptureEvtCallback();
    HsaApiTableCaptureCallback(API_EVT_PHASE_BEFORE);
    auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    HsaInterceptor.captureHsaApiTable(Table);
    HsaApiTableCaptureCallback(API_EVT_PHASE_AFTER);
    LLVM_DEBUG(llvm::dbgs() << "Captured the HSA API Tables.\n");

    HsaInterceptor.setInternalCallback(luthier::hsa::internalApiCallback);
    HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_freeze);
    HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_destroy);
    HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_load_agent_code_object);
  }
  if (Type == ROCPROFILER_HIP_COMPILER_TABLE) {
    LLVM_DEBUG(llvm::dbgs() << "Capturing the HIP Compiler API Table.\n");
    auto &HipInterceptor = hip::Interceptor::instance();
    auto *Table = static_cast<HipCompilerDispatchTable *>(Tables[0]);
    HipInterceptor.captureCompilerDispatchTable(Table);
    HipInterceptor.setInternalCallback(hip::internalApiCallback);
    HipInterceptor.enableInternalCallback(
        luthier::hip::HIP_API_ID___hipRegisterFunction);
    LLVM_DEBUG(llvm::dbgs() << "Captured the HIP Compiler API Table.\n");
  }
}

void rocprofilerFinalize(void *Data) {
  LUTHIER_LOG_FUNCTION_CALL_START
//  luthier::Controller::instance().getAtApiTableReleaseEvtCallback()(
//      API_EVT_PHASE_BEFORE);
//  luthier::hsa::Interceptor::instance().uninstallApiTables();
//  luthier::Controller::instance().getAtApiTableReleaseEvtCallback()(
//      API_EVT_PHASE_AFTER);
  LUTHIER_LOG_FUNCTION_CALL_END
}

Controller::Controller() : Singleton<Controller>() {
  // Initialize all the singletons
  CG = new CodeGenerator();
  COM = new ToolExecutableManager();
  CL = new CodeLifter();
  TM = new TargetManager();
  HipInterceptor = new hip::Interceptor();
}

Controller::~Controller() {
  delete CG;
  delete COM;
  delete CL;
  delete TM;
  delete HipInterceptor;
}
void Controller::init() {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    atToolInit(API_EVT_PHASE_BEFORE);
    C = new luthier::Controller();
    llvm::EnablePrettyStackTrace();
    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/matinraayai/Luthier/issues/ and "
                          "include the crash backtrace.\n");
    atToolInit(API_EVT_PHASE_AFTER);
  });
  llvm::EnablePrettyStackTraceOnSigInfoForThisThread(true);
}

void Controller::finalize() {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    atFinalization(API_EVT_PHASE_BEFORE);
    delete C;
    atFinalization(API_EVT_PHASE_AFTER);
  });
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ID) {
  ID->name = "Luthier";
  rocprofiler_at_intercept_table_registration(
      luthier::apiRegistrationCallback,
      ROCPROFILER_HSA_TABLE | ROCPROFILER_HIP_COMPILER_TABLE, nullptr);

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::rocprofilerFinalize, nullptr};
  return &Cfg;
}