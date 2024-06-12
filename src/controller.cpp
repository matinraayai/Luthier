#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "controller.hpp"
#include "disassembler.hpp"
#include "hip_intercept.hpp"
#include "log.hpp"
#include "luthier/hip_trace_api.h"
#include "target_manager.hpp"

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
  if (Phase == API_EVT_PHASE_ENTER) {
    if (ApiId == hip::HIP_API_ID___hipRegisterFunction) {
      auto &COM = CodeObjectManager::instance();
      auto &LastRFuncArgs = Args.__hipRegisterFunction;
      // If the function doesn't have __luthier_wrap__ in its name then it
      // belongs to the instrumented application or HIP can manage it on its own
      // since no device function is present to strip from it
      if (llvm::StringRef(LastRFuncArgs.deviceFunction)
              .find(luthier::DeviceFunctionWrap) != llvm::StringRef::npos) {
        COM.registerInstrumentationFunctionWrapper(
            LastRFuncArgs.hostFunction, LastRFuncArgs.deviceFunction);
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
  if (Phase == API_EVT_PHASE_EXIT &&
      ApiId == HSA_API_EVT_ID_hsa_executable_freeze) {
    hsa::Executable Exec(CBData->hsa_executable_destroy.executable);
    // Cache the executable and its items
    if (auto Err = Platform::instance().cacheExecutableOnExecutableFreeze(Exec))
      llvm::report_fatal_error("Tool executable register failed");
    // Check if the executable belongs to the tool and not the app
    if (auto Err = CodeObjectManager::instance()
                       .checkIfLuthierToolExecutableAndRegister(Exec)) {
      llvm::report_fatal_error("Tool executable check failed");
    }
  }
  if (Phase == API_EVT_PHASE_EXIT &&
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
  if (Phase == API_EVT_PHASE_ENTER &&
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
    auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    luthier::hsa::Interceptor::instance().captureHsaApiTable(Table);
    luthier::hsa::atHsaApiTableLoad();
    auto &hsaInterceptor = luthier::hsa::Interceptor::instance();
    hsaInterceptor.setInternalCallback(luthier::hsa::internalApiCallback);
    hsaInterceptor.setUserCallback(luthier::hsa::atHsaEvt);
    hsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_freeze);
    hsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_destroy);
    hsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_load_agent_code_object);
    LLVM_DEBUG(llvm::dbgs() << "Captured the HSA API Tables.\n");
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
  luthier::hsa::atHsaApiTableUnload();
  luthier::hsa::Interceptor::instance().uninstallApiTables();
  LUTHIER_LOG_FUNCTION_CALL_END
}

Controller::Controller() : Singleton<Controller>() {
  CG = new CodeGenerator();
  COM = new CodeObjectManager();
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
    C = new luthier::Controller();
    llvm::EnablePrettyStackTrace();
    llvm::EnablePrettyStackTraceOnSigInfoForThisThread(true);
    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/matinraayai/Luthier/issues/ and "
                          "include the crash backtrace.\n");
  });
}

void Controller::finalize() {
  static std::once_flag Once{};
  std::call_once(Once, []() { delete C; });
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