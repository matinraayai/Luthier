#include "global_singleton_manager.hpp"
#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "hip_intercept.hpp"
#include "log.hpp"
#include "luthier/hip_trace_api.h"
#include "target_manager.hpp"

#include "luthier/luthier.h"
#include "luthier/types.h"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace luthier {

template <>
GlobalSingletonManager *Singleton<GlobalSingletonManager>::Instance{nullptr};

GlobalSingletonManager *GlobalSingletonManager::GSM{nullptr};

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
  }
}

void rocprofilerFinalize(void *Data) {
  LUTHIER_LOG_FUNCTION_CALL_START
  luthier::hsa::atHsaApiTableUnload();
  luthier::hsa::Interceptor::instance().uninstallApiTables();
  LUTHIER_LOG_FUNCTION_CALL_END
}

GlobalSingletonManager::GlobalSingletonManager()
    : Singleton<GlobalSingletonManager>() {
  CG = new CodeGenerator();
  COM = new CodeObjectManager();
  CL = new CodeLifter();
  TM = new TargetManager();
  HipInterceptor = new hip::Interceptor();
  if (!HipInterceptor->isEnabled())
    llvm::report_fatal_error("HIP Interceptor failed to initialize.");
  HipInterceptor->setInternalCallback(hip::internalApiCallback);
  HipInterceptor->enableInternalCallback(
      luthier::hip::HIP_API_ID___hipRegisterFunction);
}

GlobalSingletonManager::~GlobalSingletonManager() {
  delete CG;
  delete COM;
  delete CL;
  delete TM;
  delete HipInterceptor;
}
void GlobalSingletonManager::init() {
  static std::once_flag Once{};
  std::call_once(Once, []() { GSM = new luthier::GlobalSingletonManager(); });
}

void GlobalSingletonManager::finalize() {
  static std::once_flag Once{};
  std::call_once(Once, []() { delete GSM; });
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ID) {
  ID->name = "Luthier";
  rocprofiler_at_intercept_table_registration(luthier::apiRegistrationCallback,
                                              ROCPROFILER_HSA_TABLE, nullptr);

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::rocprofilerFinalize, nullptr};
  return &Cfg;
}