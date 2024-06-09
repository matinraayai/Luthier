#include "global_singleton_manager.hpp"
#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "hip_intercept.hpp"
#include "log.hpp"
#include "luthier/hip_trace_api.h"
#include "target_manager.hpp"

#include <luthier/types.h>

namespace luthier {

template <>
GlobalSingletonManager *Singleton<GlobalSingletonManager>::Instance{nullptr};

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
  delete TM;
  delete HipInterceptor;
}
} // namespace luthier