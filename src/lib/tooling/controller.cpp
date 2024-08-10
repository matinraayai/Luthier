//===-- controller.cpp - Luthier tool's Controller Logic implementation ---===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the \c Controller singleton class logic, as well as
/// any other interactions that needs to happen with the rocprofiler library.
//===----------------------------------------------------------------------===//

#include "tooling/controller.hpp"
#include "common/log.hpp"
#include "hip/HipCompilerApiInterceptor.hpp"
#include "hip/HipRuntimeApiInterceptor.hpp"
#include "hsa/hsa_executable.hpp"
#include "tooling_common/code_generator.hpp"
#include "tooling_common/code_lifter.hpp"
#include "tooling_common/intrinsic/ReadReg.hpp"
#include "tooling_common/intrinsic/WriteReg.hpp"
#include "tooling_common/target_manager.hpp"
#include "tooling_common/tool_executable_manager.hpp"

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
static void internalApiCallback(ApiEvtArgs *Args, ApiEvtPhase Phase,
                                ApiEvtID ApiId) {
  LUTHIER_LOG_FUNCTION_CALL_START
  if (Phase == API_EVT_PHASE_BEFORE) {
    if (ApiId == HIP_COMPILER_API_EVT_ID___hipRegisterFunction) {
      auto &COM = ToolExecutableManager::instance();
      auto &LastRFuncArgs = Args->__hipRegisterFunction;
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
                         hsa::ApiEvtID ApiId) {
  LUTHIER_LOG_FUNCTION_CALL_START
  if (Phase == API_EVT_PHASE_AFTER &&
      ApiId == HSA_API_EVT_ID_hsa_executable_freeze) {
    hsa::Executable Exec(CBData->hsa_executable_destroy.executable);
    // Cache the executable and its items
    LUTHIER_REPORT_FATAL_ON_ERROR(
        Platform::instance().cacheExecutableOnExecutableFreeze(Exec));
    // Check if the executable belongs to the tool and not the app
    LUTHIER_REPORT_FATAL_ON_ERROR(
        ToolExecutableManager::instance().registerIfLuthierToolExecutable(
            Exec));
  }
  if (Phase == API_EVT_PHASE_AFTER &&
      ApiId == HSA_API_EVT_ID_hsa_executable_load_agent_code_object) {
    // because the output of hsa_executable_load_agent_code_object can be set to
    // nullptr by the app, we have to access it by iterating over the LCOs of
    // the Exec it was created for
    hsa::Executable Exec(
        CBData->hsa_executable_load_agent_code_object.executable);
    LUTHIER_REPORT_FATAL_ON_ERROR(
        Platform::instance().cacheExecutableOnLoadedCodeObjectCreation(Exec));
  }
  if (Phase == API_EVT_PHASE_BEFORE &&
      ApiId == HSA_API_EVT_ID_hsa_executable_destroy) {
    hsa::Executable Exec(CBData->hsa_executable_destroy.executable);
    LUTHIER_REPORT_FATAL_ON_ERROR(
        CodeLifter::instance().invalidateCachedExecutableItems(Exec));

    LUTHIER_REPORT_FATAL_ON_ERROR(
        ToolExecutableManager::instance().unregisterIfLuthierToolExecutable(
            Exec));

    LUTHIER_REPORT_FATAL_ON_ERROR(
        Platform::instance().invalidateExecutableOnExecutableDestroy(Exec));
  }
  LUTHIER_LOG_FUNCTION_CALL_END
}
} // namespace hsa

static void parseEnvVariableArgs() {

  llvm::StringMap<llvm::cl::Option *> &Map = llvm::cl::getRegisteredOptions();
  // Disable machine verifier since it takes too much time + it causes
  // issues with live-in registers in code generator
  reinterpret_cast<llvm::cl::opt<llvm::cl::boolOrDefault> *>(
      Map["verify-machineinstrs"])
      ->setValue(llvm::cl::BOU_FALSE, true);

  auto Argv = "";
  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_ASSERTION(llvm::cl::ParseCommandLineOptions(
          0, &Argv, "Luthier, An AMD GPU Dynamic Binary Instrumentation Tool",
          &llvm::errs(), "LUTHIER_ARGS")));
}

static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                    uint64_t LibVersion, uint64_t LibInstance,
                                    void **Tables, uint64_t NumTables,
                                    void *Data) {
  // Argument parsing is done the first time any of the API tables have been
  // intercepted;
  // This is because at this point, all statically-defined cl arguments should
  // be initialized by now
  static std::once_flag ArgParseOnceFlag;
  std::call_once(ArgParseOnceFlag, parseEnvVariableArgs);

  if (Type == ROCPROFILER_HSA_TABLE) {
    auto &HsaInterceptor = luthier::hsa::HsaRuntimeInterceptor::instance();
    auto &HsaApiTableCaptureCallback =
        Controller::instance().getAtHSAApiTableCaptureEvtCallback();
    HsaApiTableCaptureCallback(API_EVT_PHASE_BEFORE);
    LLVM_DEBUG(llvm::dbgs() << "Capturing the HSA API Tables.\n");
    auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.captureApiTable(Table));
    HsaApiTableCaptureCallback(API_EVT_PHASE_AFTER);
    LLVM_DEBUG(llvm::dbgs() << "Captured the HSA API Tables.\n");

    HsaInterceptor.setInternalCallback(luthier::hsa::internalApiCallback);
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_freeze));
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_destroy));
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_load_agent_code_object));
  }
  if (Type == ROCPROFILER_HIP_COMPILER_TABLE) {
    LLVM_DEBUG(llvm::dbgs() << "Capturing the HIP Compiler API Table.\n");
    auto &HipCompilerInterceptor =
        luthier::hip::HipCompilerApiInterceptor::instance();
    auto *Table = static_cast<HipCompilerDispatchTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(
        HipCompilerInterceptor.captureApiTable(Table));
    HipCompilerInterceptor.setInternalCallback(hip::internalApiCallback);

    LUTHIER_REPORT_FATAL_ON_ERROR(HipCompilerInterceptor.enableInternalCallback(
        hip::HIP_COMPILER_API_EVT_ID___hipRegisterFunction));
    LLVM_DEBUG(llvm::dbgs() << "Captured the HIP Compiler API Table.\n");
  }
  if (Type == ROCPROFILER_HIP_RUNTIME_TABLE) {
    LLVM_DEBUG(llvm::dbgs() << "Capturing the HIP Runtime API Table.\n");
    auto &HipRuntimeInterceptor =
        luthier::hip::HipRuntimeApiInterceptor::instance();
    auto *Table = static_cast<HipDispatchTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(HipRuntimeInterceptor.captureApiTable(Table));
    HipRuntimeInterceptor.setInternalCallback(hip::internalApiCallback);
    LLVM_DEBUG(llvm::dbgs() << "Captured the HIP Runtime API Table.\n");
  }
}

void rocprofilerFinalize(void *Data){
    LUTHIER_LOG_FUNCTION_CALL_START
        // TODO: place this somewhere else
        //   luthier::Controller::instance().getAtApiTableReleaseEvtCallback()(
        //       API_EVT_PHASE_BEFORE);
        //   luthier::hsa::Interceptor::instance().uninstallApiTables();
        //   luthier::Controller::instance().getAtApiTableReleaseEvtCallback()(
        //       API_EVT_PHASE_AFTER);
        LUTHIER_LOG_FUNCTION_CALL_END}

Controller::Controller()
    : Singleton<Controller>() {
  // Initialize all the singletons
  TM = new TargetManager();
  HsaPlatform = new hsa::Platform();
  CG = new CodeGenerator();
  COM = new ToolExecutableManager();
  CL = new CodeLifter();
  HsaInterceptor = new hsa::HsaRuntimeInterceptor();
  HipCompilerInterceptor = new hip::HipCompilerApiInterceptor();
  HipRuntimeInterceptor = new hip::HipRuntimeApiInterceptor();
  // Register Luthier intrinsics with the Code Generator
  CG->registerIntrinsic("luthier::readReg",
                        {readRegIRProcessor, readRegMIRProcessor});
  CG->registerIntrinsic("luthier::writeReg",
                        {writeRegIRProcessor, writeRegMIRProcessor});
}

Controller::~Controller() {
  delete CG;
  delete CL;
  delete TM;
  delete COM;
  delete HsaPlatform;
  delete HsaInterceptor;
  delete HipCompilerInterceptor;
  delete HipRuntimeInterceptor;
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

namespace hsa {
void setAtApiTableCaptureEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback) {
  Controller::instance().setAtHSAApiTableCaptureEvtCallback(Callback);
}
} // namespace hsa
void setAtApiTableReleaseEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback) {
  Controller::instance().setAtApiTableReleaseEvtCallback(Callback);
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ID) {
  ID->name = "Luthier";
  rocprofiler_at_intercept_table_registration(
      luthier::apiRegistrationCallback,
      ROCPROFILER_HSA_TABLE | ROCPROFILER_HIP_COMPILER_TABLE |
          ROCPROFILER_HIP_RUNTIME_TABLE,
      nullptr);

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::rocprofilerFinalize, nullptr};
  return &Cfg;
}