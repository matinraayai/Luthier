//===-- Controller.cpp - Luthier tool's Controller Logic implementation ---===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
///
/// \file
/// This file implements the \c Controller singleton class logic, as well as
/// registration of the Luthier tool with the rocprofiler library.
//===----------------------------------------------------------------------===//
#include "tooling/Controller.hpp"
#include "common/Log.hpp"
#include "hip/HipCompilerApiInterceptor.hpp"
#include "hip/HipRuntimeApiInterceptor.hpp"
#include "hsa/Executable.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "luthier/luthier.h"
#include "luthier/types.h"
#include "tooling_common/CodeGenerator.hpp"
#include "tooling_common/CodeLifter.hpp"
#include "tooling_common/TargetManager.hpp"
#include "tooling_common/ToolExecutableLoader.hpp"
#include "tooling_common/intrinsic/ImplicitArgPtr.hpp"
#include "tooling_common/intrinsic/ReadReg.hpp"
#include "tooling_common/intrinsic/SAtomicAdd.hpp"
#include "tooling_common/intrinsic/WriteExec.hpp"
#include "tooling_common/intrinsic/WriteReg.hpp"
#include <llvm/Support/Error.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-controller"

namespace luthier {

template <> Controller *Singleton<Controller>::Instance{nullptr};

/// Where the controller singleton is stored
static Controller *C{nullptr};

Controller::Controller() {
  // Initialize all Luthier singletons
  TM = new TargetManager();
  HsaPlatform = new hsa::ExecutableBackedObjectsCache();
  CG = new CodeGenerator();
  TEL = new ToolExecutableLoader();
  CL = new CodeLifter();
  HsaInterceptor = new hsa::HsaRuntimeInterceptor();
  HipCompilerInterceptor = new hip::HipCompilerApiInterceptor();
  HipRuntimeInterceptor = new hip::HipRuntimeApiInterceptor();
  // Register Luthier intrinsics with the Code Generator
  CG->registerIntrinsic("luthier::readReg",
                        {readRegIRProcessor, readRegMIRProcessor});
  CG->registerIntrinsic("luthier::writeReg",
                        {writeRegIRProcessor, writeRegMIRProcessor});
  CG->registerIntrinsic("luthier::writeExec",
                        {writeExecIRProcessor, writeExecMIRProcessor});
  CG->registerIntrinsic(
      "luthier::implicitArgPtr",
      {implicitArgPtrIRProcessor, implicitArgPtrMIRProcessor});
  CG->registerIntrinsic("luthier::sAtomicAdd",
                        {sAtomicAddIRProcessor, sAtomicAddMIRProcessor});
}

Controller::~Controller() {
  delete CG;
  delete CL;
  delete TM;
  delete TEL;
  delete HsaPlatform;
  delete HsaInterceptor;
  delete HipCompilerInterceptor;
  delete HipRuntimeInterceptor;
}

namespace hip {
/// Tooling library's internal HIP API callback
static void internalApiCallback(ApiEvtArgs *Args, const ApiEvtPhase Phase,
                                const ApiEvtID ApiId) {
  LLVM_DEBUG(llvm::dbgs() << "HIP Internal callback for API " << ApiId << "\n");
  if (Phase == API_EVT_PHASE_BEFORE) {
    if (ApiId == HIP_COMPILER_API_EVT_ID___hipRegisterFunction) {
      LLVM_DEBUG(llvm::dbgs()
                 << "__hipRegisterFunction args: "
                 << Args->__hipRegisterFunction.deviceFunction << ".\n");
      auto &TEL = ToolExecutableLoader::instance();
      auto &LastRFuncArgs = Args->__hipRegisterFunction;
      // Look for kernels that serve as handles for hooks and register them with
      // the tool executable loader
      if (llvm::StringRef(LastRFuncArgs.deviceFunction)
              .find(HookHandlePrefix) != llvm::StringRef::npos) {
        TEL.registerInstrumentationHookWrapper(LastRFuncArgs.hostFunction,
                                               LastRFuncArgs.deviceFunction);
      }
    }
  }
}
} // namespace hip

namespace hsa {
/// Luthier's internal HSA API callback; Mainly used to notify singletons about
/// creation of loaded code objects, freezing and destruction of executables
void internalApiCallback(ApiEvtArgs *Args, const ApiEvtPhase Phase,
                         const ApiEvtID ApiId) {
  LUTHIER_LOG_FUNCTION_CALL_START
  if (Phase == API_EVT_PHASE_AFTER &&
      ApiId == HSA_API_EVT_ID_hsa_executable_freeze) {
    const hsa::Executable Exec(Args->hsa_executable_destroy.executable);
    // Cache the executable and its items
    LUTHIER_REPORT_FATAL_ON_ERROR(ExecutableBackedObjectsCache::instance()
                                      .cacheExecutableOnExecutableFreeze(Exec));
    // Check if the executable belongs to the tool and not the app
    LUTHIER_REPORT_FATAL_ON_ERROR(
        ToolExecutableLoader::instance().registerIfLuthierToolExecutable(Exec));
  }
  if (Phase == API_EVT_PHASE_AFTER &&
      ApiId == HSA_API_EVT_ID_hsa_executable_load_agent_code_object) {
    // because the output of hsa_executable_load_agent_code_object can be set to
    // nullptr by the app, we have to access it by iterating over the LCOs of
    // the Exec it was created for
    const hsa::Executable Exec(
        Args->hsa_executable_load_agent_code_object.executable);
    LUTHIER_REPORT_FATAL_ON_ERROR(
        ExecutableBackedObjectsCache::instance()
            .cacheExecutableOnLoadedCodeObjectCreation(Exec));
  }
  if (Phase == API_EVT_PHASE_BEFORE &&
      ApiId == HSA_API_EVT_ID_hsa_executable_destroy) {
    hsa::Executable Exec(Args->hsa_executable_destroy.executable);
    LUTHIER_REPORT_FATAL_ON_ERROR(
        CodeLifter::instance().invalidateCachedExecutableItems(Exec));

    LUTHIER_REPORT_FATAL_ON_ERROR(
        ToolExecutableLoader::instance().unregisterIfLuthierToolExecutable(
            Exec));

    LUTHIER_REPORT_FATAL_ON_ERROR(
        ExecutableBackedObjectsCache::instance()
            .invalidateExecutableOnExecutableDestroy(Exec));
  }
  LUTHIER_LOG_FUNCTION_CALL_END
}
} // namespace hsa

static void parseEnvVariableArgs() {
  llvm::StringMap<llvm::cl::Option *> &RegisteredOpts =
      llvm::cl::getRegisteredOptions();
  // Disable machine verifier by default since it takes too much time +
  // it causes issues with live-in registers in code generator
  reinterpret_cast<llvm::cl::opt<llvm::cl::boolOrDefault> *>(
      RegisteredOpts["verify-machineinstrs"])
      ->setValue(llvm::cl::BOU_FALSE, true);

  const auto Argv = "";
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      llvm::cl::ParseCommandLineOptions(
          0, &Argv, "Luthier, An AMD GPU Dynamic Binary Instrumentation Tool",
          &llvm::errs(), "LUTHIER_ARGS"),
      "Failed to parse the command line arguments"));
}

void rocprofilerFinalize(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    atToolFini(API_EVT_PHASE_BEFORE);
    delete C;
    atToolFini(API_EVT_PHASE_AFTER);
  });
}

void rocprofilerInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    atToolInit(API_EVT_PHASE_BEFORE);
    C = new Controller();
    llvm::EnablePrettyStackTrace();
    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/matinraayai/Luthier/issues/ and "
                          "include the crash error message and backtrace.\n");
    llvm::sys::AddSignalHandler(rocprofilerFinalize, nullptr);
    atToolInit(API_EVT_PHASE_AFTER);
  });
}

static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                    uint64_t LibVersion, uint64_t LibInstance,
                                    void **Tables, uint64_t NumTables,
                                    void *Data) {
  // Initialize the Luthier tooling (if not already initialized)
  rocprofilerInit(Data);
  if (Type == ROCPROFILER_HSA_TABLE) {
    auto &HsaInterceptor = hsa::HsaRuntimeInterceptor::instance();
    auto &HsaApiTableCaptureCallback =
        Controller::instance().getAtHSAApiTableCaptureEvtCallback();
    HsaApiTableCaptureCallback(API_EVT_PHASE_BEFORE);
    LLVM_DEBUG(llvm::dbgs() << "Capturing the HSA API Tables.\n");
    auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.captureApiTable(Table));
    HsaApiTableCaptureCallback(API_EVT_PHASE_AFTER);
    LLVM_DEBUG(llvm::dbgs() << "Captured the HSA API Tables.\n");

    HsaInterceptor.setInternalCallback(hsa::internalApiCallback);
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
    auto &HipRuntimeInterceptor = hip::HipRuntimeApiInterceptor::instance();
    auto *Table = static_cast<HipDispatchTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(HipRuntimeInterceptor.captureApiTable(Table));
    HipRuntimeInterceptor.setInternalCallback(hip::internalApiCallback);
    LLVM_DEBUG(llvm::dbgs() << "Captured the HIP Runtime API Table.\n");
  }
}

namespace hsa {
void setAtApiTableCaptureEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback) {
  Controller::instance().setAtHSAApiTableCaptureEvtCallback(Callback);
}
} // namespace hsa

void setAtApiTableReleaseEvtCallback(
    const std::function<void(ApiEvtPhase)> &Callback) {}
} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ClientID) {
  // Parse the luthier arguments here so that LLVM_DEBUG works inside this
  // function
  static std::once_flag ArgParseOnceFlag;
  std::call_once(ArgParseOnceFlag, luthier::parseEnvVariableArgs);

  LLVM_DEBUG(const uint32_t RocProfVerMajor = Version / 10000;
             const uint32_t RocProfVerMinor = (Version % 10000) / 100;
             const uint32_t RocProfVerPatch = Version % 100;
             llvm::dbgs() << "Registering Luthier tool with rocprofiler-sdk.\n";
             llvm::dbgs() << "Rocprofiler-sdk version: " << RocProfVerMajor
                          << "." << RocProfVerMinor << "." << RocProfVerPatch
                          << " (" << RuntimeVersion << ")" << "\n";
             llvm::dbgs() << "Tool priority: " << Priority << ";\n";);
  // TODO: Make tools define a function that returns the name of the tool
  // instead of assigning plain "Luthier" as the tool name
  ClientID->name = "Luthier";
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
