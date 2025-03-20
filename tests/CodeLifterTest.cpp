//
// Created by User on 2/13/2025.
//

#include "hip/HipCompilerApiInterceptor.hpp"
#include "hip/HipRuntimeApiInterceptor.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
// #include <gtest/gtest.h>
#include <amd_comgr/amd_comgr.h>
#include <cstdint>
#include <luthier/common/ErrorCheck.h>
#include <luthier/llvm/streams.h>
#include <luthier/types.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Support/FileSystem.h>
#include <luthier/comgr/ComgrError.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <string>
#include <tooling/Controller.hpp>
#include <tooling_common/CodeLifter.hpp>
#include <tooling_common/ToolExecutableLoader.hpp>

#include <TestingUtils.h>

static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                    uint64_t LibVersion, uint64_t LibInstance,
                                    void **Tables, uint64_t NumTables,
                                    void *Data) {
  if (Type == ROCPROFILER_HSA_TABLE) {
    auto &HsaInterceptor = luthier::hsa::HsaRuntimeInterceptor::instance();
    auto &HsaApiTableCaptureCallback =
        luthier::Controller::instance().getAtHSAApiTableCaptureEvtCallback();
    HsaApiTableCaptureCallback(luthier::API_EVT_PHASE_BEFORE);
    auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.captureApiTable(Table));
    HsaApiTableCaptureCallback(luthier::API_EVT_PHASE_AFTER);

  }
}

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ClientID) {
  rocprofiler_at_intercept_table_registration(
          apiRegistrationCallback,
          ROCPROFILER_HSA_TABLE,
          nullptr);

  // TODO - double check that this is ok
  static auto Cfg = rocprofiler_tool_configure_result_t{
    sizeof(rocprofiler_tool_configure_result_t),
    nullptr, nullptr, nullptr};
  return &Cfg;
}

using namespace luthier;
class CodeLifterTest {
  std::string m_filepath;

  /// \c CodeLifter \c Singleton instance
  CodeLifter *CL{nullptr};

  /// \c TargetManager \c Singleton instance
  TargetManager *TM{nullptr};

  /// \c hsa::HsaRuntimeInterceptor \c Singleton instance
  hsa::HsaRuntimeInterceptor *HsaInterceptor{nullptr};

  /// \c hsa::ExecutableBackedObjectsCache \c Singleton instance
  hsa::ExecutableBackedObjectsCache *HsaPlatform{nullptr};

  public:
    // TODO test for valid filepath?
    CodeLifterTest(std::string filepath) : m_filepath(filepath) {}

    llvm::Error setupTest() {
      LUTHIER_RETURN_ON_ERROR(hsa::init());
      // debugging print, for initial run
      llvm::outs() << "HSA initialized!\n";
      TM = new TargetManager();
      HsaInterceptor = new hsa::HsaRuntimeInterceptor();
      HsaPlatform = new hsa::ExecutableBackedObjectsCache();
      CL = new CodeLifter();

      llvm::outs() << "Luthier intrinsics registered!\n";
      return llvm::Error::success();
    }

    llvm::Error tearDownTest() {
      LUTHIER_RETURN_ON_ERROR(hsa::shutdown());
      llvm::outs() << "HSA shutdown!\n";

      delete CL;
      delete HsaPlatform;
      delete HsaInterceptor;
      delete TM;
      llvm::outs() << "Memory freed!\n";
      return llvm::Error::success();
    }

    llvm::Error runTest() {
      LUTHIER_REPORT_FATAL_ON_ERROR(setupTest());

      // use testing utils
      LUTHIER_REPORT_FATAL_ON_ERROR(compile_and_link(m_filepath));

      LUTHIER_REPORT_FATAL_ON_ERROR(tearDownTest());

      return llvm::Error::success();
    }
};

int main()
{
  CodeLifterTest codeLifterTest("main-hip-amdgcn-amd-amdhsa-gfx908.s");
  // TODO - this will be replaced with a gtest TEST macro, eventually
  codeLifterTest.runTest();
  return 0;
}