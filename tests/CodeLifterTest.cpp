//
// Created by User on 2/13/2025.
//

#include "hip/HipCompilerApiInterceptor.hpp"
#include "hip/HipRuntimeApiInterceptor.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include <gtest/gtest.h>
#include <amd_comgr/amd_comgr.h>
#include <cstdint>
#include <include/luthier/common/ErrorCheck.h>
#include <include/luthier/llvm/streams.h>
#include <include/luthier/types.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Support/FileSystem.h>
#include <luthier/comgr/ComgrError.h>
#include <luthier/common/ErrorCheck.h>
#include <rocprofiler-sdk/registration.h>
#include <string>
#include <tooling/Controller.hpp>
#include <tooling_common/CodeLifter.hpp>
#include <tooling_common/ToolExecutableLoader.hpp>

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

  static auto Cfg = rocprofiler_tool_configure_result_t{
    sizeof(rocprofiler_tool_configure_result_t),
    &rocprofilerServiceInit, &toolingLibraryFini, nullptr};
  return &Cfg;
}

using namespace luthier;
class ComgrTest {
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
    ComgrTest(std::string filepath) : m_filepath(filepath) {}

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

      // read in .s file
      llvm::ErrorOr <std::unique_ptr<llvm::MemoryBuffer>> BuffOrErr =
          llvm::MemoryBuffer::getFile(m_filepath);

      // https://github.com/ROCm/llvm-project/blob/7addc3557e2d6e0a1aa133d625c62e5ee04bc5bf/llvm/tools/llvm-dwarfdump/llvm-dwarfdump.cpp#L322
      if (BuffOrErr.getError()) {
        llvm::errs() << "Error opening file: " << m_filepath << "\n";
        // TODO - handle this error
      }

      std::unique_ptr<llvm::MemoryBuffer> Buff = std::move(BuffOrErr.get());

      // TODO - move this logic into utils file for unit tests
      amd_comgr_data_t DataIn, DataReloc;
      amd_comgr_data_set_t DataSetIn, DataSetOut, DataSetReloc;
      amd_comgr_action_info_t DataAction;

      // create data in set
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_ERROR_CHECK(
        amd_comgr_create_data_set(&DataSetIn)));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataIn)));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_set_data(DataIn, Buff->getBuffer().size(),
          Buff->getBuffer().data())));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_set_data_name(DataIn, m_filepath.c_str())));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_data_set_add(DataSetIn, DataIn)));

      // create action info and set ISA name??
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_create_action_info(&DataAction)));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx908")));

      // compile source to executable
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_create_data_set(&DataSetOut)));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_action_info_set_option_list(DataAction, NULL, 0)));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_do_action(
          AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE, DataAction, DataSetIn, DataSetOut)));

      // link to executable on drive
      amd_comgr_data_set_t DataSetLinked;
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_create_data_set(&DataSetLinked)));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
          DataAction, DataSetOut, DataSetLinked)));

      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
        amd_comgr_create_data_set(&DataSetReloc)));

      /*
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_ERROR_CHECK(
        amd_comgr_action_data_get_data(DataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE,
            0, &DataReloc)));

      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_COMGR_ERROR_CHECK(
        amd_comgr_get_data()));
        */

      LUTHIER_REPORT_FATAL_ON_ERROR(tearDownTest());

      return llvm::Error::success();
    }
};

int main()
{
  ComgrTest comgrTest("main-hip-amdgcn-amd-amdhsa-gfx908.s");
  // TODO - this will be replaced with a gtest TEST macro, eventually
  comgrTest.runTest();
  return 0;
}