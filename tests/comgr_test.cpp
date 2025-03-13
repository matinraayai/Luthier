//
// Created by User on 2/13/2025.
//

#include <algorithm>
#include <cstdint>
// TODO - fix CMakeLists and include paths, this is horrible...
#include "../../../../AppData/Local/JetBrains/CLion2024.3/.docker/2024_3/Docker/containers_rc_northeastern_edu_luthier_luthier-dev-llvm-20-shared-rocm-6_3_1_latest/opt/rocm/include/amd_comgr/amd_comgr.h"
#include "../include/luthier/common/ErrorCheck.h"
#include "../src/include/hsa/HsaRuntimeInterceptor.hpp"
#include "../src/include/hsa/hsa.hpp"
#include "../src/include/tooling/Controller.hpp"
#include "../src/include/tooling_common/CodeLifter.hpp"
#include "hip/HipCompilerApiInterceptor.hpp"
#include "hip/HipRuntimeApiInterceptor.hpp"
#include "tooling/Controller.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
// TODO how to include these better?
#include "../src/include/intrinsic/WriteExec.hpp"
#include "../src/include/intrinsic/ReadReg.hpp"
#include "../src/include/intrinsic/ImplicitArgPtr.hpp"
#include "../src/include/intrinsic/WriteReg.hpp"
#include "../src/include/intrinsic/SAtomicAdd.hpp"

#include "tooling_common/CodeGenerator.hpp"
#include <amd_comgr/amd_comgr.h>
#include <cmake-build-debug-docker/include/luthier/hsa/TraceApi.h>
#include <common/Log.hpp>
#include <hsa/ExecutableBackedObjectsCache.hpp>
#include <include/luthier/llvm/streams.h>
#include <include/luthier/types.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Support/FileSystem.h>

// TODO fix this include as well?
#include "../../../../AppData/Local/JetBrains/CLion2024.3/.docker/2024_3/Docker/containers_rc_northeastern_edu_luthier_luthier-dev-llvm-20-shared-rocm-6_3_1_latest/opt/rocm/include/rocprofiler-sdk/registration.h"

#include <string>
#include <tooling_common/ToolExecutableLoader.hpp>
#include <vector>

struct HsaApiTable;

static void internalApiCallback(luthier::hsa::ApiEvtArgs *Args, const luthier::ApiEvtPhase Phase,
                                const luthier::hsa::ApiEvtID ApiId) {
  if (Phase == luthier::API_EVT_PHASE_AFTER &&
     ApiId == luthier::hsa::HSA_API_EVT_ID_hsa_executable_freeze) {
    const luthier::hsa::Executable Exec(Args->hsa_executable_destroy.executable);
    // Cache the executable and its items
    LUTHIER_REPORT_FATAL_ON_ERROR(ExecutableBackedObjectsCache::instance()
                                      .cacheExecutableOnExecutableFreeze(Exec));
    // Check if the executable belongs to the tool and not the app
    LUTHIER_REPORT_FATAL_ON_ERROR(
        luthier::ToolExecutableLoader::instance().registerIfLuthierToolExecutable(Exec));
     }
}

// need this too apparently
// copy above the 'using namepsace luthier'?
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

    HsaInterceptor.setInternalCallback(internalApiCallback);
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_freeze));
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_destroy));
    LUTHIER_REPORT_FATAL_ON_ERROR(HsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_load_agent_code_object));
  }
}

// NEED THIS!!!
extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ClientID) {
  rocprofiler_at_intercept_table_registration(
          apiRegistrationCallback,
          ROCPROFILER_HSA_TABLE, // only need HSA table?
          nullptr);

  static auto Cfg = rocprofiler_tool_configure_result_t{
    sizeof(rocprofiler_tool_configure_result_t),
    &rocprofilerServiceInit, &toolingLibraryFini, nullptr};
  return &Cfg;
}

using namespace luthier;
class ComgrTest {
  private:
    std::string m_filepath;

  // COPIED FROM CONTROLLER HPP

  /// \c CodeGenerator \c Singleton instance
  CodeGenerator *CG{nullptr};

  /// \c ToolExecutableLoader \c Singleton instance
  ToolExecutableLoader *TEL{nullptr};

  /// \c CodeLifter \c Singleton instance
  CodeLifter *CL{nullptr};

  /// \c TargetManager \c Singleton instance
  TargetManager *TM{nullptr};

  /// \c hip::HipCompilerApiInterceptor \c Singleton instance
  hip::HipCompilerApiInterceptor *HipCompilerInterceptor{nullptr};

  /// \c hip::HipRuntimeApiInterceptor \c Singleton instance
  hip::HipRuntimeApiInterceptor *HipRuntimeInterceptor{nullptr};

  /// \c hsa::HsaRuntimeInterceptor \c Singleton instance
  hsa::HsaRuntimeInterceptor *HsaInterceptor{nullptr};

  /// \c hsa::ExecutableBackedObjectsCache \c Singleton instance
  hsa::ExecutableBackedObjectsCache *HsaPlatform{nullptr};

  public:
    // TODO test for valid filepath?
    ComgrTest(std::string filepath) : m_filepath(filepath) {}

    int setupTest() {
      // MUST CALL BEFORE TEST
      hsa::init();
      llvm::outs() << "HSA initialized!\n";
      // Initialize all Luthier singletons
      TM = new TargetManager();
      // TODO - is this compiler interceptor needed?
      HipCompilerInterceptor = new hip::HipCompilerApiInterceptor();
      HsaInterceptor = new hsa::HsaRuntimeInterceptor();
      HsaPlatform = new hsa::ExecutableBackedObjectsCache();
      HipRuntimeInterceptor = new hip::HipRuntimeApiInterceptor();
      TEL = new ToolExecutableLoader();
      CL = new CodeLifter();
      CG = new CodeGenerator();

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

      llvm::outs() << "Luthier intrinsics registered!\n";
      return 0;
    }

    int tearDownTest() {
      // MUST CALL AFTER TEST COMPLETES
      hsa::shutdown();
      llvm::outs() << "HSA shutdown!\n";

      // delete everything
      delete CG;
      delete CL;
      delete TEL;
      delete HsaPlatform;
      delete HipRuntimeInterceptor;
      delete HsaInterceptor;
      delete HipCompilerInterceptor;
      delete TM;
      llvm::outs() << "Memory freed!\n";
      return 0;
    }

    int runTest() {
      setupTest();

      // process .s file

      // open as llvm fd stream?
      std::error_code EC;
      llvm::raw_fd_stream out(m_filepath, EC);

      // make buffer for file TODO figure out how to efficiently do this??
      std::vector<char> sFileBytes;
      char c;
      while (out.read(&c, 1) != -1) {
        sFileBytes.push_back(c);
      }

      amd_comgr_data_t DataIn;
      amd_comgr_data_set_t DataSetIn, DataSetOut;
      amd_comgr_action_info_t DataAction;

      // create data in set
      amd_comgr_create_data_set(&DataSetIn);
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataIn); // TODO make sure right flag?
      amd_comgr_set_data(DataIn, sFileBytes.size(), sFileBytes.data());
      amd_comgr_set_data_name(DataIn, m_filepath.c_str());
      amd_comgr_data_set_add(DataSetIn, DataIn);

      // create action info and set ISA name??
      amd_comgr_create_action_info(&DataAction);
      // TODO set language and verify ISA name is correct?
      amd_comgr_action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx908");

      // compile source to executable

      amd_comgr_create_data_set(&DataSetOut);
      amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
      amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE, DataAction, DataSetIn, DataSetOut);


      // link to executable on drive
      // use AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE???
      amd_comgr_data_set_t DataSetLinked;
      amd_comgr_create_data_set(&DataSetLinked);
      // TODO - can I reuse the same 'Action' object here?
      amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, DataAction, DataSetOut, DataSetLinked);

      // test if it is an elf file with given command
      // readelf -a mydumped_exec.hsaco

      /* CL provides an executable lift???
      *llvm::Expected<const LiftedRepresentation &>
        CodeLifter::lift(const hsa::Executable &Exec) {
       */

      tearDownTest();

      return 0;
    }
};

int main()
{
  ComgrTest comgrTest("main-hip-amdgcn-amd-amdhsa-gfx908.s");
  comgrTest.runTest();
  return 0;
}