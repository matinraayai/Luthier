//===-- code_lifter_test.cpp - Unit Test for the Luthier Code Lifter ------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// longer description
//===----------------------------------------------------------------------===//

// #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <doctest/doctest.h>
#include "unittest_common.hpp"
#include "tooling_common/code_lifter.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>

/*
int factorial(int number) {
  return number <= 1 ? number : factorial(number - 1) * number;
}

TEST_CASE("testing the factorial function") {
  CHECK(factorial(0) == 1);
  CHECK(factorial(1) == 1);
  CHECK(factorial(2) == 2);
  CHECK(factorial(3) == 6);
  CHECK(factorial(10) == 3628800);
}
*/


static llvm::cl::opt<std::string> 
CodeObjF(llvm::cl::Positional, llvm::cl::Required, 
         llvm::cl::desc("<input code object file>"));

static llvm::cl::opt<std::string> 
OutputDir("o", llvm::cl::init("-"),
          llvm::cl::value_desc("Directory name"),
          llvm::cl::desc("Directory to store MIR output file"));

HsaApiTable CapturedTable;

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, 
                                    "Unit Test for the Luthier Code Lifter\n");
  llvm::outs() << "RUNNING CODE LIFTER TEST\n\n";

  /// Create output file for MIR
  std::string OutFileName;
  OutputDir == "-" ? 
      OutFileName = unittest::getMIRFileName(CodeObjF) :
      OutFileName = OutputDir + '/' + unittest::getMIRFileName(CodeObjF);
  std::error_code EC;
  llvm::raw_fd_ostream 
      OutFile(llvm::StringRef(OutFileName), EC);
  if (EC) {
    llvm::errs() << "Error when opening output file\n\n";
    return -1;
  }
  OutFile << "# RUN: llc -mtriple=amdgcn -mcpu=gfx908 --verify-machineinstrs -run-pass verify -o - %s | FileCheck -check-prefixes=GCN %s";
  OutFile << "\n\n";

  /// Load input code object
  auto CodeObjBuf = llvm::MemoryBuffer::getFile(CodeObjF);
  UNITTEST_RETURN_ON_ERROR(CodeObjBuf.getError());
  
  /// Create HSA interceptor to capture API table
  auto &TestInterceptor = luthier::hsa::Interceptor::instance();
  
  /// Create Luthier components neccessary to run the component under test
  /// In this case, the code lifter 
  auto *TargetManager   = new luthier::TargetManager();
  auto *TestPlatform    = new luthier::hsa::Platform();
  auto *Lifter          = new luthier::CodeLifter();

  UNITTEST_RETURN_ON_ERROR(luthier::hsa::init());
  if (TestInterceptor.captureHsaApiTable(&CapturedTable)) {
    llvm::outs() << "Successfully captured the HSA API Table\n\n";
  } else {
    llvm::outs() << "Failed to capture the HSA API Table\n\n";
    return -1;
  }

  auto Reader = luthier::hsa::CodeObjectReader
                       ::createFromMemory(CodeObjBuf.get()->getBuffer());
  UNITTEST_RETURN_ON_ERROR(Reader.takeError());
  auto ExecToLift = luthier::hsa::Executable::create();
  UNITTEST_RETURN_ON_ERROR(ExecToLift.takeError());
  llvm::SmallVector<luthier::hsa::GpuAgent> Agents;
  UNITTEST_RETURN_ON_ERROR(luthier::hsa::getGpuAgents(Agents));
  auto LCO = ExecToLift->loadAgentCodeObject(*Reader, Agents[0]);
  UNITTEST_RETURN_ON_ERROR(LCO.takeError());
  UNITTEST_RETURN_ON_ERROR(ExecToLift->freeze());
  auto LiftedCodeObj = Lifter->lift(*ExecToLift);
  UNITTEST_RETURN_ON_ERROR(LiftedCodeObj.takeError());

  llvm::outs() << "Dump the LLVM Functions found in the executable to: "
               << OutFileName << "\n\n";
  for (auto &[EXEC, MF] : LiftedCodeObj->functions()) {
    llvm::printMIR(OutFile, *MF);
  }

  UNITTEST_RETURN_ON_ERROR(luthier::hsa::shutdown());
  
  delete Lifter;
  delete TestPlatform;
  delete TargetManager;
  TestInterceptor.uninstallApiTables();

  llvm::outs() << "CODE LIFTER TEST FINISHED\n\n";
  return 0;
}

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ID) {
  rocprofiler_at_intercept_table_registration(
      unittest::saveHsaApiTable,
      ROCPROFILER_HSA_TABLE, &CapturedTable);
  static auto Cfg = rocprofiler_tool_configure_result_t{
    sizeof(rocprofiler_tool_configure_result_t), &unittest::toolInit,
                                                 &unittest::toolFini,
                                                 nullptr};
  return &Cfg;
}

