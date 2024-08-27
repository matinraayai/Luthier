//===-- code_lifter_test.cpp - Unit Test for the Luthier Code Lifter ------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// longer description
//===----------------------------------------------------------------------===//

#include "hsa/hsa.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/CodeObjectReader.hpp"
#include "hsa/Executable.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "luthier/luthier.h"
#include "luthier/types.h"
#include "common/Error.hpp"
#include "common/ObjectUtils.hpp"

#include "unittest_common.hpp"
#include "tooling_common/CodeLifter.hpp"
#include <string>

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
  
  /// Create Luthier singletons neccessary to run the component under test
  auto *TargetManager  = new luthier::TargetManager();
  auto *HsaPlatform    = new luthier::hsa::ExecutableBackedObjectsCache();
  auto *Lifter         = new luthier::CodeLifter();
  auto *HsaInterceptor = new luthier::hsa::HsaRuntimeInterceptor();

  UNITTEST_RETURN_ON_ERROR(luthier::hsa::init());
  
  llvm::outs() << "Attempting to capture API Table\n\n";
  UNITTEST_RETURN_ON_ERROR(HsaInterceptor->captureApiTable(&CapturedTable));
  llvm::outs() << "Successfully captured the HSA API Table\n\n";

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
  
  HsaInterceptor->uninstallApiTables();
  llvm::outs() << "HSA Shutdown called and API Tables uninstalled\n\n";
  
  delete Lifter;
  delete TargetManager;
  delete HsaPlatform;
  delete HsaInterceptor;

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

