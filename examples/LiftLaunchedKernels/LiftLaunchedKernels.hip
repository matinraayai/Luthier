//===-- LiftLaunchedKernels.hip --------------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// Lifts every kernel observed on dispatch and prints the resulting MIR.
/// No instrumentation is performed — \c runInstrumentationPass is a no-op.
/// The tool uses \c CodeDiscoveryPass to materialize the dispatched kernel
/// into the target module, then dumps the module to stderr.
//===----------------------------------------------------------------------===//
#include <hip/hip_runtime.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/raw_ostream.h>
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSATooling/HSATool.h>
#include <luthier/HSATooling/HsaMemoryAllocationAccessor.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/CodeDiscoveryPass.h>
#include <luthier/ToolCodeGen/EntryPoint.h>
#include <luthier/ToolCodeGen/IPPredicatedCFG.h>
#include <luthier/ToolCodeGen/InitialEntryPointAnalysis.h>
#include <luthier/ToolCodeGen/InitialExecutionPointAnalysis.h>
#include <luthier/ToolCodeGen/InstrumentationPMDriver.h>
#include <luthier/ToolCodeGen/InstrumentationPass.h>
#include <luthier/ToolCodeGen/MemoryAllocationAccessor.h>
#include <luthier/ToolCodeGen/MetadataParserAnalysis.h>
#include <luthier/ToolCodeGen/PrePostAmbleEmitter.h>
#include <luthier/ToolCodeGen/TraceCallGraph.h>
#include <mutex>
#include <unordered_set>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lift-launched-kernels-tool"

struct LiftLaunchedKernelsTool
    : public luthier::HSATool<LiftLaunchedKernelsTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  struct Options {
    llvm::cl::OptionCategory Cat{"Lift Launched Kernels Tool Options"};
    luthier::CodeDiscoveryPassOptions DiscoveryOpts;
  } Opts;

  std::recursive_mutex Mutex;

  /// KD addresses already printed, so we don't dump the same kernel twice.
  std::unordered_set<uint64_t> PrintedKDs;

  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  /// No-op: the tool only lifts and prints, never injects payloads.
  luthier::InstrumentationPreservedAnalyses
  runInstrumentationPass(llvm::Module &, llvm::ModuleAnalysisManager &,
                         llvm::MachineFunction &,
                         llvm::FunctionAnalysisManager &) {
    return luthier::InstrumentationPreservedAnalyses{};
  }
};

static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot{
    nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};

void LiftLaunchedKernelsTool::onPackets(
    const hsa_queue_t &, uint64_t,
    llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
    hsa_amd_queue_intercept_packet_writer PacketWriter) {
  for (const auto &Packet : Packets) {
    const auto *DispatchPacket = Packet.asKernelDispatch();
    if (!DispatchPacket) {
      PacketWriter(&Packet, 1);
      continue;
    }

    std::lock_guard<std::recursive_mutex> Lock(Mutex);

    if (PrintedKDs.insert(DispatchPacket->kernel_object).second) {
      llvm::LLVMContext Ctx;
      auto M = std::make_unique<llvm::Module>("luthier.target", Ctx);
      M->setTargetTriple(llvm::Triple("amdgcn-amd-amdhsa"));

      // MAM must die before FAM (its proxy-result destructor calls
      // `FAM->clear()`). C++ destroys in reverse declaration order.
      llvm::LoopAnalysisManager LAM;
      llvm::FunctionAnalysisManager FAM;
      llvm::CGSCCAnalysisManager CGAM;
      llvm::ModuleAnalysisManager MAM;
      llvm::PassBuilder PB;
      PB.registerModuleAnalyses(MAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      const auto &KD =
          *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
              DispatchPacket->kernel_object);

      MAM.registerPass([&] {
        return luthier::InitialEntryPointAnalysis(
            [&](llvm::Module &, llvm::ModuleAnalysisManager &) {
              return luthier::EntryPoint(KD);
            });
      });
      MAM.registerPass([&] {
        return luthier::InitialExecutionPointAnalysis(
            [&](llvm::Module &, llvm::ModuleAnalysisManager &)
                -> const llvm::amdhsa::kernel_descriptor_t & { return KD; });
      });
      MAM.registerPass([&] {
        return luthier::MemoryAllocationAnalysis(
            std::make_unique<luthier::HsaMemoryAllocationAccessor>(
                *this, *::CoreApiSnapshot, *::AmdExtSnapshot,
                ::LoaderSnapshot->getTable()));
      });
      luthier::amdgpu::hsamd::MetadataParser MetadataParserInstance;
      MAM.registerPass([&] {
        return luthier::MetadataParserAnalysis(MetadataParserInstance);
      });
      MAM.registerPass([] { return luthier::TraceCallGraphAnalysis(); });
      MAM.registerPass(
          [] { return luthier::FunctionPreambleDescriptorAnalysis(); });
      MAM.registerPass([] { return luthier::IPPredCFGAnalysis(); });

      llvm::ModulePassManager MPM;
      MPM.addPass(luthier::CodeDiscoveryPass(Opts.DiscoveryOpts));
      MPM.run(*M, MAM);

      luthier::errs() << "Lifted kernel for kernel_object 0x"
                      << llvm::Twine::utohexstr(DispatchPacket->kernel_object)
                      << ":\n";
      M->print(luthier::errs(), /*AAW=*/nullptr);
      luthier::errs() << "\n";
    }
    PacketWriter(&Packet, 1);
  }
}

namespace luthier {

static void atToolFini(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    luthier::errs() << "Luthier lift launched kernels tool is terminating!\n";
    LiftLaunchedKernelsTool::destroyInstance();
    delete LoaderSnapshot;
    LoaderSnapshot = nullptr;
    delete AmdExtSnapshot;
    AmdExtSnapshot = nullptr;
    delete CoreApiSnapshot;
    CoreApiSnapshot = nullptr;
    luthier::finalizeStreams();
  });
}

static void atToolFini() { atToolFini(nullptr); }

static void atToolInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    luthier::initializeStreams();
    luthier::errs() << "Luthier lift launched kernels tool is launching!\n";
    std::atexit(atToolFini);

    llvm::Error Err = llvm::Error::success();
    CoreApiSnapshot =
        new luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>(Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
    AmdExtSnapshot =
        new luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable>(Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
    LoaderSnapshot = new luthier::rocprofiler::HsaExtensionTableSnapshot<
        HSA_EXTENSION_AMD_LOADER>(Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    LiftLaunchedKernelsTool::createInstance(*CoreApiSnapshot, *AmdExtSnapshot,
                                            *LoaderSnapshot, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(
            0, &Argv, "Luthier Lift Launched Kernel Tool", &luthier::errs(),
            nullptr, "LUTHIER_ARGS"),
        "Failed to parse the command line arguments."));
    llvm::EnablePrettyStackTrace();
    luthier::printStackTraceOnFatalSignal();
    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/matinraayai/Luthier/issues/ and "
                          "include the crash error message and backtrace.\n");
    llvm::sys::AddSignalHandler(atToolFini, nullptr);
  });
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ClientID) {
  luthier::atToolInit(nullptr);

  LLVM_DEBUG(const uint32_t RocProfVerMajor = Version / 10000;
             const uint32_t RocProfVerMinor = (Version % 10000) / 100;
             const uint32_t RocProfVerPatch = Version % 100;
             llvm::dbgs() << "Registering Luthier tool with rocprofiler-sdk.\n";
             llvm::dbgs() << "Rocprofiler-sdk version: " << RocProfVerMajor
                          << "." << RocProfVerMinor << "." << RocProfVerPatch
                          << " (" << RuntimeVersion << ")\n";
             llvm::dbgs() << "Tool priority: " << Priority << ";\n";);
  ClientID->name = "Luthier Lift Launched Kernels tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
