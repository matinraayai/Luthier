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
#include <luthier/ToolCodeGen/InstrumentationPassBuilder.h>
#include <luthier/ToolCodeGen/MemoryAllocationAccessor.h>
#include <luthier/ToolCodeGen/MetadataParserAnalysis.h>
#include <luthier/ToolCodeGen/Prototype.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <mutex>
#include <unordered_set>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lift-launched-kernels-tool"

struct LiftLaunchedKernelsTool
    : public luthier::HSATool<LiftLaunchedKernelsTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  struct Options {
    llvm::cl::OptionCategory Cat{"Lift Launched Kernels Tool Options"};
  } Opts;

  std::recursive_mutex Mutex;

  /// KD addresses already printed, so we don't dump the same kernel twice.
  std::unordered_set<uint64_t> PrintedKDs;

  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  /// No-op: the tool only lifts and prints, never injects payloads. Nothing is
  /// mutated, so every analysis stays valid.
  llvm::PreservedAnalyses run(luthier::Prototype &,
                              luthier::PrototypeAnalysisManager &) {
    return llvm::PreservedAnalyses::all();
  }
};

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(LiftLaunchedKernelsTool)

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
      const auto &KD =
          *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
              DispatchPacket->kernel_object);

      std::unique_ptr<llvm::TargetMachine> TM;
      LUTHIER_REPORT_FATAL_ON_ERROR(buildTargetMachineForKD(&KD).moveInto(TM));

      llvm::LLVMContext Ctx;
      auto TargetM = std::make_unique<llvm::Module>("luthier.target", Ctx);
      TargetM->setTargetTriple(TM->getTargetTriple());
      TargetM->setDataLayout(TM->createDataLayout());
      auto IModuleM = std::make_unique<llvm::Module>("luthier.imodule", Ctx);
      IModuleM->setTargetTriple(TM->getTargetTriple());
      IModuleM->setDataLayout(TM->createDataLayout());

      // Code discovery reads the dispatch's entry point off the target module
      // rather than from a resolver callback.
      luthier::setInitialEntryPoint(*TargetM, luthier::EntryPoint(KD));
      luthier::setInitialExecutionPoint(*TargetM, KD);

      // CodeDiscoveryPass is a Prototype pass, so it needs both modules even
      // though this tool never injects anything into the instrumentation one.
      luthier::Prototype IP(std::move(TargetM), std::move(IModuleM));

      llvm::MachineModuleInfo MMI(TM.get());

      // Each module of the prototype gets its own managers, so that a pass over
      // one module cannot clear the other's cached results. Innermost manager
      // first: an inner analysis-manager proxy's destructor clears the manager
      // it proxies, so an outer manager must be declared after everything it
      // proxies.
      llvm::LoopAnalysisManager TargetLAM, ILAM;
      llvm::FunctionAnalysisManager TargetFAM, IFAM;
      llvm::CGSCCAnalysisManager TargetCGAM, ICGAM;
      llvm::MachineFunctionAnalysisManager TargetMFAM, IMFAM;
      llvm::ModuleAnalysisManager TargetMAM, IMAM;
      luthier::PrototypeAnalysisManager IPAM;

      const luthier::InstrumentationPassBuilder::ModuleAnalysisManagers
          TargetAMs{TargetMAM, TargetCGAM, TargetFAM, TargetLAM, TargetMFAM};
      const luthier::InstrumentationPassBuilder::ModuleAnalysisManagers IAMs{
          IMAM, ICGAM, IFAM, ILAM, IMFAM};

      luthier::InstrumentationPassBuilder PB(*TM);
      PB.registerPrototypeAnalyses(IPAM);
      PB.registerAnalyses(TargetAMs);
      PB.registerAnalyses(IAMs);
      PB.crossRegisterProxies(IPAM, TargetAMs, IAMs);

      luthier::amdgpu::hsamd::MetadataParser MetadataParserInstance;
      for (llvm::ModuleAnalysisManager *M : {&TargetMAM, &IMAM}) {
        M->registerPass([&] { return llvm::MachineModuleAnalysis(MMI); });
        M->registerPass([] { return luthier::InitialEntryPointAnalysis(); });
        M->registerPass(
            [] { return luthier::InitialExecutionPointAnalysis(); });
        M->registerPass([&] {
          return luthier::MemoryAllocationAnalysis(
              std::make_unique<luthier::HsaMemoryAllocationAccessor>(
                  *this, *::CoreApiSnapshot, *::AmdExtSnapshot,
                  *::LoaderSnapshot));
        });
        M->registerPass([&] {
          return luthier::MetadataParserAnalysis(MetadataParserInstance);
        });
      }

      luthier::PrototypePassManager IPPM;
      IPPM.addPass(luthier::CodeDiscoveryPass());
      IPPM.run(IP, IPAM);

      luthier::errs() << "Lifted kernel for kernel_object 0x"
                      << llvm::Twine::utohexstr(DispatchPacket->kernel_object)
                      << ":\n";
      IP.getTargetModule().print(luthier::errs(), /*AAW=*/nullptr);
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
