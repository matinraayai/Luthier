//===-- KernelInstrument.hip ----------------------------------------------===//
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
/// Sample kernel-instrumentation tool demonstrating the \c writeReg intrinsic.
/// Inserts an \c instrumentationHook before a fixed offset in the entry MBB.
//===----------------------------------------------------------------------===//
#include <SIDefines.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSATooling/HSATool.h>
#include <luthier/HSATooling/HsaMemoryAllocationAccessor.h>
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/CodeDiscoveryPass.h>
#include <luthier/ToolCodeGen/EntryPoint.h>
#include <luthier/ToolCodeGen/FunctionAnnotations.h>
#include <luthier/ToolCodeGen/IPPredicatedCFG.h>
#include <luthier/ToolCodeGen/InitialEntryPointAnalysis.h>
#include <luthier/ToolCodeGen/InitialExecutionPointAnalysis.h>
#include <luthier/ToolCodeGen/MemoryAllocationAccessor.h>
#include <luthier/ToolCodeGen/MetadataParserAnalysis.h>
#include <luthier/ToolCodeGen/NewPMAsmPrinter.h>
#include <luthier/ToolCodeGen/PrePostAmbleEmitter.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <mutex>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-kernel-instrument-tool"

namespace {

struct KernelInstrumentTool
    : public luthier::HSATool<KernelInstrumentTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  __attribute__((managed)) static uint64_t WaveID;
  __attribute__((managed)) static hsa_dim3_t GroupIDs;
  __attribute__((managed)) static uint64_t PositionInGroup;

  __attribute__((device, used)) static void
  instrumentationHook(llvm::MCRegister Reg, uint32_t Val);

  struct Options {
    llvm::cl::OptionCategory Cat{"Kernel Instrument Tool Options"};
    luthier::InstrumentationPMDriverOptions DriverOpts;
    luthier::CodeDiscoveryPassOptions DiscoveryOpts;
  } Opts;

  std::mutex Mutex;
  uint32_t NumKernelLaunched = 0;

  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM);
};

__attribute__((managed)) uint64_t KernelInstrumentTool::WaveID = 0;
__attribute__((managed)) hsa_dim3_t KernelInstrumentTool::GroupIDs{0, 0, 0};
__attribute__((managed)) uint64_t KernelInstrumentTool::PositionInGroup{0};

__attribute__((device, used)) void
KernelInstrumentTool::instrumentationHook(llvm::MCRegister Reg, uint32_t Val) {
  luthier::writeReg(Reg, Val);
}

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(KernelInstrumentTool)

static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot{
    nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};

static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
runInstrumentationPipelineForDispatch(
    KernelInstrumentTool &T, const hsa_kernel_dispatch_packet_t &Packet) {
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

  const auto &KD = *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
      Packet.kernel_object);

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
            T, *::CoreApiSnapshot, *::AmdExtSnapshot,
            ::LoaderSnapshot->getTable()));
  });
  luthier::amdgpu::hsamd::MetadataParser MetadataParserInstance;
  MAM.registerPass(
      [&] { return luthier::MetadataParserAnalysis(MetadataParserInstance); });
  MAM.registerPass([] { return luthier::LuthierCallGraphAnalysis(); });
  MAM.registerPass(
      [] { return luthier::FunctionPreambleDescriptorAnalysis(); });
  MAM.registerPass([] { return luthier::IPPredCFGAnalysis(); });

  llvm::SmallVector<char, 0> ObjBuf;
  llvm::raw_svector_ostream ObjOS(ObjBuf);

  llvm::ModulePassManager MPM;
  MPM.addPass(luthier::CodeDiscoveryPass(T.Opts.DiscoveryOpts));
  MPM.addPass(T.buildPipeline(T.Opts.DriverOpts));
  MPM.addPass(
      luthier::NewPMAsmPrinter(llvm::CodeGenFileType::ObjectFile, ObjOS));
  MPM.run(*M, MAM);

  return std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(ObjBuf), "luthier.instrumented",
      /*RequiresNullTerminator=*/false);
}

llvm::PreservedAnalyses
KernelInstrumentTool::run(luthier::Prototype &P,
                          luthier::PrototypeAnalysisManager &PAM) {
  llvm::Module &IModule = P.getInstrumentationModule();
  auto &Ctx = IModule.getContext();
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    if (MF.empty() || MF.front().empty())
      return;
    // Replicate the original tool's "fourth instruction in the entry MBB"
    // insertion point — chase ->getNextNode() three times from the first
    // instruction.
    auto *Begin = &*MF.front().begin();
    auto *Tgt = Begin->getNextNode();
    if (!Tgt)
      return;
    auto *T2 = Tgt->getNextNode();
    if (!T2)
      return;
    auto *T3 = T2->getNextNode();
    if (!T3)
      return;

    LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
        &KernelInstrumentTool::instrumentationHook, P, PAM, *T3,
        {llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), llvm::AMDGPU::SCC),
         llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1)}));
  });
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<luthier::TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.abandon<luthier::InjectedPayloadAndInstPointAnalysis>();
  PA.abandon<luthier::PrototypeCallGraphAnalysis>();
  return PA;
}

void KernelInstrumentTool::onPackets(
    const hsa_queue_t &, uint64_t,
    llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
    hsa_amd_queue_intercept_packet_writer PacketWriter) {
  std::vector<luthier::hsa::AqlPacket> AppPackets(Packets.begin(),
                                                  Packets.end());
  for (auto &Packet : AppPackets) {
    auto *DispatchPacket = Packet.asKernelDispatch();
    if (!DispatchPacket) {
      PacketWriter(&Packet, 1);
      continue;
    }
    std::lock_guard<std::mutex> Lock(Mutex);

    const auto *KD =
        reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
            DispatchPacket->kernel_object);
    if (auto MissErr = this->overrideWithInstrumented(*DispatchPacket, 0)) {
      llvm::consumeError(std::move(MissErr));
      std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
      LUTHIER_REPORT_FATAL_ON_ERROR(
          runInstrumentationPipelineForDispatch(*this, *DispatchPacket)
              .moveInto(ObjBuf));
      LUTHIER_REPORT_FATAL_ON_ERROR(
          this->loadInstrumented(std::move(ObjBuf), KD, 0).takeError());
      LUTHIER_REPORT_FATAL_ON_ERROR(
          this->overrideWithInstrumented(*DispatchPacket, 0));
    }
    PacketWriter(DispatchPacket, 1);
    NumKernelLaunched++;
  }
}

} // anonymous namespace

namespace luthier {

static void atToolFini(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    luthier::outs() << llvm::formatv("WaveID: {0}\n",
                                     KernelInstrumentTool::WaveID);
    luthier::outs() << llvm::formatv(
        "Group ID: ({0}, {1}, {2})\n", KernelInstrumentTool::GroupIDs.z,
        KernelInstrumentTool::GroupIDs.y, KernelInstrumentTool::GroupIDs.x);
    luthier::outs() << llvm::formatv("Position in Group: {0}\n",
                                     KernelInstrumentTool::PositionInGroup);
    luthier::outs() << "Kernel Instrument Tool is terminating!\n";
    KernelInstrumentTool::destroyInstance();
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
    luthier::outs() << "Kernel instrument tool is launching.\n";
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

    KernelInstrumentTool::createInstance(*CoreApiSnapshot, *AmdExtSnapshot,
                                         *LoaderSnapshot, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(
            0, &Argv, "Luthier Kernel Instrument Tool", &luthier::errs(),
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
  ClientID->name = "Luthier Kernel Instrument tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
