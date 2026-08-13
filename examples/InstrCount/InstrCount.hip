//===-- InstrCount.hip - Instruction Count Example -------------*- C++ -*-===//
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
/// Instruction-counter tool migrated to the per-tool \c HSATool singleton
/// design. Inspired by NVBit's instruction counter. The tool intercepts AQL
/// kernel-dispatch packets, runs the instrumentation pipeline against the
/// dispatched kernel (\c CodeDiscoveryPass → \c InstrumentationPMDriver →
/// \c NewPMAsmPrinter), loads the resulting relocatable through
/// \c InstrumentedKernelLoaderAndLauncher, and overrides the dispatched
/// packet to point at the instrumented variant.
//===----------------------------------------------------------------------===//
#include <SIInstrInfo.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/Agent.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSA/ISA.h>
#include <luthier/HSATooling/HSATool.h>
#include <luthier/HSATooling/HsaMemoryAllocationAccessor.h>
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/LLVM/CodeGenHelpers.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/CodeDiscoveryPass.h>
#include <luthier/ToolCodeGen/EntryPoint.h>
#include <luthier/ToolCodeGen/FunctionAnnotations.h>
#include <luthier/ToolCodeGen/IPPredicatedCFG.h>
#include <luthier/ToolCodeGen/InitialEntryPointAnalysis.h>
#include <luthier/ToolCodeGen/InitialExecutionPointAnalysis.h>
#include <luthier/ToolCodeGen/InstructionTracesAnalysis.h>
#include <luthier/ToolCodeGen/MemoryAllocationAccessor.h>
#include <luthier/ToolCodeGen/MetadataParserAnalysis.h>
#include <luthier/ToolCodeGen/NewPMAsmPrinter.h>
#include <luthier/ToolCodeGen/PrePostAmbleEmitter.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <memory>
#include <mutex>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-instr-counter-tool"

//===----------------------------------------------------------------------===//
// InstrCountTool — per-tool singleton subclass of HSATool
//===----------------------------------------------------------------------===//

struct InstrCountTool
    : public luthier::HSATool<InstrCountTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  //===---------------------------------------------------------------===//
  // Device-side state. Hooks and device globals live as static members
  // of the tool class so the host shadow handles are addressable as
  // \c &InstrCountTool::countInstructionsScalar / \c &InstrCountTool::Counter
  // and HIP-Clang's \c __hipRegisterFunction / \c __hipRegisterVar emit
  // entries for them into the loader's slot table.
  //===---------------------------------------------------------------===//

  /// Per-agent instruction counter, device-resident. Read after each
  /// kernel via \c hsa_memory_copy from the host (see \c onPackets);
  /// `__managed__` was previously used but its host-shadow pointer
  /// indirection adds a memory load to every increment site that
  /// drives RA toward a calling-convention mismatch on
  /// `luthier::sAtomicAdd`. A plain `__device__` global avoids the
  /// indirection — host I/O goes through HSA explicit copies.
  __attribute__((device)) static uint64_t Counter;

  /// Wave-fronts-as-warps counter hook. Atomically increments \c Counter
  /// by either 1 (wavefront-level) or by the active-lane count.
  __attribute__((device, used)) static void
  countInstructionsVector(bool CountWaveFrontLevel);

  /// Scalar counter hook. Single-lane atomic increment of \c Counter.
  __attribute__((device, used)) static void countInstructionsScalar();

  //===---------------------------------------------------------------===//
  // Host-side state. All previously file-static fields move here as
  // non-static members so destruction order is controlled by \c delete
  // on the tool.
  //===---------------------------------------------------------------===//

  /// Per-tool CLI options. Owns its own \c OptionCategory plus the
  /// \c InstrumentationPMDriverOptions consumed by the pipeline.
  struct Options {
    llvm::cl::OptionCategory Cat{"Instruction Count Tool Options"};

    llvm::cl::opt<unsigned> InstrBeginInterval{
        "instr-begin-interval",
        llvm::cl::desc("Beginning of the instruction interval where to "
                       "apply instrumentation, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> InstrEndInterval{
        "instr-end-interval",
        llvm::cl::desc("End of the instruction interval where to "
                       "apply instrumentation, exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> KernelBeginInterval{
        "kernel-begin-interval",
        llvm::cl::desc("Beginning of the kernel interval to apply "
                       "instrumentation, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> KernelEndInterval{
        "kernel-end-interval",
        llvm::cl::desc(
            "End of the kernel interval to apply instrumentation, exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<bool> CountWavefrontLevel{
        "count-wavefront-level",
        llvm::cl::desc(
            "Whether to count number of instructions at the wavefront "
            "level or thread level"),
        llvm::cl::init(false), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<bool> DemangleKernelNames{
        "demangle-kernel-names",
        llvm::cl::desc("Whether to demangle kernel names before printing"),
        llvm::cl::init(true), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

  } Opts;

  std::recursive_mutex Mutex;

  /// Number of kernels dispatched so far.
  uint32_t NumKernelLaunched = 0;

  /// Cumulative instruction count across all observed kernel dispatches.
  uint64_t TotalNumInstructions = 0;

  /// Cross-MF instruction index for the current \c runInstrumentationPass
  /// sweep. Reset at the start of each per-dispatch pipeline.
  unsigned NextInstrIdx = 0;

  //===---------------------------------------------------------------===//
  // PacketMonitorTrait callback. Called for every AQL packet submitted
  // on any intercepted queue.
  //===---------------------------------------------------------------===//
  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  //===---------------------------------------------------------------===//
  // InjectedPayloadCreationPass callback. Invoked by the
  // InstrumentationPMDriver for each MachineFunction in the discovered
  // target module.
  //===---------------------------------------------------------------===//

  /// Prototype-level entry point: payload creation reads the target module's
  /// MIR and writes into the instrumentation module, so it needs both halves of
  /// the prototype.
  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// Out-of-line static device member definitions
//===----------------------------------------------------------------------===//

__attribute__((device)) uint64_t InstrCountTool::Counter;

__attribute__((device, used)) void
InstrCountTool::countInstructionsVector(bool CountWaveFrontLevel) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  uint32_t NumActiveThreads = __popcll(ExecMask);

  if (FirstActiveThreadId == LaneId) {
    if (CountWaveFrontLevel) {
      if (NumActiveThreads > 0)
        atomicAdd(&InstrCountTool::Counter, 1);
    } else {
      atomicAdd(&InstrCountTool::Counter, NumActiveThreads);
    }
  }
}

__attribute__((device, used)) void InstrCountTool::countInstructionsScalar() {
  (void)luthier::sAtomicAdd(&InstrCountTool::Counter, 1UL);
}

//===----------------------------------------------------------------------===//
// File-scope statics: API-table snapshots + the tool itself. The snapshots
// must outlive the tool (the tool ctor takes them by reference) and must be
// constructed during rocprofiler-sdk's configure phase before any other
// component requests them.
//===----------------------------------------------------------------------===//

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(InstrCountTool)

static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot{
    nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};

//===----------------------------------------------------------------------===//
// InstrCountTool::runInstrumentationPass
//===----------------------------------------------------------------------===//

llvm::PreservedAnalyses
InstrCountTool::run(luthier::Prototype &P,
                    luthier::PrototypeAnalysisManager &PAM) {
  llvm::Module &IModule = P.getInstrumentationModule();
  NextInstrIdx = 0;
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    luthier::errs() << "[InstrCountTool] runInstrumentationPass for MF '"
                    << MF.getName() << "' starting at instr idx "
                    << NextInstrIdx << "\n";
    unsigned StartIdx = NextInstrIdx;
    auto *CountWavefrontLevelConstVal = llvm::ConstantInt::getBool(
        IModule.getContext(), Opts.CountWavefrontLevel);
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        unsigned I = NextInstrIdx++;
        if (I < Opts.InstrBeginInterval || I >= Opts.InstrEndInterval)
          continue;

        const void *HostHandle = nullptr;
        llvm::SmallVector<InjectedPayloadCreationPass::PayloadArg, 1> Args;
        if (luthier::isScalar(MI) || luthier::isLaneAccess(MI)) {
          HostHandle = reinterpret_cast<const void *>(
              &InstrCountTool::countInstructionsScalar);
        } else {
          HostHandle = reinterpret_cast<const void *>(
              &InstrCountTool::countInstructionsVector);
          Args.push_back(CountWavefrontLevelConstVal);
        }
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->createInjectedPayload(HostHandle, IModule, MI, Args));
      }
    }
    luthier::errs() << "[InstrCountTool] runInstrumentationPass for MF '"
                    << MF.getName() << "' visited " << (NextInstrIdx - StartIdx)
                    << " MI(s)\n";
  });
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  // Payload creation only adds functions to the instrumentation module and
  // attaches instrumentation-point metadata to target MIs. No MachineFunction
  // is created or destroyed, so every cached MachineFunctionAnalysis result —
  // including the lifted target MIR this pass just walked — stays valid.
  PA.preserve<luthier::TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<luthier::IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  return PA;
}

//===----------------------------------------------------------------------===//
// InstrCountTool::onPackets
//===----------------------------------------------------------------------===//

void InstrCountTool::onPackets(
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

    std::lock_guard<std::recursive_mutex> Lock(Mutex);

    const auto Core = ::CoreApiSnapshot->getTable();
    const auto AmdExt = ::AmdExtSnapshot->getTable();

    const bool ActiveRegion = NumKernelLaunched >= Opts.KernelBeginInterval &&
                              NumKernelLaunched < Opts.KernelEndInterval;

    // The launcher's cache is keyed by the *original* kernel descriptor, and
    // overrideWithInstrumented rewrites Packet.kernel_object in place. Capture
    // the key before that happens: the counter read-back below needs it, and by
    // then DispatchPacket->kernel_object names the instrumented variant.
    const auto *OriginalKD =
        reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
            DispatchPacket->kernel_object);

    bool MustDestroySignalAfterLaunch = false;
    if (DispatchPacket->completion_signal.handle == 0) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
          AmdExt.callFunction<&::AmdExtTable::hsa_amd_signal_create_fn>(
              0, 0, nullptr, 0, &DispatchPacket->completion_signal),
          "Failed to create a signal for the dispatch packet."));
      MustDestroySignalAfterLaunch = true;
    }
    const hsa_signal_value_t SignalValue =
        Core.callFunction<&::CoreApiTable::hsa_signal_load_scacquire_fn>(
            DispatchPacket->completion_signal);

    if (ActiveRegion) {
      luthier::errs() << "Kernel is being instrumented.\n";

      // If the (kernel_object, preset) pair isn't already cached, run the
      // pipeline and load the resulting relocatable into the launcher.
      // overrideWithInstrumented succeeds iff the cache has the variant.
      const auto *KD = OriginalKD;
      // Probe the cache; on miss, run the pipeline + load and re-try the
      // override. The probe Error MUST be bound to a local so its
      // destructor doesn't trip the unhandled-error abort.
      if (auto MissErr =
              this->overrideWithInstrumented(*DispatchPacket, /*Preset=*/0)) {
        llvm::consumeError(std::move(MissErr));
        std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->runInstrumentationPipelineForDispatch(*KD).moveInto(ObjBuf));
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->loadInstrumented(std::move(ObjBuf), KD, 0).takeError());
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->overrideWithInstrumented(*DispatchPacket, 0));
      }
    }

    auto T1 = std::chrono::high_resolution_clock::now();
    constexpr auto Timeout =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::seconds{300})
            .count();
    PacketWriter(DispatchPacket, 1);

    const hsa_signal_value_t CurrentSignal =
        Core.callFunction<&::CoreApiTable::hsa_signal_wait_relaxed_fn>(
            DispatchPacket->completion_signal, HSA_SIGNAL_CONDITION_LT,
            SignalValue, Timeout, HSA_WAIT_STATE_ACTIVE);
    if (CurrentSignal >= SignalValue)
      llvm::report_fatal_error(
          llvm::formatv("Wait timeout reached for signal handle {0:x}",
                        DispatchPacket->completion_signal.handle));

    if (MustDestroySignalAfterLaunch) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
          Core.callFunction<&::CoreApiTable::hsa_signal_destroy_fn>(
              DispatchPacket->completion_signal),
          "Failed to destroy the dispatch packet's completion signal."));
    }

    // Read Counter back from the device and accumulate. Counter is now
    // `__device__` (was `__managed__`), so host access goes through
    // an explicit `hsa_memory_copy` from the symbol's per-agent device
    // address. Agent comes from the same `hsa_amd_pointer_info(KD)`
    // call the dispatch path already uses; counter address comes from
    // the device-tool code loader's HSA executable symbol record.
    if (ActiveRegion) {
      const auto *KernelDesc = OriginalKD;
      hsa_amd_pointer_info_t CounterAgentInfo{};
      CounterAgentInfo.size = sizeof(CounterAgentInfo);
      hsa_status_t PInfoStatus = AmdExt.callFunction<hsa_amd_pointer_info>(
          const_cast<void *>(reinterpret_cast<const void *>(KernelDesc)),
          &CounterAgentInfo,
          /*alloc=*/nullptr, /*num_agents_accessible=*/nullptr,
          /*accessible=*/nullptr);
      if (PInfoStatus == HSA_STATUS_SUCCESS &&
          CounterAgentInfo.type != HSA_EXT_POINTER_TYPE_UNKNOWN) {
        hsa_agent_t Agent = CounterAgentInfo.agentOwner;
        auto SymOrErr =
            this->lookupGlobalVariable(&InstrCountTool::Counter, KernelDesc);
        if (!SymOrErr) {
          llvm::consumeError(SymOrErr.takeError());
        } else {
          auto AddrOrErr =
              luthier::hsa::executableSymbolGetAddress(Core, *SymOrErr);
          if (!AddrOrErr) {
            llvm::consumeError(AddrOrErr.takeError());
          } else {
            uint64_t CounterSnapshot = 0;
            hsa_status_t RC = Core.callFunction<hsa_memory_copy>(
                &CounterSnapshot, reinterpret_cast<void *>(*AddrOrErr),
                sizeof(CounterSnapshot));
            if (RC == HSA_STATUS_SUCCESS) {
              TotalNumInstructions += CounterSnapshot;
              // Zero the device-side counter for the next dispatch.
              const uint64_t Zero = 0;
              (void)Core.callFunction<hsa_memory_copy>(
                  reinterpret_cast<void *>(*AddrOrErr), &Zero, sizeof(Zero));
            }
          }
        }
      }
    }

    (void)T1;
    NumKernelLaunched++;
  }
}

//===----------------------------------------------------------------------===//
// rocprofiler-sdk init / fini
//===----------------------------------------------------------------------===//

namespace luthier {

static void atToolFini(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    InstrCountTool::withInstance([](InstrCountTool &T) {
      luthier::errs() << "Total instructions executed: "
                      << T.TotalNumInstructions << "\n";
    });
    InstrCountTool::destroyInstance();
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
    luthier::errs() << "Instruction counter tool is launching.\n";
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

    InstrCountTool::createInstance(*CoreApiSnapshot, *AmdExtSnapshot,
                                   *LoaderSnapshot, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(
            0, &Argv, "Luthier ROCm Instruction Counter Tool", &luthier::errs(),
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
  ClientID->name = "Luthier Instruction Counter tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
