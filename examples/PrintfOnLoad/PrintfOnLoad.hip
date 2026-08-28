//===-- PrintfOnLoad.hip - Printf-Before-Load Example ----------*- C++ -*-===//
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
/// Minimal Luthier tool that injects one \c printf call ahead of every
/// load-carrying machine instruction the target kernel executes. Modeled on
/// \c InstrCount.hip: same singleton \c HSATool, same on-packet override,
/// same payload-creation callback — only the per-MI predicate (\c mayLoad)
/// and the payload (\c printf instead of an atomic counter) change.
///
/// The device payload prints one line per wavefront that reaches a load, so
/// output volume is <tt>(load MIs) × (waves)</tt> rather than
/// <tt>× (active lanes)</tt>. Even that can be a lot; a real workload will
/// flood stdout. The intent here is to demonstrate that the
/// \c ExtendedKernargBuffer wiring propagates the printf hostcall buffer
/// end-to-end, not to be quiet.
//===----------------------------------------------------------------------===//
#include <chrono>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/Agent.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSATooling/HSATool.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <memory>
#include <mutex>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-printf-on-load-tool"

//===----------------------------------------------------------------------===//
// PrintfOnLoadTool — per-tool singleton subclass of HSATool
//===----------------------------------------------------------------------===//

struct PrintfOnLoadTool
    : public luthier::HSATool<PrintfOnLoadTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  //===---------------------------------------------------------------===//
  // Device-side state. The hook is a single member function; there is no
  // device global for this tool because printf carries every observation
  // through the hostcall service.
  //===---------------------------------------------------------------===//

  /// Payload for every load-carrying target MI. Prints one line per
  /// wavefront that reaches the site. Guarded on the first active lane
  /// so a wave with N active lanes still emits one record, not N.
  ///
  /// The bisection variants below carve the payload into single-piece
  /// bodies so we can localize which construct triggers the codegen
  /// "illegal VGPR to SGPR copy" error. Selection is controlled by
  /// \c Opts.PayloadPiece (see \c PayloadPieceOpt).
  __attribute__((device, used)) static void printfOnLoad();
  __attribute__((device, used)) static void payloadEmpty();
  __attribute__((device, used)) static void payloadPrintfOnly();
  __attribute__((device, used)) static void payloadReadExec();
  __attribute__((device, used)) static void payloadLaneId();
  __attribute__((device, used)) static void payloadFfsll();
  __attribute__((device, used)) static void payloadGuardOnly();

  //===---------------------------------------------------------------===//
  // Host-side CLI + bookkeeping.
  //===---------------------------------------------------------------===//

  struct Options {
    llvm::cl::OptionCategory Cat{"Printf-on-Load Tool Options"};

    llvm::cl::opt<unsigned> InstrBeginInterval{
        "instr-begin-interval",
        llvm::cl::desc("First load MI index (inclusive) to instrument, per "
                       "kernel"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> InstrEndInterval{
        "instr-end-interval",
        llvm::cl::desc("End of the load-MI interval (exclusive)"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> KernelBeginInterval{
        "kernel-begin-interval",
        llvm::cl::desc("First dispatched kernel (inclusive) to instrument"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> KernelEndInterval{
        "kernel-end-interval",
        llvm::cl::desc("End of the dispatched-kernel interval (exclusive)"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    /// Which payload variant to inject. Used to bisect which construct in
    /// the full \c printfOnLoad body triggers the "illegal VGPR to SGPR
    /// copy" codegen error. The variants share the load-MI predicate; only
    /// the callee changes.
    enum PayloadPiece {
      Full,        // full printfOnLoad (guard + printf)
      Empty,       // return; — nothing but epilogue
      PrintfOnly,  // printf(...), no guard
      ReadExec,    // consume __builtin_amdgcn_read_exec()
      LaneId,      // consume __lane_id()+1
      Ffsll,       // consume __ffsll(exec)
      GuardOnly,   // full lane guard, no printf
    };
    llvm::cl::opt<PayloadPiece> Piece{
        "payload-piece",
        llvm::cl::desc("Which slice of the printf-on-load payload to use "
                       "(for codegen bisection)"),
        llvm::cl::values(
            clEnumValN(Full, "full", "full body (default)"),
            clEnumValN(Empty, "empty", "empty body"),
            clEnumValN(PrintfOnly, "printf-only", "printf() with no guard"),
            clEnumValN(ReadExec, "read-exec", "__builtin_amdgcn_read_exec() only"),
            clEnumValN(LaneId, "lane-id", "__lane_id()+1 only"),
            clEnumValN(Ffsll, "ffsll", "__ffsll(exec) only"),
            clEnumValN(GuardOnly, "guard-only",
                       "full lane guard, no printf call")),
        llvm::cl::init(Full), llvm::cl::NotHidden, llvm::cl::cat(Cat)};
  } Opts;

  std::recursive_mutex Mutex;

  /// Number of kernels dispatched so far.
  uint32_t NumKernelLaunched = 0;

  /// Cross-MF load-MI index for the current sweep. Reset per dispatch.
  unsigned NextLoadIdx = 0;

  //===---------------------------------------------------------------===//
  // PacketMonitorTrait callback.
  //===---------------------------------------------------------------===//
  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  //===---------------------------------------------------------------===//
  // InjectedPayloadCreationPass callback.
  //===---------------------------------------------------------------===//
  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// Out-of-line static device member definitions
//===----------------------------------------------------------------------===//

__attribute__((device, used)) void PrintfOnLoadTool::printfOnLoad() {
  // Emit exactly one hostcall per wave. Any lane would produce the same
  // record (the payload is called in convergent context at the site), so
  // gate on the first active lane and let the rest fall through as no-ops.
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  const uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId != LaneId)
    return;

  // A constant-string printf keeps the payload's IR footprint minimal — no
  // dispatch-geometry reads, no varargs materialization — while still
  // exercising the hostcall service the extended kernarg buffer wires up.
  printf("[luthier-printf-on-load] load reached\n");
}

// Bisection variants. Each captures one construct from \c printfOnLoad so
// we can pin down which lowering step triggers the codegen error. The
// \c volatile sink stops the optimizer from folding the observed value
// away before codegen sees it.
__attribute__((device, noinline)) void
PrintfOnLoadTool::payloadEmpty() {}

__attribute__((device, noinline)) void
PrintfOnLoadTool::payloadPrintfOnly() {
  printf("[luthier-printf-on-load] load reached\n");
}

__attribute__((device, noinline)) void
PrintfOnLoadTool::payloadReadExec() {
  volatile unsigned long long int Sink = __builtin_amdgcn_read_exec();
  (void)Sink;
}

__attribute__((device, noinline)) void
PrintfOnLoadTool::payloadLaneId() {
  volatile uint32_t Sink = __lane_id() + 1;
  (void)Sink;
}

__attribute__((device, noinline)) void
PrintfOnLoadTool::payloadFfsll() {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  volatile uint32_t Sink = __ffsll(ExecMask);
  (void)Sink;
}

__attribute__((device, noinline)) void
PrintfOnLoadTool::payloadGuardOnly() {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  const uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId != LaneId)
    return;
  volatile uint32_t Sink = 1;
  (void)Sink;
}

//===----------------------------------------------------------------------===//
// File-scope statics: API-table snapshots + the tool itself.
//===----------------------------------------------------------------------===//

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(PrintfOnLoadTool)

static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot{
    nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};

//===----------------------------------------------------------------------===//
// PrintfOnLoadTool::run — payload creation
//===----------------------------------------------------------------------===//

llvm::PreservedAnalyses
PrintfOnLoadTool::run(luthier::Prototype &P,
                      luthier::PrototypeAnalysisManager &PAM) {
  llvm::Module &IModule = P.getInstrumentationModule();
  (void)IModule;
  NextLoadIdx = 0;
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    // Skip HIP runtime helper kernels (\c __amd_rocclr_fillBufferAligned,
    // \c __amd_rocclr_copyBuffer, …). Their MIR trips a downstream RA
    // liveness bug when this tool's printf payload is inlined into them
    // ("Use of $vcc_hi does not have a corresponding definition on every
    // path"). They are also HIP-runtime scaffolding rather than user code,
    // so instrumenting them is not what a demo tool wants to do. The
    // \c __amd_rocclr_ prefix is the stable name HIP emits for these
    // helpers.
    if (MF.getName().starts_with("__amd_rocclr_")) {
      luthier::errs() << "[PrintfOnLoadTool] skipping HIP runtime helper '"
                      << MF.getName() << "'\n";
      return;
    }
    luthier::errs() << "[PrintfOnLoadTool] payload creation for MF '"
                    << MF.getName() << "' starting at load idx "
                    << NextLoadIdx << "\n";
    unsigned StartIdx = NextLoadIdx;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        // The predicate that turns InstrCount into a printf-before-load
        // tool: the MI is worth instrumenting iff it has a load memory
        // operand. This mirrors what LLVM's own DebugInfoPass /
        // MachineScheduler consult when they want the "does this touch
        // memory-in?" bit — no need to enumerate every AMDGPU load
        // opcode.
        if (!MI.mayLoad())
          continue;
        unsigned I = NextLoadIdx++;
        if (I < Opts.InstrBeginInterval || I >= Opts.InstrEndInterval)
          continue;

        // Route through the CLI-selected payload slice so we can bisect
        // which construct in the full body triggers the codegen error.
        const void *PayloadFn = [&]() -> const void * {
          switch (Opts.Piece) {
          case Options::Full:
            return reinterpret_cast<const void *>(
                &PrintfOnLoadTool::printfOnLoad);
          case Options::Empty:
            return reinterpret_cast<const void *>(
                &PrintfOnLoadTool::payloadEmpty);
          case Options::PrintfOnly:
            return reinterpret_cast<const void *>(
                &PrintfOnLoadTool::payloadPrintfOnly);
          case Options::ReadExec:
            return reinterpret_cast<const void *>(
                &PrintfOnLoadTool::payloadReadExec);
          case Options::LaneId:
            return reinterpret_cast<const void *>(
                &PrintfOnLoadTool::payloadLaneId);
          case Options::Ffsll:
            return reinterpret_cast<const void *>(
                &PrintfOnLoadTool::payloadFfsll);
          case Options::GuardOnly:
            return reinterpret_cast<const void *>(
                &PrintfOnLoadTool::payloadGuardOnly);
          }
          return reinterpret_cast<const void *>(
              &PrintfOnLoadTool::printfOnLoad);
        }();

        // The empty ArrayRef spelling has to be explicit — the sibling
        // lambda-taking overload of createInjectedPayload also accepts a
        // brace-initializer at that position.
        LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
            PayloadFn, P, PAM, MI,
            llvm::ArrayRef<luthier::PayloadArg>{}));
      }
    }
    luthier::errs() << "[PrintfOnLoadTool] payload creation for MF '"
                    << MF.getName() << "' picked "
                    << (NextLoadIdx - StartIdx) << " load MI(s)\n";
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

//===----------------------------------------------------------------------===//
// PrintfOnLoadTool::onPackets
//===----------------------------------------------------------------------===//

void PrintfOnLoadTool::onPackets(
    const hsa_queue_t &Queue, uint64_t,
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

    // The launcher owns the extended kernarg buffer for the dispatch that
    // follows — the returned handle has to outlive the dispatch. Move it
    // into KernargBuf below when the override succeeds; a default-constructed
    // handle covers the code path where nothing was overridden.
    luthier::ExtendedKernargBuffer KernargBuf;
    if (ActiveRegion) {
      // Cache miss → run the pipeline, load the relocatable, and retry the
      // override. On hit, the override installs the extended kernarg buffer
      // in place. Same protocol as InstrCount.
      auto FirstTry =
          this->overrideWithInstrumented(*DispatchPacket, Queue, /*Preset=*/0);
      if (!FirstTry) {
        llvm::consumeError(FirstTry.takeError());
        std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->runInstrumentationPipelineForDispatch(*OriginalKD)
                .moveInto(ObjBuf));
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->loadInstrumented(std::move(ObjBuf), OriginalKD, 0)
                .takeError());
        auto Retry =
            this->overrideWithInstrumented(*DispatchPacket, Queue, 0);
        LUTHIER_REPORT_FATAL_ON_ERROR(Retry.takeError());
        KernargBuf = std::move(*Retry);
      } else {
        KernargBuf = std::move(*FirstTry);
      }
    }

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
    PrintfOnLoadTool::withInstance([](PrintfOnLoadTool &T) {
      luthier::errs() << "Printf-on-load tool observed "
                      << T.NumKernelLaunched << " dispatched kernel(s).\n";
    });
    PrintfOnLoadTool::destroyInstance();
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
    luthier::errs() << "Printf-on-load tool is launching.\n";
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

    PrintfOnLoadTool::createInstance(*CoreApiSnapshot, *AmdExtSnapshot,
                                     *LoaderSnapshot, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(
            0, &Argv, "Luthier ROCm Printf-on-Load Tool", &luthier::errs(),
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
             llvm::dbgs()
             << "Registering Luthier printf-on-load tool with rocprofiler-sdk.\n";
             llvm::dbgs() << "Rocprofiler-sdk version: " << RocProfVerMajor
                          << "." << RocProfVerMinor << "." << RocProfVerPatch
                          << " (" << RuntimeVersion << ")\n";
             llvm::dbgs() << "Tool priority: " << Priority << ";\n";);
  ClientID->name = "Luthier Printf-on-Load tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
