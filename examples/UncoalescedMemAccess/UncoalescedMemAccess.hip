//===-- UncoalescedMemAccess.hip - Uncoalesced memory accesses --*- C++ -*-===//
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
/// Counts uncoalesced global memory accesses.
///
/// Every global/flat memory instruction is traced with
/// \c luthier::comms::MemoryTrace: while the kernel runs, each access reaches
/// the CPU as (kernel-argument buffer, element index) through fine-grained
/// pinned host memory. \c CoalescingHandler then rebuilds the addresses and
/// decides which accesses were coalesced.
///
/// <b>Uncoalesced.</b> The hardware issues a 64-lane VMEM instruction 16
/// lanes per cycle. A group is coalesced if it touches no more 128-byte cache
/// lines than its bytes need; a wave-instruction is uncoalesced if any group
/// wastes a line. This is what the hardware counters measure:
/// \code
///   uncoalesced wave-instructions = SQ_INSTS_VMEM - TD_COALESCABLE_WAVEFRONT_sum
/// \endcode
///
/// <b>Usage.</b>
/// \code
///   LD_PRELOAD=<build>/lib/libLuthierTooling.so:\
///   <build>/examples/UncoalescedMemAccess/libLuthierUncoalescedMemAccess.so ./app
/// \endcode
/// Options (in \c LUTHIER_ARGS):
///   --comms-sub-buffers=N           sub-buffers, power of two (256)
///   --comms-sub-buffer-bytes=N      capacity of each (65536)
///   --comms-element-size=N          bytes per array element (4)
///   --comms-workgroup-ids=ttmp|sva|none
///                                   where hooks read the workgroup id (ttmp)
///   --lanes-per-cycle=N             coalescing granularity (16)
///   --cache-line-bytes=N            cache line size (128)
///   --kernel-begin-interval / --kernel-end-interval
///
/// <b>Validated</b> on a 1024^3 MFMA SGEMM (gfx908) against rocprofv3:
/// SQ_INSTS_VMEM, the coalesced and the uncoalesced counts match exactly.
//===----------------------------------------------------------------------===//
#include "CoalescingHandler.h"

#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <luthier/Comms/MemoryTrace.h>
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/Rocprofiler/RocprofilerSdkLibrary.h>
#include <luthier/ToolCodeGen/InitialEntryPointAnalysis.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <luthier/Tooling/Tool.h>

#include <map>
#include <mutex>
#include <set>
#include <tuple>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-uncoalesced-mem-access"

namespace comms = luthier::comms;

/// Which workgroups and compute units sent messages, to check the workgroup
/// ids and hardware ids are real.
class WorkgroupCensus final : public comms::MemoryAccessHandler {
public:
  void handleAccesses(const comms::WaveHeader &H,
                      llvm::ArrayRef<comms::AccessRecord>) override {
    ++Messages[{H.BlockIdxX, H.BlockIdxY, H.BlockIdxZ}];
    // GFX9 HW_ID: cu_id [11:8], sh_id [12], se_id [15:13].
    if (H.HwId != 0)
      ComputeUnits.insert(H.HwId >> 8 & 0xffu);
  }
  void report(llvm::raw_ostream &OS) const {
    if (Messages.empty() ||
        std::get<0>(Messages.begin()->first) == comms::UnknownBlockIdx) {
      OS << "[uncoalesced] workgroup ids: not reported\n";
      return;
    }
    uint32_t MaxX = 0, MaxY = 0, MaxZ = 0;
    uint64_t MinMsgs = UINT64_MAX, MaxMsgs = 0;
    for (const auto &[Wg, N] : Messages) {
      MaxX = std::max(MaxX, std::get<0>(Wg));
      MaxY = std::max(MaxY, std::get<1>(Wg));
      MaxZ = std::max(MaxZ, std::get<2>(Wg));
      MinMsgs = std::min(MinMsgs, N);
      MaxMsgs = std::max(MaxMsgs, N);
    }
    OS << llvm::formatv("[uncoalesced] workgroup ids: {0} distinct, "
                        "x<={1} y<={2} z<={3}, {4}-{5} messages each\n",
                        Messages.size(), MaxX, MaxY, MaxZ, MinMsgs, MaxMsgs);
    std::set<unsigned> Engines;
    for (unsigned C : ComputeUnits)
      Engines.insert(C >> 5);
    OS << llvm::formatv("[uncoalesced] hardware ids: {0} distinct compute "
                        "units on {1} shader engines\n",
                        ComputeUnits.size(), Engines.size());
  }

private:
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint64_t> Messages;
  std::set<unsigned> ComputeUnits; ///< (se, sh, cu) packed as HW_ID[15:8]
};

struct UncoalescedMemAccessTool
    : public luthier::Tool<UncoalescedMemAccessTool, llvm::MachineFunction>,
      public comms::MemoryTrace<UncoalescedMemAccessTool> {
  using Tool::Tool;

  struct Options {
    llvm::cl::OptionCategory Cat{"Uncoalesced Memory Access Tool Options"};
    llvm::cl::opt<unsigned> NumSubBuffers{
        "comms-sub-buffers", llvm::cl::desc("Sub-buffers (power of two)"),
        llvm::cl::init(256), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> SubBufferBytes{
        "comms-sub-buffer-bytes", llvm::cl::desc("Bytes per sub-buffer"),
        llvm::cl::init(64 * 1024), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> ElementSize{
        "comms-element-size",
        llvm::cl::desc("Bytes per array element (power of two)"),
        llvm::cl::init(4), llvm::cl::cat(Cat)};
    llvm::cl::opt<WorkgroupIdSource> WorkgroupIds{
        "comms-workgroup-ids", llvm::cl::desc("Where to read workgroup ids"),
        llvm::cl::values(
            clEnumValN(WorkgroupIdSource::TTMP, "ttmp", "TTMP8/9/10"),
            clEnumValN(WorkgroupIdSource::SVA, "sva", "luthier::readSVA"),
            clEnumValN(WorkgroupIdSource::None, "none", "not reported")),
        llvm::cl::init(WorkgroupIdSource::TTMP), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> LanesPerCycle{
        "lanes-per-cycle", llvm::cl::desc("Lanes issued together (1-64)"),
        llvm::cl::init(16), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> CacheLineBytes{
        "cache-line-bytes", llvm::cl::desc("Cache line size in bytes"),
        llvm::cl::init(128), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> KernelBeginInterval{
        "kernel-begin-interval",
        llvm::cl::desc("First kernel to instrument, inclusive"),
        llvm::cl::init(0), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> KernelEndInterval{
        "kernel-end-interval",
        llvm::cl::desc("Last kernel to instrument, exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::cat(Cat)};
  } Opts;

  std::recursive_mutex Mutex;
  uint32_t NumKernelLaunched = 0;
  std::unique_ptr<CoalescingHandler> Coalescing;
  WorkgroupCensus Census;

  /// Applies the options; called once they are parsed.
  void configure() {
    setConfig({Opts.NumSubBuffers, Opts.SubBufferBytes, Opts.ElementSize,
               Opts.WorkgroupIds});
    Coalescing =
        std::make_unique<CoalescingHandler>(Opts.LanesPerCycle,
                                            Opts.CacheLineBytes);
    addHandler(*Coalescing);
    addHandler(Census);
  }

  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM);

  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  void printReport() const;
};

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(UncoalescedMemAccessTool)

static luthier::DynamicLibrary *RocprofilerLib{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot{
    nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};

//===----------------------------------------------------------------------===//
// Instrumentation: trace every global memory instruction
//===----------------------------------------------------------------------===//

llvm::PreservedAnalyses
UncoalescedMemAccessTool::run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM) {
  // A device-function prototype is loaded under its kernel's code object,
  // which already defines the tracer's globals: bind to those instead of
  // defining second copies.
  {
    llvm::Module &IModule = P.getInstrumentationModule();
    auto &TargetMAM =
        PAM.getResult<luthier::TargetModuleAnalysisManagerPrototypeProxy>(P)
            .getManager();
    if (!TargetMAM
             .getResult<luthier::InitialEntryPointAnalysis>(P.getTargetModule())
             .getInitialEntryPoint()
             .isKernel()) {
      llvm::SmallPtrSet<llvm::Constant *, 8> Externalized;
      for (llvm::GlobalVariable &GV : IModule.globals()) {
        if (GV.isDeclaration() || GV.hasLocalLinkage() ||
            GV.getName().starts_with("llvm."))
          continue;
        GV.setInitializer(nullptr);
        GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
        GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
        GV.setDSOLocal(false);
        Externalized.insert(&GV);
      }
      llvm::removeFromUsedLists(IModule, [&](llvm::Constant *C) {
        return C && Externalized.contains(C->stripPointerCasts());
      });
    }
  }

  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    unsigned Traced = 0;
    for (llvm::MachineBasicBlock &MBB : MF)
      for (llvm::MachineInstr &MI : MBB)
        if (isTraceable(MI)) {
          LUTHIER_REPORT_FATAL_ON_ERROR(traceAccess(P, PAM, MI));
          ++Traced;
        }
    luthier::errs() << "[uncoalesced] " << MF.getName() << ": traced "
                    << Traced << " memory instruction(s)\n";
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
// Dispatch handling
//===----------------------------------------------------------------------===//

void UncoalescedMemAccessTool::onPackets(
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
    const bool Active = NumKernelLaunched >= Opts.KernelBeginInterval &&
                        NumKernelLaunched < Opts.KernelEndInterval;
    // The override rewrites the packet; keep the application's version.
    const hsa_kernel_dispatch_packet_t AppPacket = *DispatchPacket;
    const auto *KD = reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
        DispatchPacket->kernel_object);

    bool MustDestroySignal = false;
    if (DispatchPacket->completion_signal.handle == 0) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
          AmdExt.callFunction<&::AmdExtTable::hsa_amd_signal_create_fn>(
              0, 0, nullptr, 0, &DispatchPacket->completion_signal),
          "Failed to create a signal for the dispatch packet."));
      MustDestroySignal = true;
    }
    const hsa_signal_value_t SignalValue =
        Core.callFunction<&::CoreApiTable::hsa_signal_load_scacquire_fn>(
            DispatchPacket->completion_signal);

    luthier::ExtendedKernargBuffer KernargBuf;
    if (Active) {
      auto FirstTry = overrideWithInstrumented(*DispatchPacket, Queue, 0);
      if (!FirstTry) {
        llvm::consumeError(FirstTry.takeError());
        std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
        LUTHIER_REPORT_FATAL_ON_ERROR(
            runInstrumentationPipelineForDispatch(*KD).moveInto(ObjBuf));
        LUTHIER_REPORT_FATAL_ON_ERROR(
            loadInstrumented(std::move(ObjBuf), KD, 0).takeError());
        auto Retry = overrideWithInstrumented(*DispatchPacket, Queue, 0);
        LUTHIER_REPORT_FATAL_ON_ERROR(Retry.takeError());
        KernargBuf = std::move(*Retry);
      } else {
        KernargBuf = std::move(*FirstTry);
      }
      LUTHIER_REPORT_FATAL_ON_ERROR(beginDispatch(AppPacket, *KD));
    }

    // Generous: waves may stall while the CPU drains sub-buffers.
    constexpr auto Timeout =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::seconds{1800})
            .count();
    PacketWriter(DispatchPacket, 1);
    const hsa_signal_value_t CurrentSignal =
        Core.callFunction<&::CoreApiTable::hsa_signal_wait_relaxed_fn>(
            DispatchPacket->completion_signal, HSA_SIGNAL_CONDITION_LT,
            SignalValue, Timeout, HSA_WAIT_STATE_ACTIVE);
    if (Active)
      endDispatch();
    if (CurrentSignal >= SignalValue)
      llvm::report_fatal_error(
          llvm::formatv("Wait timeout reached for signal handle {0:x}",
                        DispatchPacket->completion_signal.handle));
    if (MustDestroySignal)
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
          Core.callFunction<&::CoreApiTable::hsa_signal_destroy_fn>(
              DispatchPacket->completion_signal),
          "Failed to destroy the dispatch packet's completion signal."));
    NumKernelLaunched++;
  }
}

void UncoalescedMemAccessTool::printReport() const {
  auto &OS = luthier::errs();
  const comms::MessageProcessor *P = processor();
  if (!P) {
    OS << "[uncoalesced] no kernel was instrumented\n";
    return;
  }
  const double MiB = double(P->bytes()) / (1024.0 * 1024.0);
  OS << llvm::formatv("[uncoalesced] {0} messages, {1:F1} MiB in {2:F2} s, "
                      "{3} mid-kernel hand-off(s)\n",
                      P->messages(), MiB, P->seconds(), P->handOffs());
  if (P->malformed())
    OS << "[uncoalesced] WARNING: " << P->malformed()
       << " malformed message(s) skipped\n";
  if (droppedMessages())
    OS << "[uncoalesced] WARNING: messages dropped (larger than a "
          "sub-buffer)\n";
  Census.report(OS);
  Coalescing->report(OS, sites());
}

//===----------------------------------------------------------------------===//
// rocprofiler-sdk init / fini
//===----------------------------------------------------------------------===//

namespace luthier {

static void atToolFini(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    UncoalescedMemAccessTool::withInstance(
        [](UncoalescedMemAccessTool &T) { T.printReport(); });
    UncoalescedMemAccessTool::destroyInstance(); // frees the shared buffers
    delete LoaderSnapshot;
    LoaderSnapshot = nullptr;
    delete AmdExtSnapshot;
    AmdExtSnapshot = nullptr;
    delete CoreApiSnapshot;
    CoreApiSnapshot = nullptr;
    delete RocprofilerLib;
    RocprofilerLib = nullptr;
    luthier::finalizeStreams();
  });
}

static void atToolFini() { atToolFini(nullptr); }

static void atToolFatalSignal(void *) { luthier::finalizeStreams(); }

static void atToolInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    luthier::initializeStreams();
    luthier::errs() << "Uncoalesced memory access tool is launching.\n";
    std::atexit(atToolFini);

    llvm::Error Err = llvm::Error::success();
    auto RocprofilerLibOrErr =
        luthier::rocprofiler::openRocprofilerSdkLibrary();
    if (!RocprofilerLibOrErr)
      llvm::report_fatal_error(RocprofilerLibOrErr.takeError());
    RocprofilerLib =
        new luthier::DynamicLibrary(std::move(*RocprofilerLibOrErr));
    CoreApiSnapshot =
        new luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>(
            *RocprofilerLib, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
    AmdExtSnapshot =
        new luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable>(
            *RocprofilerLib, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
    LoaderSnapshot = new luthier::rocprofiler::HsaExtensionTableSnapshot<
        HSA_EXTENSION_AMD_LOADER>(*RocprofilerLib, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
    UncoalescedMemAccessTool::createInstance(*RocprofilerLib, *CoreApiSnapshot,
                                             *AmdExtSnapshot, *LoaderSnapshot,
                                             Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(
            0, &Argv, "Luthier Uncoalesced Memory Access Tool",
            &luthier::errs(), nullptr, "LUTHIER_ARGS"),
        "Failed to parse the command line arguments."));
    UncoalescedMemAccessTool::withInstance([](UncoalescedMemAccessTool &T) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          llvm::isPowerOf2_32(T.Opts.ElementSize) && T.Opts.LanesPerCycle >= 1 &&
              T.Opts.LanesPerCycle <= 64 && T.Opts.CacheLineBytes >= 1,
          "--comms-element-size must be a power of two, --lanes-per-cycle "
          "1..64 and --cache-line-bytes positive"));
      T.configure();
    });
    llvm::EnablePrettyStackTrace();
    luthier::printStackTraceOnFatalSignal();
    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/matinraayai/Luthier/issues/ and "
                          "include the crash error message and backtrace.\n");
    llvm::sys::AddSignalHandler(atToolFatalSignal, nullptr);
  });
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t,
                      rocprofiler_client_id_t *ClientID) {
  luthier::atToolInit(nullptr);
  ClientID->name = "Luthier Uncoalesced Memory Access tool";
  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
