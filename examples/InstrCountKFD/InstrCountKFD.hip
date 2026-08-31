//===-- InstrCountKFD.hip - instruction counting below the runtime ---------===//
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
/// Counts the instructions an application executes, where the application never
/// uses HSA.
///
/// \par What this is, relative to ../InstrCount
/// The same tool with a different door. The device-side hooks, the payload
/// injection and the instrument-on-dispatch sequence are the HSA example's,
/// unchanged -- which is the point: instrumenting an application that talks to
/// the driver directly is not a different kind of instrumentation, only a
/// different way of finding out that a kernel is about to run.
///
/// Three things do differ, and each is forced rather than chosen.
///
/// \par 1. Dispatches arrive from the driver, not from HSA
/// \c KfdPacketMonitorTrait delivers packets copied out of the ring buffer the
/// preloaded wrapper substituted, so \c onDispatchPacket replaces \c onPackets.
/// The packet handed over is a staged copy that the wrapper writes to the GPU's
/// ring afterwards, so rewriting \c kernel_object in place is what makes the GPU
/// run the instrumented kernel.
///
/// \par 2. HSA has to be brought up, and only after the application has started
/// Loading an instrumented kernel is done by HSA, and HSA cannot simply be
/// initialized in such a process: three per-process driver resources collide with
/// an application that already claimed them. \c KFDTool::ensureHsaInitialized
/// deals with all three, and is called on the first dispatch rather than at load
/// time -- because initializing first does not avoid the collisions, it only
/// moves them onto the application, which is not ours to break.
///
/// \par 3. Counters are read at exit, not per dispatch
/// The HSA example creates a completion signal, substitutes it into the packet,
/// waits on it and reads the counter for that kernel. That cannot be done here:
/// \c completion_signal in this packet is the application's own KFD event
/// mailbox, not an HSA signal, and substituting one risks hanging the
/// application's own completion tracking. So this accumulates and reports totals
/// when the tool is destroyed, by which time the application has finished. The
/// cost is per-kernel attribution; the benefit is not interfering with how the
/// application discovers its own work is done.
//===----------------------------------------------------------------------===//
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/ExecutableSymbol.h>
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/KFD/KFDTool.h>
#include <luthier/LLVM/CodeGenHelpers.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/MIRConvenience.h>
#include <luthier/ToolCodeGen/Prototype.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>

#include <hip/hip_runtime.h>

#include <mutex>
#include <vector>

//===----------------------------------------------------------------------===//
// The tool
//===----------------------------------------------------------------------===//

struct InstrCountKfdTool
    : public luthier::KFDTool<InstrCountKfdTool, llvm::MachineFunction>,
      public luthier::ToolDeviceCodeOffloadParserTrait<InstrCountKfdTool> {

  /// The offload parser is opt-in on \c KFDTool -- an analysis-only tool has no
  /// device code and should not be forced to be a HIP translation unit -- so a
  /// tool that does have hooks inherits it explicitly and threads \p Err into
  /// both bases.
  InstrCountKfdTool(
      typename luthier::Singleton<InstrCountKfdTool>::CreationKey Key,
      const luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
      const luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
      const luthier::rocprofiler::HsaExtensionTableSnapshot<
          HSA_EXTENSION_AMD_LOADER> &VenLoader,
      llvm::Error &Err)
      : luthier::KFDTool<InstrCountKfdTool, llvm::MachineFunction>(
            Key, CoreApi, AmdExt, VenLoader, Err),
        luthier::ToolDeviceCodeOffloadParserTrait<InstrCountKfdTool>(Err) {}

  //===---------------------------------------------------------------===//
  // Device-side state, identical to the HSA example's.
  //===---------------------------------------------------------------===//

  /// Device-resident counter. A plain \c __device__ global rather than
  /// \c __managed__ for the reason the HSA example gives: the host-shadow
  /// indirection adds a load at every increment site and pushes register
  /// allocation toward a calling-convention mismatch on the atomic.
  __attribute__((device)) static uint64_t Counter;

  /// Vector hook: one atomic add per wavefront, or per active lane.
  __attribute__((device, used)) static void
  countInstructionsVector(bool CountWaveFrontLevel);

  /// Scalar hook: a single-lane atomic add.
  __attribute__((device, used)) static void countInstructionsScalar();

  //===---------------------------------------------------------------===//
  // Host-side state
  //===---------------------------------------------------------------===//

  struct Options {
    llvm::cl::OptionCategory Cat{"KFD Instruction Count Tool Options"};

    llvm::cl::opt<bool> CountWavefrontLevel{
        "count-wavefront-level",
        llvm::cl::desc("Count one instruction per wavefront rather than one per "
                       "active lane"),
        llvm::cl::init(false), llvm::cl::NotHidden, llvm::cl::cat(Cat)};
  };

  Options Opts;

  /// Every kernel descriptor an instrumented variant was loaded for. Kept
  /// because the counters live in those variants' executables and are read at
  /// exit; the launcher's cache is keyed the same way but does not expose an
  /// iterator, and keeping a list here avoids adding one for a single consumer.
  std::vector<const llvm::amdhsa::kernel_descriptor_t *> Instrumented;

  std::mutex Mutex;

  unsigned NextInstrIdx{0};

  /// Payload injection. Identical to the HSA example's: walk the lifted MIR and
  /// attach a counting hook to every instruction.
  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM);

  /// A dispatch, as copied out of the application's ring buffer.

  /// \brief Print the two kernel descriptors side by side.
  ///
  /// Diagnostic for the case where a substituted kernel never launches: the
  /// command processor reads the descriptor before running anything, and it can
  /// refuse a descriptor whose register or segment requests do not fit -- which
  /// looks from outside exactly like a stalled queue, with no error anywhere.
  ///
  /// The descriptors are read with \c hsa_memory_copy rather than dereferenced,
  /// because the instrumented one lives in device memory that the host may not
  /// have mapped.
  void dumpDescriptors(const llvm::amdhsa::kernel_descriptor_t *Original,
                       uint64_t InstrumentedKO,
                       const hsa_kernel_dispatch_packet_t &Packet) {
    const auto Core = this->getCoreApiTableSnapshot().getTable();
    llvm::amdhsa::kernel_descriptor_t Orig{}, Instr{};
    const bool OrigOk = Core.callFunction<hsa_memory_copy>(
                            &Orig, Original, sizeof(Orig)) == HSA_STATUS_SUCCESS;
    const bool InstrOk =
        Core.callFunction<hsa_memory_copy>(
            &Instr, reinterpret_cast<const void *>(InstrumentedKO),
            sizeof(Instr)) == HSA_STATUS_SUCCESS;

    auto Row = [&](const char *Name, uint64_t A, uint64_t B) {
      luthier::errs() << llvm::formatv("[KD] {0,-32} {1,18:x} {2,18:x}{3}\n", Name,
                                       A, B, A == B ? "" : "   <-- differs");
    };
    luthier::errs() << llvm::formatv(
        "[KD] {0,-32} {1,18} {2,18}\n", "field",
        OrigOk ? "application" : "application(unread)",
        InstrOk ? "instrumented" : "instrumented(unread)");
    Row("kernel_object", reinterpret_cast<uint64_t>(Original), InstrumentedKO);
    Row("group_segment_fixed_size", Orig.group_segment_fixed_size,
        Instr.group_segment_fixed_size);
    Row("private_segment_fixed_size", Orig.private_segment_fixed_size,
        Instr.private_segment_fixed_size);
    Row("kernarg_size", Orig.kernarg_size, Instr.kernarg_size);
    Row("kernel_code_entry_byte_offset",
        static_cast<uint64_t>(Orig.kernel_code_entry_byte_offset),
        static_cast<uint64_t>(Instr.kernel_code_entry_byte_offset));
    Row("compute_pgm_rsrc1", Orig.compute_pgm_rsrc1, Instr.compute_pgm_rsrc1);
    Row("compute_pgm_rsrc2", Orig.compute_pgm_rsrc2, Instr.compute_pgm_rsrc2);
    Row("compute_pgm_rsrc3", Orig.compute_pgm_rsrc3, Instr.compute_pgm_rsrc3);
    Row("kernel_code_properties", Orig.kernel_code_properties,
        Instr.kernel_code_properties);
    luthier::errs() << llvm::formatv(
        "[KD] packet: group_segment_size={0} private_segment_size={1} "
        "kernarg_address={2:x} grid={3}x{4}x{5} wg={6}x{7}x{8}\n",
        Packet.group_segment_size, Packet.private_segment_size,
        reinterpret_cast<uint64_t>(Packet.kernarg_address), Packet.grid_size_x,
        Packet.grid_size_y, Packet.grid_size_z, Packet.workgroup_size_x,
        Packet.workgroup_size_y, Packet.workgroup_size_z);
  }

  void onDispatchPacket(const luthier::kfd::QueueInfo &Q, uint64_t,
                        luthier::hsa::AqlPacket &Packet);

  /// Read every counter and report. Called at teardown, when the application
  /// has finished and nothing is still writing them.
  void reportTotals();
};

//===----------------------------------------------------------------------===//
// Device code
//===----------------------------------------------------------------------===//

__attribute__((device)) uint64_t InstrCountKfdTool::Counter;

__attribute__((device, used)) void
InstrCountKfdTool::countInstructionsVector(bool CountWaveFrontLevel) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  uint32_t NumActiveThreads = __popcll(ExecMask);

  if (FirstActiveThreadId == LaneId) {
    if (CountWaveFrontLevel) {
      if (NumActiveThreads > 0)
        atomicAdd(&InstrCountKfdTool::Counter, 1);
    } else {
      atomicAdd(&InstrCountKfdTool::Counter, NumActiveThreads);
    }
  }
}

__attribute__((device, used)) void InstrCountKfdTool::countInstructionsScalar() {
  atomicAdd(&InstrCountKfdTool::Counter, 1ULL);
}

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(InstrCountKfdTool)

//===----------------------------------------------------------------------===//
// Payload injection
//===----------------------------------------------------------------------===//

llvm::PreservedAnalyses
InstrCountKfdTool::run(luthier::Prototype &P,
                       luthier::PrototypeAnalysisManager &PAM) {
  llvm::Module &IModule = P.getInstrumentationModule();
  NextInstrIdx = 0;
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    auto *CountWavefrontLevelConstVal =
        llvm::ConstantInt::getBool(IModule.getContext(), Opts.CountWavefrontLevel);
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        NextInstrIdx++;
        const void *HostHandle = nullptr;
        llvm::SmallVector<luthier::PayloadArg, 1> Args;
        if (luthier::isScalar(MI) || luthier::isLaneAccess(MI)) {
          HostHandle = reinterpret_cast<const void *>(
              &InstrCountKfdTool::countInstructionsScalar);
        } else {
          HostHandle = reinterpret_cast<const void *>(
              &InstrCountKfdTool::countInstructionsVector);
          Args.push_back(CountWavefrontLevelConstVal);
        }
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->createInjectedPayload(HostHandle, P, PAM, MI, Args));
      }
    }
  });

  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  // Payload creation only adds functions to the instrumentation module and
  // attaches metadata to target MIs; no MachineFunction is created or destroyed,
  // so the lifted target MIR this pass just walked stays valid.
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

void InstrCountKfdTool::onDispatchPacket(const luthier::kfd::QueueInfo &Q,
                                         uint64_t,
                                         luthier::hsa::AqlPacket &Packet) {
  auto *Dispatch = Packet.asKernelDispatch();
  if (Dispatch == nullptr)
    return; // a barrier or something else; copied through untouched

  std::lock_guard<std::mutex> Lock(Mutex);

  // Everything below is the tool's own work, and all of it reaches the driver
  // through the same ioctl the wrapper intercepts. Without this the runtime's
  // queues and allocations would be taken for the application's -- and a queue
  // of ours that got wrapped would feed our own dispatches back into this
  // callback.
  //
  // Process-wide rather than per-thread, and covering the whole body rather than
  // just initialization: measured, the runtime creates a queue during
  // loadInstrumented -- a blit queue, to copy the code object to the device --
  // long after hsa_init has returned, and on a thread of its own. A per-thread
  // region around initialization caught neither.
  //
  // The cost is that an application queue created concurrently with this callback
  // is excluded too. Acceptable while the window is one dispatch's worth of work;
  // it would not be if this ran for the life of the process.
  luthier::kfd::ProcessWideToolRegion ToolWork;

  // First dispatch brings HSA up. Deliberately here and not at load time: the
  // application has to claim the driver's per-process resources first.
  if (llvm::Error Err = ensureHsaInitialized()) {
    luthier::errs() << "[InstrCountKFD] HSA could not be initialized, so no "
                       "kernel can be loaded: "
                    << llvm::toString(std::move(Err)) << "\n";
    return;
  }

  // Capture the key before overriding: overrideWithInstrumented rewrites
  // kernel_object in place, after which the packet names the instrumented
  // variant rather than the original the cache is keyed by.
  const auto *OriginalKD =
      reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
          Dispatch->kernel_object);

  const uint32_t PrivateBefore = Dispatch->private_segment_size;

  // Probe the cache; on a miss, lift, instrument, load, and retry. The probe's
  // Error must be bound to a local so its destructor does not trip the
  // unhandled-error abort.
  if (auto MissErr = this->overrideWithInstrumented(*Dispatch, /*Preset=*/0)) {
    llvm::consumeError(std::move(MissErr));

    // The device to load onto comes from the queue, not from the descriptor:
    // an application that allocates through the driver leaves
    // hsa_amd_pointer_info reporting the descriptor as owned by nothing.
    auto AgentOrErr = agentForCurrentDispatch();
    if (!AgentOrErr) {
      luthier::errs() << "[InstrCountKFD] " << llvm::toString(AgentOrErr.takeError())
                      << "\n";
      return;
    }

    std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
    if (llvm::Error Err =
            this->runInstrumentationPipelineForDispatch(*OriginalKD)
                .moveInto(ObjBuf)) {
      luthier::errs() << llvm::formatv(
          "[InstrCountKFD] gpu {0} kernel_object {1:x}: instrumentation failed: "
          "{2}\n",
          Q.GpuId, Dispatch->kernel_object, llvm::toString(std::move(Err)));
      // Left uninstrumented rather than fatal: the application's own kernel is
      // still in the packet, so it runs correctly and only the measurement is
      // missing. A tool preloaded into someone else's program should not take it
      // down over that.
      return;
    }
    // Write the linked code object out for offline inspection. The generated
    // code is otherwise only ever in memory, and questions like "does this
    // kernel reach scratch through a buffer descriptor or through flat
    // addressing?" are answered by disassembling it, not by reading fields.
    if (const char *Path = getenv("LUTHIER_INSTRCOUNT_KFD_DUMP_OBJ")) {
      std::error_code EC;
      llvm::raw_fd_ostream Out(Path, EC);
      if (EC)
        luthier::errs() << llvm::formatv(
            "[InstrCountKFD] could not write {0}: {1}\n", Path, EC.message());
      else
        Out << ObjBuf->getBuffer();
    }

    if (llvm::Error Err =
            this->loadInstrumented(std::move(ObjBuf), OriginalKD, 0, *AgentOrErr)
                .takeError()) {
      luthier::errs() << llvm::formatv(
          "[InstrCountKFD] gpu {0} kernel_object {1:x}: loading failed: {2}\n",
          Q.GpuId, Dispatch->kernel_object, llvm::toString(std::move(Err)));
      return;
    }
    Instrumented.push_back(OriginalKD);

    // A control, not a feature: do everything up to and including loading the
    // instrumented kernel, then leave the application's own kernel in the
    // packet. Separates "loading onto a live KFD queue's device disturbs the
    // application" from "the substituted kernel does not run", which otherwise
    // look identical from outside.
    if (getenv("LUTHIER_INSTRCOUNT_KFD_NO_OVERRIDE") != nullptr) {
      luthier::errs() << "[InstrCountKFD] loaded but not substituting "
                         "(LUTHIER_INSTRCOUNT_KFD_NO_OVERRIDE)\n";
      return;
    }

    if (llvm::Error Err = this->overrideWithInstrumented(*Dispatch, 0)) {
      luthier::errs() << llvm::formatv(
          "[InstrCountKFD] gpu {0} kernel_object {1:x}: could not substitute "
          "the instrumented kernel: {2}\n",
          Q.GpuId, Dispatch->kernel_object, llvm::toString(std::move(Err)));
      return;
    }
    luthier::errs() << llvm::formatv(
        "[InstrCountKFD] gpu {0}: instrumented kernel_object {1:x}; private "
        "segment {2} -> {3} bytes\n",
        Q.GpuId, reinterpret_cast<uint64_t>(OriginalKD), PrivateBefore,
        Dispatch->private_segment_size);

    // Withdraw a private-segment request the instrumented kernel does not use.
    //
    // This is required for correctness, not a tuning knob. Enabling a private
    // segment buffer shifts every other user SGPR up by four, so a kernel whose
    // code still reads its arguments from s[0:1] gets the scratch descriptor
    // instead of its argument pointer. Measured on gfx908: without this the
    // dispatch runs but computes the wrong answer, and the counter reports 1
    // instruction instead of 8.
    //
    // Only when the application's kernel needed no private segment at all. That
    // is the case where the request is known to be spurious and where the
    // application, needing no scratch, never set any up. A kernel that genuinely
    // spills must keep its request -- clearing it there would take away memory
    // the code really uses.
    if (PrivateBefore == 0 && Dispatch->private_segment_size > 0) {
      const auto Core = this->getCoreApiTableSnapshot().getTable();
      llvm::amdhsa::kernel_descriptor_t KD{};
      auto *Dst = reinterpret_cast<void *>(Dispatch->kernel_object);
      if (Core.callFunction<hsa_memory_copy>(&KD, Dst, sizeof(KD)) ==
          HSA_STATUS_SUCCESS) {
        KD.kernel_code_properties &= ~static_cast<uint16_t>(1u); // buffer
        KD.compute_pgm_rsrc2 &= ~1u;                            // enable bit
        if (Core.callFunction<hsa_memory_copy>(Dst, &KD, sizeof(KD)) !=
            HSA_STATUS_SUCCESS)
          luthier::errs()
              << "[InstrCountKFD] could not withdraw the private-segment "
                 "request; the instrumented kernel will read its arguments "
                 "from the wrong registers\n";
      }
    }

    if (getenv("LUTHIER_INSTRCOUNT_KFD_DUMP_KD") != nullptr) {
      uint32_t Node = 0;
      this->getCoreApiTableSnapshot().getTable().callFunction<hsa_agent_get_info>(
          *AgentOrErr, static_cast<hsa_agent_info_t>(
                           HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
          &Node);
      luthier::errs() << llvm::formatv(
          "[KD] loaded onto agent handle {0:x}, driver node {1} (application is "
          "on gpu_id {2})\n",
          AgentOrErr->handle, Node, Q.GpuId);
      dumpDescriptors(OriginalKD, Dispatch->kernel_object, *Dispatch);
    }
  }
}

void InstrCountKfdTool::reportTotals() {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (Instrumented.empty()) {
    luthier::errs() << "[InstrCountKFD] TOTAL kernels=0 instructions=0\n";
    return;
  }

  const auto Core = this->getCoreApiTableSnapshot().getTable();
  uint64_t Total = 0;
  unsigned Read = 0;
  for (const auto *KD : Instrumented) {
    auto SymOrErr = this->lookupGlobalVariable(&InstrCountKfdTool::Counter, KD);
    if (!SymOrErr) {
      llvm::consumeError(SymOrErr.takeError());
      continue;
    }
    auto AddrOrErr = luthier::hsa::executableSymbolGetAddress(Core, *SymOrErr);
    if (!AddrOrErr) {
      llvm::consumeError(AddrOrErr.takeError());
      continue;
    }
    uint64_t Snapshot = 0;
    if (Core.callFunction<hsa_memory_copy>(
            &Snapshot, reinterpret_cast<void *>(*AddrOrErr),
            sizeof(Snapshot)) != HSA_STATUS_SUCCESS)
      continue;
    Total += Snapshot;
    Read++;
  }
  luthier::errs() << llvm::formatv(
      "[InstrCountKFD] TOTAL kernels={0} counters_read={1} instructions={2}\n",
      Instrumented.size(), Read, Total);
}

//===----------------------------------------------------------------------===//
// Tool lifetime
//===----------------------------------------------------------------------===//

namespace {

luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable> *CoreSnap{nullptr};
luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnap{nullptr};
luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnap{nullptr};

void atToolFini() {
  (void)InstrCountKfdTool::withInstance(
      [](InstrCountKfdTool &T) { T.reportTotals(); });
}

/// Stood up from a library constructor rather than from
/// \c rocprofiler_configure. Configure fires while HSA is initializing, and here
/// HSA initializes on the first dispatch -- a dispatch this tool only sees
/// because constructing it registered a packet callback. Waiting for configure
/// would mean never registering and never seeing anything.
void attach() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    luthier::initializeStreams();
    std::atexit(atToolFini);

    // Make LLVM's -debug / -debug-only reachable here, the way ../InstrCount
    // does it: the tool has no argv of its own, so the options are read from
    // LUTHIER_ARGS. Registered before the tool is built, so pass-level output
    // covers instrumentation from the first dispatch onwards.
    //
    // This was the one instrumenting path with no way to ask a pass what it
    // decided, which is why the user-SGPR force-enable had to be inferred from
    // disassembling the generated object instead of read off a log.
    const char *Argv = "";
    luthier::registerDebugCLOptions();
    if (!llvm::cl::ParseCommandLineOptions(0, &Argv,
                                           "Luthier KFD Instruction Counter",
                                           &luthier::errs(), nullptr,
                                           "LUTHIER_ARGS"))
      luthier::errs()
          << "[InstrCountKFD] could not parse LUTHIER_ARGS; continuing\n";
    llvm::EnablePrettyStackTrace();
    luthier::printStackTraceOnFatalSignal();

    llvm::Error Err = llvm::Error::success();
    CoreSnap =
        new luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>(Err);
    AmdExtSnap =
        new luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable>(Err);
    LoaderSnap = new luthier::rocprofiler::HsaExtensionTableSnapshot<
        HSA_EXTENSION_AMD_LOADER>(Err);
    if (Err) {
      luthier::errs() << "[InstrCountKFD] could not request the HSA API tables: "
                      << llvm::toString(std::move(Err)) << "\n";
      return;
    }

    InstrCountKfdTool::createInstance(*CoreSnap, *AmdExtSnap, *LoaderSnap, Err);
    if (Err) {
      luthier::errs() << "[InstrCountKFD] could not attach: "
                      << llvm::toString(std::move(Err)) << "\n";
      return;
    }
    luthier::errs() << "[InstrCountKFD] attached\n";
  });
}

__attribute__((constructor)) void attachFromConstructor() { attach(); }

} // namespace

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t,
                      rocprofiler_client_id_t *ClientID) {
  ClientID->name = "Luthier KFD Instruction Counter tool";
  // Does no work: the tool is already up, because by the time this fires HSA is
  // mid-initialization. Exported so rocprofiler recognises a client here.
  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr, nullptr, nullptr};
  return &Cfg;
}
