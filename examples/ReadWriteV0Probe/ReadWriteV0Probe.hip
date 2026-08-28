//===-- ReadWriteV0Probe.hip - Read/Write v0 Probe Tool ---------*- C++ -*-===//
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
/// Two-mode Luthier probe used to exercise the payload side-effects analysis /
/// preserve-live-regs interaction on \c v0:
///
///   --v0-mode=read   → payload reads \c v0 into a vreg and atomically adds
///                       it to a single device-resident counter. Verifies that
///                       when the payload declares a phys-reg read, Luthier
///                       does not synthesize a second read of the same reg for
///                       preservation.
///   --v0-mode=write  → payload writes a constant into \c v0. Verifies that
///                       when the payload declares a phys-reg write, Luthier
///                       does not preserve the reg across the payload — the
///                       original value must be trampled.
///
/// The tool inserts the selected payload before every target MI so the
/// interaction is exercised at every instrumentation site.
//===----------------------------------------------------------------------===//
#include <SIInstrInfo.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/Agent.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSATooling/HSATool.h>
#include <luthier/Intrinsic/IntrinsicCalls.h>
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <memory>
#include <mutex>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-read-write-v0-probe"

//===----------------------------------------------------------------------===//
// ReadWriteV0ProbeTool — per-tool singleton subclass of HSATool
//===----------------------------------------------------------------------===//

struct ReadWriteV0ProbeTool
    : public luthier::HSATool<ReadWriteV0ProbeTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  //===---------------------------------------------------------------===//
  // Device-side state.
  //===---------------------------------------------------------------===//

  /// Device-resident accumulator. Payload atomically adds \c readReg(v0)
  /// into it on every instrumented site (read mode). Read back to host
  /// via \c hsa_memory_copy after each dispatch.
  __attribute__((device)) static uint64_t V0Accumulator;

  /// Read-mode hook: receives the value of \c v0 pre-read by the
  /// createInjectedPayload wrapper (via a \c RegArg PayloadArg → the
  /// wrapper emits a \c luthier::readReg call, which IS inside a
  /// function tagged \c InjectedPayloadAttribute). Atomically adds the
  /// value into the counter. Keeping the \c readReg call in the wrapper
  /// (not this hook) matters: IntrinsicMIRLoweringPass rejects phys-reg
  /// access from non-payload functions, and force-inlining does not
  /// delete the original hook symbol, so a hook that touches phys-regs
  /// itself would still leave a stray, non-payload copy behind.
  __attribute__((device, used)) static void readV0Payload(uint32_t V);

  /// Write-mode hook: same routing — the wrapper is what performs the
  /// \c writeReg (the RegArg PayloadArg is asymmetric today; see
  /// \c run(): for writes we still have to force the wrapper to build
  /// a \c writeReg call ourselves via the Builder overload of
  /// createInjectedPayload).
  __attribute__((device, used)) static void writeV0Payload();

  //===---------------------------------------------------------------===//
  // Host-side CLI + bookkeeping.
  //===---------------------------------------------------------------===//

  enum V0Mode { Read, Write };

  /// Target physreg selector — mirrors the register-family sweep the
  /// user asked for. Each enumerator maps to an AMDGPU MCRegister enum
  /// at run() time (see \c targetRegisterID).
  enum TargetReg {
    V0, V1, V2, V3, V4,
    S0, S1, S2, S3, S4, S5, S6, S7, S8,
    Scc, Vcc
  };

  struct Options {
    llvm::cl::OptionCategory Cat{"Read/Write v0 Probe Tool Options"};

    llvm::cl::opt<V0Mode> Mode{
        "v0-mode",
        llvm::cl::desc("Which payload to inject"),
        llvm::cl::values(
            clEnumValN(Read, "read",
                       "read the target reg and atomicAdd into counter"),
            clEnumValN(Write, "write",
                       "overwrite the target reg with a constant")),
        llvm::cl::init(Read), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<TargetReg> Target{
        "target-reg",
        llvm::cl::desc("Which physical register the payload should probe"),
        llvm::cl::values(
            clEnumValN(V0, "v0", "VGPR0"),
            clEnumValN(V1, "v1", "VGPR1"),
            clEnumValN(V2, "v2", "VGPR2"),
            clEnumValN(V3, "v3", "VGPR3"),
            clEnumValN(V4, "v4", "VGPR4"),
            clEnumValN(S0, "s0", "SGPR0"),
            clEnumValN(S1, "s1", "SGPR1"),
            clEnumValN(S2, "s2", "SGPR2"),
            clEnumValN(S3, "s3", "SGPR3"),
            clEnumValN(S4, "s4", "SGPR4"),
            clEnumValN(S5, "s5", "SGPR5"),
            clEnumValN(S6, "s6", "SGPR6"),
            clEnumValN(S7, "s7", "SGPR7"),
            clEnumValN(S8, "s8", "SGPR8"),
            clEnumValN(Scc, "scc", "SCC"),
            clEnumValN(Vcc, "vcc", "VCC")),
        llvm::cl::init(V0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> InstrBeginInterval{
        "instr-begin-interval",
        llvm::cl::desc("First MI index (inclusive) to instrument, per kernel"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};

    llvm::cl::opt<unsigned> InstrEndInterval{
        "instr-end-interval",
        llvm::cl::desc("End of the MI interval (exclusive)"),
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

    llvm::cl::opt<std::string> DumpInstrumentedBinaryTo{
        "dump-instrumented-binary-to",
        llvm::cl::desc(
            "If non-empty, write the raw ELF object of every freshly "
            "compiled instrumented dispatch to this path (overwritten each "
            "time). Use with objdump to inspect the emitted v0 reads/writes."),
        llvm::cl::init(""), llvm::cl::NotHidden, llvm::cl::cat(Cat)};
  } Opts;

  std::recursive_mutex Mutex;

  /// Number of kernels dispatched so far.
  uint32_t NumKernelLaunched = 0;

  /// Cumulative sum of the per-dispatch \c V0Accumulator snapshots
  /// (read mode only).
  uint64_t TotalAccumulator = 0;

  /// Cross-MF instruction index for the current pipeline sweep. Reset per
  /// dispatch.
  unsigned NextInstrIdx = 0;

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

__attribute__((device)) uint64_t ReadWriteV0ProbeTool::V0Accumulator;

__attribute__((device, used)) void
ReadWriteV0ProbeTool::readV0Payload(uint32_t V) {
  atomicAdd(&ReadWriteV0ProbeTool::V0Accumulator, static_cast<uint64_t>(V));
}

__attribute__((device, used)) void ReadWriteV0ProbeTool::writeV0Payload() {
  // The wrapper-emitted writeReg (see run()) is what actually clobbers
  // v0; this hook exists only to give the wrapper a target function to
  // inline into so the whole payload gets the InjectedPayloadAttribute.
}

//===----------------------------------------------------------------------===//
// File-scope statics: API-table snapshots + the tool itself.
//===----------------------------------------------------------------------===//

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(ReadWriteV0ProbeTool)

static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot{
    nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};

//===----------------------------------------------------------------------===//
// ReadWriteV0ProbeTool::run — payload creation
//===----------------------------------------------------------------------===//

llvm::PreservedAnalyses
ReadWriteV0ProbeTool::run(luthier::Prototype &P,
                          luthier::PrototypeAnalysisManager &PAM) {
  llvm::Module &IModule = P.getInstrumentationModule();
  auto &Ctx = IModule.getContext();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  // Enum → AMDGPU MCRegister ID. Kept in one place so both the RegArg
  // (read) and the hand-built writeReg (write) paths agree, and so the
  // list is trivially extendable to more registers.
  llvm::MCRegister TargetReg;
  switch (Opts.Target) {
  case V0: TargetReg = llvm::AMDGPU::VGPR0; break;
  case V1: TargetReg = llvm::AMDGPU::VGPR1; break;
  case V2: TargetReg = llvm::AMDGPU::VGPR2; break;
  case V3: TargetReg = llvm::AMDGPU::VGPR3; break;
  case V4: TargetReg = llvm::AMDGPU::VGPR4; break;
  case S0: TargetReg = llvm::AMDGPU::SGPR0; break;
  case S1: TargetReg = llvm::AMDGPU::SGPR1; break;
  case S2: TargetReg = llvm::AMDGPU::SGPR2; break;
  case S3: TargetReg = llvm::AMDGPU::SGPR3; break;
  case S4: TargetReg = llvm::AMDGPU::SGPR4; break;
  case S5: TargetReg = llvm::AMDGPU::SGPR5; break;
  case S6: TargetReg = llvm::AMDGPU::SGPR6; break;
  case S7: TargetReg = llvm::AMDGPU::SGPR7; break;
  case S8: TargetReg = llvm::AMDGPU::SGPR8; break;
  case Scc: TargetReg = llvm::AMDGPU::SCC; break;
  // \c VCC is the 64-bit combined register (\c VCC_LO + \c VCC_HI). Its
  // wave32 half is \c VCC_LO, which is a proper 32-bit SGPR alias. The
  // probe's hook takes \c uint32_t, so route "vcc" to \c VCC_LO to
  // sidestep the readReg / writeReg 64/32 size mismatch on wave32
  // subtargets.
  case Vcc: TargetReg = llvm::AMDGPU::VCC_LO; break;
  }
  NextInstrIdx = 0;
  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    // Skip HIP runtime helper kernels — instrumenting them just floods
    // dispatches unrelated to the workload under test. Same rationale as
    // PrintfOnLoad.
    if (MF.getName().starts_with("__amd_rocclr_")) {
      luthier::errs() << "[ReadWriteV0ProbeTool] skipping HIP runtime helper '"
                      << MF.getName() << "'\n";
      return;
    }
    luthier::errs() << "[ReadWriteV0ProbeTool] payload creation for MF '"
                    << MF.getName() << "' starting at instr idx "
                    << NextInstrIdx << "\n";
    unsigned StartIdx = NextInstrIdx;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        unsigned I = NextInstrIdx++;
        if (I < Opts.InstrBeginInterval || I >= Opts.InstrEndInterval)
          continue;
        if (Opts.Mode == Read) {
          // RegArg{TargetReg, i32}: the wrapper createInjectedPayload
          // builds for us emits a luthier::readReg call up front and
          // passes the loaded value as the hook's uint32_t arg. The
          // readReg call therefore lives inside the InjectedPayload-
          // attributed wrapper, not inside the hook function, which is
          // what IntrinsicMIRLoweringPass demands.
          luthier::PayloadArg Args[]{luthier::RegArg{TargetReg, I32}};
          LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
              &ReadWriteV0ProbeTool::readV0Payload, P, PAM, MI, Args));
        } else {
          // Write mode: build the wrapper body by hand — emit a
          // luthier::writeReg(TargetReg, 0xDEADBEEF) directly, then the
          // (empty) hook, so the writeReg lives in the wrapper for the
          // same reason as above.
          LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
              &ReadWriteV0ProbeTool::writeV0Payload, P, PAM, MI,
              [&](llvm::Function &Hook,
                  llvm::IRBuilderBase &Builder) -> llvm::Error {
                // writeReg call: (Reg, Val). Arg order matters — writeReg's
                // IR processor pulls the destination reg from arg 0 as a
                // ConstantInt, then the value to write from arg 1.
                (void)luthier::insertCallToIntrinsic(
                    IModule, Builder, "luthier::writeReg",
                    *llvm::Type::getVoidTy(Ctx),
                    static_cast<uint32_t>(TargetReg.id()), 0xDEADBEEFu);
                llvm::CallInst *HookCall = Builder.CreateCall(&Hook);
                llvm::InlineFunctionInfo IFI;
                llvm::InlineResult IR =
                    llvm::InlineFunction(*HookCall, IFI);
                if (!IR.isSuccess())
                  return LUTHIER_MAKE_GENERIC_ERROR(
                      std::string("Failed to force-inline writeV0Payload: ") +
                      IR.getFailureReason());
                return llvm::Error::success();
              }));
        }
      }
    }
    luthier::errs() << "[ReadWriteV0ProbeTool] payload creation for MF '"
                    << MF.getName() << "' visited "
                    << (NextInstrIdx - StartIdx) << " MI(s)\n";
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
// ReadWriteV0ProbeTool::onPackets
//===----------------------------------------------------------------------===//

void ReadWriteV0ProbeTool::onPackets(
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

    // overrideWithInstrumented rewrites Packet.kernel_object in place; the
    // counter read-back below needs the ORIGINAL descriptor to key the
    // lookup, so capture it first (same protocol as InstrCount).
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

    luthier::ExtendedKernargBuffer KernargBuf;
    if (ActiveRegion) {
      auto FirstTry =
          this->overrideWithInstrumented(*DispatchPacket, Queue, /*Preset=*/0);
      if (!FirstTry) {
        llvm::consumeError(FirstTry.takeError());
        std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->runInstrumentationPipelineForDispatch(*OriginalKD)
                .moveInto(ObjBuf));
        // Dump the raw ELF of the freshly-compiled variant when the user
        // asked. Written before loadInstrumented (which consumes the
        // buffer). If the file cannot be opened, just skip — this is a
        // debug aid, not a functional requirement.
        if (!Opts.DumpInstrumentedBinaryTo.empty()) {
          std::error_code EC;
          llvm::raw_fd_ostream OS(Opts.DumpInstrumentedBinaryTo, EC);
          if (!EC)
            OS.write(ObjBuf->getBufferStart(), ObjBuf->getBufferSize());
        }
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

    // In read mode, read V0Accumulator back for this dispatch and reset
    // it. Same access pattern as InstrCount::Counter.
    if (ActiveRegion && Opts.Mode == Read) {
      hsa_amd_pointer_info_t CounterAgentInfo{};
      CounterAgentInfo.size = sizeof(CounterAgentInfo);
      hsa_status_t PInfoStatus = AmdExt.callFunction<hsa_amd_pointer_info>(
          const_cast<void *>(reinterpret_cast<const void *>(OriginalKD)),
          &CounterAgentInfo,
          /*alloc=*/nullptr, /*num_agents_accessible=*/nullptr,
          /*accessible=*/nullptr);
      if (PInfoStatus == HSA_STATUS_SUCCESS &&
          CounterAgentInfo.type != HSA_EXT_POINTER_TYPE_UNKNOWN) {
        auto SymOrErr = this->lookupGlobalVariable(
            &ReadWriteV0ProbeTool::V0Accumulator, OriginalKD);
        if (!SymOrErr) {
          llvm::consumeError(SymOrErr.takeError());
        } else {
          auto AddrOrErr =
              luthier::hsa::executableSymbolGetAddress(Core, *SymOrErr);
          if (!AddrOrErr) {
            llvm::consumeError(AddrOrErr.takeError());
          } else {
            uint64_t Snapshot = 0;
            hsa_status_t RC = Core.callFunction<hsa_memory_copy>(
                &Snapshot, reinterpret_cast<void *>(*AddrOrErr),
                sizeof(Snapshot));
            if (RC == HSA_STATUS_SUCCESS) {
              TotalAccumulator += Snapshot;
              const uint64_t Zero = 0;
              (void)Core.callFunction<hsa_memory_copy>(
                  reinterpret_cast<void *>(*AddrOrErr), &Zero, sizeof(Zero));
            }
          }
        }
      }
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
    ReadWriteV0ProbeTool::withInstance([](ReadWriteV0ProbeTool &T) {
      if (T.Opts.Mode == ReadWriteV0ProbeTool::Read) {
        luthier::errs()
            << "Total v0 accumulator across all dispatches: "
            << T.TotalAccumulator << "\n";
      }
      luthier::errs() << "Read/Write v0 probe observed "
                      << T.NumKernelLaunched << " dispatched kernel(s).\n";
    });
    ReadWriteV0ProbeTool::destroyInstance();
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
    luthier::errs() << "Read/Write v0 probe tool is launching.\n";
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

    ReadWriteV0ProbeTool::createInstance(*CoreApiSnapshot, *AmdExtSnapshot,
                                         *LoaderSnapshot, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(
            0, &Argv, "Luthier Read/Write v0 Probe Tool", &luthier::errs(),
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
             << "Registering Luthier read/write v0 probe with rocprofiler-sdk.\n";
             llvm::dbgs() << "Rocprofiler-sdk version: " << RocProfVerMajor
                          << "." << RocProfVerMinor << "." << RocProfVerPatch
                          << " (" << RuntimeVersion << ")\n";
             llvm::dbgs() << "Tool priority: " << Priority << ";\n";);
  ClientID->name = "Luthier Read/Write v0 Probe tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
