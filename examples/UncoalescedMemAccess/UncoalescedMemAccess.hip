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
/// Counts uncoalesced global loads and stores.
///
/// Every global load/store gets a hook that sends each lane's address to the
/// CPU while the kernel runs, through \c luthier::comms::Channel. On the CPU,
/// the lanes of each wave-instruction are split into groups of
/// \c --lanes-per-cycle (the hardware issues 16 lanes per cycle); a group is
/// coalesced if it touches no more \c --cache-line-bytes lines than its bytes
/// need, and a wave-instruction is uncoalesced if any group wastes a line.
///
/// \code
///   LD_PRELOAD=<build>/lib/libLuthierTooling.so:\
///   <build>/examples/UncoalescedMemAccess/libLuthierUncoalescedMemAccess.so ./app
///   LUTHIER_ARGS="--lanes-per-cycle=16 --cache-line-bytes=64"   # optional
/// \endcode
//===----------------------------------------------------------------------===//
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <luthier/Comms/Channel.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/Intrinsic/IntrinsicCalls.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/Rocprofiler/RocprofilerSdkLibrary.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <luthier/Tooling/Tool.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace comms = luthier::comms;

static llvm::cl::opt<unsigned>
    LanesPerCycle("lanes-per-cycle", llvm::cl::desc("Lanes issued together"),
                  llvm::cl::init(16));
static llvm::cl::opt<unsigned>
    CacheLineBytes("cache-line-bytes", llvm::cl::desc("Cache line size"),
                   llvm::cl::init(128));

//===----------------------------------------------------------------------===//
// The message: what travels from the GPU to the CPU
//===----------------------------------------------------------------------===//
//
// One message per wave each time a traced load/store runs (Protocol.h):
//
//   uint64_t Exec        which lanes were active                   8 bytes
//   uint32_t InstrIdx    which traced instruction (+4 padding)     8 bytes
//   uint32_t Data[2*A]   each active lane's 64-bit address, as
//                        2 dwords (send<2>): low halves, then high  8 * A
//
// A full 64-lane wave sends 8 + 8 + 64 * 8 = 528 bytes.
//
// InstrIdx is a plain number, not a pointer. Example: the pass instruments a
// kernel's global_load_dword first and its global_store_dword second, so
// they get InstrIdx 0 and 1 and Instructions[0] / Instructions[1] on the CPU
// hold their MIR line and width (4 bytes). When a wave of 64 lanes runs the
// store, the GPU sends Exec = 0xffffffffffffffff, InstrIdx = 1 and 64
// addresses; the CPU looks up Instructions[1] to know each lane moved 4
// bytes. Whatever is fixed per instruction stays on the CPU this way.

/// What the CPU knows about each traced instruction; indexed by InstrIdx.
struct TracedInstruction {
  std::string Kernel;
  std::string MIR;
  uint32_t Width; ///< Bytes per lane, e.g. 4 for dword, 16 for dwordx4.
};

//===----------------------------------------------------------------------===//
// The CPU side: decides, per wave-instruction, whether it was coalesced
//===----------------------------------------------------------------------===//

struct CoalescingHandler final : comms::MessageHandler<uint32_t, 2> {
  std::vector<TracedInstruction> Instructions;
  struct Stats {
    uint64_t Waves = 0, Uncoalesced = 0, Lines = 0, Ideal = 0;
  };
  std::map<uint32_t, Stats> PerInstr; ///< Keyed by InstrIdx.

  void handle(uint64_t Exec, const uint32_t &InstrIdx,
              llvm::ArrayRef<uint32_t> Data) override {
    const uint64_t Width = Instructions[InstrIdx].Width;
    const uint32_t ActiveLanes = uint32_t(Data.size() / 2);
    // The hardware issues the lanes LanesPerCycle at a time. Collect, for
    // each such group, the cache lines its lanes touch and the bytes it moves.
    std::vector<uint64_t> GroupLines[64];
    uint64_t GroupBytes[64] = {};
    for (uint32_t I = 0; I < ActiveLanes; ++I) {
      // The i-th address belongs to the i-th set bit of Exec; its low half
      // is Data[I], its high half Data[ActiveLanes + I].
      const unsigned Lane = unsigned(__builtin_ctzll(Exec));
      Exec &= Exec - 1;
      const uint64_t Address =
          uint64_t(Data[ActiveLanes + I]) << 32 | Data[I];
      const unsigned G = Lane / LanesPerCycle;
      GroupBytes[G] += Width;
      for (uint64_t Line = Address / CacheLineBytes;
           Line <= (Address + Width - 1) / CacheLineBytes; ++Line)
        GroupLines[G].push_back(Line);
    }
    // A group is coalesced if it touches no more lines than its bytes need;
    // the wave-instruction is uncoalesced if any group wastes a line.
    Stats &S = PerInstr[InstrIdx];
    ++S.Waves;
    bool Wasted = false;
    for (unsigned G = 0; G < 64; ++G) {
      auto &L = GroupLines[G];
      std::sort(L.begin(), L.end());
      const uint64_t Lines = std::unique(L.begin(), L.end()) - L.begin();
      const uint64_t Ideal = (GroupBytes[G] + CacheLineBytes - 1) / CacheLineBytes;
      S.Lines += Lines;
      S.Ideal += Ideal;
      Wasted |= Lines > Ideal;
    }
    S.Uncoalesced += Wasted;
  }

  void report(llvm::raw_ostream &OS) const {
    OS << llvm::formatv("[uncoalesced] {0}-lane groups, {1}-byte lines\n",
                        unsigned(LanesPerCycle), unsigned(CacheLineBytes));
    std::map<std::string, Stats> PerKernel;
    for (const auto &[Idx, S] : PerInstr) {
      OS << llvm::formatv("{0,10} waves {1,10} uncoalesced  {2}: {3}\n",
                          S.Waves, S.Uncoalesced, Instructions[Idx].Kernel,
                          Instructions[Idx].MIR);
      Stats &K = PerKernel[Instructions[Idx].Kernel];
      K.Waves += S.Waves;
      K.Uncoalesced += S.Uncoalesced;
      K.Lines += S.Lines;
    }
    for (const auto &[Kernel, K] : PerKernel)
      OS << llvm::formatv("[uncoalesced] kernel {0}: VMEM {1} coalesced {2} "
                          "uncoalesced {3} lines {4}\n",
                          Kernel, K.Waves, K.Waves - K.Uncoalesced,
                          K.Uncoalesced, K.Lines);
  }
};

struct UncoalescedMemAccessTool
    : luthier::Tool<UncoalescedMemAccessTool, llvm::MachineFunction>,
      comms::Channel<UncoalescedMemAccessTool, uint32_t, 2> {
  using Tool::Tool;
  CoalescingHandler Coalescing;
  std::recursive_mutex Mutex;

  /// The device side, run by every active lane just before the traced
  /// load/store: rebuild the lane's address the way the instruction does
  /// (64-bit VGPR pair + immediate offset) and send it as 2 dwords, with the
  /// instruction's \p InstrIdx once for the wave.
  __attribute__((device, used)) static void
  onAccess(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
           uint32_t InstrIdx) {
    const uint64_t Address =
        (uint64_t(AddrHi) << 32 | AddrLo) + int64_t(int32_t(ImmOffset));
    const uint32_t Lane[2] = {uint32_t(Address), uint32_t(Address >> 32)};
    send<2>(InstrIdx, Lane);
  }

  /// Instrumentation: a call to onAccess before every global load/store
  /// whose address is in a VGPR pair (<tt>global_load v, v[a:a+1], off</tt>).
  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM) {
    P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
      const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
      const auto &TRI = *ST.getRegisterInfo();
      for (llvm::MachineBasicBlock &MBB : MF)
        for (llvm::MachineInstr &MI : MBB) {
          // Only vector global memory instructions: global_* loads and
          // stores with the whole address in a VGPR pair. The SGPR-base form
          // (global_load v, v_off, s[b:b+1]) is skipped.
          const unsigned Opc = MI.getOpcode();
          if (!llvm::SIInstrInfo::isFLATGlobal(MI) || !MI.mayLoadOrStore() ||
              llvm::AMDGPU::hasNamedOperand(Opc, llvm::AMDGPU::OpName::saddr))
            continue;
          auto Operand = [&](llvm::AMDGPU::OpName Name) -> llvm::MachineOperand & {
            return MI.getOperand(llvm::AMDGPU::getNamedOperandIdx(Opc, Name));
          };

          // The address operands: the VGPR pair, and the immediate offset
          // the instruction adds to it.
          const llvm::MCRegister VAddr =
              Operand(llvm::AMDGPU::OpName::vaddr).getReg().asMCReg();
          const int64_t Imm = Operand(llvm::AMDGPU::OpName::offset).getImm();

          // Bytes each lane moves: the size of the data register (vdata for
          // a store, vdst for a load), e.g. 4 for dword, 16 for dwordx4.
          const auto Data = MI.mayStore() ? llvm::AMDGPU::OpName::vdata
                                          : llvm::AMDGPU::OpName::vdst;
          const uint32_t Width =
              TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(
                  Operand(Data).getReg().asMCReg())) / 8;

          // Record the instruction on the CPU. Its position in Instructions
          // is the InstrIdx the payload will send back.
          const uint32_t InstrIdx = uint32_t(Coalescing.Instructions.size());
          std::string MIR;
          llvm::raw_string_ostream(MIR) << MI;
          Coalescing.Instructions.push_back({MF.getName().str(), MIR, Width});

          // Build the payload Luthier inserts before MI. Luthier's intrinsics
          // (readReg) are only lowered inside the payload, so the registers
          // are read here and passed to onAccess, which is then inlined.
          LUTHIER_REPORT_FATAL_ON_ERROR(createInjectedPayload(
              &UncoalescedMemAccessTool::onAccess, P, PAM, MI,
              [&](llvm::Function &Hook, llvm::IRBuilderBase &B) -> llvm::Error {
                llvm::Module &M = *Hook.getParent();
                // The value a register holds at this point in the app.
                auto ReadReg = [&](llvm::MCRegister R) {
                  return luthier::insertCallToIntrinsic(
                      M, B, "luthier::readReg", *B.getInt32Ty(),
                      uint32_t(R.id()));
                };
                // onAccess(AddrLo, AddrHi, ImmOffset, InstrIdx): the two
                // halves of the address register pair, then two constants.
                // InstrIdx is an integer index into Instructions, not a
                // pointer.
                llvm::Value *Args[] = {
                    ReadReg(TRI.getSubReg(VAddr, llvm::AMDGPU::sub0)),
                    ReadReg(TRI.getSubReg(VAddr, llvm::AMDGPU::sub1)),
                    B.getInt32(uint32_t(Imm)), B.getInt32(InstrIdx)};
                llvm::InlineFunctionInfo IFI;
                llvm::InlineFunction(*B.CreateCall(&Hook, Args), IFI);
                return llvm::Error::success();
              }));
        }
    });
    // Payloads were added: keep Luthier's target/instrumentation module
    // analyses, drop only the ones the new payloads change.
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

  /// Every dispatch: swap in the instrumented kernel, open the channel, run,
  /// and drain what is left.
  void onPackets(const hsa_queue_t &Queue, uint64_t,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter) {
    std::vector<luthier::hsa::AqlPacket> AppPackets(Packets.begin(),
                                                    Packets.end());
    for (auto &Packet : AppPackets) {
      auto *Dispatch = Packet.asKernelDispatch();
      if (!Dispatch) {
        PacketWriter(&Packet, 1);
        continue;
      }
      std::lock_guard Lock(Mutex);
      const auto Core = getCoreApiTableSnapshot().getTable();
      const auto AmdExt = getAmdExtTableSnapshot().getTable();
      const auto *KD = reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
          Dispatch->kernel_object);

      // We wait for the kernel, so it needs a completion signal.
      const bool OwnSignal = Dispatch->completion_signal.handle == 0;
      if (OwnSignal)
        AmdExt.callFunction<&::AmdExtTable::hsa_amd_signal_create_fn>(
            0, 0, nullptr, 0, &Dispatch->completion_signal);

      // Instrument the kernel the first time it is launched.
      auto Kernargs = overrideWithInstrumented(*Dispatch, Queue, 0);
      if (!Kernargs) {
        llvm::consumeError(Kernargs.takeError());
        std::unique_ptr<llvm::MemoryBuffer> Obj;
        LUTHIER_REPORT_FATAL_ON_ERROR(
            runInstrumentationPipelineForDispatch(*KD).moveInto(Obj));
        LUTHIER_REPORT_FATAL_ON_ERROR(
            loadInstrumented(std::move(Obj), KD, 0).takeError());
        Kernargs = overrideWithInstrumented(*Dispatch, Queue, 0);
        LUTHIER_REPORT_FATAL_ON_ERROR(Kernargs.takeError());
      }

      LUTHIER_REPORT_FATAL_ON_ERROR(openChannel(*KD));
      const auto Before =
          Core.callFunction<&::CoreApiTable::hsa_signal_load_scacquire_fn>(
              Dispatch->completion_signal);
      PacketWriter(Dispatch, 1);
      Core.callFunction<&::CoreApiTable::hsa_signal_wait_relaxed_fn>(
          Dispatch->completion_signal, HSA_SIGNAL_CONDITION_LT, Before,
          UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
      closeChannel();

      if (OwnSignal)
        Core.callFunction<&::CoreApiTable::hsa_signal_destroy_fn>(
            Dispatch->completion_signal);
    }
  }
};

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(UncoalescedMemAccessTool)

//===----------------------------------------------------------------------===//
// rocprofiler-sdk init / fini: the same in every Luthier tool
//===----------------------------------------------------------------------===//

static luthier::DynamicLibrary *RocprofilerLib;
static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable> *Core;
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExt;
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *Loader;

static void atToolFini(void *);

static void atToolInit() {
  luthier::initializeStreams();
  luthier::printStackTraceOnFatalSignal();
  // Report at exit, while the options above are still alive.
  std::atexit([] { atToolFini(nullptr); });
  const char *Argv = "";
  llvm::cl::ParseCommandLineOptions(0, &Argv, "", &luthier::errs(), nullptr,
                                    "LUTHIER_ARGS");
  llvm::Error Err = llvm::Error::success();
  RocprofilerLib = new luthier::DynamicLibrary(
      llvm::cantFail(luthier::rocprofiler::openRocprofilerSdkLibrary()));
  Core = new std::remove_pointer_t<decltype(Core)>(*RocprofilerLib, Err);
  AmdExt = new std::remove_pointer_t<decltype(AmdExt)>(*RocprofilerLib, Err);
  Loader = new std::remove_pointer_t<decltype(Loader)>(*RocprofilerLib, Err);
  UncoalescedMemAccessTool::createInstance(*RocprofilerLib, *Core, *AmdExt,
                                           *Loader, Err);
  LUTHIER_REPORT_FATAL_ON_ERROR(std::move(Err));
  UncoalescedMemAccessTool::withInstance(
      [](UncoalescedMemAccessTool &T) { T.addHandler(T.Coalescing); });
}

static void atToolFini(void *) {
  static std::once_flag Once;
  std::call_once(Once, [] {
    UncoalescedMemAccessTool::withInstance([](UncoalescedMemAccessTool &T) {
      T.Coalescing.report(luthier::errs());
    });
    UncoalescedMemAccessTool::destroyInstance();
    luthier::finalizeStreams();
  });
}

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t,
                      rocprofiler_client_id_t *ClientID) {
  atToolInit();
  ClientID->name = "Luthier Uncoalesced Memory Access tool";
  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr, &atToolFini,
      nullptr};
  return &Cfg;
}
