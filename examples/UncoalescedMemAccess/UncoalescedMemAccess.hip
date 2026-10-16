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
/// Every global load/store gets a hook that sends each lane's address, with a
/// dh_comms-style header (timestamp, source line, access kind and size), to
/// the CPU while the kernel runs, through \c luthier::comms::Channel. On the
/// CPU,
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
#include <luthier/ToolCodeGen/MemoryAllocationAccessor.h>
#include <luthier/ToolCodeGen/TargetMachineInstrMDNode.h>
#include <luthier/Tooling/Tool.h>
#include <llvm/DebugInfo/DWARF/DWARFContext.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <llvm/Support/xxhash.h>

#include <algorithm>
#include <map>
#include <optional>
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
//   uint64_t   Exec        which lanes were active (added by Comms)  8 bytes
//   WaveHeader             once per wave, see below                 32 bytes
//   uint32_t   Data[2*A]   each active lane's 64-bit address as 2
//                          dwords (send<2>): low halves, then high   8 * A
//
// A full 64-lane wave sends 8 + 32 + 64 * 8 = 552 bytes.
//
// WaveHeader carries the same information as dh_comms' wave_header_t
// (dh_comms/include/data_headers.h); what Comms already sends, or what
// follows from it, is not repeated:
//
//   dh_comms wave_header_t      here
//   exec                        Exec (sent by Comms)
//   data_size                   popcount(Exec) * 2 dwords (implied)
//   active_lane_count           popcount(Exec) (implied)
//   is_vector_message           always: every active lane sends (implied)
//   has_lane_headers            never: no per-lane header (implied)
//   timestamp                   WaveHeader::Timestamp
//   dwarf_fname_hash            WaveHeader::FileHash
//   dwarf_line, dwarf_column    WaveHeader::Line, WaveHeader::Column
//   user_type, user_data        WaveHeader::UserType, WaveHeader::UserData
//
// The DWARF fields and UserData are known when the instruction is
// instrumented, so the instrumentation pass bakes them into the payload as
// constants (see run()); only Timestamp is read on the GPU.

/// dh_comms' user_type for a memory address message (message_type::address).
constexpr uint32_t AddressMessage = 0;

/// The per-wave part of every message.
struct WaveHeader {
  uint64_t Timestamp; ///< GPU clock when the wave reached the access.
  uint64_t FileHash;  ///< Hash of the source file name; 0 without -g.
  uint32_t Line;      ///< Source line of the access; 0 without -g.
  uint32_t Column;    ///< Source column of the access; 0 without -g.
  uint32_t UserType;  ///< What the lane data is: AddressMessage.
  /// dh_comms' layout for addresses (v_submit_address): [1:0] 1 = read,
  /// 2 = write, 3 = both; [5:2] address space (llvm::AMDGPUAS); [21:6]
  /// bytes per lane, e.g. 4 for dword, 16 for dwordx4.
  uint32_t UserData;
};

/// Builds WaveHeader::UserData.
constexpr uint32_t packUserData(bool Reads, bool Writes, unsigned AddrSpace,
                                unsigned Bytes) {
  return uint32_t(Reads) | uint32_t(Writes) << 1 | (AddrSpace & 0xfu) << 2 |
         (Bytes & 0xffffu) << 6;
}

//===----------------------------------------------------------------------===//
// The CPU side: counts uncoalesced wave-instructions
//===----------------------------------------------------------------------===//

struct CoalescingHandler final : comms::MessageHandler<WaveHeader, 2> {
  uint64_t Waves = 0, Uncoalesced = 0, Lines = 0, Ideal = 0;

  void handle(uint64_t Exec, const WaveHeader &H,
              llvm::ArrayRef<uint32_t> Data) override {
    if (H.UserType != AddressMessage)
      return;
    const uint64_t Width = H.UserData >> 6 & 0xffffu;
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
    ++Waves;
    bool Wasted = false;
    for (unsigned G = 0; G < 64; ++G) {
      auto &L = GroupLines[G];
      std::sort(L.begin(), L.end());
      const uint64_t Touched = std::unique(L.begin(), L.end()) - L.begin();
      const uint64_t Needed =
          (GroupBytes[G] + CacheLineBytes - 1) / CacheLineBytes;
      Lines += Touched;
      Ideal += Needed;
      Wasted |= Touched > Needed;
    }
    Uncoalesced += Wasted;
  }

  void report(llvm::raw_ostream &OS) const {
    OS << llvm::formatv("[uncoalesced] {0}-lane groups, {1}-byte lines\n",
                        unsigned(LanesPerCycle), unsigned(CacheLineBytes));
    OS << llvm::formatv("[uncoalesced] VMEM wave-instructions: {0}  "
                        "coalesced: {1}  uncoalesced: {2}\n",
                        Waves, Waves - Uncoalesced, Uncoalesced);
    OS << llvm::formatv("[uncoalesced] cache lines: {0} touched, {1} needed\n",
                        Lines, Ideal);
  }
};

struct UncoalescedMemAccessTool
    : luthier::Tool<UncoalescedMemAccessTool, llvm::MachineFunction>,
      comms::Channel<UncoalescedMemAccessTool, WaveHeader, 2> {
  using Tool::Tool;
  CoalescingHandler Coalescing;
  std::recursive_mutex Mutex;

  /// The device side, run by every active lane just before the traced
  /// load/store: rebuild the lane's address the way the instruction does
  /// (64-bit VGPR pair + immediate offset) and send it as 2 dwords, with the
  /// header once for the wave. Everything but the address and the clock is
  /// a constant of the payload.
  __attribute__((device, used)) static void
  onAccess(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
           uint32_t FileHashLo, uint32_t FileHashHi, uint32_t Line,
           uint32_t Column, uint32_t UserData) {
    // Read the clock first so the submission itself barely perturbs it.
    const WaveHeader H{__builtin_readcyclecounter(),
                       uint64_t(FileHashHi) << 32 | FileHashLo,
                       Line,
                       Column,
                       AddressMessage,
                       UserData};
    const uint64_t Address =
        (uint64_t(AddrHi) << 32 | AddrLo) + int64_t(int32_t(ImmOffset));
    const uint32_t Lane[2] = {uint32_t(Address), uint32_t(Address >> 32)};
    send<2>(H, Lane);
  }

  //===--------------------------------------------------------------------===//
  // Where a traced instruction is in the application's source
  //===--------------------------------------------------------------------===//
  //
  // Luthier lifts the application's code into MachineInstrs, one per
  // original instruction, and tags each with a TargetMachineInstrMDNode (the
  // "pcsections !N" seen in MIR) holding the device address it was loaded
  // at. From that address:
  //
  //   1. MemoryAllocationAnalysis asks the HSA loader which loaded code
  //      object contains it, and returns the code object (an llvm::object
  //      ELF file) and the base address it was loaded at;
  //   2. address - load base = the instruction's address in the ELF, the one
  //      llvm-objdump -d prints (code objects are loaded from address 0);
  //   3. if the application was built with -g, llvm::DWARFContext turns that
  //      ELF address into file:line:column, the same way llvm-symbolizer
  //      does.

  /// One DWARF reader per application code object, built on first use.
  std::map<const llvm::object::ObjectFile *,
           std::unique_ptr<llvm::DWARFContext>>
      Dwarf;

  /// The source location of the application instruction \p MI was lifted
  /// from, or an empty one if it is unknown (no -g).
  llvm::DILineInfo sourceOf(const llvm::MachineInstr &MI,
                            const luthier::MemoryAllocationAccessor &Memory) {
    // The device address of the original instruction, from Luthier's tag.
    auto *Tag = luthier::TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
    const std::optional<uint64_t> PC =
        Tag ? Tag->getTraceInstrAddress() : std::nullopt;
    if (!PC)
      return {};
    // The loaded code object containing it, and the address it starts at.
    auto Alloc = Memory.getAllocationDescriptor(*PC);
    if (!Alloc) {
      llvm::consumeError(Alloc.takeError());
      return {};
    }
    const auto *Obj = Alloc->getAllocationCodeObject();
    if (!Obj)
      return {};
    const uint64_t ElfAddress =
        *PC - reinterpret_cast<uint64_t>(Alloc->getDeviceAllocation().data());
    // ELF address -> file:line:column, if the code object has DWARF (-g).
    auto &Ctx = Dwarf[Obj];
    if (!Ctx)
      Ctx = llvm::DWARFContext::create(*Obj);
    auto Info = Ctx->getLineInfoForAddress(
        {ElfAddress, llvm::object::SectionedAddress::UndefSection},
        llvm::DILineInfoSpecifier(
            llvm::DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath,
            llvm::DILineInfoSpecifier::FunctionNameKind::None));
    return Info ? *Info : llvm::DILineInfo{};
  }

  /// Instrumentation: a call to onAccess before every global load/store
  /// whose address is in a VGPR pair (<tt>global_load v, v[a:a+1], off</tt>).
  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM) {
    // Maps an application device address to its loaded code object.
    const luthier::MemoryAllocationAccessor &Memory =
        PAM.getResult<luthier::TargetModuleAnalysisManagerPrototypeProxy>(P)
            .getManager()
            .getResult<luthier::MemoryAllocationAnalysis>(P.getTargetModule())
            .getAccessor();

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
          const unsigned Width =
              TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(
                  Operand(Data).getReg().asMCReg())) / 8;

          // The header fields known now, from MI itself and the
          // application's DWARF; they become constants of the payload.
          const llvm::DILineInfo Src = sourceOf(MI, Memory);
          const uint64_t FileHash =
              Src.Line ? llvm::xxh3_64bits(Src.FileName) : 0;
          const uint32_t UserData =
              packUserData(MI.mayLoad(), MI.mayStore(),
                           llvm::AMDGPUAS::GLOBAL_ADDRESS, Width);

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
                // onAccess(AddrLo, AddrHi, ImmOffset, FileHashLo,
                // FileHashHi, Line, Column, UserData): the address register
                // pair, then constants.
                llvm::Value *Args[] = {
                    ReadReg(TRI.getSubReg(VAddr, llvm::AMDGPU::sub0)),
                    ReadReg(TRI.getSubReg(VAddr, llvm::AMDGPU::sub1)),
                    B.getInt32(uint32_t(Imm)),
                    B.getInt32(uint32_t(FileHash)),
                    B.getInt32(uint32_t(FileHash >> 32)),
                    B.getInt32(Src.Line),
                    B.getInt32(Src.Column),
                    B.getInt32(UserData)};
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
