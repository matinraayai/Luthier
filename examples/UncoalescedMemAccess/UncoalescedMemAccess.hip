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
/// Every global memory instruction (\c global_*) gets a hook that sends its access to
/// the CPU, while the kernel runs, through the generic GPU-to-CPU channel
/// \c luthier::comms::Channel. Each access is sent as (kernel-argument
/// buffer, element index): before every dispatch the host resolves the
/// kernel's pointer arguments to their allocations, and the hook maps each
/// address into one of them. \c CoalescingHandler then rebuilds the
/// addresses and decides which accesses were coalesced.
///
/// <b>Uncoalesced.</b> The hardware issues a VMEM instruction 16 lanes per
/// cycle. A group is coalesced if it touches no more 128-byte cache
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
///   --lanes-per-cycle=N             coalescing granularity (16)
///   --cache-line-bytes=N            cache line size (128)
///   --kernel-begin-interval / --kernel-end-interval
///
/// The report names each traced instruction by its function, basic block and
/// MIR text, so it can be found in the kernel's MIR.
///
/// <b>Targets.</b> CDNA (GFX9) and RDNA (GFX10-12, wave32 or wave64). Each
/// message carries the wave size and the shader engine / shader array /
/// CU (WGP on RDNA) of the wave. Run on gfx908; RDNA is compile-tested only.
//===----------------------------------------------------------------------===//
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
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <luthier/Comms/Channel.h>
#include <luthier/Comms/WorkgroupId.h>
#include <luthier/Intrinsic/IntrinsicCalls.h>
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

#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-uncoalesced-mem-access"

namespace comms = luthier::comms;

//===----------------------------------------------------------------------===//
// Memory-access tracing: what one access looks like on the wire
//===----------------------------------------------------------------------===//

/// \c WaveHeader::Tag of memory-access messages.
constexpr uint32_t MemoryAccessTag = 0x4d454d00; // "MEM\0"

/// Upper bound on kernel-argument buffers resolved per dispatch.
constexpr uint32_t MaxTrackedBuffers = 16;

/// \c AccessRecord::BufferId of an address in no tracked buffer.
constexpr uint32_t UnresolvedBuffer = 0xffffffffu;

/// One lane's access: where it landed, not the address.
struct AccessRecord {
  uint32_t ElementIndex; ///< (address - buffer start) / element size.
  uint32_t BufferId;     ///< Kernel-argument slot, or \c UnresolvedBuffer.
};

/// The dispatch's kernel-argument buffers, as the hooks see them.
struct BufferTable {
  uint32_t ElementSizeLog2; ///< Index = byte offset >> ElementSizeLog2.
  uint32_t NumBuffers;
  uint64_t BufferBase[MaxTrackedBuffers];
  uint64_t BufferEnd[MaxTrackedBuffers]; ///< One past the last byte.
  uint32_t BufferArgSlot[MaxTrackedBuffers];
};

/// A kernel-argument buffer resolved for one dispatch.
struct TrackedBuffer {
  uint64_t Base;    ///< The pointer the kernel was given.
  uint64_t End;     ///< End of its allocation.
  uint32_t ArgSlot; ///< 8-byte kernel-argument slot it was passed in.
};

/// A traced instruction, indexed by \c WaveHeader::Site.
struct TracedSite {
  std::string Function; ///< Machine function containing it.
  unsigned Block;       ///< Its basic block: \c bb.<Block> in the MIR.
  std::string MIR;      ///< The instruction as printed in the function's MIR.
  uint32_t Width;       ///< Bytes accessed per lane; also \c WaveHeader::Info.
};

/// Base for analyses of memory-access messages: decodes the lane data into
/// \c AccessRecord s and ignores messages with other tags.
class MemoryAccessHandler : public comms::MessageHandler {
public:
  /// Called before each dispatch with that dispatch's buffers.
  virtual void beginDispatch(llvm::ArrayRef<TrackedBuffer> Buffers,
                             uint32_t ElementSizeLog2) {}

  /// \p Records holds one entry per active lane; the i-th belongs to the
  /// i-th set bit of \c Header.Exec.
  virtual void handleAccesses(const comms::WaveHeader &Header,
                              llvm::ArrayRef<AccessRecord> Records) = 0;

  void handle(const comms::WaveHeader &H,
              llvm::ArrayRef<uint32_t> Data) final {
    if (H.Tag != MemoryAccessTag || H.DwordsPerLane != 2)
      return;
    Records.resize(H.ActiveLanes);
    for (uint32_t L = 0; L < H.ActiveLanes; ++L)
      Records[L] = {Data[L], Data[H.ActiveLanes + L]};
    handleAccesses(H, Records);
  }

private:
  std::vector<AccessRecord> Records;
};

//===----------------------------------------------------------------------===//
// Memory-access tracing: what is recorded and how the hook is called
//===----------------------------------------------------------------------===//

/// \c TracedSite of \p MI: its function, its MIR text and its width.
static TracedSite describeSite(const llvm::MachineInstr &MI) {
  const auto &ST = MI.getMF()->getSubtarget<llvm::GCNSubtarget>();
  const llvm::SIRegisterInfo &TRI = *ST.getRegisterInfo();
  TracedSite S{MI.getMF()->getName().str(),
               unsigned(MI.getParent()->getNumber()), "", 0};
  llvm::raw_string_ostream OS(S.MIR);
  MI.print(OS, /*IsStandalone=*/false, /*SkipOpers=*/false,
           /*SkipDebugLoc=*/true, /*AddNewLine=*/false, ST.getInstrInfo());
  // Drop Luthier's bookkeeping metadata (pcsections, ...) after the operands.
  if (size_t Meta = S.MIR.find(", pcsections"); Meta != std::string::npos)
    S.MIR.resize(Meta);
  for (auto Name : {llvm::AMDGPU::OpName::vdata, llvm::AMDGPU::OpName::vdst}) {
    const int Idx = llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name);
    if (Idx < 0)
      continue;
    const llvm::MCRegister R = MI.getOperand(Idx).getReg().asMCReg();
    S.Width = TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(R)) / 8;
    break;
  }
  return S;
}

/// Global memory instructions (\c global_*). Flat (\c flat_*) and scratch
/// instructions are not traced.
static bool isTraceable(const llvm::MachineInstr &MI) {
  return llvm::SIInstrInfo::isFLATGlobal(MI) && MI.mayLoadOrStore() &&
         llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                       llvm::AMDGPU::OpName::vaddr);
}

/// Builds, at the payload's insertion point, the call of \p Hook
/// (\c onAccess or \c onAccessScalarBase) for the access of \p MI, then
/// inlines it. Luthier lowers its intrinsics only inside the payload, so
/// every register read happens here.
static llvm::Error buildAccessHookCall(llvm::Function &Hook,
                                       llvm::IRBuilderBase &B,
                                       const llvm::MachineInstr &MI,
                                       uint32_t Site, uint32_t Width) {
  const auto &ST = MI.getMF()->getSubtarget<llvm::GCNSubtarget>();
  const llvm::SIRegisterInfo &TRI = *ST.getRegisterInfo();
  llvm::Module &M = *Hook.getParent();
  llvm::Type &I32 = *B.getInt32Ty();
  auto ReadReg = [&](llvm::MCRegister R) -> llvm::Value * {
    return luthier::insertCallToIntrinsic(M, B, "luthier::readReg", I32,
                                          uint32_t(R.id()));
  };
  auto Reg = [&](llvm::AMDGPU::OpName Name) {
    return MI.getOperand(llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name))
        .getReg()
        .asMCReg();
  };
  const int OffIdx = llvm::AMDGPU::getNamedOperandIdx(
      MI.getOpcode(), llvm::AMDGPU::OpName::offset);
  const int64_t Imm = OffIdx >= 0 ? MI.getOperand(OffIdx).getImm() : 0;
  const bool ScalarBase = llvm::AMDGPU::hasNamedOperand(
      MI.getOpcode(), llvm::AMDGPU::OpName::saddr);
  const llvm::MCRegister Base = Reg(ScalarBase ? llvm::AMDGPU::OpName::saddr
                                               : llvm::AMDGPU::OpName::vaddr);

  llvm::SmallVector<llvm::Value *, 10> Args{
      ReadReg(TRI.getSubReg(Base, llvm::AMDGPU::sub0)),
      ReadReg(TRI.getSubReg(Base, llvm::AMDGPU::sub1))};
  if (ScalarBase)
    Args.push_back(ReadReg(Reg(llvm::AMDGPU::OpName::vaddr)));
  Args.append({B.getInt32(uint32_t(Imm)), B.getInt32(Site), B.getInt32(Width)});
  comms::appendWorkgroupIdArgs(ST, M, B, Args);

  llvm::CallInst *Call = B.CreateCall(&Hook, Args);
  llvm::InlineFunctionInfo IFI;
  llvm::InlineResult IR = llvm::InlineFunction(*Call, IFI);
  return IR.isSuccess() ? llvm::Error::success()
                        : LUTHIER_MAKE_GENERIC_ERROR(
                              "failed to inline the access hook: " +
                              std::string(IR.getFailureReason()));
}

//===----------------------------------------------------------------------===//
// Analyses of the accesses
//===----------------------------------------------------------------------===//

/// Decides, for every executed memory instruction of every wave, whether the
/// access was coalesced. The hardware issues a VMEM instruction in groups of
/// \c LanesPerCycle lanes; a group is coalesced if it touches no more cache
/// lines than its bytes need (ceil(bytes / line size)), and a
/// wave-instruction is uncoalesced if any group wastes a line. Addresses are
/// rebuilt from (buffer, element index) and the dispatch's buffer bases.
class CoalescingHandler final : public MemoryAccessHandler {
public:
  CoalescingHandler(uint32_t LanesPerCycle, uint32_t CacheLineBytes)
      : LanesPerCycle(LanesPerCycle), CacheLineBytes(CacheLineBytes) {}

  void beginDispatch(llvm::ArrayRef<TrackedBuffer> Buffers,
                     uint32_t ElemSizeLog2) override {
    ElementSizeLog2 = ElemSizeLog2;
    BaseBySlot.fill(0);
    for (const auto &B : Buffers)
      if (B.ArgSlot < BaseBySlot.size())
        BaseBySlot[B.ArgSlot] = B.Base;
  }

  void handleAccesses(
      const comms::WaveHeader &H,
      llvm::ArrayRef<AccessRecord> Records) override {
    Stats &S = PerSite[H.Site];
    ++S.Waves;
    const uint64_t Width = H.Info ? H.Info : 4; // bytes per lane

    // Lines touched and bytes moved by each group of lanes issued together.
    llvm::SmallVector<uint64_t, 16> GroupLines[64];
    uint64_t GroupBytes[64] = {};
    uint64_t Exec = H.Exec; // i-th record <-> i-th set bit
    for (const auto &R : Records) {
      const unsigned Lane = unsigned(__builtin_ctzll(Exec));
      Exec &= Exec - 1;
      if (R.BufferId >= BaseBySlot.size()) {
        ++S.Unresolved;
        continue;
      }
      const uint64_t Addr =
          BaseBySlot[R.BufferId] + (uint64_t(R.ElementIndex) << ElementSizeLog2);
      const unsigned G = Lane / LanesPerCycle;
      GroupBytes[G] += Width;
      for (uint64_t L = Addr / CacheLineBytes;
           L <= (Addr + Width - 1) / CacheLineBytes; ++L)
        GroupLines[G].push_back(L);
    }

    bool Wasted = false;
    for (unsigned G = 0; G < 64; ++G) {
      if (GroupBytes[G] == 0)
        continue;
      auto &L = GroupLines[G];
      std::sort(L.begin(), L.end());
      const uint64_t Lines = std::unique(L.begin(), L.end()) - L.begin();
      const uint64_t Ideal =
          (GroupBytes[G] + CacheLineBytes - 1) / CacheLineBytes;
      S.Lines += Lines;
      S.IdealLines += Ideal;
      Wasted |= Lines > Ideal;
    }
    S.Uncoalesced += Wasted;
  }

  /// One line per traced site, then per-kernel and overall totals.
  void report(llvm::raw_ostream &OS,
              llvm::ArrayRef<TracedSite> Sites) const {
    OS << llvm::formatv("[uncoalesced] {0}-lane groups, {1}-byte lines\n",
                        LanesPerCycle, CacheLineBytes);
    OS << llvm::formatv("{0,4} {1,10} {2,12} {3,9} {4,9}  {5}\n", "site",
                        "waves", "uncoalesced", "lines/wv", "ideal/wv",
                        "function, block: MIR");
    uint64_t Waves = 0, Uncoalesced = 0, Lines = 0, Ideal = 0, Unresolved = 0;
    struct KernelTotals {
      uint64_t Waves = 0, Uncoalesced = 0, Lines = 0;
    };
    std::map<std::string, KernelTotals> PerKernel;
    for (const auto &[Site, S] : PerSite) {
      const TracedSite *T = Site < Sites.size() ? &Sites[Site] : nullptr;
      OS << llvm::formatv("{0,4} {1,10} {2,12} {3,9:F2} {4,9:F2}  ", Site,
                          S.Waves, S.Uncoalesced,
                          double(S.Lines) / double(S.Waves),
                          double(S.IdealLines) / double(S.Waves));
      if (T)
        OS << T->Function << ", bb." << T->Block << ": " << T->MIR << "\n";
      else
        OS << "?\n";
      Waves += S.Waves;
      Uncoalesced += S.Uncoalesced;
      Lines += S.Lines;
      Ideal += S.IdealLines;
      Unresolved += S.Unresolved;
      auto &K = PerKernel[T ? T->Function : "?"];
      K.Waves += S.Waves;
      K.Uncoalesced += S.Uncoalesced;
      K.Lines += S.Lines;
    }
    for (const auto &[Kernel, K] : PerKernel)
      OS << llvm::formatv("[uncoalesced] kernel {0}: VMEM {1} coalesced {2} "
                          "uncoalesced {3} lines {4}\n",
                          Kernel, K.Waves, K.Waves - K.Uncoalesced,
                          K.Uncoalesced, K.Lines);
    OS << llvm::formatv("[uncoalesced] VMEM wave-instructions: {0}  "
                        "coalesced: {1}  uncoalesced: {2}\n",
                        Waves, Waves - Uncoalesced, Uncoalesced);
    OS << llvm::formatv("[uncoalesced] cache lines: {0} touched, {1} needed\n",
                        Lines, Ideal);
    if (Unresolved)
      OS << llvm::formatv("[uncoalesced] WARNING: {0} lanes accessed no "
                          "tracked buffer and were skipped\n",
                          Unresolved);
  }

private:
  struct Stats {
    uint64_t Waves = 0, Uncoalesced = 0, Lines = 0, IdealLines = 0,
             Unresolved = 0;
  };
  uint32_t LanesPerCycle;
  uint32_t CacheLineBytes;
  uint32_t ElementSizeLog2 = 2;
  std::array<uint64_t, MaxTrackedBuffers> BaseBySlot{};
  std::map<uint32_t, Stats> PerSite;
};

/// Which workgroups and compute units sent messages, to check the workgroup
/// ids and hardware ids are real.
class WorkgroupCensus final : public MemoryAccessHandler {
public:
  void handleAccesses(const comms::WaveHeader &H,
                      llvm::ArrayRef<AccessRecord>) override {
    ++Messages[{H.BlockIdxX, H.BlockIdxY, H.BlockIdxZ}];
    if (H.CuId != comms::UnknownHwId)
      ComputeUnits.insert({H.SeId, H.SaId, H.CuId});
    WaveSizes.insert(H.WaveSize);
  }
  void report(llvm::raw_ostream &OS) const {
    if (Messages.empty())
      return;
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
    std::set<unsigned> Engines, Arrays;
    for (const auto &[Se, Sa, Cu] : ComputeUnits) {
      Engines.insert(Se);
      Arrays.insert(Se << 8 | Sa);
    }
    OS << llvm::formatv("[uncoalesced] hardware ids: {0} distinct CUs (WGPs "
                        "on RDNA) in {1} shader arrays on {2} shader "
                        "engines\n",
                        ComputeUnits.size(), Arrays.size(), Engines.size());
    for (unsigned W : WaveSizes)
      OS << "[uncoalesced] wave size: " << W << "\n";
  }

private:
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint64_t> Messages;
  std::set<std::tuple<uint8_t, uint8_t, uint8_t>> ComputeUnits; ///< SE, SA, CU
  std::set<unsigned> WaveSizes;
};

struct UncoalescedMemAccessTool
    : public luthier::Tool<UncoalescedMemAccessTool, llvm::MachineFunction>,
      public comms::Channel<UncoalescedMemAccessTool> {
  using Tool::Tool;

  //===--------------------------------------------------------------------===//
  // Device code
  //===--------------------------------------------------------------------===//

  /// Set by the host before each traced dispatch.
  __attribute__((device)) static BufferTable Table;

  /// Address in a 64-bit VGPR pair: <tt>global_load v, v[a:a+1], off</tt>.
  __attribute__((device, used)) static void
  onAccess(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
           uint32_t Site, uint32_t Width, uint32_t BlockX, uint32_t BlockY,
           uint32_t BlockZ) {
    sendAccess((uint64_t(AddrHi) << 32 | AddrLo) + int64_t(int32_t(ImmOffset)),
               {MemoryAccessTag, Site, Width, BlockX, BlockY, BlockZ,
                comms::device::readHwLocation()});
  }

  /// Scalar base plus 32-bit vector offset:
  /// <tt>global_load v, v_off, s[b:b+1]</tt>.
  __attribute__((device, used)) static void
  onAccessScalarBase(uint32_t BaseLo, uint32_t BaseHi, uint32_t VOffset,
                     uint32_t ImmOffset, uint32_t Site, uint32_t Width,
                     uint32_t BlockX, uint32_t BlockY, uint32_t BlockZ) {
    sendAccess((uint64_t(BaseHi) << 32 | BaseLo) + VOffset +
                   int64_t(int32_t(ImmOffset)),
               {MemoryAccessTag, Site, Width, BlockX, BlockY, BlockZ,
                comms::device::readHwLocation()});
  }

  /// Maps \p Address to (buffer, element index) and sends it.
  __attribute__((device, always_inline)) static void
  sendAccess(uint64_t Address, const comms::device::MessageInfo &M) {
    // Buffers never overlap, so at most one matches; no early exit keeps the
    // trip count uniform.
    uint32_t Record[2] = {0u, UnresolvedBuffer};
    for (uint32_t I = 0; I < Table.NumBuffers; ++I) {
      if (Address >= Table.BufferBase[I] && Address < Table.BufferEnd[I]) {
        Record[0] =
            uint32_t((Address - Table.BufferBase[I]) >> Table.ElementSizeLog2);
        Record[1] = Table.BufferArgSlot[I];
      }
    }
    send<2>(subBufferFor(M.BlockIdxX, M.BlockIdxY, M.BlockIdxZ), M, Record);
  }

  //===--------------------------------------------------------------------===//
  // Host code
  //===--------------------------------------------------------------------===//

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
  std::vector<TracedSite> Sites;
  std::vector<MemoryAccessHandler *> AccessHandlers;
  std::unique_ptr<CoalescingHandler> Coalescing;
  WorkgroupCensus Census;

  /// Applies the options; called once they are parsed.
  void configure() {
    setChannelConfig({Opts.NumSubBuffers, Opts.SubBufferBytes});
    Coalescing =
        std::make_unique<CoalescingHandler>(Opts.LanesPerCycle,
                                            Opts.CacheLineBytes);
    for (MemoryAccessHandler *H :
         {static_cast<MemoryAccessHandler *>(Coalescing.get()),
          static_cast<MemoryAccessHandler *>(&Census)}) {
      addHandler(*H);
      AccessHandlers.push_back(H);
    }
  }

  /// Injects a hook before \p MI that sends its access to the host.
  llvm::Error traceAccess(luthier::Prototype &P,
                          luthier::PrototypeAnalysisManager &PAM,
                          llvm::MachineInstr &MI) {
    const uint32_t Site = uint32_t(Sites.size());
    Sites.push_back(describeSite(MI));
    const uint32_t Width = Sites.back().Width;
    auto Build = [&](llvm::Function &Hook,
                     llvm::IRBuilderBase &B) -> llvm::Error {
      return buildAccessHookCall(Hook, B, MI, Site, Width);
    };
    llvm::function_ref<llvm::Error(llvm::Function &, llvm::IRBuilderBase &)>
        BuildRef(Build);
    return llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                         llvm::AMDGPU::OpName::saddr)
               ? createInjectedPayload(&UncoalescedMemAccessTool::onAccessScalarBase,
                                       P, PAM, MI, BuildRef)
               : createInjectedPayload(&UncoalescedMemAccessTool::onAccess, P,
                                       PAM, MI, BuildRef);
  }

  /// Opens the channel and tells the hooks and handlers where this
  /// dispatch's buffers are. \p AppPacket is the application's packet
  /// (before the override); \p KD its original kernel descriptor.
  llvm::Error beginTrace(const hsa_kernel_dispatch_packet_t &AppPacket,
                         const llvm::amdhsa::kernel_descriptor_t &KD) {
    LUTHIER_RETURN_ON_ERROR(openChannel(KD));
    const auto Tracked = kernelArgBuffers(AppPacket, KD);
    BufferTable T{};
    T.ElementSizeLog2 = llvm::Log2_32(Opts.ElementSize);
    T.NumBuffers = uint32_t(Tracked.size());
    for (uint32_t I = 0; I < T.NumBuffers; ++I) {
      T.BufferBase[I] = Tracked[I].Base;
      T.BufferEnd[I] = Tracked[I].End;
      T.BufferArgSlot[I] = Tracked[I].ArgSlot;
    }
    LUTHIER_RETURN_ON_ERROR(
        writeDeviceGlobal(&UncoalescedMemAccessTool::Table, &T, sizeof(T), KD));
    for (MemoryAccessHandler *H : AccessHandlers)
      H->beginDispatch(Tracked, T.ElementSizeLog2);
    return llvm::Error::success();
  }

  /// The kernel's pointer arguments that point into live allocations. Only
  /// reads the kernel arguments; non-pointer slots are skipped.
  llvm::SmallVector<TrackedBuffer, MaxTrackedBuffers>
  kernelArgBuffers(const hsa_kernel_dispatch_packet_t &Packet,
                   const llvm::amdhsa::kernel_descriptor_t &KD) {
    const auto Core = getCoreApiTableSnapshot().getTable();
    const auto AmdExt = getAmdExtTableSnapshot().getTable();
    llvm::SmallVector<TrackedBuffer, MaxTrackedBuffers> Out;
    const uint32_t Slots =
        std::min<uint32_t>(MaxTrackedBuffers, KD.kernarg_size / 8);
    uint64_t Args[MaxTrackedBuffers] = {};
    if (Slots == 0 ||
        Core.callFunction<hsa_memory_copy>(Args, Packet.kernarg_address,
                                           Slots * sizeof(uint64_t)) !=
            HSA_STATUS_SUCCESS)
      return Out;
    for (uint32_t Slot = 0; Slot < Slots; ++Slot) {
      if (Args[Slot] == 0)
        continue;
      hsa_amd_pointer_info_t Info{};
      Info.size = sizeof(Info);
      if (AmdExt.callFunction<hsa_amd_pointer_info>(
              reinterpret_cast<void *>(Args[Slot]), &Info, nullptr, nullptr,
              nullptr) != HSA_STATUS_SUCCESS ||
          Info.type == HSA_EXT_POINTER_TYPE_UNKNOWN)
        continue;
      const uint64_t AllocBase =
          reinterpret_cast<uint64_t>(Info.agentBaseAddress);
      const uint64_t AllocEnd = AllocBase + Info.sizeInBytes;
      if (Args[Slot] >= AllocBase && Args[Slot] < AllocEnd)
        Out.push_back({Args[Slot], AllocEnd, Slot});
    }
    return Out;
  }

  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM);

  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  void printReport() const;
};

__attribute__((device)) BufferTable UncoalescedMemAccessTool::Table;

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
      LUTHIER_REPORT_FATAL_ON_ERROR(beginTrace(AppPacket, *KD));
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
      closeChannel();
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
  Coalescing->report(OS, Sites);
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
