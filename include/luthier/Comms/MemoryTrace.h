//===-- MemoryTrace.h - Stream memory accesses to the host ------*- C++ -*-===//
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
/// Reusable memory-access tracing for Luthier tools. A tool derives from
/// \c MemoryTrace<Tool> to get the device state, the hooks, and the host
/// plumbing; it only decides which instructions to trace and what to do with
/// the accesses (a \c MemoryAccessHandler).
///
/// Each traced access is sent as (kernel-argument buffer, element index)
/// rather than as an address: before every dispatch the host resolves the
/// kernel's pointer arguments to their allocations, and the device maps each
/// address into one of them.
///
/// Usage, in the tool:
/// \code
///   struct MyTool : luthier::Tool<MyTool, llvm::MachineFunction>,
///                   luthier::comms::MemoryTrace<MyTool> { ... };
///   // instrumentation pass:   traceAccess(P, PAM, MI) on each MI
///   // onPackets, per dispatch: beginDispatch(...) / endDispatch()
/// \endcode
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_MEMORY_TRACE_H
#define LUTHIER_COMMS_MEMORY_TRACE_H

#include "luthier/Comms/Device.h"
#include "luthier/Comms/MessageProcessor.h"
#include "luthier/Comms/SharedBuffers.h"

#include <hip/hip_runtime.h>
#include <llvm/Support/AMDGPUAddrSpace.h>

#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <luthier/HSA/hsa.h>
#include <luthier/Intrinsic/IntrinsicCalls.h>
#include <luthier/Intrinsic/ReadHwReg.h>
#include <luthier/Intrinsic/ScalarValueArgument.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/ToolCodeGen/Prototype.h>

#include <string>
#include <vector>

namespace luthier::comms {

//===----------------------------------------------------------------------===//
// What one access looks like on the wire
//===----------------------------------------------------------------------===//

/// \c WaveHeader::Tag of memory-access messages.
inline constexpr uint32_t MemoryAccessTag = 0x4d454d00; // "MEM\0"

/// Upper bound on kernel-argument buffers resolved per dispatch.
inline constexpr uint32_t MaxTrackedBuffers = 16;

/// \c AccessRecord::BufferId of an address in no tracked buffer.
inline constexpr uint32_t UnresolvedBuffer = 0xffffffffu;

/// One lane's access: where it landed, not the address.
struct AccessRecord {
  uint32_t ElementIndex; ///< (address - buffer start) / element size.
  uint32_t BufferId;     ///< Kernel-argument slot, or \c UnresolvedBuffer.
};

/// Packs an access into \c WaveHeader::Info:
/// [1:0] 1 = read, 2 = write, 3 = both; [5:2] \c llvm::AMDGPUAS address
/// space; [21:6] width in bytes.
constexpr uint32_t packAccessInfo(bool Reads, bool Writes, unsigned AddrSpace,
                                  unsigned WidthBytes) {
  return (uint32_t(Reads) | uint32_t(Writes) << 1) |
         ((AddrSpace & 0xfu) << 2) | (uint32_t(WidthBytes) << 6);
}
constexpr bool accessReads(uint32_t Info) { return Info & 1u; }
constexpr bool accessWrites(uint32_t Info) { return Info & 2u; }
constexpr unsigned accessAddrSpace(uint32_t Info) { return (Info >> 2) & 0xfu; }
constexpr unsigned accessWidth(uint32_t Info) { return Info >> 6; }

/// Where the device finds everything for one dispatch.
struct TraceState {
  ChannelDescriptor Channel;
  uint32_t ElementSizeLog2; ///< Index = byte offset >> ElementSizeLog2.
  uint32_t NumBuffers;
  uint64_t BufferBase[MaxTrackedBuffers];
  uint64_t BufferEnd[MaxTrackedBuffers]; ///< One past the last byte.
  uint32_t BufferArgSlot[MaxTrackedBuffers];
};

//===----------------------------------------------------------------------===//
// Host side: consuming accesses
//===----------------------------------------------------------------------===//

/// A kernel-argument buffer resolved for one dispatch.
struct TrackedBuffer {
  uint64_t Base;    ///< The pointer the kernel was given.
  uint64_t End;     ///< End of its allocation.
  uint32_t ArgSlot; ///< 8-byte kernel-argument slot it was passed in.
};

/// A traced instruction, indexed by \c WaveHeader::Site.
struct TracedSite {
  std::string Kernel;
  std::string Opcode;
  uint32_t Info; ///< packAccessInfo()
};

/// Base for analyses of memory-access messages.
class MemoryAccessHandler : public MessageHandler {
public:
  /// Called before each dispatch with that dispatch's buffers.
  virtual void beginDispatch(llvm::ArrayRef<TrackedBuffer> Buffers,
                             uint32_t ElementSizeLog2) {}

  /// \p Records holds one entry per active lane; the i-th belongs to the
  /// i-th set bit of \c Header.Exec.
  virtual void handleAccesses(const WaveHeader &Header,
                              llvm::ArrayRef<AccessRecord> Records) = 0;

  void handle(const WaveHeader &H, llvm::ArrayRef<uint32_t> Data) final {
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
// The tracer
//===----------------------------------------------------------------------===//

template <typename ToolT> class MemoryTrace {
public:
  //===--------------------------------------------------------------------===//
  // Device code
  //===--------------------------------------------------------------------===//

  /// Set by the host before each traced dispatch.
  __attribute__((device)) static TraceState State;

  /// One lock per sub-buffer, serialising the waves that share it.
  __attribute__((device)) static uint32_t Locks[MaxSubBuffers];

  /// Address in a 64-bit VGPR pair: <tt>global_load v, v[a:a+1], off</tt>.
  __attribute__((device, used)) static void
  onAccess(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
           uint32_t Site, uint32_t Info, uint32_t BlockX, uint32_t BlockY,
           uint32_t BlockZ, uint32_t HwId) {
    send((uint64_t(AddrHi) << 32 | AddrLo) + int64_t(int32_t(ImmOffset)),
         {MemoryAccessTag, Site, Info, BlockX, BlockY, BlockZ, HwId});
  }

  /// Scalar base plus 32-bit vector offset:
  /// <tt>global_load v, v_off, s[b:b+1]</tt>.
  __attribute__((device, used)) static void
  onAccessScalarBase(uint32_t BaseLo, uint32_t BaseHi, uint32_t VOffset,
                     uint32_t ImmOffset, uint32_t Site, uint32_t Info,
                     uint32_t BlockX, uint32_t BlockY, uint32_t BlockZ,
                     uint32_t HwId) {
    send((uint64_t(BaseHi) << 32 | BaseLo) + VOffset +
             int64_t(int32_t(ImmOffset)),
         {MemoryAccessTag, Site, Info, BlockX, BlockY, BlockZ, HwId});
  }

private:
  __attribute__((device, always_inline)) static void
  send(uint64_t Address, const device::MessageInfo &M) {
    // Address -> (buffer, element index). Buffers never overlap, so at most
    // one matches; no early exit keeps the trip count uniform.
    uint32_t Record[2] = {0u, UnresolvedBuffer};
    for (uint32_t I = 0; I < State.NumBuffers; ++I) {
      if (Address >= State.BufferBase[I] && Address < State.BufferEnd[I]) {
        Record[0] =
            uint32_t((Address - State.BufferBase[I]) >> State.ElementSizeLog2);
        Record[1] = State.BufferArgSlot[I];
      }
    }
    // Any wave-uniform choice of sub-buffer is correct; spread waves by
    // workgroup when known, else by the first lane's address page.
    const uint32_t Key =
        M.BlockIdxX != UnknownBlockIdx
            ? M.BlockIdxX + 7919u * M.BlockIdxY + 104729u * M.BlockIdxZ
            : device::broadcast(uint32_t(Address >> 12));
    device::submit<2>(State.Channel, Locks, Key * 2654435761u >> 8, M,
                      Record);
  }

public:
  //===--------------------------------------------------------------------===//
  // Host code
  //===--------------------------------------------------------------------===//

  /// Where the hooks' workgroup id comes from.
  enum class WorkgroupIdSource {
    None, ///< Not reported.
    /// Trap temporaries \c TTMP8 / \c TTMP9 / \c TTMP10, which hold the
    /// workgroup id x / y / z (verified on gfx908).
    TTMP,
    /// \c luthier::readSVA(WORKGROUP_ID_*). Returns garbage on gfx908 with
    /// the current Luthier; kept so the issue can be reproduced.
    SVA,
  };

  struct Config {
    uint32_t NumSubBuffers = 256;
    uint64_t SubBufferBytes = 64 * 1024;
    uint32_t ElementSize = 4; ///< Power of two.
    WorkgroupIdSource WorkgroupIds = WorkgroupIdSource::TTMP;
  };

  void setConfig(const Config &C) { Cfg = C; }
  void addHandler(MemoryAccessHandler &H) { Handlers.push_back(&H); }
  llvm::ArrayRef<TracedSite> sites() const { return Sites; }
  const MessageProcessor *processor() const { return Processor.get(); }
  bool droppedMessages() const { return Dropped; }

  /// Global or flat (not scratch) instructions that access memory.
  static bool isTraceable(const llvm::MachineInstr &MI) {
    return llvm::SIInstrInfo::isFLAT(MI) &&
           !llvm::SIInstrInfo::isFLATScratch(MI) && MI.mayLoadOrStore() &&
           llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                         llvm::AMDGPU::OpName::vaddr);
  }

  /// Injects a hook before \p MI that sends its access to the host.
  llvm::Error traceAccess(Prototype &P, PrototypeAnalysisManager &PAM,
                          llvm::MachineInstr &MI) {
    const auto &ST = MI.getMF()->getSubtarget<llvm::GCNSubtarget>();
    const llvm::SIRegisterInfo &TRI = *ST.getRegisterInfo();
    const unsigned Opc = MI.getOpcode();
    const uint32_t Site = uint32_t(Sites.size());
    const uint32_t Info = describeAccess(MI, TRI);
    Sites.push_back({MI.getMF()->getName().str(),
                     ST.getInstrInfo()->getName(Opc).str(), Info});

    const llvm::MCRegister VAddr =
        operandReg(MI, llvm::AMDGPU::OpName::vaddr);
    const int SAddrIdx =
        llvm::AMDGPU::getNamedOperandIdx(Opc, llvm::AMDGPU::OpName::saddr);
    const int OffIdx =
        llvm::AMDGPU::getNamedOperandIdx(Opc, llvm::AMDGPU::OpName::offset);
    const int64_t Imm = OffIdx >= 0 ? MI.getOperand(OffIdx).getImm() : 0;
    const bool ScalarBase = SAddrIdx >= 0;
    const llvm::MCRegister Base =
        ScalarBase ? MI.getOperand(SAddrIdx).getReg().asMCReg() : VAddr;
    const WorkgroupIdSource Wg = Cfg.WorkgroupIds;

    // Arguments are built inside the payload, where Luthier lowers its
    // intrinsics, and the hook is then inlined there.
    auto Build = [&](llvm::Function &Hook,
                     llvm::IRBuilderBase &B) -> llvm::Error {
      llvm::Module &M = *Hook.getParent();
      llvm::Type &I32 = *B.getInt32Ty();
      auto ReadReg = [&](llvm::MCRegister R) -> llvm::Value * {
        return insertCallToIntrinsic(M, B, "luthier::readReg", I32,
                                     uint32_t(R.id()));
      };
      auto ReadSVA = [&](ScalarValueArgument SA) -> llvm::Value * {
        return insertCallToIntrinsic(M, B, "luthier::readSVA", I32,
                                     uint8_t(SA));
      };
      auto Const = [&](uint32_t V) { return B.getInt32(V); };

      llvm::SmallVector<llvm::Value *, 10> Args{
          ReadReg(TRI.getSubReg(Base, llvm::AMDGPU::sub0)),
          ReadReg(TRI.getSubReg(Base, llvm::AMDGPU::sub1))};
      if (ScalarBase)
        Args.push_back(ReadReg(VAddr));
      Args.append({Const(uint32_t(Imm)), Const(Site), Const(Info)});
      switch (Wg) {
      case WorkgroupIdSource::SVA:
        Args.append({ReadSVA(WORKGROUP_ID_X), ReadSVA(WORKGROUP_ID_Y),
                     ReadSVA(WORKGROUP_ID_Z)});
        break;
      case WorkgroupIdSource::TTMP:
        Args.append({ReadReg(llvm::AMDGPU::TTMP8), ReadReg(llvm::AMDGPU::TTMP9),
                     ReadReg(llvm::AMDGPU::TTMP10)});
        break;
      case WorkgroupIdSource::None:
        Args.append(3, Const(UnknownBlockIdx));
        break;
      }
      // Wave / SIMD / CU / shader-engine ids of the executing wave.
      Args.push_back(insertCallToIntrinsic(M, B, "luthier::readHwReg", I32,
                                           uint16_t(HwRegHwIdGfx9)));

      llvm::CallInst *Call = B.CreateCall(&Hook, Args);
      llvm::InlineFunctionInfo IFI;
      llvm::InlineResult IR = llvm::InlineFunction(*Call, IFI);
      return IR.isSuccess() ? llvm::Error::success()
                            : LUTHIER_MAKE_GENERIC_ERROR(
                                  "comms: failed to inline the access hook: " +
                                  std::string(IR.getFailureReason()));
    };
    return ScalarBase ? self().createInjectedPayload(
                            &MemoryTrace::onAccessScalarBase, P, PAM, MI,
                            llvm::function_ref<llvm::Error(
                                llvm::Function &, llvm::IRBuilderBase &)>(
                                Build))
                      : self().createInjectedPayload(
                            &MemoryTrace::onAccess, P, PAM, MI,
                            llvm::function_ref<llvm::Error(
                                llvm::Function &, llvm::IRBuilderBase &)>(
                                Build));
  }

  /// Points the instrumented kernel at empty buffers and starts draining.
  /// Call after the dispatch was overridden with its instrumented variant
  /// and before it is submitted. \p AppPacket is the application's packet
  /// (before the override); \p KD its original kernel descriptor.
  llvm::Error beginDispatch(const hsa_kernel_dispatch_packet_t &AppPacket,
                            const llvm::amdhsa::kernel_descriptor_t &KD) {
    const auto Core = self().getCoreApiTableSnapshot().getTable();
    const auto AmdExt = self().getAmdExtTableSnapshot().getTable();
    if (!Processor) {
      auto BuffersOrErr = SharedBuffers::create(
          Core, AmdExt, {Cfg.NumSubBuffers, Cfg.SubBufferBytes});
      LUTHIER_RETURN_ON_ERROR(BuffersOrErr.takeError());
      Buffers = std::move(*BuffersOrErr);
      std::vector<MessageHandler *> Hs(Handlers.begin(), Handlers.end());
      Processor = std::make_unique<MessageProcessor>(*Buffers, std::move(Hs));
    }
    Buffers->reset();

    const auto Tracked = kernelArgBuffers(AppPacket, KD);
    TraceState S{};
    S.Channel = Buffers->describe();
    S.ElementSizeLog2 = llvm::Log2_32(Cfg.ElementSize);
    S.NumBuffers = uint32_t(Tracked.size());
    for (uint32_t I = 0; I < S.NumBuffers; ++I) {
      S.BufferBase[I] = Tracked[I].Base;
      S.BufferEnd[I] = Tracked[I].End;
      S.BufferArgSlot[I] = Tracked[I].ArgSlot;
    }
    LUTHIER_RETURN_ON_ERROR(writeDeviceGlobal(&MemoryTrace::State, &S,
                                              sizeof(S), KD));
    static const uint32_t ZeroLocks[MaxSubBuffers] = {};
    LUTHIER_RETURN_ON_ERROR(writeDeviceGlobal(&MemoryTrace::Locks, ZeroLocks,
                                              sizeof(ZeroLocks), KD));
    for (MemoryAccessHandler *H : Handlers)
      H->beginDispatch(Tracked, S.ElementSizeLog2);
    Processor->start();
    return llvm::Error::success();
  }

  /// Drains what is left once the dispatch has completed.
  void endDispatch() {
    Processor->stop();
    Dropped |= Buffers->errorBits() & 1u;
  }

  /// The kernel's pointer arguments that point into live allocations. Only
  /// reads the kernel arguments; non-pointer slots are skipped.
  llvm::SmallVector<TrackedBuffer, MaxTrackedBuffers>
  kernelArgBuffers(const hsa_kernel_dispatch_packet_t &Packet,
                   const llvm::amdhsa::kernel_descriptor_t &KD) {
    const auto Core = self().getCoreApiTableSnapshot().getTable();
    const auto AmdExt = self().getAmdExtTableSnapshot().getTable();
    llvm::SmallVector<TrackedBuffer, MaxTrackedBuffers> Out;
    const uint32_t Slots = std::min<uint32_t>(MaxTrackedBuffers,
                                              KD.kernarg_size / 8);
    uint64_t Args[MaxTrackedBuffers] = {};
    if (Slots == 0 ||
        Core.template callFunction<hsa_memory_copy>(Args, Packet.kernarg_address,
                                           Slots * sizeof(uint64_t)) !=
            HSA_STATUS_SUCCESS)
      return Out;
    for (uint32_t Slot = 0; Slot < Slots; ++Slot) {
      if (Args[Slot] == 0)
        continue;
      hsa_amd_pointer_info_t Info{};
      Info.size = sizeof(Info);
      if (AmdExt.template callFunction<hsa_amd_pointer_info>(
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

private:
  ToolT &self() { return static_cast<ToolT &>(*this); }

  template <typename OpNameT>
  static llvm::MCRegister operandReg(const llvm::MachineInstr &MI,
                                     OpNameT Name) {
    return MI
        .getOperand(llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name))
        .getReg()
        .asMCReg();
  }

  /// Read/write, address space and width of a FLAT-family access.
  static uint32_t describeAccess(const llvm::MachineInstr &MI,
                                 const llvm::SIRegisterInfo &TRI) {
    unsigned Width = 0;
    for (auto Name : {llvm::AMDGPU::OpName::vdata, llvm::AMDGPU::OpName::vdst}) {
      if (!llvm::AMDGPU::hasNamedOperand(MI.getOpcode(), Name))
        continue;
      const llvm::MCRegister R = operandReg(MI, Name);
      Width = TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(R)) / 8;
      break;
    }
    return packAccessInfo(MI.mayLoad(), MI.mayStore(),
                          llvm::SIInstrInfo::isFLATGlobal(MI)
                              ? llvm::AMDGPUAS::GLOBAL_ADDRESS
                              : llvm::AMDGPUAS::FLAT_ADDRESS,
                          Width);
  }

  template <typename T>
  llvm::Error writeDeviceGlobal(T *Var, const void *Src, size_t Bytes,
                                const llvm::amdhsa::kernel_descriptor_t &KD) {
    const auto Core = self().getCoreApiTableSnapshot().getTable();
    auto SymOrErr = self().lookupGlobalVariable(Var, &KD);
    LUTHIER_RETURN_ON_ERROR(SymOrErr.takeError());
    auto AddrOrErr = hsa::executableSymbolGetAddress(Core, *SymOrErr);
    LUTHIER_RETURN_ON_ERROR(AddrOrErr.takeError());
    return LUTHIER_HSA_CALL_ERROR_CHECK(
        Core.template callFunction<hsa_memory_copy>(
            reinterpret_cast<void *>(*AddrOrErr), Src, Bytes),
        "comms: failed to write a device global");
  }

  Config Cfg;
  std::vector<MemoryAccessHandler *> Handlers;
  std::vector<TracedSite> Sites;
  std::unique_ptr<SharedBuffers> Buffers;
  std::unique_ptr<MessageProcessor> Processor;
  bool Dropped = false;
};

template <typename ToolT>
__attribute__((device)) TraceState MemoryTrace<ToolT>::State;
template <typename ToolT>
__attribute__((device)) uint32_t MemoryTrace<ToolT>::Locks[MaxSubBuffers];

} // namespace luthier::comms

#endif
