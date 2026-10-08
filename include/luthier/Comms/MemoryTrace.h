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
  std::string Function; ///< Machine function containing it.
  unsigned Block;       ///< Its basic block: \c bb.<Block> in the MIR.
  std::string MIR;      ///< The instruction as printed in the function's MIR.
  uint32_t Width;       ///< Bytes accessed per lane; also \c WaveHeader::Info.
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
// Instrumentation: what is recorded and how the hook is called
//===----------------------------------------------------------------------===//

/// \c TracedSite of \p MI: its function, its MIR text and its width.
inline TracedSite describeSite(const llvm::MachineInstr &MI) {
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

/// Builds, at the payload's insertion point, the call of \p Hook
/// (\c onAccess or \c onAccessScalarBase) for the access of \p MI, then
/// inlines it. Luthier lowers its intrinsics only inside the payload, so
/// every register read happens here.
inline llvm::Error buildAccessHookCall(llvm::Function &Hook,
                                       llvm::IRBuilderBase &B,
                                       const llvm::MachineInstr &MI,
                                       uint32_t Site, uint32_t Width) {
  const auto &ST = MI.getMF()->getSubtarget<llvm::GCNSubtarget>();
  const llvm::SIRegisterInfo &TRI = *ST.getRegisterInfo();
  llvm::Module &M = *Hook.getParent();
  llvm::Type &I32 = *B.getInt32Ty();
  auto ReadReg = [&](llvm::MCRegister R) -> llvm::Value * {
    return insertCallToIntrinsic(M, B, "luthier::readReg", I32,
                                 uint32_t(R.id()));
  };
  auto Reg = [&](llvm::AMDGPU::OpName Name) {
    return MI.getOperand(llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name))
        .getReg()
        .asMCReg();
  };
  const int OffIdx =
      llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(), llvm::AMDGPU::OpName::offset);
  const int64_t Imm = OffIdx >= 0 ? MI.getOperand(OffIdx).getImm() : 0;
  const bool ScalarBase =
      llvm::AMDGPU::hasNamedOperand(MI.getOpcode(), llvm::AMDGPU::OpName::saddr);
  const llvm::MCRegister Base = Reg(ScalarBase ? llvm::AMDGPU::OpName::saddr
                                               : llvm::AMDGPU::OpName::vaddr);

  llvm::SmallVector<llvm::Value *, 10> Args{
      ReadReg(TRI.getSubReg(Base, llvm::AMDGPU::sub0)),
      ReadReg(TRI.getSubReg(Base, llvm::AMDGPU::sub1))};
  if (ScalarBase)
    Args.push_back(ReadReg(Reg(llvm::AMDGPU::OpName::vaddr)));
  Args.append({B.getInt32(uint32_t(Imm)), B.getInt32(Site), B.getInt32(Width)});

  // Workgroup id x / y / z. TTMPs ("trap temporaries") are SGPRs reserved for
  // the trap handler, which application code never writes; at wave launch
  // the hardware/firmware leaves the workgroup id in them: TTMP8 / 9 / 10
  // up to GFX11 (verified on gfx908), and with architected SGPRs (GFX12)
  // TTMP9 = x, TTMP7 = y | z << 16.
  if (ST.hasArchitectedSGPRs()) {
    llvm::Value *YZ = ReadReg(llvm::AMDGPU::TTMP7);
    Args.append({ReadReg(llvm::AMDGPU::TTMP9), B.CreateAnd(YZ, 0xffff),
                 B.CreateLShr(YZ, 16)});
  } else {
    Args.append({ReadReg(llvm::AMDGPU::TTMP8), ReadReg(llvm::AMDGPU::TTMP9),
                 ReadReg(llvm::AMDGPU::TTMP10)});
  }

  // Engine / array / CU (WGP) of the wave; decoded by device::decodeHwId.
  Args.push_back(insertCallToIntrinsic(
      M, B, "luthier::readHwReg", I32,
      uint16_t(ST.getGeneration() >= llvm::AMDGPUSubtarget::GFX10
                   ? HwRegHwId1Gfx10
                   : HwRegHwIdGfx9)));

  llvm::CallInst *Call = B.CreateCall(&Hook, Args);
  llvm::InlineFunctionInfo IFI;
  llvm::InlineResult IR = llvm::InlineFunction(*Call, IFI);
  return IR.isSuccess() ? llvm::Error::success()
                        : LUTHIER_MAKE_GENERIC_ERROR(
                              "comms: failed to inline the access hook: " +
                              std::string(IR.getFailureReason()));
}

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
  /// \p HwId is the raw hardware-id register (see \c device::decodeHwId).
  __attribute__((device, used)) static void
  onAccess(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
           uint32_t Site, uint32_t Width, uint32_t BlockX, uint32_t BlockY,
           uint32_t BlockZ, uint32_t HwId) {
    send((uint64_t(AddrHi) << 32 | AddrLo) + int64_t(int32_t(ImmOffset)),
         {MemoryAccessTag, Site, Width, BlockX, BlockY, BlockZ,
          device::decodeHwId(HwId)});
  }

  /// Scalar base plus 32-bit vector offset:
  /// <tt>global_load v, v_off, s[b:b+1]</tt>.
  __attribute__((device, used)) static void
  onAccessScalarBase(uint32_t BaseLo, uint32_t BaseHi, uint32_t VOffset,
                     uint32_t ImmOffset, uint32_t Site, uint32_t Width,
                     uint32_t BlockX, uint32_t BlockY, uint32_t BlockZ,
                     uint32_t HwId) {
    send((uint64_t(BaseHi) << 32 | BaseLo) + VOffset +
             int64_t(int32_t(ImmOffset)),
         {MemoryAccessTag, Site, Width, BlockX, BlockY, BlockZ,
          device::decodeHwId(HwId)});
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
    // workgroup.
    const uint32_t Key =
        M.BlockIdxX + 7919u * M.BlockIdxY + 104729u * M.BlockIdxZ;
    device::submit<2>(State.Channel, Locks, Key * 2654435761u >> 8, M,
                      Record);
  }

public:
  //===--------------------------------------------------------------------===//
  // Host code
  //===--------------------------------------------------------------------===//

  struct Config {
    uint32_t NumSubBuffers = 256;
    uint64_t SubBufferBytes = 64 * 1024;
    uint32_t ElementSize = 4; ///< Power of two.
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
    const uint32_t Site = uint32_t(Sites.size());
    Sites.push_back(describeSite(MI));
    auto Build = [&](llvm::Function &Hook,
                     llvm::IRBuilderBase &B) -> llvm::Error {
      return buildAccessHookCall(Hook, B, MI, Site, Sites.back().Width);
    };
    llvm::function_ref<llvm::Error(llvm::Function &, llvm::IRBuilderBase &)>
        BuildRef(Build);
    return llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                         llvm::AMDGPU::OpName::saddr)
               ? self().createInjectedPayload(&MemoryTrace::onAccessScalarBase,
                                              P, PAM, MI, BuildRef)
               : self().createInjectedPayload(&MemoryTrace::onAccess, P, PAM,
                                              MI, BuildRef);
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
