//===-- DhCommsTool.hip - Stream memory indices from GPU to CPU -*- C++ -*-===//
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
/// A Luthier tool that sends every global-memory access of an instrumented
/// kernel to the CPU while the kernel runs, using AMD's dh_comms protocol.
///
/// For each access the GPU sends *where* it landed, not the address: the
/// kernel-argument buffer and the element index inside it. The host resolves
/// the buffers before each dispatch; see \c resolveKernelArgBuffers.
///
/// Reading order:
///   1. \c DhCommsTool::run          what gets instrumented, with which hook
///   2. \c DhCommsTool::onAccess*    the hooks (device code)
///   3. dh_comms/DeviceSubmit.h      how a hook writes a message
///   4. \c DhCommsTool::onPackets    per-dispatch setup, start/stop draining
///   5. dh_comms/MessageProcessor.h  how the CPU drains the buffers
///   6. dh_comms/AccessIndexHandler.h what the CPU does with each message
///
/// Options (via LUTHIER_ARGS):
///   --dh-comms-sub-buffers=N        sub-buffers, power of two (default 256)
///   --dh-comms-sub-buffer-bytes=N   capacity of each (default 65536)
///   --dh-comms-element-size=N       bytes per array element (default 4)
///   --dh-comms-row-length=N         row length for (row, col) (default 1024)
///   --dh-comms-num-rows=N           rows, for the range check (default 1024)
///   --kernel-begin-interval / --kernel-end-interval
//===----------------------------------------------------------------------===//
#include "dh_comms/AccessIndexHandler.h"
#include "dh_comms/CoalescingHandler.h"
#include "dh_comms/DeviceSubmit.h"
#include "dh_comms/MessageProcessor.h"
#include "dh_comms/SharedBuffers.h"

#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <luthier/Common/Debug.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/Stacktrace.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/ExecutableSymbol.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSATooling/HSATool.h>
#include <luthier/HSATooling/HsaMemoryAllocationAccessor.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/InitialEntryPointAnalysis.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>

#include <memory>
#include <mutex>
#include <vector>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-dh-comms"

namespace dhc = luthier::dh_comms;

struct DhCommsTool
    : public luthier::HSATool<DhCommsTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  /// No indirect calls to resolve in the kernels this tool targets.
  static void patchPCUsagesHostCallback(std::uint64_t Out[2],
                                        const std::uint64_t /*In*/[7]) {
    Out[0] = 0;
    Out[1] = 0;
  }
  luthier::PatchPCUsagesPass::TargetAddressHostResolverFn
  getPatchPCUsagesHostCallback() const {
    return &DhCommsTool::patchPCUsagesHostCallback;
  }

  //===------------------------------------------------------------------===//
  // Device state
  //===------------------------------------------------------------------===//

  /// Where the shared buffers are and how to index. Copied in by the host
  /// before every instrumented dispatch.
  __attribute__((device)) static dhc::DeviceDescriptor Desc;

  /// One lock per sub-buffer, serialising the waves that share it. Device
  /// memory: only waves touch it (dh_comms' atomic_flags_d_).
  __attribute__((device)) static uint32_t Locks[dhc::MaxSubBuffers];

  //===------------------------------------------------------------------===//
  // Hooks: one per addressing form of a FLAT/GLOBAL instruction
  //===------------------------------------------------------------------===//

  /// Address in a 64-bit VGPR pair: <tt>global_load v, v[a:a+1], off</tt>.
  __attribute__((device, used)) static void
  onAccess(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
           uint32_t InstrId, uint32_t AccessInfo);

  /// Scalar base plus 32-bit vector offset: <tt>global_load v, v_off, s[b:b+1]</tt>.
  __attribute__((device, used)) static void
  onAccessScalarBase(uint32_t BaseLo, uint32_t BaseHi, uint32_t VOffset,
                     uint32_t ImmOffset, uint32_t InstrId, uint32_t AccessInfo);

  /// Diagnostic twins (--dh-comms-noop-hooks): same arguments, same register
  /// reads, none of the dh_comms protocol. Isolates injection effects from
  /// protocol effects. The guarded store keeps the arguments alive.
  __attribute__((device)) static uint32_t NoopSink;
  __attribute__((device, used)) static void
  onAccessNoop(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
               uint32_t InstrId, uint32_t AccessInfo);
  __attribute__((device, used)) static void
  onAccessScalarBaseNoop(uint32_t BaseLo, uint32_t BaseHi, uint32_t VOffset,
                         uint32_t ImmOffset, uint32_t InstrId,
                         uint32_t AccessInfo);

  //===------------------------------------------------------------------===//
  // Host state
  //===------------------------------------------------------------------===//

  struct Options {
    llvm::cl::OptionCategory Cat{"dh_comms Tool Options"};
    llvm::cl::opt<unsigned> NumSubBuffers{
        "dh-comms-sub-buffers",
        llvm::cl::desc("Number of sub-buffers (power of two)"),
        llvm::cl::init(256), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> SubBufferBytes{
        "dh-comms-sub-buffer-bytes",
        llvm::cl::desc("Capacity of each sub-buffer in bytes"),
        llvm::cl::init(64 * 1024), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> ElementSize{
        "dh-comms-element-size",
        llvm::cl::desc("Bytes per array element; index = byte offset / size"),
        llvm::cl::init(4), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> RowLength{
        "dh-comms-row-length",
        llvm::cl::desc("Row length used to split an index into (row, col)"),
        llvm::cl::init(1024), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> NumRows{
        "dh-comms-num-rows",
        llvm::cl::desc("Number of rows, for the in-range check"),
        llvm::cl::init(1024), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> LanesPerCycle{
        "dh-comms-lanes-per-cycle",
        llvm::cl::desc("Lanes the hardware issues together for one VMEM "
                       "instruction (coalescing granularity)"),
        llvm::cl::init(16), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> CacheLineBytes{
        "dh-comms-cache-line-bytes",
        llvm::cl::desc("Cache line size for the coalescing analysis"),
        llvm::cl::init(128), llvm::cl::cat(Cat)};
    llvm::cl::opt<bool> NoopHooks{
        "dh-comms-noop-hooks",
        llvm::cl::desc("Diagnostic: inject argument-only hooks instead of "
                       "the dh_comms protocol"),
        llvm::cl::init(false), llvm::cl::cat(Cat)};
    llvm::cl::opt<std::string> DumpObject{
        "dh-comms-dump-object",
        llvm::cl::desc("Diagnostic: write each instrumented code object to "
                       "<path>.<dispatch>.o"),
        llvm::cl::init(""), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> InstrBeginInterval{
        "instr-begin-interval",
        llvm::cl::desc("First instrumentation point (InstrId), inclusive"),
        llvm::cl::init(0), llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> InstrEndInterval{
        "instr-end-interval",
        llvm::cl::desc("Last instrumentation point (InstrId), exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::cat(Cat)};
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

  /// What each instrumentation point is, indexed by the InstrId the GPU
  /// sends back in WaveHeader::InstrId.
  std::vector<dhc::InstrumentedAccess> Accesses;

  std::unique_ptr<dhc::SharedBuffers> Buffers;
  std::unique_ptr<dhc::AccessIndexHandler> Handler;
  std::unique_ptr<dhc::CoalescingHandler> Coalescing;
  std::unique_ptr<dhc::MessageProcessor> Processor;
  uint32_t DroppedMessageErrors = 0;

  /// Instrumentation pass: inject a hook before every global/flat access.
  llvm::PreservedAnalyses run(luthier::Prototype &P,
                              luthier::PrototypeAnalysisManager &PAM);

  /// Called for every AQL packet the application submits.
  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  /// Prints what reached the CPU and whether every index is in range.
  void printReport() const;

private:
  /// Creates the shared buffers, handler and processor on first use.
  void ensureCommsReady();

  /// Fills \p D's indexing half: every kernel argument that points into a
  /// live allocation becomes a tracked buffer.
  void resolveKernelArgBuffers(const hsa_kernel_dispatch_packet_t &Packet,
                               const llvm::amdhsa::kernel_descriptor_t &KD,
                               dhc::DeviceDescriptor &D);

  /// Copies \p Bytes from host memory into the device global \p Var of the
  /// instrumented code object that \p KD dispatches.
  template <typename T>
  void writeDeviceGlobal(const T &Var, const void *Src, size_t Bytes,
                         const llvm::amdhsa::kernel_descriptor_t *KD);
};

//===----------------------------------------------------------------------===//
// Device code
//===----------------------------------------------------------------------===//

__attribute__((device)) dhc::DeviceDescriptor DhCommsTool::Desc;
__attribute__((device)) uint32_t DhCommsTool::Locks[dhc::MaxSubBuffers];
__attribute__((device)) uint32_t DhCommsTool::NoopSink;

__attribute__((device, used)) void
DhCommsTool::onAccessNoop(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
                          uint32_t InstrId, uint32_t AccessInfo) {
  if ((AddrLo ^ AddrHi ^ ImmOffset ^ InstrId ^ AccessInfo) == 0x5eed5eedu)
    NoopSink = 1;
}

__attribute__((device, used)) void DhCommsTool::onAccessScalarBaseNoop(
    uint32_t BaseLo, uint32_t BaseHi, uint32_t VOffset, uint32_t ImmOffset,
    uint32_t InstrId, uint32_t AccessInfo) {
  if ((BaseLo ^ BaseHi ^ VOffset ^ ImmOffset ^ InstrId ^ AccessInfo) ==
      0x5eed5eedu)
    NoopSink = 1;
}

__attribute__((device, used)) void
DhCommsTool::onAccess(uint32_t AddrLo, uint32_t AddrHi, uint32_t ImmOffset,
                      uint32_t InstrId, uint32_t AccessInfo) {
  const uint64_t Address = ((uint64_t(AddrHi) << 32) | AddrLo) +
                           int64_t(int32_t(ImmOffset));
  dhc::device::submitMemoryIndex(Desc, Locks, Address,
                                 InstrId, AccessInfo);
}

__attribute__((device, used)) void DhCommsTool::onAccessScalarBase(
    uint32_t BaseLo, uint32_t BaseHi, uint32_t VOffset, uint32_t ImmOffset,
    uint32_t InstrId, uint32_t AccessInfo) {
  const uint64_t Address = ((uint64_t(BaseHi) << 32) | BaseLo) + VOffset +
                           int64_t(int32_t(ImmOffset));
  dhc::device::submitMemoryIndex(Desc, Locks, Address,
                                 InstrId, AccessInfo);
}


//===----------------------------------------------------------------------===//
// API-table snapshots; must outlive the tool.
//===----------------------------------------------------------------------===//

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(DhCommsTool)

static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot{
    nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};

//===----------------------------------------------------------------------===//
// Instrumentation
//===----------------------------------------------------------------------===//

/// Read/write kind, address space and width of a FLAT-family access.
static uint32_t describeAccess(const llvm::MachineInstr &MI,
                               const llvm::SIRegisterInfo &TRI) {
  const uint8_t Rw = MI.mayLoad() && MI.mayStore() ? dhc::memory_access::ReadWrite
                     : MI.mayLoad()                ? dhc::memory_access::Read
                                                   : dhc::memory_access::Write;
  const uint8_t Space = llvm::SIInstrInfo::isFLATGlobal(MI)
                            ? dhc::address_space::Global
                            : dhc::address_space::Flat;
  // Width of the data moved: the stored value, or else the loaded one.
  uint16_t Width = 0;
  for (auto Name : {llvm::AMDGPU::OpName::vdata, llvm::AMDGPU::OpName::vdst}) {
    const int Idx = llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name);
    if (Idx < 0)
      continue;
    const llvm::MCRegister Reg = MI.getOperand(Idx).getReg().asMCReg();
    Width = uint16_t(TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(Reg)) / 8);
    break;
  }
  return dhc::packAccessInfo(Rw, Space, Width);
}

llvm::PreservedAnalyses
DhCommsTool::run(luthier::Prototype &P,
                 luthier::PrototypeAnalysisManager &PAM) {
  llvm::Module &IModule = P.getInstrumentationModule();
  llvm::LLVMContext &Ctx = IModule.getContext();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);

  // A device-function prototype is loaded under its kernel's code object,
  // which already defines Desc and Locks: bind to those rather than defining
  // second copies (same as InstrCount).
  {
    auto &TargetMAM =
        PAM.getResult<luthier::TargetModuleAnalysisManagerPrototypeProxy>(P)
            .getManager();
    const luthier::EntryPoint EP =
        TargetMAM.getResult<luthier::InitialEntryPointAnalysis>(
                     P.getTargetModule())
            .getInitialEntryPoint();
    if (!EP.isKernel()) {
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
    const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
    const llvm::SIInstrInfo &TII = *ST.getInstrInfo();
    const llvm::SIRegisterInfo &TRI = *ST.getRegisterInfo();
    unsigned Instrumented = 0;

    for (llvm::MachineBasicBlock &MBB : MF) {
      for (llvm::MachineInstr &MI : MBB) {
        // Global memory only: FLAT/GLOBAL instructions, never scratch.
        if (!llvm::SIInstrInfo::isFLAT(MI) ||
            llvm::SIInstrInfo::isFLATScratch(MI) || !MI.mayLoadOrStore())
          continue;
        const unsigned Opc = MI.getOpcode();
        const int VAddrIdx =
            llvm::AMDGPU::getNamedOperandIdx(Opc, llvm::AMDGPU::OpName::vaddr);
        if (VAddrIdx < 0)
          continue;
        const int SAddrIdx =
            llvm::AMDGPU::getNamedOperandIdx(Opc, llvm::AMDGPU::OpName::saddr);
        const int OffsetIdx =
            llvm::AMDGPU::getNamedOperandIdx(Opc, llvm::AMDGPU::OpName::offset);
        const int64_t Imm =
            OffsetIdx >= 0 ? MI.getOperand(OffsetIdx).getImm() : 0;

        const uint32_t InstrId = uint32_t(Accesses.size());
        const uint32_t Info = describeAccess(MI, TRI);
        Accesses.push_back({MF.getName().str(), TII.getName(Opc).str(), Info});
        if (InstrId < Opts.InstrBeginInterval ||
            InstrId >= Opts.InstrEndInterval)
          continue;

        auto Const = [&](uint64_t V) -> luthier::PayloadArg {
          return llvm::ConstantInt::get(I32, V);
        };
        auto Lo = [&](llvm::MCRegister R) -> luthier::PayloadArg {
          return luthier::RegArg{TRI.getSubReg(R, llvm::AMDGPU::sub0), I32};
        };
        auto Hi = [&](llvm::MCRegister R) -> luthier::PayloadArg {
          return luthier::RegArg{TRI.getSubReg(R, llvm::AMDGPU::sub1), I32};
        };
        const llvm::MCRegister VAddr =
            MI.getOperand(VAddrIdx).getReg().asMCReg();

        if (SAddrIdx >= 0) {
          const llvm::MCRegister SAddr =
              MI.getOperand(SAddrIdx).getReg().asMCReg();
          luthier::PayloadArg Args[]{Lo(SAddr), Hi(SAddr),
                                     luthier::RegArg{VAddr, I32},
                                     Const(uint32_t(Imm)), Const(InstrId),
                                     Const(Info)};
          LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
              Opts.NoopHooks ? &DhCommsTool::onAccessScalarBaseNoop
                             : &DhCommsTool::onAccessScalarBase,
              P, PAM, MI, Args));
        } else {
          luthier::PayloadArg Args[]{Lo(VAddr), Hi(VAddr),
                                     Const(uint32_t(Imm)), Const(InstrId),
                                     Const(Info)};
          LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
              Opts.NoopHooks ? &DhCommsTool::onAccessNoop
                             : &DhCommsTool::onAccess,
              P, PAM, MI, Args));
        }
        ++Instrumented;
      }
    }
    luthier::errs() << "[dh_comms] " << MF.getName() << ": instrumented "
                    << Instrumented << " global memory instruction(s)\n";
  });

  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  // Payload creation only adds functions and instrumentation-point
  // metadata; no MachineFunction is created or destroyed.
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
// Per-dispatch setup
//===----------------------------------------------------------------------===//

void DhCommsTool::ensureCommsReady() {
  if (Processor)
    return;
  auto BuffersOrErr = dhc::SharedBuffers::create(
      ::CoreApiSnapshot->getTable(), ::AmdExtSnapshot->getTable(),
      {Opts.NumSubBuffers, Opts.SubBufferBytes});
  LUTHIER_REPORT_FATAL_ON_ERROR(BuffersOrErr.takeError());
  Buffers = std::move(*BuffersOrErr);
  Handler = std::make_unique<dhc::AccessIndexHandler>(Opts.RowLength,
                                                      Opts.NumRows);
  Coalescing = std::make_unique<dhc::CoalescingHandler>(Opts.LanesPerCycle,
                                                        Opts.CacheLineBytes);
  Processor = std::make_unique<dhc::MessageProcessor>(
      *Buffers,
      std::vector<dhc::MessageHandler *>{Handler.get(), Coalescing.get()});
  luthier::errs() << llvm::formatv(
      "[dh_comms] {0} sub-buffers x {1} bytes of fine-grained host memory\n",
      Buffers->numSubBuffers(), Buffers->subBufferCapacity());
}

void DhCommsTool::resolveKernelArgBuffers(
    const hsa_kernel_dispatch_packet_t &Packet,
    const llvm::amdhsa::kernel_descriptor_t &KD, dhc::DeviceDescriptor &D) {
  const auto Core = ::CoreApiSnapshot->getTable();
  const auto AmdExt = ::AmdExtSnapshot->getTable();

  // Treat each 8-byte kernel-argument slot as a candidate pointer; keep the
  // ones the runtime knows as allocations. Non-pointer slots are skipped.
  const uint32_t Slots =
      std::min<uint32_t>(dhc::MaxTrackedBuffers, KD.kernarg_size / 8);
  uint64_t Args[dhc::MaxTrackedBuffers] = {};
  if (Slots == 0 ||
      Core.callFunction<hsa_memory_copy>(Args, Packet.kernarg_address,
                                         Slots * sizeof(uint64_t)) !=
          HSA_STATUS_SUCCESS)
    return;

  D.NumTrackedBuffers = 0;
  for (uint32_t Slot = 0; Slot < Slots; ++Slot) {
    const uint64_t Ptr = Args[Slot];
    if (Ptr == 0)
      continue;
    hsa_amd_pointer_info_t Info{};
    Info.size = sizeof(Info);
    if (AmdExt.callFunction<hsa_amd_pointer_info>(
            reinterpret_cast<void *>(Ptr), &Info, nullptr, nullptr,
            nullptr) != HSA_STATUS_SUCCESS ||
        Info.type == HSA_EXT_POINTER_TYPE_UNKNOWN)
      continue;
    const uint64_t AllocBase = reinterpret_cast<uint64_t>(Info.agentBaseAddress);
    const uint64_t AllocEnd = AllocBase + Info.sizeInBytes;
    if (Ptr < AllocBase || Ptr >= AllocEnd)
      continue;
    // Index from the pointer the kernel was given, not the allocation start.
    const uint32_t N = D.NumTrackedBuffers++;
    D.BufferBase[N] = Ptr;
    D.BufferEnd[N] = AllocEnd;
    D.BufferArgSlot[N] = Slot;
    luthier::errs() << llvm::formatv(
        "[dh_comms]   arg{0}: buffer of {1} bytes\n", Slot, AllocEnd - Ptr);
  }
}

template <typename T>
void DhCommsTool::writeDeviceGlobal(
    const T &Var, const void *Src, size_t Bytes,
    const llvm::amdhsa::kernel_descriptor_t *KD) {
  const auto Core = ::CoreApiSnapshot->getTable();
  auto SymOrErr = this->lookupGlobalVariable(&Var, KD);
  LUTHIER_REPORT_FATAL_ON_ERROR(SymOrErr.takeError());
  auto AddrOrErr = luthier::hsa::executableSymbolGetAddress(Core, *SymOrErr);
  LUTHIER_REPORT_FATAL_ON_ERROR(AddrOrErr.takeError());
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      Core.callFunction<hsa_memory_copy>(reinterpret_cast<void *>(*AddrOrErr),
                                         Src, Bytes),
      "dh_comms: failed to write a device global"));
}

void DhCommsTool::onPackets(
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

    // overrideWithInstrumented rewrites the packet, so take what we need
    // from the application's version first.
    const auto *OriginalKD =
        reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
            DispatchPacket->kernel_object);
    const hsa_kernel_dispatch_packet_t AppPacket = *DispatchPacket;

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
    if (ActiveRegion) {
      // Instrument on first sight of this kernel, then dispatch the variant.
      auto FirstTry =
          this->overrideWithInstrumented(*DispatchPacket, Queue, 0);
      if (!FirstTry) {
        llvm::consumeError(FirstTry.takeError());
        std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->runInstrumentationPipelineForDispatch(*OriginalKD)
                .moveInto(ObjBuf));
        if (!Opts.DumpObject.empty()) {
          const std::string Path =
              llvm::formatv("{0}.{1}.o", Opts.DumpObject.getValue(),
                            NumKernelLaunched);
          std::error_code EC;
          llvm::raw_fd_ostream OS(Path, EC);
          if (!EC) {
            OS.write(ObjBuf->getBufferStart(), ObjBuf->getBufferSize());
            luthier::errs() << "[dh_comms] wrote instrumented object " << Path
                            << "\n";
          }
        }
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->loadInstrumented(std::move(ObjBuf), OriginalKD, 0)
                .takeError());
        auto Retry = this->overrideWithInstrumented(*DispatchPacket, Queue, 0);
        LUTHIER_REPORT_FATAL_ON_ERROR(Retry.takeError());
        KernargBuf = std::move(*Retry);
      } else {
        KernargBuf = std::move(*FirstTry);
      }

      // Point the instrumented kernel at fresh, empty shared buffers.
      ensureCommsReady();
      Buffers->reset();
      dhc::DeviceDescriptor D{};
      Buffers->describe(D);
      D.NumGroupsX = (AppPacket.grid_size_x + AppPacket.workgroup_size_x - 1) /
                     AppPacket.workgroup_size_x;
      D.NumGroupsY = (AppPacket.grid_size_y + AppPacket.workgroup_size_y - 1) /
                     AppPacket.workgroup_size_y;
      D.ElementSizeLog2 = llvm::Log2_32(Opts.ElementSize);
      luthier::errs() << "[dh_comms] dispatch " << NumKernelLaunched
                      << ": resolving kernel-argument buffers\n";
      resolveKernelArgBuffers(AppPacket, *OriginalKD, D);
      writeDeviceGlobal(DhCommsTool::Desc, &D, sizeof(D), OriginalKD);
      Coalescing->beginDispatch(D);
      static const uint32_t ZeroLocks[dhc::MaxSubBuffers] = {};
      writeDeviceGlobal(DhCommsTool::Locks, ZeroLocks, sizeof(ZeroLocks),
                        OriginalKD);
      Processor->start();
    }

    // Generous: the kernel may stall while the CPU drains sub-buffers.
    constexpr auto Timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::seconds{1800})
                                 .count();
    PacketWriter(DispatchPacket, 1);
    const hsa_signal_value_t CurrentSignal =
        Core.callFunction<&::CoreApiTable::hsa_signal_wait_relaxed_fn>(
            DispatchPacket->completion_signal, HSA_SIGNAL_CONDITION_LT,
            SignalValue, Timeout, HSA_WAIT_STATE_ACTIVE);

    if (ActiveRegion) {
      Processor->stop(); // final drain of partially filled sub-buffers
      if (Buffers->errorBits() & 1)
        ++DroppedMessageErrors;
    }
    if (CurrentSignal >= SignalValue)
      llvm::report_fatal_error(
          llvm::formatv("Wait timeout reached for signal handle {0:x}",
                        DispatchPacket->completion_signal.handle));
    if (MustDestroySignal) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
          Core.callFunction<&::CoreApiTable::hsa_signal_destroy_fn>(
              DispatchPacket->completion_signal),
          "Failed to destroy the dispatch packet's completion signal."));
    }
    NumKernelLaunched++;
  }
}

//===----------------------------------------------------------------------===//
// Report
//===----------------------------------------------------------------------===//

void DhCommsTool::printReport() const {
  auto &OS = luthier::errs();
  OS << "\n[dh_comms] ===== memory indices received on the CPU =====\n";
  if (!Processor) {
    OS << "[dh_comms] no kernel was instrumented\n";
    return;
  }
  const double MiB = double(Processor->bytesProcessed()) / (1024.0 * 1024.0);
  const double Secs = Processor->secondsActive();
  OS << llvm::formatv("[dh_comms] {0} messages, {1:F1} MiB in {2:F2} s "
                      "({3:F1} MiB/s), {4} mid-kernel hand-off(s)\n",
                      Processor->messagesProcessed(), MiB, Secs,
                      Secs > 0 ? MiB / Secs : 0.0, Processor->handOffs());
  if (Processor->malformedSubBuffers())
    OS << "[dh_comms] WARNING: " << Processor->malformedSubBuffers()
       << " malformed message(s) skipped\n";
  if (DroppedMessageErrors)
    OS << "[dh_comms] WARNING: messages dropped (larger than a sub-buffer)\n";
  OS << llvm::formatv("[dh_comms] row = index / {0}, col = index % {0}; "
                      "in range = every row < {1} and every col < {0}\n",
                      unsigned(Opts.RowLength), unsigned(Opts.NumRows));
  const bool Ok = Handler->report(OS, Accesses);
  Coalescing->report(OS, Accesses);
  OS << "[dh_comms] verdict: "
     << (Ok ? "all indices in range" : "SOME INDICES OUT OF RANGE") << "\n";
}

//===----------------------------------------------------------------------===//
// rocprofiler-sdk init / fini
//===----------------------------------------------------------------------===//

namespace luthier {

static void atToolFini(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    DhCommsTool::withInstance([](DhCommsTool &T) { T.printReport(); });
    DhCommsTool::destroyInstance(); // frees the shared buffers
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

static void atToolFatalSignal(void *) { luthier::finalizeStreams(); }

static void atToolInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    luthier::initializeStreams();
    luthier::errs() << "dh_comms memory-index tool is launching.\n";
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

    DhCommsTool::createInstance(*CoreApiSnapshot, *AmdExtSnapshot,
                                *LoaderSnapshot, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(0, &Argv, "Luthier dh_comms Tool",
                                          &luthier::errs(), nullptr,
                                          "LUTHIER_ARGS"),
        "Failed to parse the command line arguments."));
    DhCommsTool::withInstance([](DhCommsTool &T) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          llvm::isPowerOf2_32(T.Opts.ElementSize),
          "--dh-comms-element-size must be a power of two"));
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          T.Opts.LanesPerCycle >= 1 && T.Opts.LanesPerCycle <= 64 &&
              T.Opts.CacheLineBytes >= 1,
          "--dh-comms-lanes-per-cycle must be 1..64 and "
          "--dh-comms-cache-line-bytes positive"));
    });
    llvm::EnablePrettyStackTrace();
    luthier::printStackTraceOnFatalSignal();
    llvm::sys::AddSignalHandler(atToolFatalSignal, nullptr);
  });
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t,
                      rocprofiler_client_id_t *ClientID) {
  luthier::atToolInit(nullptr);
  ClientID->name = "Luthier dh_comms tool";
  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
