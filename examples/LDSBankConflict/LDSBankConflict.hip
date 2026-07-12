//===-- LDSBankConflict.hip -------------------------------------*- C++ -*-===//
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
/// LDS bank-conflict detection tool migrated to the per-tool \c HSATool
/// singleton design. Inspects single-/double-address DS_* instructions and
/// inserts hooks that compute per-wavefront bank-conflict counts.
//===----------------------------------------------------------------------===//
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
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
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/ExecutableSymbol.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSATooling/HSATool.h>
#include <luthier/HSATooling/HsaMemoryAllocationAccessor.h>
#include <luthier/LLVM/streams.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/ToolCodeGen/CodeDiscoveryPass.h>
#include <luthier/ToolCodeGen/EntryPoint.h>
#include <luthier/ToolCodeGen/FunctionAnnotations.h>
#include <luthier/ToolCodeGen/InitialEntryPointAnalysis.h>
#include <luthier/ToolCodeGen/InitialExecutionPointAnalysis.h>
#include <luthier/ToolCodeGen/InstrumentationPMDriver.h>
#include <luthier/ToolCodeGen/InstrumentationPass.h>
#include <luthier/ToolCodeGen/IPPredicatedCFG.h>
#include <luthier/ToolCodeGen/TraceCallGraph.h>
#include <luthier/ToolCodeGen/MemoryAllocationAccessor.h>
#include <luthier/ToolCodeGen/MetadataParserAnalysis.h>
#include <luthier/ToolCodeGen/NewPMAsmPrinter.h>
#include <luthier/ToolCodeGen/PrePostAmbleEmitter.h>
#include <mutex>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-example-lds-bank-conflict"

struct LDSBankConflictTool
    : public luthier::HSATool<LDSBankConflictTool, llvm::MachineFunction> {
  using HSATool::HSATool;

  __attribute__((device)) static float NumBankConflicts;
  __attribute__((device)) static uint64_t NumSingleAddressLDSInstructions;
  __attribute__((device)) static uint64_t NumDoubleAddressLDSInstructions;

  __attribute__((device, used)) static void countUniqueBanksPerWarpSingleAddress(
      unsigned M0, unsigned VAddr, short Offset, short CacheSizeLog2);
  __attribute__((device, used)) static void countUniqueBanksPerWarpDoubleAddress(
      unsigned M0, unsigned VAddr, uint8_t Offset0, uint8_t Offset1,
      short CacheSizeLog2);
  __attribute__((device, used)) static void
  countUniqueBanksPerWarpDoubleAddressTwoSource32(
      unsigned M0, unsigned VAddr, uint16_t Offset0, uint16_t Offset1,
      uint16_t CacheSizeLog2);
  __attribute__((device, used)) static void
  countUniqueBanksPerWarpDoubleAddressTwoSource64(
      unsigned M0, unsigned VAddr, uint16_t Offset0, uint16_t Offset1,
      uint16_t CacheSizeLog2);

  struct Options {
    llvm::cl::OptionCategory Cat{"LDS Bank Conflict Options"};
    llvm::cl::opt<unsigned> InstrBeginInterval{
        "instr-begin-interval", llvm::cl::desc("Begin interval, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> InstrEndInterval{
        "instr-end-interval", llvm::cl::desc("End interval, inclusive"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> KernelBeginInterval{
        "kernel-start-interval", llvm::cl::desc("Kernel begin, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden, llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> KernelEndInterval{
        "kernel-end-interval", llvm::cl::desc("Kernel end, exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(Cat)};
    llvm::cl::opt<bool> DemangleKernelNames{
        "demangle-kernel-names",
        llvm::cl::desc("Demangle kernel names before printing"),
        llvm::cl::init(true), llvm::cl::NotHidden, llvm::cl::cat(Cat)};
    llvm::cl::opt<unsigned> Log2BankSize{
        "log-two-bank-size", llvm::cl::desc("Log2 of the LDS bank size"),
        llvm::cl::init(11), llvm::cl::NotHidden, llvm::cl::cat(Cat)};
    luthier::InstrumentationPMDriverOptions DriverOpts;
    luthier::CodeDiscoveryPassOptions DiscoveryOpts;
  } Opts;

  std::recursive_mutex Mutex;
  uint32_t NumKernelLaunched = 0;
  double TotalNumBankConflicts = 0;
  uint64_t TotalNumSingleAddressLDSInstructions = 0;
  uint64_t TotalNumDoubleAddressLDSInstructions = 0;
  unsigned NextInstrIdx = 0;

  /// InstrumentationPipelineTrait callback (optional): reset the per-dispatch
  /// instruction counter before each sweep.
  void onBeginDispatchInstrumentation() { NextInstrIdx = 0; }

  void onPackets(const hsa_queue_t &Queue, uint64_t PacketIdx,
                 llvm::ArrayRef<luthier::hsa::AqlPacket> Packets,
                 hsa_amd_queue_intercept_packet_writer PacketWriter);

  luthier::InstrumentationPreservedAnalyses
  runInstrumentationPass(llvm::Module &IModule, llvm::ModuleAnalysisManager &,
                         llvm::MachineFunction &MF,
                         llvm::FunctionAnalysisManager &);
};

__attribute__((device)) float LDSBankConflictTool::NumBankConflicts;
__attribute__((device)) uint64_t LDSBankConflictTool::NumSingleAddressLDSInstructions;
__attribute__((device)) uint64_t LDSBankConflictTool::NumDoubleAddressLDSInstructions;

__attribute__((device, used)) void
LDSBankConflictTool::countUniqueBanksPerWarpSingleAddress(unsigned M0,
                                                          unsigned VAddr,
                                                          short Offset,
                                                          short CacheSizeLog2) {
  unsigned long long ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  unsigned FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId)
    atomicAdd(&LDSBankConflictTool::NumSingleAddressLDSInstructions, 1);
  unsigned Address = M0 + VAddr + Offset;
  unsigned BankIdx = Address >> CacheSizeLog2;
  float NumWorkItemsWithSameBankID =
      static_cast<float>(__popcll(__match_any(BankIdx)));
  atomicAdd(&LDSBankConflictTool::NumBankConflicts,
            1.f / NumWorkItemsWithSameBankID);
}

__attribute__((device, used)) void
LDSBankConflictTool::countUniqueBanksPerWarpDoubleAddress(
    unsigned M0, unsigned VAddr, uint8_t Offset0, uint8_t Offset1,
    short CacheSizeLog2) {
  unsigned long long ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  unsigned FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId)
    atomicAdd(&LDSBankConflictTool::NumDoubleAddressLDSInstructions, 1);
  unsigned Address0 = M0 + VAddr + (Offset0 << 2);
  unsigned Address1 = M0 + VAddr + (Offset1 << 2);
  unsigned BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned BankIdx1 = Address1 >> CacheSizeLog2;
  float NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  float NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned B0i = __shfl_sync(ExecMask, BankIdx0, I);
      if (B0i == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned B1i = __shfl_sync(ExecMask, BankIdx1, I);
      if (B1i == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }
  atomicAdd(&LDSBankConflictTool::NumBankConflicts,
            1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&LDSBankConflictTool::NumBankConflicts,
            1.f / NumWorkItemsWithSameBankID1);
}

__attribute__((device, used)) void
LDSBankConflictTool::countUniqueBanksPerWarpDoubleAddressTwoSource32(
    unsigned M0, unsigned VAddr, uint16_t Offset0, uint16_t Offset1,
    uint16_t CacheSizeLog2) {
  unsigned long long ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  unsigned FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId)
    atomicAdd(&LDSBankConflictTool::NumDoubleAddressLDSInstructions, 1);
  unsigned Address0 = M0 + VAddr + (Offset0 << 8);
  unsigned Address1 = M0 + VAddr + (Offset1 << 8);
  unsigned BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned BankIdx1 = Address1 >> CacheSizeLog2;
  float NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  float NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned B0i = __shfl_sync(ExecMask, BankIdx0, I);
      if (B0i == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned B1i = __shfl_sync(ExecMask, BankIdx1, I);
      if (B1i == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }
  atomicAdd(&LDSBankConflictTool::NumBankConflicts,
            1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&LDSBankConflictTool::NumBankConflicts,
            1.f / NumWorkItemsWithSameBankID1);
}

__attribute__((device, used)) void
LDSBankConflictTool::countUniqueBanksPerWarpDoubleAddressTwoSource64(
    unsigned M0, unsigned VAddr, uint16_t Offset0, uint16_t Offset1,
    uint16_t CacheSizeLog2) {
  unsigned long long ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  unsigned FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId)
    atomicAdd(&LDSBankConflictTool::NumDoubleAddressLDSInstructions, 1);
  unsigned Address0 = M0 + VAddr + (Offset0 << 9);
  unsigned Address1 = M0 + VAddr + (Offset1 << 9);
  unsigned BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned BankIdx1 = Address1 >> CacheSizeLog2;
  float NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  float NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned B0i = __shfl_sync(ExecMask, BankIdx0, I);
      if (B0i == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned B1i = __shfl_sync(ExecMask, BankIdx1, I);
      if (B1i == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }
  atomicAdd(&LDSBankConflictTool::NumBankConflicts,
            1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&LDSBankConflictTool::NumBankConflicts,
            1.f / NumWorkItemsWithSameBankID1);
}

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(LDSBankConflictTool)

static luthier::rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
    *CoreApiSnapshot{nullptr};
static luthier::rocprofiler::HsaApiTableSnapshot<::AmdExtTable>
    *AmdExtSnapshot{nullptr};
static luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot{nullptr};


luthier::InstrumentationPreservedAnalyses
LDSBankConflictTool::runInstrumentationPass(llvm::Module &IModule,
                                            llvm::ModuleAnalysisManager &,
                                            llvm::MachineFunction &MF,
                                            llvm::FunctionAnalysisManager &) {
  auto &Ctx = IModule.getContext();
  auto *BankSizeArg = llvm::ConstantInt::get(llvm::Type::getInt16Ty(Ctx),
                                              Opts.Log2BankSize, false);

  for (auto &MBB : MF) {
    const auto &STI = MF.getSubtarget<llvm::GCNSubtarget>();
    const auto *TII = STI.getInstrInfo();
    for (auto &MI : MBB) {
      unsigned I = NextInstrIdx++;
      if (I < Opts.InstrBeginInterval || I >= Opts.InstrEndInterval)
        continue;
      if (!llvm::SIInstrInfo::isDS(MI) || TII->isAlwaysGDS(MI.getOpcode()) ||
          TII->hasModifiersSet(MI, llvm::AMDGPU::OpName::gds))
        continue;

      if (llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                        llvm::AMDGPU::OpName::offset) &&
          llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                        llvm::AMDGPU::OpName::addr)) {
        int AddrOpIdx = llvm::AMDGPU::getNamedOperandIdx(
            MI.getOpcode(), llvm::AMDGPU::OpName::addr);
        llvm::MCRegister AddrOpVGPR =
            MI.getOperand(AddrOpIdx).getReg().asMCReg();
        int OffsetOpIdx = llvm::AMDGPU::getNamedOperandIdx(
            MI.getOpcode(), llvm::AMDGPU::OpName::offset);
        auto *ShortArg = llvm::ConstantInt::get(
            llvm::Type::getInt16Ty(Ctx),
            MI.getOperand(OffsetOpIdx).getImm(), false);
        LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
            &LDSBankConflictTool::countUniqueBanksPerWarpSingleAddress, IModule,
            MI,
            {InjectedPayloadCreationPass::RegArg{llvm::AMDGPU::M0,
                                                  llvm::Type::getInt32Ty(Ctx)},
             InjectedPayloadCreationPass::RegArg{AddrOpVGPR,
                                                  llvm::Type::getInt32Ty(Ctx)},
             ShortArg, BankSizeArg}));
      }

      if (llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                        llvm::AMDGPU::OpName::offset0) &&
          llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                        llvm::AMDGPU::OpName::offset1) &&
          llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                        llvm::AMDGPU::OpName::addr)) {
        int AddrOpIdx = llvm::AMDGPU::getNamedOperandIdx(
            MI.getOpcode(), llvm::AMDGPU::OpName::addr);
        llvm::MCRegister AddrOpVGPR =
            MI.getOperand(AddrOpIdx).getReg().asMCReg();
        int O0Idx = llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                                     llvm::AMDGPU::OpName::offset0);
        int O1Idx = llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                                     llvm::AMDGPU::OpName::offset1);

        const void *HostHandle = reinterpret_cast<const void *>(
            &LDSBankConflictTool::countUniqueBanksPerWarpDoubleAddress);
        llvm::Type *OffsetTy = llvm::Type::getInt8Ty(Ctx);
        if (MI.getOpcode() == llvm::AMDGPU::DS_READ2ST64_B32 ||
            MI.getOpcode() == llvm::AMDGPU::DS_WRITE2ST64_B32 ||
            MI.getOpcode() == llvm::AMDGPU::DS_WRXCHG2ST64_RTN_B32) {
          HostHandle = reinterpret_cast<const void *>(
              &LDSBankConflictTool::
                  countUniqueBanksPerWarpDoubleAddressTwoSource32);
          OffsetTy = llvm::Type::getInt16Ty(Ctx);
        } else if (MI.getOpcode() == llvm::AMDGPU::DS_READ2ST64_B64 ||
                   MI.getOpcode() == llvm::AMDGPU::DS_WRITE2ST64_B64 ||
                   MI.getOpcode() == llvm::AMDGPU::DS_WRXCHG2ST64_RTN_B64) {
          HostHandle = reinterpret_cast<const void *>(
              &LDSBankConflictTool::
                  countUniqueBanksPerWarpDoubleAddressTwoSource64);
          OffsetTy = llvm::Type::getInt16Ty(Ctx);
        }
        auto *O0 = llvm::ConstantInt::get(
            OffsetTy, MI.getOperand(O0Idx).getImm(), false);
        auto *O1 = llvm::ConstantInt::get(
            OffsetTy, MI.getOperand(O1Idx).getImm(), false);
        LUTHIER_REPORT_FATAL_ON_ERROR(this->createInjectedPayload(
            HostHandle, IModule, MI,
            {InjectedPayloadCreationPass::RegArg{llvm::AMDGPU::M0,
                                                  llvm::Type::getInt32Ty(Ctx)},
             InjectedPayloadCreationPass::RegArg{AddrOpVGPR,
                                                  llvm::Type::getInt32Ty(Ctx)},
             O0, O1, BankSizeArg}));
      }
    }
  }
  return luthier::InstrumentationPreservedAnalyses{};
}

void LDSBankConflictTool::onPackets(
    const hsa_queue_t &, uint64_t,
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

    if (ActiveRegion) {
      luthier::errs() << "Kernel is being instrumented.\n";
      const auto *KD =
          reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
              DispatchPacket->kernel_object);
      if (auto MissErr = this->overrideWithInstrumented(*DispatchPacket, 0)) {
        llvm::consumeError(std::move(MissErr));
        std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->runInstrumentationPipelineForDispatch(*KD, Opts.DriverOpts,
                                                        Opts.DiscoveryOpts)
                .moveInto(ObjBuf));
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->loadInstrumented(std::move(ObjBuf), KD, 0).takeError());
        LUTHIER_REPORT_FATAL_ON_ERROR(
            this->overrideWithInstrumented(*DispatchPacket, 0));
      }
    }

    constexpr auto Timeout =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::seconds{300})
            .count();
    PacketWriter(DispatchPacket, 1);
    const hsa_signal_value_t Current =
        Core.callFunction<&::CoreApiTable::hsa_signal_wait_relaxed_fn>(
            DispatchPacket->completion_signal, HSA_SIGNAL_CONDITION_LT,
            SignalValue, Timeout, HSA_WAIT_STATE_ACTIVE);
    if (Current >= SignalValue)
      llvm::report_fatal_error(llvm::formatv(
          "Wait timeout for signal {0:x}",
          DispatchPacket->completion_signal.handle));
    if (MustDestroySignalAfterLaunch) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
          Core.callFunction<&::CoreApiTable::hsa_signal_destroy_fn>(
              DispatchPacket->completion_signal),
          "Failed to destroy the dispatch packet's completion signal."));
    }

    // Read the device-side counters back into the host accumulators and zero
    // them for the next dispatch. The injected hooks atomicAdd into these
    // `__device__` globals; without reading them back here the TotalNum*
    // values printed in atToolFini stay zero. Mirrors the InstrCount tool.
    if (ActiveRegion) {
      const auto *KernelDesc =
          reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
              DispatchPacket->kernel_object);
      hsa_amd_pointer_info_t PtrInfo{};
      PtrInfo.size = sizeof(PtrInfo);
      hsa_status_t PInfoStatus = AmdExt.callFunction<hsa_amd_pointer_info>(
          const_cast<void *>(reinterpret_cast<const void *>(KernelDesc)),
          &PtrInfo, /*alloc=*/nullptr, /*num_agents_accessible=*/nullptr,
          /*accessible=*/nullptr);
      if (PInfoStatus == HSA_STATUS_SUCCESS &&
          PtrInfo.type != HSA_EXT_POINTER_TYPE_UNKNOWN) {
        hsa_agent_t Agent = PtrInfo.agentOwner;
        // Read a device global into Acc (matched type), then zero it on device.
        auto readAndZero = [&](auto *Sym, auto &Acc) {
          using T = std::remove_reference_t<decltype(Acc)>;
          auto SymOrErr = this->lookupGlobalVariable(Sym, KernelDesc);
          if (!SymOrErr) {
            llvm::consumeError(SymOrErr.takeError());
            return;
          }
          auto AddrOrErr =
              luthier::hsa::executableSymbolGetAddress(Core, *SymOrErr);
          if (!AddrOrErr) {
            llvm::consumeError(AddrOrErr.takeError());
            return;
          }
          T Snapshot{};
          if (Core.callFunction<hsa_memory_copy>(
                  &Snapshot, reinterpret_cast<void *>(*AddrOrErr),
                  sizeof(Snapshot)) != HSA_STATUS_SUCCESS)
            return;
          Acc += Snapshot;
          const T Zero{};
          (void)Core.callFunction<hsa_memory_copy>(
              reinterpret_cast<void *>(*AddrOrErr), &Zero, sizeof(Zero));
        };
        uint64_t SingleAddr = 0, DoubleAddr = 0;
        float BankConflicts = 0.f;
        readAndZero(&LDSBankConflictTool::NumSingleAddressLDSInstructions,
                    SingleAddr);
        readAndZero(&LDSBankConflictTool::NumDoubleAddressLDSInstructions,
                    DoubleAddr);
        readAndZero(&LDSBankConflictTool::NumBankConflicts, BankConflicts);
        TotalNumSingleAddressLDSInstructions += SingleAddr;
        TotalNumDoubleAddressLDSInstructions += DoubleAddr;
        TotalNumBankConflicts += BankConflicts;
      }
    }
    NumKernelLaunched++;
  }
}

namespace luthier {

static void atToolFini(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    LDSBankConflictTool::withInstance([](LDSBankConflictTool &T) {
      luthier::errs() << "Total number of LDS instructions: "
                      << (T.TotalNumSingleAddressLDSInstructions +
                          T.TotalNumDoubleAddressLDSInstructions)
                      << " (single-address: "
                      << T.TotalNumSingleAddressLDSInstructions
                      << ", double-address: "
                      << T.TotalNumDoubleAddressLDSInstructions << ")\n";
      luthier::errs() << "Total number of bank conflicts detected: "
                      << T.TotalNumBankConflicts << "\n";
    });
    LDSBankConflictTool::destroyInstance();
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
    luthier::errs() << "LDS Bank Conflict detection tool is launching.\n";
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

    LDSBankConflictTool::createInstance(*CoreApiSnapshot, *AmdExtSnapshot,
                                        *LoaderSnapshot, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));

    const auto Argv = "";
    luthier::registerDebugCLOptions();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(0, &Argv,
                                          "Luthier LDS Bank Conflict Tool",
                                          &luthier::errs(), nullptr,
                                          "LUTHIER_ARGS"),
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
             llvm::dbgs() << "Registering Luthier tool with rocprofiler-sdk.\n";
             llvm::dbgs() << "Rocprofiler-sdk version: " << RocProfVerMajor
                          << "." << RocProfVerMinor << "." << RocProfVerPatch
                          << " (" << RuntimeVersion << ")\n";
             llvm::dbgs() << "Tool priority: " << Priority << ";\n";);
  ClientID->name = "Luthier LDS Bank Conflict tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
