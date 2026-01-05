//===-- LDSBankConflict.hip -------------------------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// This file implements a sample LDS Bank Conflict detector in wave-fronts
/// using Luthier.
/// The tool was inspired by NVBit paper's cache simulator example.
//===----------------------------------------------------------------------===//
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <luthier/HIP/HipError.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/LLVM/streams.h>
#include <luthier/luthier.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/HSATooling/Context.h>
#include <mutex>
#include <thread>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-example-lds-bank-conflict"

using namespace luthier;

//===----------------------------------------------------------------------===//
// Commandline arguments for the tool
//===----------------------------------------------------------------------===//
static llvm::cl::OptionCategory *LDSBankConflictToolOptionCategory;

static llvm::cl::opt<unsigned int> *InstrBeginInterval;

static llvm::cl::opt<unsigned int> *InstrEndInterval;

static llvm::cl::opt<unsigned int> *KernelBeginInterval;

static llvm::cl::opt<unsigned int> *KernelEndInterval;

static llvm::cl::opt<bool> *DemangleKernelNames;

static llvm::cl::opt<unsigned> *Log2BankSize;

//===----------------------------------------------------------------------===//
// Global variables of the tool
//===----------------------------------------------------------------------===//
/// Where the controller singleton is stored
static Context *C{nullptr};

/// Where the HIP runtime table is stored
static rocprofiler::HipApiTableSnapshot<ROCPROFILER_HIP_RUNTIME_TABLE>
    *HipRuntimeTableSnapshot{nullptr};

/// Per-kernel buffer to store the number of bank conflicts detected for
/// the currently running kernel
__attribute__((device)) float NumBankConflicts;

/// The final place on host to store the total number of bank conflicts
/// detected for all kernels
static uint64_t TotalNumBankConflicts = 0;

/// Place for the currently running kernel to write the number of single-address
/// LDS instructions it has executed so far
__attribute__((device)) uint64_t NumSingleAddressLDSInstructions;

/// Total number of single-address LDS instructions executed so far
static uint64_t TotalNumSingleAddressLDSInstructions = 0;

/// Place for the currently running kernel to write the number of double-address
/// LDS instructions it has executed so far
__attribute__((device)) uint64_t NumDoubleAddressLDSInstructions;

/// Total number of double-address LDS instructions executed so far
static uint64_t TotalNumDoubleAddressLDSInstructions = 0;

/// Number of kernels launched so far
static uint32_t NumKernelLaunched = 0;

std::recursive_mutex Mutex;

MARK_LUTHIER_DEVICE_MODULE

///
/// \param M0
/// \param VAddr
/// \param Offset
/// \param CacheSizeLog2
LUTHIER_HOOK_ANNOTATE
countUniqueBanksPerWarpSingleAddress(unsigned int M0, unsigned int VAddr,
                                     short Offset, short CacheSizeLog2) {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  // Get the position of the thread in the current wavefront (1-index)
  const uint32_t LaneId = __lane_id() + 1;
  // Get the first active thread id inside this wavefront
  unsigned int FirstActiveThreadId = __ffsll(ExecMask);
  // Have only the first active thread perform the atomic add
  if (FirstActiveThreadId == LaneId) {
    atomicAdd(&NumSingleAddressLDSInstructions, 1);
  }

  // Calculate the LDS address accessed by this instruction
  unsigned int Address = M0 + VAddr + Offset;
  // Calculate this work-item's bank ID
  unsigned int BankIdx = Address >> CacheSizeLog2;
  // Count how many other work-items in the warp have the same bank ID
  auto NumWorkItemsWithSameBankID =
      static_cast<float>(__popcll(__match_any(BankIdx)));
  // Each thread with the same bank ID will add 1 / NumWorkItemsWithSameBankID
  // Incrementing the final value by 1.
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID);
}

LUTHIER_EXPORT_HOOK_HANDLE(countUniqueBanksPerWarpSingleAddress);

///
/// \param M0
/// \param VAddr
/// \param Offset0
/// \param Offset1
/// \param CacheSizeLog2
LUTHIER_HOOK_ANNOTATE
countUniqueBanksPerWarpDoubleAddress(unsigned int M0, unsigned int VAddr,
                                     uint8_t Offset0, uint8_t Offset1,
                                     short CacheSizeLog2) {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  // Get the position of the thread in the current wavefront (1-index)
  const uint32_t LaneId = __lane_id() + 1;
  // Get the first active thread id inside this wavefront
  unsigned int FirstActiveThreadId = __ffsll(ExecMask);
  // Have only the first active thread perform the atomic add
  if (FirstActiveThreadId == LaneId) {
    atomicAdd(&NumDoubleAddressLDSInstructions, 1);
  }

  // Calculate the LDS addresses accessed by this instruction
  unsigned int Address0 = M0 + VAddr + (Offset0 << 2);

  unsigned int Address1 = M0 + VAddr + (Offset1 << 2);
  // Calculate this work-item's bank IDs
  unsigned int BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned int BankIdx1 = Address1 >> CacheSizeLog2;
  // Count how many other work-items in the warp have the same bank ID
  auto NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  auto NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  // We also have to count the conflict between BankIdx0 of
  // this thread and BankIdx1 of other threads
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned int BankIdx0IthThread = __shfl_sync(ExecMask, BankIdx0, I);
      if (BankIdx0IthThread == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned int BankIdx1IthThread = __shfl_sync(ExecMask, BankIdx1, I);
      if (BankIdx1IthThread == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }

  // Each thread with the same bank ID will add 1 / NumWorkItemsWithSameBankID
  // Incrementing the final value by 1.
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID1);
}
LUTHIER_EXPORT_HOOK_HANDLE(countUniqueBanksPerWarpDoubleAddress);

LUTHIER_HOOK_ANNOTATE countUniqueBanksPerWarpDoubleAddressTwoSource32(
    unsigned int M0, unsigned int VAddr, uint16_t Offset0, uint16_t Offset1,
    uint8_t CacheSizeLog2) {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  // Get the position of the thread in the current wavefront (1-index)
  const uint32_t LaneId = __lane_id() + 1;
  // Get the first active thread id inside this wavefront
  unsigned int FirstActiveThreadId = __ffsll(ExecMask);
  // Have only the first active thread perform the atomic add
  if (FirstActiveThreadId == LaneId) {
    atomicAdd(&NumDoubleAddressLDSInstructions, 1);
  }

  // Calculate the LDS addresses accessed by this instruction
  unsigned int Address0 = M0 + VAddr + (Offset0 << 8);

  unsigned int Address1 = M0 + VAddr + (Offset1 << 8);
  // Calculate this work-item's bank IDs
  unsigned int BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned int BankIdx1 = Address1 >> CacheSizeLog2;
  // Count how many other work-items in the warp have the same bank ID
  auto NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  auto NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  // We also have to count the conflict between BankIdx0 of
  // this thread and BankIdx1 of other threads
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned int BankIdx0IthThread = __shfl_sync(ExecMask, BankIdx0, I);
      if (BankIdx0IthThread == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned int BankIdx1IthThread = __shfl_sync(ExecMask, BankIdx1, I);
      if (BankIdx1IthThread == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }

  // Each thread with the same bank ID will add 1 / NumWorkItemsWithSameBankID
  // Incrementing the final value by 1.
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID1);
}
LUTHIER_EXPORT_HOOK_HANDLE(countUniqueBanksPerWarpDoubleAddressTwoSource32);

LUTHIER_HOOK_ANNOTATE countUniqueBanksPerWarpDoubleAddressTwoSource64(
    unsigned int M0, unsigned int VAddr, uint16_t Offset0, uint16_t Offset1,
    uint8_t CacheSizeLog2) {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  // Get the position of the thread in the current wavefront (1-index)
  const uint32_t LaneId = __lane_id() + 1;
  // Get the first active thread id inside this wavefront
  unsigned int FirstActiveThreadId = __ffsll(ExecMask);
  // Have only the first active thread perform the atomic add
  if (FirstActiveThreadId == LaneId) {
    atomicAdd(&NumDoubleAddressLDSInstructions, 1);
  }

  // Calculate the LDS addresses accessed by this instruction
  unsigned int Address0 = M0 + VAddr + (Offset0 << 9);

  unsigned int Address1 = M0 + VAddr + (Offset1 << 9);
  // Calculate this work-item's bank IDs
  unsigned int BankIdx0 = Address0 >> CacheSizeLog2;
  unsigned int BankIdx1 = Address1 >> CacheSizeLog2;
  // Count how many other work-items in the warp have the same bank ID
  auto NumWorkItemsWithSameBankID0 =
      static_cast<float>(__popcll(__match_any(BankIdx0)));
  auto NumWorkItemsWithSameBankID1 =
      static_cast<float>(__popcll(__match_any(BankIdx1)));
  // We also have to count the conflict between BankIdx0 of
  // this thread and BankIdx1 of other threads
  for (uint32_t I = 0; I < 64; I++) {
    if (LaneId != I) {
      unsigned int BankIdx0IthThread = __shfl_sync(ExecMask, BankIdx0, I);
      if (BankIdx0IthThread == NumWorkItemsWithSameBankID1)
        NumWorkItemsWithSameBankID1++;
      unsigned int BankIdx1IthThread = __shfl_sync(ExecMask, BankIdx1, I);
      if (BankIdx1IthThread == NumWorkItemsWithSameBankID0)
        NumWorkItemsWithSameBankID0++;
    }
  }

  // Each thread with the same bank ID will add 1 / NumWorkItemsWithSameBankID
  // Incrementing the final value by 1.
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID0);
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID1);
}

LUTHIER_EXPORT_HOOK_HANDLE(countUniqueBanksPerWarpDoubleAddressTwoSource64);

static llvm::Error instrumentSingleAddressDSOps(InstrumentationTask &IT,
                                                llvm::LLVMContext &Ctx,
                                                llvm::MachineInstr &MI,
                                                llvm::ConstantInt *BankSize) {
  if (llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                    llvm::AMDGPU::OpName::offset) &&
      llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                    llvm::AMDGPU::OpName::addr)) {
    // Get the address operand of the LDS operation
    int AddrOpIdx = llvm::AMDGPU::getNamedOperandIdx(
        MI.getOpcode(), llvm::AMDGPU::OpName::addr);
    llvm::MCRegister AddrOpVGPR = MI.getOperand(AddrOpIdx).getReg().asMCReg();
    // Get the offset of the operation
    int OffsetOpIdx = llvm::AMDGPU::getNamedOperandIdx(
        MI.getOpcode(), llvm::AMDGPU::OpName::offset);
    unsigned Short = MI.getOperand(OffsetOpIdx).getImm();
    auto *ShortArg =
        llvm::ConstantInt::get(llvm::Type::getInt16Ty(Ctx), Short, false);
    LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
        MI, LUTHIER_GET_HOOK_HANDLE(countUniqueBanksPerWarpSingleAddress),
        {llvm::AMDGPU::M0, AddrOpVGPR, ShortArg, BankSize}));
  }
  return llvm::Error::success();
}

static llvm::Error instrumentDoubleAddressDSOps(InstrumentationTask &IT,
                                                llvm::LLVMContext &Ctx,
                                                llvm::MachineInstr &MI,
                                                llvm::ConstantInt *BankSize) {
  if (llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                    llvm::AMDGPU::OpName::offset0) &&
      llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                    llvm::AMDGPU::OpName::offset1) &&
      llvm::AMDGPU::hasNamedOperand(MI.getOpcode(),
                                    llvm::AMDGPU::OpName::addr)) {
    // Get the address operand of the LDS operation
    int AddrOpIdx = llvm::AMDGPU::getNamedOperandIdx(
        MI.getOpcode(), llvm::AMDGPU::OpName::addr);
    llvm::MCRegister AddrOpVGPR = MI.getOperand(AddrOpIdx).getReg().asMCReg();
    // Get the offset of the operation
    int Offset0OpIdx = llvm::AMDGPU::getNamedOperandIdx(
        MI.getOpcode(), llvm::AMDGPU::OpName::offset0);
    int Offset1OpIdx = llvm::AMDGPU::getNamedOperandIdx(
        MI.getOpcode(), llvm::AMDGPU::OpName::offset1);
    if (MI.getOpcode() == llvm::AMDGPU::DS_READ2ST64_B32 ||
        MI.getOpcode() == llvm::AMDGPU::DS_WRITE2ST64_B32 ||
        MI.getOpcode() == llvm::AMDGPU::DS_WRXCHG2ST64_RTN_B32) {
      auto *Offset0Arg =
          llvm::ConstantInt::get(llvm::Type::getInt16Ty(Ctx),
                                 MI.getOperand(Offset0OpIdx).getImm(), false);
      auto *Offset1Arg =
          llvm::ConstantInt::get(llvm::Type::getInt16Ty(Ctx),
                                 MI.getOperand(Offset1OpIdx).getImm(), false);
      LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
          MI,
          LUTHIER_GET_HOOK_HANDLE(
              countUniqueBanksPerWarpDoubleAddressTwoSource32),
          {llvm::AMDGPU::M0, AddrOpVGPR, Offset0Arg, Offset1Arg, BankSize}));

    } else if (MI.getOpcode() == llvm::AMDGPU::DS_READ2ST64_B64 ||
               MI.getOpcode() == llvm::AMDGPU::DS_WRITE2ST64_B64 ||
               MI.getOpcode() == llvm::AMDGPU::DS_WRXCHG2ST64_RTN_B64) {
      auto *Offset0Arg =
          llvm::ConstantInt::get(llvm::Type::getInt16Ty(Ctx),
                                 MI.getOperand(Offset0OpIdx).getImm(), false);
      auto *Offset1Arg =
          llvm::ConstantInt::get(llvm::Type::getInt16Ty(Ctx),
                                 MI.getOperand(Offset1OpIdx).getImm(), false);
      LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
          MI,
          LUTHIER_GET_HOOK_HANDLE(
              countUniqueBanksPerWarpDoubleAddressTwoSource64),
          {llvm::AMDGPU::M0, AddrOpVGPR, Offset0Arg, Offset1Arg, BankSize}));
    } else {
      auto *Offset0Arg =
          llvm::ConstantInt::get(llvm::Type::getInt8Ty(Ctx),
                                 MI.getOperand(Offset0OpIdx).getImm(), false);
      auto *Offset1Arg =
          llvm::ConstantInt::get(llvm::Type::getInt8Ty(Ctx),
                                 MI.getOperand(Offset1OpIdx).getImm(), false);
      LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
          MI, LUTHIER_GET_HOOK_HANDLE(countUniqueBanksPerWarpDoubleAddress),
          {llvm::AMDGPU::M0, AddrOpVGPR, Offset0Arg, Offset1Arg, BankSize}));
    }
  }
  return llvm::Error::success();
}

static llvm::Error instrumentationLoop(InstrumentationTask &IT,
                                       LiftedRepresentation &LR) {
  unsigned int I = 0;
  return LR.iterateAllDefinedFunctionTypes(
      [&](const hsa::LoadedCodeObjectSymbol &Sym,
          llvm::MachineFunction &MF) -> llvm::Error {
        for (auto &MBB : MF) {
          const auto &STI = MF.getSubtarget<llvm::GCNSubtarget>();
          const auto *TII = STI.getInstrInfo();
          auto &Ctx = LR.getContext();
          auto *BankSizeArg = llvm::ConstantInt::get(
              llvm::Type::getInt16Ty(Ctx), *Log2BankSize, false);
          for (auto &MI : MBB) {
            // Ignore instructions not inside an interval
            // Ignore GDS operations
            if (I >= *InstrBeginInterval && I < *InstrEndInterval &&
                llvm::SIInstrInfo::isDS(MI) &&
                !TII->isAlwaysGDS(MI.getOpcode()) &&
                !TII->hasModifiersSet(MI, llvm::AMDGPU::OpName::gds)) {

              // Handle single source LDS access
              LUTHIER_RETURN_ON_ERROR(
                  instrumentSingleAddressDSOps(IT, Ctx, MI, BankSizeArg));
              // Handle double source LDS access
              LUTHIER_RETURN_ON_ERROR(
                  instrumentDoubleAddressDSOps(IT, Ctx, MI, BankSizeArg));
            }
            I++;
          }
        }
        return llvm::Error::success();
      });
}

static void
instrumentAllFunctionsOfLR(const hsa::LoadedCodeObjectKernel &KernelSymbol) {
  auto LR = lift(KernelSymbol);
  LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());

  LUTHIER_REPORT_FATAL_ON_ERROR(instrumentAndLoad(
      KernelSymbol, *LR, instrumentationLoop, "lds bank conflict"));
}

template <typename T> static llvm::Error zeroDeviceBuffer(T &DevBuffer) {
  T HostBuffer = 0;
  T *DevBufferAddr;
  const auto HsaCoreApiTable = C->getHsaCoreTable();

  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HIP_CALL_ERROR_CHECK(
      HipRuntimeTableSnapshot->getTable()
          .callFunction<&::HipDispatchTable::hipGetSymbolAddress_fn>(
              (void **)&DevBufferAddr, &DevBuffer),
      "Failed to get the symbol associated with the device buffer"));
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      HsaCoreApiTable.callFunction<hsa_memory_copy>(DevBufferAddr, &HostBuffer,
                                                    sizeof(T)),
      "Failed to copy zeros to the device buffer");
}

template <typename T>
static llvm::Expected<T> copyDeviceBufferToHost(T &DevBuffer) {
  T *DevBufferAddr;
  T HostBuffer;
  const auto HsaCoreApiTable = C->getHsaCoreTable();
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HIP_CALL_ERROR_CHECK(
      HipRuntimeTableSnapshot->getTable()
          .callFunction<&::HipDispatchTable::hipGetSymbolAddress_fn>(
              (void **)&DevBufferAddr, &DevBuffer),
      "Failed to get the symbol associated with the device buffer"));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      HsaCoreApiTable.callFunction<hsa_memory_copy>(&HostBuffer, DevBufferAddr,
                                                    sizeof(T)),
      "Failed to copy the device buffer back to host"));
  return HostBuffer;
}

static void
atPacketDispatchCallback(const hsa_queue_t &Queue, uint64_t PacketIdx,
                         llvm::ArrayRef<hsa::AqlPacket> Packets,
                         hsa_amd_queue_intercept_packet_writer PacketWriter) {
  // Copy the app packets because we are going to modify the packets field
  // of the argument, and it will be easier to mutate
  std::vector<hsa::AqlPacket> AppPackets(Packets);
  for (auto &Packet : AppPackets) {
    if (auto *DispatchPacket = Packet.asKernelDispatch()) {
      Mutex.lock();

      const auto HsaCoreApiTable = C->getHsaCoreTable();
      const auto HsaAmdExtTable = C->getHsaAmdExtTable();
      const auto &LoaderTable = C->getHsaLoaderTable();

      /// This is set to true if the original dispatch packet didn't have a
      /// signal field set. In this scenario, the tool creates a signal so that
      /// it can wait on the end of the kernel, and must destroy the signal
      /// after waiting on the kernel
      bool MustDestroySignalAfterLaunch{false};

      /// The value of the signal to wait for after kernel launches
      hsa_signal_value_t SignalValue{0};

      /// Keeps track of whether we are in the interval to profile or not
      bool ActiveRegion{false};

      /// If we are entering to a kernel launch:
      /// 1. Lock the mutex to prevent multiple kernels to run concurrently
      /// 2. Instrument the kernel if no already instrumented
      /// 3. Select whether the instrumented kernel will run or not
      /// 4. Reset the managed Counter variable
      // Get the kernel that is about to get launched
      std::unique_ptr<hsa::LoadedCodeObjectKernel> KernelBeingInstrumented;
      LUTHIER_REPORT_FATAL_ON_ERROR(
          hsa::KernelDescriptor::fromKernelObject(DispatchPacket->kernel_object)
              ->getLoadedCodeObjectKernelSymbol(HsaCoreApiTable, LoaderTable)
              .moveInto(KernelBeingInstrumented));

      auto KernelName = KernelBeingInstrumented->getName();
      LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());
      std::string KernelNameToBePrinted = DemangleKernelNames
                                              ? llvm::demangle(*KernelName)
                                              : std::string(*KernelName);
      luthier::errs() << "Kernel " << KernelNameToBePrinted
                      << " is being launched\n";
      // Get the packet's completion signal and its value

      // Create a signal if the application didn't create one for this
      // packet
      if (DispatchPacket->completion_signal.handle == 0) {
        if (HsaAmdExtTable.callFunction<hsa_amd_signal_create>(
                SignalValue, 0, nullptr, 0,
                &DispatchPacket->completion_signal) != HSA_STATUS_SUCCESS)
          llvm::report_fatal_error("Failed to create a signal for the "
                                   "dispatch packet.");
        MustDestroySignalAfterLaunch = true;
      } else {
        MustDestroySignalAfterLaunch = false;
      }
      SignalValue = HsaCoreApiTable.callFunction<hsa_signal_load_scacquire>(
          DispatchPacket->completion_signal);

      if (NumKernelLaunched >= *KernelBeginInterval &&
          NumKernelLaunched < *KernelEndInterval) {
        ActiveRegion = true;
      } else {
        ActiveRegion = false;
      }
      luthier::errs() << "Active region: " << ActiveRegion << "\n";

      if (ActiveRegion) {
        luthier::errs() << "Kernel is being instrumented.\n";

        auto IsKernelInstrumented =
            isKernelInstrumented(*KernelBeingInstrumented, "lds bank conflict");
        LUTHIER_REPORT_FATAL_ON_ERROR(IsKernelInstrumented.takeError());
        if (!*IsKernelInstrumented) {

          instrumentAllFunctionsOfLR(*KernelBeingInstrumented);
        }

        LUTHIER_REPORT_FATAL_ON_ERROR(luthier::overrideWithInstrumented(
            *DispatchPacket, "lds bank conflict"));

        // Zero the device variables

        LUTHIER_REPORT_FATAL_ON_ERROR(zeroDeviceBuffer(NumBankConflicts));

        LUTHIER_REPORT_FATAL_ON_ERROR(
            zeroDeviceBuffer(NumSingleAddressLDSInstructions));

        LUTHIER_REPORT_FATAL_ON_ERROR(
            zeroDeviceBuffer(NumDoubleAddressLDSInstructions));
      }
      luthier::errs() << "Active region in the exit callback: " << ActiveRegion
                      << "\n";
      constexpr auto Timeout =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::seconds{300})
              .count();
      PacketWriter(DispatchPacket, 1);
      auto CurrentSignalValue =
          HsaCoreApiTable.callFunction<hsa_signal_wait_relaxed>(
              DispatchPacket->completion_signal, HSA_SIGNAL_CONDITION_LT,
              SignalValue, Timeout, HSA_WAIT_STATE_ACTIVE);
      if (CurrentSignalValue >= SignalValue)
        llvm::report_fatal_error(
            llvm::formatv("Wait timeout reached for signal handle {0:x},"
                          "Expected "
                          "value: {1}, Actual Value: {2}",
                          DispatchPacket->completion_signal.handle,
                          CurrentSignalValue - 1, SignalValue));
      if (MustDestroySignalAfterLaunch) {
        if (HsaCoreApiTable.callFunction<hsa_signal_destroy>(
                DispatchPacket->completion_signal) != HSA_STATUS_SUCCESS) {
          llvm::report_fatal_error("Failed to destroy the completed kernel's"
                                   "signal");
        }
      }
      if (ActiveRegion) {
        luthier::errs() << "Kernel is being waited on\n";
        /// If we are exiting a kernel launch:
        /// 1. Wait on the signal of the kernel being launched
        /// 2. Get the number of wave fronts in the kernel
        /// 3. Print the counter for this kernel
        /// 4. Release the lock

        // Get the kernel that was launched

        auto KernelName = KernelBeingInstrumented->getName();
        LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());
        std::string KernelNameToBePrinted = DemangleKernelNames
                                                ? llvm::demangle(*KernelName)
                                                : std::string(*KernelName);

        auto KernelBankConflicts = copyDeviceBufferToHost(NumBankConflicts);

        LUTHIER_REPORT_FATAL_ON_ERROR(KernelBankConflicts.takeError());

        auto KernelSingleAddrLDSInsts =
            copyDeviceBufferToHost(NumSingleAddressLDSInstructions);

        LUTHIER_REPORT_FATAL_ON_ERROR(KernelSingleAddrLDSInsts.takeError());

        auto KernelDoubleAddrLDSInsts =
            copyDeviceBufferToHost(NumDoubleAddressLDSInstructions);

        LUTHIER_REPORT_FATAL_ON_ERROR(KernelDoubleAddrLDSInsts.takeError());

        // Add the number of LDS instructions + conflicts recorded per
        // wavefront
        TotalNumSingleAddressLDSInstructions += *KernelSingleAddrLDSInsts;
        TotalNumDoubleAddressLDSInstructions += *KernelDoubleAddrLDSInsts;
        TotalNumBankConflicts += static_cast<uint64_t>(*KernelBankConflicts);

        // Note: Grid size in HSA contains all the work items in the kernel.
        // This is different from the number of blocks (grid dim) in CUDA or
        // HIP For example, if a kernel is launched from HIP with 16 blocks,
        // each with 64 threads, this will be indicated as a grid size dim
        // of (1024, 1, 1) and workgroup size of (64, 1, 1), since 1024 = 64
        // x 16. As is evident from the example, workgroup size in HSA has
        // the same meaning as CUDA/HIP

        luthier::errs() << llvm::formatv(
            "Kernel {0} - {1}: Total Grid Size Dims: ({2}, {3}, {4}) "
            "Workgroup Dims: "
            "({5}, {6}, {7}) Dyn. LDS Size Requested: {8} LDS Single "
            "Address "
            "Instruction "
            "Count: {9}, LDS Double Address Instruction Count: {10}, Num "
            "bank conflicts: {11}, Num bank conflicts "
            "detected so far: {12}\n",
            NumKernelLaunched++, KernelNameToBePrinted,
            DispatchPacket->grid_size_x, DispatchPacket->grid_size_y,
            DispatchPacket->grid_size_z, DispatchPacket->workgroup_size_x,
            DispatchPacket->workgroup_size_y, DispatchPacket->workgroup_size_z,
            DispatchPacket->group_segment_size,
            TotalNumSingleAddressLDSInstructions,
            TotalNumDoubleAddressLDSInstructions, *KernelBankConflicts,
            TotalNumBankConflicts);
      }
      NumKernelLaunched++;
      Mutex.unlock();
    } else {
      PacketWriter(&Packet, 1);
    }
  }
}

namespace luthier {

void atToolFini(void *) {
  static std::once_flag Once{};
  std::call_once(Once, []() {
    luthier::errs() << "Total number of LDS instructions: "
                    << TotalNumSingleAddressLDSInstructions << "\n";
    luthier::errs() << "Total number of bank conflicts detected: "
                    << TotalNumBankConflicts << "\n";

    delete C;

    delete HipRuntimeTableSnapshot;

    delete InstrBeginInterval;

    delete InstrEndInterval;

    delete KernelBeginInterval;

    delete KernelEndInterval;

    delete DemangleKernelNames;

    delete Log2BankSize;

    delete LDSBankConflictToolOptionCategory;
  });
}

static void atToolFini() { atToolFini(nullptr); }

void atToolInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    luthier::errs() << "LDS Bank Conflict detection tool is launching.\n";

    std::atexit(atToolFini);

    LDSBankConflictToolOptionCategory =
        new llvm::cl::OptionCategory("LDS Bank Conflict Options");

    InstrBeginInterval = new llvm::cl::opt<unsigned int>(
        "instr-begin-interval",
        llvm::cl::desc("Beginning of the instruction interval where to "
                       "apply instrumentation, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden,
        llvm::cl::cat(*LDSBankConflictToolOptionCategory));

    InstrEndInterval = new llvm::cl::opt<unsigned int>(
        "instr-end-interval",
        llvm::cl::desc("End of the instruction interval where to apply "
                       "instrumentation, inclusive"),
        llvm::cl::init(std::numeric_limits<unsigned int>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(*LDSBankConflictToolOptionCategory));

    KernelBeginInterval = new llvm::cl::opt<unsigned int>(
        "kernel-start-interval",
        llvm::cl::desc("Beginning of the kernel interval to apply "
                       "instrumentation, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden,
        llvm::cl::cat(*LDSBankConflictToolOptionCategory));

    KernelEndInterval = new llvm::cl::opt<unsigned int>(
        "kernel-end-interval",
        llvm::cl::desc(
            "End of the kernel interval to apply instrumentation, exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned int>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(*LDSBankConflictToolOptionCategory));

    DemangleKernelNames = new llvm::cl::opt<bool>(
        "demangle-kernel-names",
        llvm::cl::desc("Whether to demangle kernel names before printing"),
        llvm::cl::init(true), llvm::cl::NotHidden,
        llvm::cl::cat(*LDSBankConflictToolOptionCategory));

    Log2BankSize = new llvm::cl::opt<unsigned>(
        "log-two-bank-size", llvm::cl::desc("Log2 of the LDS bank size"),
        llvm::cl::init(11), llvm::cl::NotHidden,
        llvm::cl::cat(*LDSBankConflictToolOptionCategory));

    const auto Argv = "";
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(
            0, &Argv, "Luthier ROCm Instruction Counter Tool", &luthier::errs(),
            nullptr, "LUTHIER_ARGS"),
        "Failed to parse the command line arguments."));
    llvm::EnablePrettyStackTrace();
    llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/matinraayai/Luthier/issues/ and "
                          "include the crash error message and backtrace.\n");
    // Add the rocprofiler finalize function as a signal handler to LLVM so that
    // it executes in case of a fatal error
    llvm::sys::AddSignalHandler(atToolFini, nullptr);

    llvm::Error Err = llvm::Error::success();
    C = new Context(atPacketDispatchCallback, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
    HipRuntimeTableSnapshot =
        new rocprofiler::HipApiTableSnapshot<ROCPROFILER_HIP_RUNTIME_TABLE>(
            Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
  });
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ClientID) {
  // Initialize the tooling library
  luthier::atToolInit(nullptr);

  LLVM_DEBUG(const uint32_t RocProfVerMajor = Version / 10000;
             const uint32_t RocProfVerMinor = (Version % 10000) / 100;
             const uint32_t RocProfVerPatch = Version % 100;
             llvm::dbgs() << "Registering Luthier tool with rocprofiler-sdk.\n";
             llvm::dbgs() << "Rocprofiler-sdk version: " << RocProfVerMajor
                          << "." << RocProfVerMinor << "." << RocProfVerPatch
                          << " (" << RuntimeVersion << ")"
                          << "\n";
             llvm::dbgs() << "Tool priority: " << Priority << ";\n";);
  ClientID->name = "Luthier LDS Bank Conflict tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
