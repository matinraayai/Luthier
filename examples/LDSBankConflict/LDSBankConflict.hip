//===-- LDSBankConflict.hip - Opcode Histogram Example ----------*- C++ -*-===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
#include <luthier/luthier.h>
#include <mutex>
#include <thread>
// undef ICMP_NE macro that was ill-defined in HIP headers
#undef ICMP_NE
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-opcode-hist-tool"

using namespace luthier;

//===----------------------------------------------------------------------===//
// Commandline arguments for the tool
//===----------------------------------------------------------------------===//

/// Define an option category for the instruction count tool
llvm::cl::OptionCategory
    LDSBankConflictToolOptionCategory("LDS Bank Conflict Options");

static llvm::cl::opt<unsigned int> InstrBeginInterval(
    "instr-begin-interval",
    llvm::cl::desc("Beginning of the instruction interval where to "
                   "apply instrumentation, inclusive"),
    llvm::cl::init(0), llvm::cl::NotHidden,
    llvm::cl::cat(LDSBankConflictToolOptionCategory));

static llvm::cl::opt<unsigned int> InstrEndInterval(
    "instr-end-interval",
    llvm::cl::desc("End of the instruction interval where to apply "
                   "instrumentation, inclusive"),
    llvm::cl::init(std::numeric_limits<unsigned int>::max()),
    llvm::cl::NotHidden, llvm::cl::cat(LDSBankConflictToolOptionCategory));

static llvm::cl::opt<unsigned int> KernelBeginInterval(
    "kernel-start-interval",
    llvm::cl::desc(
        "Beginning of the kernel interval to apply instrumentation, inclusive"),
    llvm::cl::init(0), llvm::cl::NotHidden,
    llvm::cl::cat(LDSBankConflictToolOptionCategory));

static llvm::cl::opt<unsigned int> KernelEndInterval(
    "kernel-end-interval",
    llvm::cl::desc(
        "End of the kernel interval to apply instrumentation, exclusive"),
    llvm::cl::init(std::numeric_limits<unsigned int>::max()),
    llvm::cl::NotHidden, llvm::cl::cat(LDSBankConflictToolOptionCategory));

static llvm::cl::opt<bool> DemangleKernelNames(
    "demangle-kernel-names",
    llvm::cl::desc("Whether to demangle kernel names before printing"),
    llvm::cl::init(true), llvm::cl::NotHidden,
    llvm::cl::cat(LDSBankConflictToolOptionCategory));

__attribute__((managed)) float NumBankConflicts = 0;

static uint64_t TotalNumBankConflicts = 0;

__attribute__((managed)) uint64_t NumSingleAddressLDSInstructions = 0;

/// Total number of instructions counted
static uint64_t TotalNumSingleAddressLDSInstructions = 0;

/// Number of kernels launched so far
static uint32_t NumKernelLaunched = 0;

static bool ActiveRegion = true;

bool MustDestroySignalAfterLaunch = false;

const hsa::LoadedCodeObjectKernel *KernelBeingInstrumented{nullptr};

hsa_signal_value_t SignalValue{0};

#define BANK_SIZE_LOG_2 9

/// Mutex to protect the managed histogram
std::mutex Mutex;

MARK_LUTHIER_DEVICE_MODULE

// Record OPCode Histogram
LUTHIER_HOOK_ANNOTATE countUniqueBanksPerWarpSingleAddress(unsigned int M0,
                                                           unsigned int VAddr,
                                                           short Offset) {
  unsigned long long int VCC =
      luthier::readReg<unsigned long long int>(llvm::AMDGPU::VCC);
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
  unsigned int BankIdx = Address >> BANK_SIZE_LOG_2;
  // Count how many other work-items in the warp have the same bank ID
  auto NumWorkItemsWithSameBankID =
      static_cast<float>(__popcll(__match_any(BankIdx)));
  // Each thread with the same bank ID will add 1 / NumWorkItemsWithSameBankID
  // Incrementing the final value by 1.
  atomicAdd(&NumBankConflicts, 1.f / NumWorkItemsWithSameBankID);

  luthier::writeReg(llvm::AMDGPU::VCC, VCC);
}
LUTHIER_EXPORT_HOOK_HANDLE(countUniqueBanksPerWarpSingleAddress);

static llvm::Error instrumentationLoop(InstrumentationTask &IT,
                                       LiftedRepresentation &LR) {
  unsigned int I = 0;
  for (auto &[_, MF] : LR.functions()) {
    const auto &STI = MF->getSubtarget<llvm::GCNSubtarget>();
    const auto *TII = STI.getInstrInfo();
    auto &Ctx = LR.getContext();
    for (auto &MBB : *MF) {
      for (auto &MI : MBB) {
        if (I >= InstrBeginInterval && I < InstrEndInterval) {
          if (llvm::SIInstrInfo::isDS(MI)) {
            // Single address instruction have a VGPR address operand and an
            // offset field + modifier indicating if this is a GDS operation or
            // not
            if (!TII->isAlwaysGDS(MI.getOpcode()) ||
                !TII->hasModifiersSet(MI, llvm::AMDGPU::OpName::gds)) {
              int AddrOpIdx = llvm::AMDGPU::getNamedOperandIdx(
                  MI.getOpcode(), llvm::AMDGPU::OpName::addr);
              int OffSetOpIDx = llvm::AMDGPU::getNamedOperandIdx(
                  MI.getOpcode(), llvm::AMDGPU::OpName::offset);
              if (AddrOpIdx != 1) {
                llvm::MCRegister VAddr =
                    MI.getOperand(AddrOpIdx).getReg().asMCReg();
                unsigned Short = MI.getOperand(OffSetOpIDx).getImm();
                auto *ShortArg = llvm::ConstantInt::get(
                    llvm::Type::getInt16Ty(Ctx), Short, false);
                LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
                    MI,
                    LUTHIER_GET_HOOK_HANDLE(
                        countUniqueBanksPerWarpSingleAddress),
                    {llvm::AMDGPU::M0, VAddr, ShortArg}));
              }
            }
          }
        }
        I++;
      }
    }
  }
  return llvm::Error::success();
}

static void
instrumentAllFunctionsOfLR(const hsa::LoadedCodeObjectKernel &KernelSymbol) {
  auto LR = lift(KernelSymbol);
  LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());

  LUTHIER_REPORT_FATAL_ON_ERROR(instrumentAndLoad(
      KernelSymbol, *LR, instrumentationLoop, "lds bank conflict"));
}

static void atHsaEvt(hsa::ApiEvtArgs *CBData, ApiEvtPhase Phase,
                     hsa::ApiEvtID ApiID) {
  if (ApiID == luthier::hsa::HSA_API_EVT_ID_hsa_queue_packet_submit) {
    // Copy the app packets because we are going to modify the packets field
    // of the argument, and it will be easier to mutate
    std::vector<hsa::AqlPacket> AppPackets(
        *CBData->hsa_queue_packet_submit.packets);
    auto *ToBeSubmittedPackets = CBData->hsa_queue_packet_submit.packets;
    // Clear the packets that are about to be submitted, as we will repopulate
    // it again after analyzing the AppPackets
    ToBeSubmittedPackets->clear();
    for (auto &Packet : AppPackets) {
      hsa_packet_type_t PacketType = Packet.getPacketType();
      auto &DispatchPacket = Packet.asKernelDispatch();
      if (PacketType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        if (Phase == luthier::API_EVT_PHASE_BEFORE) {
          Mutex.lock();
          /// If we are entering to a kernel launch:
          /// 1. Lock the mutex to prevent multiple kernels to run concurrently
          /// 2. Instrument the kernel if no already instrumented
          /// 3. Select whether the instrumented kernel will run or not
          /// 4. Reset the managed Counter variable
          // Get the kernel that is about to get launched
          auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                  DispatchPacket.kernel_object)
                                  ->getLoadedCodeObjectKernelSymbol();

          LUTHIER_REPORT_FATAL_ON_ERROR(KernelSymbol.takeError());

          KernelBeingInstrumented = &(*KernelSymbol);

          auto KernelName = KernelSymbol->getName();
          LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());
          std::string KernelNameToBePrinted = DemangleKernelNames
                                                  ? llvm::demangle(*KernelName)
                                                  : std::string(*KernelName);
          llvm::outs() << "Kernel " << KernelNameToBePrinted
                       << " is being launched\n";
          // Get the packet's completion signal and its value

          // Create a signal if the application didn't create one for this
          // packet
          if (DispatchPacket.completion_signal.handle == 0) {
            if (hsa::getHsaApiTable().amd_ext_->hsa_amd_signal_create_fn(
                    SignalValue, 0, nullptr, 0,
                    &DispatchPacket.completion_signal) != HSA_STATUS_SUCCESS)
              llvm::report_fatal_error("Failed to create a signal for the "
                                       "dispatch packet.");
            MustDestroySignalAfterLaunch = true;
          } else {
            MustDestroySignalAfterLaunch = false;
          }
          SignalValue =
              hsa::getHsaApiTable().core_->hsa_signal_load_scacquire_fn(
                  DispatchPacket.completion_signal);

          if (NumKernelLaunched >= KernelBeginInterval &&
              NumKernelLaunched < KernelEndInterval &&
              !llvm::cantFail(KernelSymbol->getName()).contains("rocclr")) {
            ActiveRegion = true;
          } else {
            ActiveRegion = false;
          }
          llvm::outs() << "Active region: " << ActiveRegion << "\n";

          if (ActiveRegion) {
            llvm::outs() << "Kernel is being instrumented.\n";

            auto IsKernelInstrumented =
                isKernelInstrumented(*KernelSymbol, "lds bank conflict");
            LUTHIER_REPORT_FATAL_ON_ERROR(IsKernelInstrumented.takeError());
            if (!*IsKernelInstrumented) {

              auto LR = lift(*KernelSymbol);
              LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());

              instrumentAllFunctionsOfLR(*KernelSymbol);

              LUTHIER_REPORT_FATAL_ON_ERROR(luthier::overrideWithInstrumented(
                  DispatchPacket, "lds bank conflict"));
              // zero the managed variables

              NumBankConflicts = 0.;

              NumSingleAddressLDSInstructions = 0;
            }
          }
        } else {
          llvm::outs() << "Active region in the exit callback: " << ActiveRegion
                       << "\n";
          constexpr auto Timeout =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::seconds{300})
                  .count();
          auto CurrentSignalValue =
              hsa::getHsaApiTable().core_->hsa_signal_wait_relaxed_fn(
                  DispatchPacket.completion_signal, HSA_SIGNAL_CONDITION_LT,
                  SignalValue, Timeout, HSA_WAIT_STATE_ACTIVE);
          if (CurrentSignalValue >= SignalValue)
            llvm::report_fatal_error(
                llvm::formatv("Wait timeout reached for signal handle {0:x},"
                              "Expected "
                              "value: {1}, Actual Value: {2}",
                              DispatchPacket.completion_signal.handle,
                              CurrentSignalValue - 1, SignalValue));
          if (MustDestroySignalAfterLaunch) {
            if (hsa::getHsaApiTable().core_->hsa_signal_destroy_fn(
                    DispatchPacket.completion_signal) != HSA_STATUS_SUCCESS) {
              llvm::report_fatal_error(
                  "Failed to destroy the completed kernel's"
                  "signal");
            }
          }
          if (ActiveRegion) {
            llvm::outs() << "Kernel is being waited on\n";
            /// If we are exiting a kernel launch:
            /// 1. Wait on the signal of the kernel being launched
            /// 2. Get the number of wave fronts in the kernel
            /// 3. Print the counter for this kernel
            /// 4. Release the lock

            // Get the kernel that was launched

            auto KernelName = KernelBeingInstrumented->getName();
            LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());
            std::string KernelNameToBePrinted =
                DemangleKernelNames ? llvm::demangle(*KernelName)
                                    : std::string(*KernelName);
            // Add the number of LDS instructions + conflicts recorded per
            // wavefront
            TotalNumSingleAddressLDSInstructions +=
                NumSingleAddressLDSInstructions;
            TotalNumBankConflicts += static_cast<uint64_t>(NumBankConflicts);

            // Note: Grid size in HSA contains all the work items in the kernel.
            // This is different from the number of blocks (grid dim) in CUDA or
            // HIP For example, if a kernel is launched from HIP with 16 blocks,
            // each with 64 threads, this will be indicated as a grid size dim
            // of (1024, 1, 1) and workgroup size of (64, 1, 1), since 1024 = 64
            // x 16. As is evident from the example, workgroup size in HSA has
            // the same meaning as CUDA/HIP

            llvm::outs() << llvm::formatv(
                "Kernel {0} - {1}: Total Grid Size Dims: ({2}, {3}, {4}) "
                "Workgroup Dims: "
                "({5}, {6}, {7}) LDS Single Address Instruction "
                "Count: {8}, Num bank conflicts: {9}, Num bank conflicts "
                "detected so far: {10}\n",
                NumKernelLaunched++, KernelNameToBePrinted,
                DispatchPacket.grid_size_x, DispatchPacket.grid_size_y,
                DispatchPacket.grid_size_z, DispatchPacket.workgroup_size_x,
                DispatchPacket.workgroup_size_y,
                DispatchPacket.workgroup_size_z,
                TotalNumSingleAddressLDSInstructions, NumBankConflicts,
                TotalNumBankConflicts);
          }
          NumKernelLaunched++;
          Mutex.unlock();
        }
      }
      if (Phase == API_EVT_PHASE_BEFORE)
        ToBeSubmittedPackets->push_back(Packet);
    }
  }
}

namespace luthier {

static void atHsaApiTableCaptureCallBack(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    LUTHIER_REPORT_FATAL_ON_ERROR(hsa::enableHsaApiEvtIDCallback(
        hsa::HSA_API_EVT_ID_hsa_queue_packet_submit));
  }
}

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "LDS Bank Conflict detection tool is launching.\n";
  } else {
    // Set the callback for when the HSA API table is captured
    hsa::setAtApiTableCaptureEvtCallback(atHsaApiTableCaptureCallBack);
    // Set the HSA API callback
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
  }
}

void atToolFini(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    llvm::outs() << "Total number of LDS instructions: "
                 << TotalNumSingleAddressLDSInstructions << "\n";
    llvm::outs() << "Total number of bank conflicts detected: "
                 << TotalNumBankConflicts << "\n";
  }
}

} // namespace luthier
