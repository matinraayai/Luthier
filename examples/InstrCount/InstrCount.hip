//===-- InstrCount.hip - Instruction Count Example ------------------------===//
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
/// This file implements a sample instruction counter tool using Luthier.
/// The tool was inspired by NVBit's instruction counter.
//===----------------------------------------------------------------------===//
#include <luthier/luthier.h>
#include <mutex>
#include <thread>
// undef ICMP_NE macro that was ill-defined in HIP headers
#undef ICMP_NE
#include <SIInstrInfo.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-instr-counter-tool"

using namespace luthier;

//===----------------------------------------------------------------------===//
// Commandline arguments for the tool
//===----------------------------------------------------------------------===//

/// Define an option category for the instruction count tool
llvm::cl::OptionCategory
    InstrCountToolOptionCategory("Instruction Count Tool Options");

static llvm::cl::opt<unsigned int> InstrBeginInterval(
    "instr-begin-interval",
    llvm::cl::desc("Beginning of the instruction interval where to "
                   "apply instrumentation, inclusive"),
    llvm::cl::init(0), llvm::cl::NotHidden,
    llvm::cl::cat(InstrCountToolOptionCategory));

static llvm::cl::opt<unsigned int>
    InstrEndInterval("instr-end-interval",
                     llvm::cl::desc("End of the instruction interval where to "
                                    "apply instrumentation, exclusive"),
                     llvm::cl::init(std::numeric_limits<unsigned int>::max()),
                     llvm::cl::NotHidden,
                     llvm::cl::cat(InstrCountToolOptionCategory));

static llvm::cl::opt<unsigned int> KernelBeginInterval(
    "kernel-begin-interval",
    llvm::cl::desc(
        "Beginning of the kernel interval to apply instrumentation, inclusive"),
    llvm::cl::init(0), llvm::cl::NotHidden,
    llvm::cl::cat(InstrCountToolOptionCategory));

static llvm::cl::opt<unsigned int> KernelEndInterval(
    "kernel-end-interval",
    llvm::cl::desc(
        "End of the kernel interval to apply instrumentation, exclusive"),
    llvm::cl::init(std::numeric_limits<unsigned int>::max()),
    llvm::cl::NotHidden, llvm::cl::cat(InstrCountToolOptionCategory));

static llvm::cl::opt<bool> CountWavefrontLevel(
    "count-wavefront-level",
    llvm::cl::desc("Whether to count number of instructions at the wavefront "
                   "level or thread level"),
    llvm::cl::init(false), llvm::cl::NotHidden,
    llvm::cl::cat(InstrCountToolOptionCategory));

static llvm::cl::opt<bool> DemangleKernelNames(
    "demangle-kernel-names",
    llvm::cl::desc("Whether to demangle kernel names before printing"),
    llvm::cl::init(true), llvm::cl::NotHidden,
    llvm::cl::cat(InstrCountToolOptionCategory));

//===----------------------------------------------------------------------===//
// Global variables of the tool
//===----------------------------------------------------------------------===//

/// Number of kernels launched so far
static uint32_t NumKernelLaunched = 0;

/// Total number of instructions, incremented by the managed variable
/// \c Counter every time a kernel finishes executing
static uint64_t TotalNumInstructions = 0;

/// Kernel instruction counter
__attribute__((managed)) uint64_t Counter = 0;

/// Keeps track of whether we are in the interval to profile or not
static bool ActiveRegion = true;

/// A Mutex, used to protect the Counter value and ActiveRegion
std::mutex Mutex;

static const luthier::hsa::LoadedCodeObjectKernel *KernelBeingInstrumented{
    nullptr};

static bool MustDestroySignalAfterLaunch{false};

static hsa_signal_value_t SignalValue{0};

//===----------------------------------------------------------------------===//
// Hook definitions
//===----------------------------------------------------------------------===//

/// If the Luthier tool contains a hook, then this macro must be used to mark it
/// for Luthier's Tool Executable Manager
MARK_LUTHIER_DEVICE_MODULE

// TODO: Can the optimizations around atomic add be removed here?
LUTHIER_HOOK_ANNOTATE countInstructionsVector(bool CountWaveFrontLevel) {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  // Get the position of the thread in the current wavefront (1-index)
  const uint32_t LaneId = __lane_id() + 1;
  // Get the first active thread id inside this wavefront
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  // Get the number of active threads in this wavefront
  uint32_t NumActiveThreads = __popcll(ExecMask);

  // Have only the first active thread perform the atomic add
  if (FirstActiveThreadId == LaneId) {
    if (CountWaveFrontLevel) {
      // Num threads can be zero when accounting for predicates off
      if (NumActiveThreads > 0) {
        atomicAdd(&Counter, 1);
      }
    } else {
      atomicAdd(&Counter, NumActiveThreads);
    }
  }
}

LUTHIER_EXPORT_HOOK_HANDLE(countInstructionsVector);

LUTHIER_HOOK_ANNOTATE countInstructionsScalar() {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  // Overwrite the exec mask with one so that only a single thread is active
  luthier::writeExec(1);
  // Increment the counter by 1
  atomicAdd(&Counter, 1);
  // Restore the exec mask
  luthier::writeExec(ExecMask);
}

LUTHIER_EXPORT_HOOK_HANDLE(countInstructionsScalar);

//  Counter += 1;
//  __builtin_amdgcn_s_dcache_wb();
//    __builtin_amdgcn_s_dcache_inv();
// atomicAdd(&Counter, 5);
//  // Get the exec mask of the wavefront
//  unsigned long long int ExecMask =
//      luthier::readReg<unsigned long long int>(llvm::AMDGPU::EXEC);
//  // Get the position of the thread in the current wavefront
//  const uint32_t ThreadPosition = __lane_id();
//  // Get the first active thread id inside this wavefront
//  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
//  // Get the number of active threads in this wavefront
//  uint32_t NumActiveThreads = __popcll(ExecMask);
//
//  // Have only the first active thread perform the atomic add
//  if (FirstActiveThreadId == ThreadPosition)
//      atomicAdd(&Counter, NumActiveThreads);

//===----------------------------------------------------------------------===//
// Tool Callbacks
//===----------------------------------------------------------------------===//

static llvm::Error instrumentationLoop(InstrumentationTask &IT,
                                       LiftedRepresentation &LR) {
  // Create a constant bool indicating the CountWavefrontLevel value
  auto *CountWavefrontLevelConstVal =
      llvm::ConstantInt::getBool(LR.getContext(), CountWavefrontLevel);
  unsigned int I = 0;
  for (auto &[_, MF] : LR.functions()) {
    for (auto &MBB : *MF) {
      for (auto &MI : MBB) {
        if (I >= InstrBeginInterval && I < InstrEndInterval) {
          bool IsScalar =
              llvm::SIInstrInfo::isSOP1(MI) || llvm::SIInstrInfo::isSOP2(MI) ||
              llvm::SIInstrInfo::isSOPK(MI) || llvm::SIInstrInfo::isSOPC(MI) ||
              llvm::SIInstrInfo::isSOPP(MI) || llvm::SIInstrInfo::isSMRD(MI);
          bool IsLaneAccess =
              MI.getOpcode() == llvm::AMDGPU::V_READFIRSTLANE_B32 ||
              MI.getOpcode() == llvm::AMDGPU::V_READLANE_B32 ||
              MI.getOpcode() == llvm::AMDGPU::V_WRITELANE_B32;
          if (IsScalar || IsLaneAccess)
            LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
                MI, LUTHIER_GET_HOOK_HANDLE(countInstructionsScalar)));
          else
            LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
                MI, LUTHIER_GET_HOOK_HANDLE(countInstructionsVector),
                {CountWavefrontLevelConstVal}));
        }
        I++;
      }
    }
  }
  return llvm::Error::success();
}

static void
instrumentAllFunctionsOfLR(const hsa::LoadedCodeObjectKernel &Kernel) {
  auto LR = lift(Kernel);
  LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());
  //  llvm::outs() << "Before instrumentation: \n";
  //  for (const auto &[FuncSymbol, MF] : LR->functions()) {
  //    MF->print(llvm::outs());
  //  }

  LUTHIER_REPORT_FATAL_ON_ERROR(
      instrumentAndLoad(Kernel, *LR, instrumentationLoop, "instr_count"));

  //   Uncomment this part to print out the final assembly
  //      llvm::SmallVector<
  //          std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>,
  //          1> SourceFiles;
  //
  //      auto InstrumentedLR = luthier::instrument(*LR, instrumentationLoop);
  //      LUTHIER_REPORT_FATAL_ON_ERROR(InstrumentedLR.takeError());
  //      llvm::outs() << "after instrumentation\n";
  //      for (const auto &[FuncSymbol, MF]: InstrumentedLR->get()->functions())
  //      {
  //        MF->print(llvm::outs());
  //      }
  //      LUTHIER_REPORT_FATAL_ON_ERROR(luthier::printLiftedRepresentation(
  //          **InstrumentedLR, SourceFiles,
  //          llvm::CodeGenFileType::AssemblyFile));
  //
  //      for (const auto &[LCO, SourceFile] : SourceFiles) {
  //        llvm::outs() << "Instrumented code object for LCO "
  //                     << llvm::format_hex(LCO.handle, 8) << "\n";
  //        llvm::outs() << SourceFile;
  //        llvm::outs() << "\n";
  //      }
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
                isKernelInstrumented(*KernelSymbol, "instr_count");
            LUTHIER_REPORT_FATAL_ON_ERROR(IsKernelInstrumented.takeError());
            if (!*IsKernelInstrumented) {

              instrumentAllFunctionsOfLR(*KernelSymbol);

              LUTHIER_REPORT_FATAL_ON_ERROR(luthier::overrideWithInstrumented(
                  DispatchPacket, "instr_count"));
              // Zero the counter
              Counter = 0;
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
            TotalNumInstructions += Counter;

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
                "({5}, {6}, {7}) Kernel Instruction "
                "Count: {8}, Total Instructions counted so far: {9}\n",
                NumKernelLaunched, KernelNameToBePrinted,
                DispatchPacket.grid_size_x, DispatchPacket.grid_size_y,
                DispatchPacket.grid_size_z, DispatchPacket.workgroup_size_x,
                DispatchPacket.workgroup_size_y,
                DispatchPacket.workgroup_size_z, Counter, TotalNumInstructions);
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
    llvm::outs() << "Instruction counter tool is launching.\n";
  } else {
    // Set the callback for when the HSA API table is captured
    hsa::setAtApiTableCaptureEvtCallback(atHsaApiTableCaptureCallBack);
    // Set the HSA API callback
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
  }
}

void atToolFini(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    llvm::outs() << "Total number of instructions counted: "
                 << TotalNumInstructions << "\n";
  }
}

} // namespace luthier
