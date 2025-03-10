//===-- InstrCount.hip - Instruction Count Example -------------*- C++ -*-===//
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
/// This file implements a sample instruction counter tool using Luthier.
/// The tool was inspired by NVBit's instruction counter.
//===----------------------------------------------------------------------===//
#include <SIInstrInfo.h>
#include <chrono>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <luthier/hip/HipError.h>
#include <luthier/hsa/DisableInterceptionScope.h>
#include <luthier/hsa/HsaError.h>
#include <luthier/llvm/CodeGenHelpers.h>
#include <luthier/llvm/EagerManagedStatic.h>
#include <luthier/llvm/streams.h>
#include <luthier/luthier.h>
#include <mutex>
#include <thread>

static decltype(std::chrono::high_resolution_clock::now()) T1{};

static unsigned int KernelTotalTime{0};

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-instr-counter-tool"

using namespace luthier;

//===----------------------------------------------------------------------===//
// Commandline arguments for the tool
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory *InstrCountToolOptionCategory;

static llvm::cl::opt<unsigned int> *InstrBeginInterval;

static llvm::cl::opt<unsigned int> *InstrEndInterval;

static llvm::cl::opt<unsigned int> *KernelBeginInterval;

static llvm::cl::opt<unsigned int> *KernelEndInterval;

static llvm::cl::opt<bool> *CountWavefrontLevel;

static llvm::cl::opt<bool> *DemangleKernelNames;

//===----------------------------------------------------------------------===//
// Global variables of the tool
//===----------------------------------------------------------------------===//
/// Name of the tool
static std::string *ToolName{nullptr};

/// Number of kernels launched so far
static uint32_t NumKernelLaunched = 0;

/// Total number of instructions, incremented by the device variable
/// \c Counter every time a kernel finishes executing
static uint64_t TotalNumInstructions = 0;

/// Profiling variable on devices Kernel instruction counter
__attribute__((device)) uint64_t Counter;

/// Keeps track of whether we are in the interval to profile or not
static bool ActiveRegion = true;

/// A Mutex, used to protect the Counter value and ActiveRegion, effectively
/// makes all kernels launch sequentially
std::recursive_mutex Mutex;

/// The current kernel being instrumented
static const luthier::hsa::LoadedCodeObjectKernel *KernelBeingInstrumented{
    nullptr};

/// This is set to true if the original dispatch packet didn't have a signal
/// field set. In this scenario, the tool creates a signal so that it can
/// wait on the end of the kernel, and must destroy the signal after waiting
/// on the kernel
static bool MustDestroySignalAfterLaunch{false};

/// The value of the signal to wait for after kernel launches
static hsa_signal_value_t SignalValue{0};

//===----------------------------------------------------------------------===//
// Hook definitions
//===----------------------------------------------------------------------===//

/// If the Luthier tool contains a hook, then this macro must be used to mark it
/// for Luthier's Tool Executable Loader
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
  (void)luthier::sAtomicAdd(&Counter, 1UL);
}

LUTHIER_EXPORT_HOOK_HANDLE(countInstructionsScalar);

//===----------------------------------------------------------------------===//
// Tool Callbacks
//===----------------------------------------------------------------------===//

static llvm::Error instrumentationLoop(InstrumentationTask &IT,
                                       LiftedRepresentation &LR) {
  // Create a constant bool indicating the CountWavefrontLevel value
  auto *CountWavefrontLevelConstVal =
      llvm::ConstantInt::getBool(LR.getContext(), *CountWavefrontLevel);
  unsigned int I = 0;
  return LR.iterateAllDefinedFunctionTypes(
      [&](const hsa::LoadedCodeObjectSymbol &Sym,
          llvm::MachineFunction &MF) -> llvm::Error {
        for (auto &MBB : MF) {
          for (auto &MI : MBB) {
            if (I >= *InstrBeginInterval && I < *InstrEndInterval) {
              bool IsScalar = luthier::isScalar(MI);
              bool IsLaneAccess = luthier::isLaneAccess(MI);
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
        return llvm::Error::success();
      });
}

static llvm::Error instrumentKernel(const hsa::LoadedCodeObjectKernel &Kernel) {
  auto LR = lift(Kernel);
  LUTHIER_RETURN_ON_ERROR(LR.takeError());
  LUTHIER_RETURN_ON_ERROR(
      instrumentAndLoad(Kernel, *LR, instrumentationLoop, "instr_count"));
  return llvm::Error::success();
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
      if (auto *DispatchPacket = Packet.asKernelDispatch()) {
        if (Phase == luthier::API_EVT_PHASE_BEFORE) {
          Mutex.lock();
          /// If we are entering to a kernel launch:
          /// 1. Lock the mutex to prevent multiple kernels to run concurrently
          /// 2. Instrument the kernel if no already instrumented
          /// 3. Select whether the instrumented kernel will run or not
          /// 4. Reset the managed Counter variable

          // Get the kernel that is about to get launched
          auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                  DispatchPacket->kernel_object)
                                  ->getLoadedCodeObjectKernelSymbol();
          LUTHIER_REPORT_FATAL_ON_ERROR(KernelSymbol.takeError());

          // Keep track of the kernel being instrumented for the exit callback
          KernelBeingInstrumented = KernelSymbol->release();

          auto KernelName = KernelBeingInstrumented->getName();
          LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());
          std::string KernelNameToBePrinted = *DemangleKernelNames
                                                  ? llvm::demangle(*KernelName)
                                                  : std::string(*KernelName);
          // Get the packet's completion signal and its value

          // Create a signal if the application didn't create one for this
          // packet
          if (DispatchPacket->completion_signal.handle == 0) {
            LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
                hsa::getHsaApiTable().amd_ext_->hsa_amd_signal_create_fn(
                    SignalValue, 0, nullptr, 0,
                    &DispatchPacket->completion_signal)));
            MustDestroySignalAfterLaunch = true;
          } else {
            MustDestroySignalAfterLaunch = false;
          }
          SignalValue =
              hsa::getHsaApiTable().core_->hsa_signal_load_scacquire_fn(
                  DispatchPacket->completion_signal);

          if (NumKernelLaunched >= *KernelBeginInterval &&
              NumKernelLaunched < *KernelEndInterval) {
            ActiveRegion = true;
          } else {
            ActiveRegion = false;
          }
          if (ActiveRegion) {
            luthier::errs() << "Kernel is being instrumented.\n";

            auto IsKernelInstrumented =
                isKernelInstrumented(*KernelBeingInstrumented, "instr_count");
            LUTHIER_REPORT_FATAL_ON_ERROR(IsKernelInstrumented.takeError());
            if (!*IsKernelInstrumented) {
              LUTHIER_REPORT_FATAL_ON_ERROR(
                  instrumentKernel(*KernelBeingInstrumented));
            }
            LUTHIER_REPORT_FATAL_ON_ERROR(luthier::overrideWithInstrumented(
                *DispatchPacket, "instr_count"));
            // Zero the instruction counter device variable
            {
              luthier::hsa::DisableUserInterceptionScope Scope;
              uint64_t CounterHost = 0;
              uint64_t *CounterDevice;
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HIP_SUCCESS_CHECK(
                  hip::getSavedDispatchTable().hipGetSymbolAddress_fn(
                      (void **)&CounterDevice, &Counter)));
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
                  hsa::getHsaApiTable().core_->hsa_memory_copy_fn(
                      CounterDevice, &CounterHost, sizeof(CounterHost))));
            }
          }
          T1 = std::chrono::high_resolution_clock::now();
        } else {
          constexpr auto Timeout =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::seconds{300})
                  .count();
          auto CurrentSignalValue =
              hsa::getHsaApiTable().core_->hsa_signal_wait_relaxed_fn(
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
            LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
                hsa::getHsaApiTable().core_->hsa_signal_destroy_fn(
                    DispatchPacket->completion_signal)));
          }
          auto T2 = std::chrono::high_resolution_clock::now();
          KernelTotalTime +=
              std::chrono::duration_cast<std::chrono::microseconds>(T2 - T1)
                  .count();
          if (ActiveRegion) {
            luthier::errs() << "Kernel is being waited on.\n";
            /// If we are exiting a kernel launch:
            /// 1. Wait on the signal of the kernel being launched
            /// 2. Get the number of wave fronts in the kernel
            /// 3. Print the counter for this kernel
            /// 4. Release the lock

            // Get the kernel that was launched

            auto KernelName = KernelBeingInstrumented->getName();
            LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());
            std::string KernelNameToBePrinted =
                *DemangleKernelNames ? llvm::demangle(*KernelName)
                                     : std::string(*KernelName);
            uint64_t CounterHost = 0;
            {
              luthier::hsa::DisableUserInterceptionScope Scope;
              uint64_t *CounterDevice;
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HIP_SUCCESS_CHECK(
                  hip::getSavedDispatchTable().hipGetSymbolAddress_fn(
                      (void **)&CounterDevice, &Counter)));
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
                  hsa::getHsaApiTable().core_->hsa_memory_copy_fn(
                      &CounterHost, CounterDevice, sizeof(Counter))));
              TotalNumInstructions += CounterHost;
            }

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
                "({5}, {6}, {7}) Kernel Instruction "
                "Count: {8}, Total Instructions counted so far: {9}\n",
                NumKernelLaunched, KernelNameToBePrinted,
                DispatchPacket->grid_size_x, DispatchPacket->grid_size_y,
                DispatchPacket->grid_size_z, DispatchPacket->workgroup_size_x,
                DispatchPacket->workgroup_size_y,
                DispatchPacket->workgroup_size_z, CounterHost,
                TotalNumInstructions);
            delete KernelBeingInstrumented;
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
    InstrCountToolOptionCategory =
        new llvm::cl::OptionCategory("Instruction Count Tool Options");

    InstrBeginInterval = new llvm::cl::opt<unsigned int>(
        "instr-begin-interval",
        llvm::cl::desc("Beginning of the instruction interval where to "
                       "apply instrumentation, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden,
        llvm::cl::cat(*InstrCountToolOptionCategory));

    InstrEndInterval = new llvm::cl::opt<unsigned int>(
        "instr-end-interval",
        llvm::cl::desc("End of the instruction interval where to "
                       "apply instrumentation, exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned int>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(*InstrCountToolOptionCategory));

    KernelBeginInterval = new llvm::cl::opt<unsigned int>(
        "kernel-begin-interval",
        llvm::cl::desc("Beginning of the kernel interval to apply "
                       "instrumentation, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden,
        llvm::cl::cat(*InstrCountToolOptionCategory));

    KernelEndInterval = new llvm::cl::opt<unsigned int>(
        "kernel-end-interval",
        llvm::cl::desc(
            "End of the kernel interval to apply instrumentation, exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned int>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(*InstrCountToolOptionCategory));

    CountWavefrontLevel = new llvm::cl::opt<bool>(
        "count-wavefront-level",
        llvm::cl::desc(
            "Whether to count number of instructions at the wavefront "
            "level or thread level"),
        llvm::cl::init(false), llvm::cl::NotHidden,
        llvm::cl::cat(*InstrCountToolOptionCategory));

    DemangleKernelNames = new llvm::cl::opt<bool>(
        "demangle-kernel-names",
        llvm::cl::desc("Whether to demangle kernel names before printing"),
        llvm::cl::init(true), llvm::cl::NotHidden,
        llvm::cl::cat(*InstrCountToolOptionCategory));

    ToolName = new std::string{"luthier instruction counter tool"};
  } else {
    luthier::errs() << "Instruction counter tool is launching.\n";
    // Set the callback for when the HSA API table is captured
    hsa::setAtApiTableCaptureEvtCallback(atHsaApiTableCaptureCallBack);
    // Set the HSA API callback
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
  }
}

void atToolFini(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    delete InstrCountToolOptionCategory;

    delete InstrBeginInterval;

    delete InstrEndInterval;

    delete KernelBeginInterval;

    delete KernelEndInterval;

    delete CountWavefrontLevel;

    delete DemangleKernelNames;

    delete ToolName;

    luthier::errs() << "Total number of counted instructions: "
                    << TotalNumInstructions << ".\n";
  }
}

llvm::StringRef getToolName() { return *ToolName; }

} // namespace luthier
