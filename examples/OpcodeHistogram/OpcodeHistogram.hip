//===-- OpcodeHistogram.hip - Opcode Histogram Example ----------*- C++ -*-===//
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
/// This file implements a sample Opcode Histogram using Luthier.
/// The tool was inspired by NVBit's opcode histogram tool.
//===----------------------------------------------------------------------===//
#include <SIInstrInfo.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <luthier/hip/HipError.h>
#include <luthier/hsa/DisableInterceptionScope.h>
#include <luthier/hsa/HsaError.h>
#include <luthier/llvm/CodeGenHelpers.h>
#include <luthier/llvm/streams.h>
#include <luthier/luthier.h>
#include <mutex>
#include <thread>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-opcode-hist-tool"

using namespace luthier;

//===----------------------------------------------------------------------===//
// Commandline arguments for the tool
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory *OpcodeHistogramToolOptionCategory;

static llvm::cl::opt<unsigned int> *InstrBeginInterval;

static llvm::cl::opt<unsigned int> *InstrEndInterval;

static llvm::cl::opt<unsigned int> *KernelBeginInterval;

static llvm::cl::opt<unsigned int> *KernelEndInterval;

static llvm::cl::opt<bool> *CountWavefrontLevel;

static llvm::cl::opt<bool> *DemangleKernelNames;

/// Name of the tool
static std::string *ToolName{nullptr};

__attribute__((
    device)) uint64_t KernelHistogram[llvm::AMDGPU::INSTRUCTION_LIST_END];

static llvm::StringMap<uint64_t> *GlobalHistogram{nullptr};

/// Total number of instructions counted
static uint64_t TotalNumInstructions = 0;

/// Number of kernels launched so far
static uint32_t NumKernelLaunched = 0;

static bool ActiveRegion = true;

bool MustDestroySignalAfterLaunch = false;

const hsa::LoadedCodeObjectKernel *KernelBeingInstrumented{nullptr};

hsa_signal_value_t SignalValue{0};

const llvm::TargetInstrInfo *TII{nullptr};

/// Mutex to protect the static device buffer
std::recursive_mutex Mutex;

MARK_LUTHIER_DEVICE_MODULE

LUTHIER_HOOK_ANNOTATE countOpcodeHistogramVector(bool CountWaveFrontLevel,
                                                 int OpcodeNumber) {
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
        atomicAdd(&KernelHistogram[OpcodeNumber], 1);
      }
    } else {
      atomicAdd(&KernelHistogram[OpcodeNumber], NumActiveThreads);
    }
  }
}

LUTHIER_EXPORT_HOOK_HANDLE(countOpcodeHistogramVector);

LUTHIER_HOOK_ANNOTATE countOpcodeHistogramScalar(int OpcodeNumber) {
  luthier::sAtomicAdd(&KernelHistogram[OpcodeNumber], 1UL);
}

LUTHIER_EXPORT_HOOK_HANDLE(countOpcodeHistogramScalar);

static llvm::Error instrumentationLoop(InstrumentationTask &IT,
                                       LiftedRepresentation &LR) {
  // Create a constant bool indicating the CountWavefrontLevel value
  auto *CountWavefrontLevelConstVal =
      llvm::ConstantInt::getBool(LR.getContext(), CountWavefrontLevel);
  unsigned int I = 0;
  return LR.iterateAllDefinedFunctionTypes(
      [&](const hsa::LoadedCodeObjectSymbol &Sym,
          llvm::MachineFunction &MF) -> llvm::Error {
        for (auto &MBB : MF) {
          for (auto &MI : MBB) {
            if (I >= *InstrBeginInterval && I < *InstrEndInterval) {
              // Get the opcode number, llvm const int
              auto *OpcodeNumber = llvm::ConstantInt::get(
                  llvm::Type::getInt32Ty(LR.getContext()), MI.getOpcode(),
                  false);
              bool IsScalar = luthier::isScalar(MI);
              bool IsLaneAccess = luthier::isLaneAccess(MI);
              if (IsScalar || IsLaneAccess)
                LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
                    MI, LUTHIER_GET_HOOK_HANDLE(countOpcodeHistogramScalar),
                    {OpcodeNumber}));
              else
                LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
                    MI, LUTHIER_GET_HOOK_HANDLE(countOpcodeHistogramVector),
                    {CountWavefrontLevelConstVal, OpcodeNumber}));
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
      KernelSymbol, *LR, instrumentationLoop, "opcode histogram"));
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

          KernelBeingInstrumented = &(**KernelSymbol);

          auto KernelName = KernelBeingInstrumented->getName();
          LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());
          std::string KernelNameToBePrinted = DemangleKernelNames
                                                  ? llvm::demangle(*KernelName)
                                                  : std::string(*KernelName);
          // Get the packet's completion signal and its value

          // Create a signal if the application didn't create one for this
          // packet
          if (DispatchPacket->completion_signal.handle == 0) {
            if (hsa::getHsaApiTable().amd_ext_->hsa_amd_signal_create_fn(
                    SignalValue, 0, nullptr, 0,
                    &DispatchPacket->completion_signal) != HSA_STATUS_SUCCESS)
              llvm::report_fatal_error("Failed to create a signal for the "
                                       "dispatch packet.");
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
                isKernelInstrumented(**KernelSymbol, "opcode histogram");
            LUTHIER_REPORT_FATAL_ON_ERROR(IsKernelInstrumented.takeError());

            auto LR = lift(**KernelSymbol);
            LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());
            TII = LR->getKernelMF().getSubtarget().getInstrInfo();

            if (!*IsKernelInstrumented) {
              instrumentAllFunctionsOfLR(**KernelSymbol);
            }
            LUTHIER_REPORT_FATAL_ON_ERROR(luthier::overrideWithInstrumented(
                *DispatchPacket, "opcode histogram"));
            // Zero the instruction counter device variable
            {
              luthier::hsa::DisableUserInterceptionScope Scope;
              uint64_t HistogramHost[llvm::AMDGPU::INSTRUCTION_LIST_END]{0};
              uint64_t *HistogramDevice;
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HIP_SUCCESS_CHECK(
                  hip::getSavedDispatchTable().hipGetSymbolAddress_fn(
                      (void **)&HistogramDevice, &KernelHistogram)));
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
                  hsa::getHsaApiTable().core_->hsa_memory_copy_fn(
                      HistogramDevice, &HistogramHost, sizeof(HistogramHost))));
            }
          }
        } else {
          luthier::errs() << "Active region in the exit callback: "
                          << ActiveRegion << "\n";
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
            if (hsa::getHsaApiTable().core_->hsa_signal_destroy_fn(
                    DispatchPacket->completion_signal) != HSA_STATUS_SUCCESS) {
              llvm::report_fatal_error(
                  "Failed to destroy the completed kernel's"
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
            std::string KernelNameToBePrinted =
                DemangleKernelNames ? llvm::demangle(*KernelName)
                                    : std::string(*KernelName);
            // Copy the histogram to the global histogram
            uint64_t HistogramHost[llvm::AMDGPU::INSTRUCTION_LIST_END]{0};
            // Zero the instruction counter device variable
            {
              luthier::hsa::DisableUserInterceptionScope Scope;
              uint64_t *HistogramDevice;
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HIP_SUCCESS_CHECK(
                  hip::getSavedDispatchTable().hipGetSymbolAddress_fn(
                      (void **)&HistogramDevice, &KernelHistogram)));
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
                  hsa::getHsaApiTable().core_->hsa_memory_copy_fn(
                      &HistogramHost, HistogramDevice, sizeof(HistogramHost))));
            }

            uint64_t InstsCountedForThisKernel = 0;
            for (int i = 0; i < llvm::AMDGPU::INSTRUCTION_LIST_END; i++) {
              if (HistogramHost[i] != 0) {
                auto OpcodeName = TII->getName(i);
                if (GlobalHistogram->contains(OpcodeName))
                  (*GlobalHistogram)[OpcodeName] += HistogramHost[i];
                else
                  GlobalHistogram->insert({OpcodeName, HistogramHost[i]});
                InstsCountedForThisKernel += HistogramHost[i];
              }
            }
            TotalNumInstructions += InstsCountedForThisKernel;

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
                NumKernelLaunched++, KernelNameToBePrinted,
                DispatchPacket->grid_size_x, DispatchPacket->grid_size_y,
                DispatchPacket->grid_size_z, DispatchPacket->workgroup_size_x,
                DispatchPacket->workgroup_size_y,
                DispatchPacket->workgroup_size_z, InstsCountedForThisKernel,
                TotalNumInstructions);

            luthier::errs()
                << "Histogram for kernel " << KernelNameToBePrinted << ":\n";
            for (auto Opcode = 0; Opcode < llvm::AMDGPU::INSTRUCTION_LIST_END;
                 ++Opcode) {
              if (HistogramHost[Opcode] != 0) {
                std::string OpcodeName = TII->getName(Opcode).str();
                luthier::errs() << llvm::formatv("  {0} = {1}\n",
                                                 OpcodeName, // index
                                                 HistogramHost[Opcode]);
              }
            }
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

llvm::StringRef getToolName() { return *ToolName; }

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    luthier::errs() << "Opcode Histogram tool is launching.\n";

    /// Define an option category for the instruction count tool
    OpcodeHistogramToolOptionCategory =
        new llvm::cl::OptionCategory("Opcode Histogram Tool Options");

    InstrBeginInterval = new llvm::cl::opt<unsigned int>(
        "instr-begin-interval",
        llvm::cl::desc("Beginning of the instruction interval where to "
                       "apply instrumentation, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden,
        llvm::cl::cat(*OpcodeHistogramToolOptionCategory));

    InstrEndInterval = new llvm::cl::opt<unsigned int>(
        "instr-end-interval",
        llvm::cl::desc("End of the instruction interval where to apply "
                       "instrumentation, inclusive"),
        llvm::cl::init(std::numeric_limits<unsigned int>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(*OpcodeHistogramToolOptionCategory));

    KernelBeginInterval = new llvm::cl::opt<unsigned int>(
        "kernel-start-interval",
        llvm::cl::desc("Beginning of the kernel interval to apply "
                       "instrumentation, inclusive"),
        llvm::cl::init(0), llvm::cl::NotHidden,
        llvm::cl::cat(*OpcodeHistogramToolOptionCategory));

    KernelEndInterval = new llvm::cl::opt<unsigned int>(
        "kernel-end-interval",
        llvm::cl::desc(
            "End of the kernel interval to apply instrumentation, exclusive"),
        llvm::cl::init(std::numeric_limits<unsigned int>::max()),
        llvm::cl::NotHidden, llvm::cl::cat(*OpcodeHistogramToolOptionCategory));

    CountWavefrontLevel = new llvm::cl::opt<bool>(
        "count-wavefront-level",
        llvm::cl::desc("Count instructions at the wavefront level"),
        llvm::cl::init(false), llvm::cl::NotHidden,
        llvm::cl::cat(*OpcodeHistogramToolOptionCategory));

    DemangleKernelNames = new llvm::cl::opt<bool>(
        "demangle-kernel-names",
        llvm::cl::desc("Whether to demangle kernel names before printing"),
        llvm::cl::init(true), llvm::cl::NotHidden,
        llvm::cl::cat(*OpcodeHistogramToolOptionCategory));

    ToolName = new std::string{"luthier opcode histogram tool"};

    GlobalHistogram = new llvm::StringMap<uint64_t>();
  } else {
    // Set the callback for when the HSA API table is captured
    hsa::setAtApiTableCaptureEvtCallback(atHsaApiTableCaptureCallBack);
    // Set the HSA API callback
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
  }
}

void atToolFini(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    luthier::errs() << "Instruction Opcode Results: \n";
    for (const auto &[OpcodeName, Count] : *GlobalHistogram) {
      luthier::errs() << llvm::formatv("  {0} = {1}\n",
                                       OpcodeName, // index
                                       Count);
    }
    delete InstrBeginInterval;

    delete InstrEndInterval;

    delete KernelBeginInterval;

    delete KernelEndInterval;

    delete CountWavefrontLevel;

    delete DemangleKernelNames;

    delete OpcodeHistogramToolOptionCategory;

    delete ToolName;

    delete GlobalHistogram;
    luthier::errs() << "Total number of instructions counted: "
                    << TotalNumInstructions << "\n";
  }
}

} // namespace luthier
