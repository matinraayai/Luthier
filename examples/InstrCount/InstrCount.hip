//===-- InstrCount.hip - Instruction Count Example ------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a sample instruction counter tool using Luthier.
/// The tool was inspired by NVBit's instruction counter.
//===----------------------------------------------------------------------===//
#include <luthier/luthier.h>
#include <mutex>
// undef ICMP_NE macro that was ill-defined in HIP headers
#undef ICMP_NE
#include <llvm/IR/Constants.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_OPERAND_ENUM
#include "AMDGPUGenInstrInfo.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-instr-counter-tool"

using namespace luthier;

//===----------------------------------------------------------------------===//
// Commandline arguments for the tool
//===----------------------------------------------------------------------===//

/// Define an option
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
                     llvm::cl::init(9), llvm::cl::NotHidden,
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

/// Number of kernels launched so far
static uint32_t NumKernelLaunched = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
static uint64_t TotalNumInstructions = 0;

/// Kernel instruction counter
__attribute__((managed)) uint64_t Counter = 0;

/// Keeps track of whether we are in the interval to profile or not
static bool ActiveRegion = true;

/// Keeps track of the value of the signal of the kernel launch before it
/// is submitted
static hsa_signal_value_t SignalVal = 0;

static bool MustDestroySignalAfterLaunch = false;

/// A Mutex, used to protect the Counter value and ActiveRegion
std::mutex Mutex;

/// If the Luthier tool contains a hook, then this macro must be used to mark it
/// for Luthier's Tool Executable Manager
MARK_LUTHIER_DEVICE_MODULE

// TODO: Can this be done instead with only scalar instruction?
LUTHIER_HOOK_ANNOTATE countInstructions(bool CountWaveFrontLevel) {
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask =
      luthier::readReg<unsigned long long int>(llvm::AMDGPU::EXEC);
  // Get the position of the thread in the current wavefront
  const uint32_t ThreadPosition = getThreadPosition();
  // Get the first active thread id inside this wavefront
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  // Get the number of active threads in this wavefront
  uint32_t NumActiveThreads = __popcll(ExecMask);

  // Have only the first active thread perform the atomic add
  if (FirstActiveThreadId == ThreadPosition) {
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

LUTHIER_EXPORT_HOOK_HANDLE(countInstructions);

static llvm::Error instrumentationLoop(InstrumentationTask &IT,
                                       LiftedRepresentation &LR) {
  // Create a constant bool indicating the CountWavefrontLevel value
  auto *CountWavefrontLevelConstVal = llvm::ConstantInt::getBool(
      *LR.getContext().getContext(), CountWavefrontLevel);
  unsigned int I = 0;
  for (auto &[_, MF] : LR.functions()) {
    for (auto &MBB : *MF) {
      for (auto &MI : MBB) {
        if (I >= InstrBeginInterval && I < InstrEndInterval) {
          LUTHIER_RETURN_ON_ERROR(IT.insertHookBefore(
              MI, LUTHIER_GET_HOOK_HANDLE(countInstructions),
              {CountWavefrontLevelConstVal}));
        }
        I++;
      }
    }
  }
  return llvm::Error::success();
}

static void instrumentAllFunctionsOfExecutable(hsa_executable_t Exec) {
  auto LR = lift(Exec);
  LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());

  //  LUTHIER_REPORT_FATAL_ON_ERROR(
  //      instrumentAndLoad(Exec, *LR, instrumentationLoop, "instr_count"));

  llvm::SmallVector<
      std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>, 1>
      SourceFiles;

  auto InstrumentedLR = luthier::instrument(*LR, instrumentationLoop);
  LUTHIER_REPORT_FATAL_ON_ERROR(InstrumentedLR.takeError());
  LUTHIER_REPORT_FATAL_ON_ERROR(luthier::printLiftedRepresentation(
      **InstrumentedLR, SourceFiles, llvm::CodeGenFileType::AssemblyFile));

  for (const auto &[LCO, SourceFile] : SourceFiles) {
    llvm::outs() << "Instrumented code object for LCO "
                 << llvm::format_hex(LCO.handle, 8) << "\n";
    llvm::outs() << SourceFile;
    llvm::outs() << "\n";
  }
}

static void atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                     luthier::ApiEvtPhase Phase, luthier::hsa::ApiEvtID ApiID) {
  if (ApiID == luthier::hsa::HSA_API_EVT_ID_hsa_queue_packet_submit) {
    auto Packets = CBData->hsa_queue_packet_submit.packets;
    for (unsigned int I = 0; I < CBData->hsa_queue_packet_submit.pkt_count;
         I++) {
      auto &Packet = Packets[I];
      hsa_packet_type_t PacketType = Packet.getPacketType();

      if (PacketType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        auto &DispatchPacket = Packet.asKernelDispatch();
        // Get the kernel that is about to get launched + its completion signal
        // value
        auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                DispatchPacket.kernel_object)
                                ->getHsaExecutableSymbol();
        // Create a signal if the application didn't create one for this packet
        if (DispatchPacket.completion_signal.handle == 0) {
          hsa::getHsaApiTable().core_->hsa_signal_create_fn(
              0, 0, {}, &DispatchPacket.completion_signal);
          MustDestroySignalAfterLaunch = true;
        } else {
          MustDestroySignalAfterLaunch = false;
        }
        SignalVal = hsa::getHsaApiTable().core_->hsa_signal_load_relaxed_fn(
            DispatchPacket.completion_signal);

        LUTHIER_REPORT_FATAL_ON_ERROR(KernelSymbol.takeError());
        if (Phase == luthier::API_EVT_PHASE_BEFORE) {

          /// If we are entering to a kernel launch:
          /// 1. Lock the mutex to prevent multiple kernels to run concurrently
          /// 2. Instrument the kernel if no already instrumented
          /// 3. Select whether the instrumented kernel will run or not
          /// 4. Reset the managed Counter variable
          Mutex.lock();
          auto IsKernelInstrumented =
              isKernelInstrumented(*KernelSymbol, "instr_count");
          LUTHIER_REPORT_FATAL_ON_ERROR(IsKernelInstrumented.takeError());
          if (!*IsKernelInstrumented) {
            // Get the executable associated with this kernel
            auto Exec = hsa::getExecutableOfSymbol(*KernelSymbol);
            LUTHIER_REPORT_FATAL_ON_ERROR(Exec.takeError());

            instrumentAllFunctionsOfExecutable(*Exec);
          }

          if (NumKernelLaunched >= KernelBeginInterval &&
              NumKernelLaunched < KernelEndInterval) {
            ActiveRegion = true;
          } else {
            ActiveRegion = false;
          }

          if (ActiveRegion) {
            LUTHIER_REPORT_FATAL_ON_ERROR(luthier::overrideWithInstrumented(
                DispatchPacket, "instr_count"));
          }
          Counter = 0;
        } else {
          /// If we are exiting a kernel launch:
          /// 1. Wait on the signal of the kernel being launched
          /// 2. Get the number of wave fronts in the kernel
          /// 3. Print the counter for this kernel
          /// 4. Release the lock
          hsa::getHsaApiTable().core_->hsa_signal_wait_relaxed_fn(
              DispatchPacket.completion_signal, HSA_SIGNAL_CONDITION_EQ,
              SignalVal, std::numeric_limits<uint64_t>::max(),
              HSA_WAIT_STATE_BLOCKED);

          TotalNumInstructions += Counter;

          auto KernelName = hsa::getSymbolName(*KernelSymbol);
          LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());

          llvm::outs() << llvm::formatv(
              "Kernel {0} - {1}: Grid Dims: ({2}, {3}, {4}) Kernel Instruction "
              "Count: {5}, Total Instructions counted so far: {6}\n",
              NumKernelLaunched++, *KernelName, DispatchPacket.grid_size_x,
              DispatchPacket.grid_size_y, DispatchPacket.grid_size_z, Counter,
              TotalNumInstructions);
          if (MustDestroySignalAfterLaunch) {
            hsa::getHsaApiTable().core_->hsa_signal_destroy_fn(
                DispatchPacket.completion_signal);
          }
          Mutex.unlock();
        }
      }
    }
  }
}

namespace luthier {

static void atHsaApiTableCaptureCallBack(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    if (auto Err = luthier::hsa::enableHsaApiEvtIDCallback(
            hsa::HSA_API_EVT_ID_hsa_queue_packet_submit))
      llvm::report_fatal_error(std::move(Err), true);
  } else {
    // Parse the tool arguments
    auto Argv = "";
    llvm::cl::ParseCommandLineOptions(0, &Argv,
                                      "Kernel Instruction Counter Tool",
                                      &llvm::errs(), "INSTR_COUNT_ARGS");
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

void atFinalization(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    llvm::outs() << "Total number of instructions counted: "
                 << TotalNumInstructions << "\n";
  }
}

} // namespace luthier
