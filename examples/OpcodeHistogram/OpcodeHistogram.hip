
//===-- OpcodeHistogram.hip - Opcode Histogram Example ------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a sample Opcode Histogram using Luthier.
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
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/ADT/APFloat.h>

#define GET_REGINFO_ENUM
// #include "AMDGPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_OPERAlsND_ENUM

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-opcodehist-tool"

using namespace luthier;

//===----------------------------------------------------------------------===//
// Commandline arguments for the tool
//===----------------------------------------------------------------------===//

/// Define an option category for the instruction count tool
/* global control variables for this tool */
llvm::cl::OptionCategory
    OpcodeHistogramToolOptionCategory("Opcode Histogram Tool Options");

static llvm::cl::opt<unsigned int> InstrBeginInterval(
    "instr-begin-interval",
    llvm::cl::desc("Beginning of the instruction interval where to "
                   "apply instrumentation, inclusive"),
    llvm::cl::init(0), llvm::cl::NotHidden,
    llvm::cl::cat(OpcodeHistogramToolOptionCategory));

static llvm::cl::opt<unsigned int> InstrEndInterval(
    "instr-end-interval",
    llvm::cl::desc("End of the instruction interval where to apply "
                   "instrumentation, inclusive"),
    llvm::cl::init(std::numeric_limits<unsigned int>::max()), llvm::cl::NotHidden,
    llvm::cl::cat(OpcodeHistogramToolOptionCategory));

static llvm::cl::opt<unsigned int>KernelBeginInterval(
    "kernel-start-interval",
    llvm::cl::desc(
        "Beginning of the kernel interval to apply instrumentation, inclusive"),
    llvm::cl::init(0),
    llvm::cl::NotHidden, llvm::cl::cat(OpcodeHistogramToolOptionCategory));

static llvm::cl::opt<unsigned int> KernelEndInterval(
    "kernel-end-interval",
    llvm::cl::desc(
        "End of the kernel interval to apply instrumentation, exclusive"),
    llvm::cl::init(std::numeric_limits<unsigned int>::max()),
    llvm::cl::NotHidden, llvm::cl::cat(OpcodeHistogramToolOptionCategory));

static llvm::cl::opt<bool> CountWavefrontLevel(
    "count-wavefront-level",
    llvm::cl::desc("Count instructions at the wavefront level"),
    llvm::cl::init(true),
    llvm::cl::NotHidden,
    llvm::cl::cat(OpcodeHistogramToolOptionCategory));

static llvm::cl::opt<bool> ExcludePredOff(
    "exclude-pred-off",
    llvm::cl::desc("Exclude instructions with predicate off"),
    llvm::cl::init(false),
    llvm::cl::NotHidden,
    llvm::cl::cat(OpcodeHistogramToolOptionCategory));

static llvm::cl::opt<bool> DemangleKernelNames(
    "demangle-kernel-names",
    llvm::cl::desc("Whether to demangle kernel names before printing"),
    llvm::cl::init(true), llvm::cl::NotHidden,
    llvm::cl::cat(OpcodeHistogramToolOptionCategory));
/* kernel id counter, maintained in system memory */
uint32_t KernelId = 0;

#define MAX_OPCODES (64 * 1024)
__managed__ uint64_t KernelHistogram[MAX_OPCODES];
__managed__ uint64_t GlobalHistogram[MAX_OPCODES];

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
static uint64_t TotalNumInstructions = 0;

/// Number of kernels launched so far
static uint32_t NumKernelLaunched = 0;

static bool ActiveRegion = true;
/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
std::mutex Mutex;

MARK_LUTHIER_DEVICE_MODULE

//Record OPCode Histogram
LUTHIER_HOOK_ANNOTATE countOpcodeHistogramVector(bool CountWaveFrontLevel, int OpcodeNumber) {
    //Check the opcode is in the map or not, if not, add it to the map
 
    //Get the exec mask of the wavefront
    unsigned long long int ExecMask = 
        luthier::readReg<unsigned long long int>(llvm::AMDGPU::EXEC);
    // Get the position of the thread in the current wavefront (1-index)
    const uint32_t LaneId = __lane_id() + 1;
    //Get the first active thread id inside this wavefront
    uint32_t FirstActiveThreadId = __ffsll(ExecMask);
    //Get the number of active threads in this wavefront
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
  // Get the exec mask of the wavefront
  unsigned long long int ExecMask =
      luthier::readReg<unsigned long long int>(llvm::AMDGPU::EXEC);
  // Overwrite the exec mask with one so that only a single thread is active
  luthier::writeReg(llvm::AMDGPU::EXEC, 1);
  //Check the opcode is in the map or not, if not, add it to the map
  // if (InstrOpcodeToNumMap.find(OpcodeName) == InstrOpcodeToNumMap.end()) {
  //     InstrOpcodeToNumMap[OpcodeName] = InstrOpcodeToNumMap.size();
  // }
  // //Get the opcode number
  // int OpcodeNumber = InstrOpcodeToNumMap[OpcodeName];
  // Increment the counter for this opcode
  atomicAdd(&KernelHistogram[OpcodeNumber], 1);
  // Restore the exec mask
  luthier::writeReg(llvm::AMDGPU::EXEC, ExecMask);
}
LUTHIER_EXPORT_HOOK_HANDLE(countOpcodeHistogramScalar);

static llvm::Error instrumentationLoop(InstrumentationTask &IT,
                                        LiftedRepresentation &LR){
      // Create a constant bool indicating the CountWavefrontLevel value
  auto *CountWavefrontLevelConstVal = llvm::ConstantInt::getBool(
      *LR.getContext().getContext(), CountWavefrontLevel);
  unsigned int I = 0;
  // const auto llvm::TargetInstrInfo *TII;
  for (auto &[_, MF] : LR.functions()) {
    for (auto &MBB : *MF) {
        const auto &STI = MF->getSubtarget();
        const llvm::TargetInstrInfo *TII = STI.getInstrInfo();
      for (auto &MI : MBB) {
        if (I >= InstrBeginInterval && I < InstrEndInterval) {
          std::string OpcodeName = TII -> getName(MI.getOpcode()).str();
          //Get the opcode number, llvm const int 
          auto * OpcodeNumber = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*LR.getContext().getContext()), MI.getOpcode(), false);
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
  }
  return llvm::Error::success();
}

static void instrumentAllFunctionsOfExecutable(hsa_executable_t Exec) {
  auto LR = lift(Exec);
  LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());

  LUTHIER_REPORT_FATAL_ON_ERROR(
      instrumentAndLoad(Exec, *LR, instrumentationLoop, "OpcodeHistogram"));
}

static void
atKernelCompletion(hsa_signal_t Signal, hsa_signal_value_t Value,
                   bool MustDestroyAfterWait,
                   hsa_kernel_dispatch_packet_t SubmittedDispatchPacket) {
  constexpr auto Timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::seconds{300})
                               .count();
  auto SignalValue = hsa::getHsaApiTable().core_->hsa_signal_wait_scacquire_fn(
      Signal, HSA_SIGNAL_CONDITION_LT, Value, Timeout, HSA_WAIT_STATE_ACTIVE);
  if (SignalValue >= Value)
    llvm::report_fatal_error(
        llvm::formatv("Wait timeout reached for signal handle {0:x}, Expected "
                      "value: {1}, Actual Value: {2}",
                      Signal.handle, Value - 1, SignalValue));
  if (MustDestroyAfterWait) {
    if (hsa::getHsaApiTable().core_->hsa_signal_destroy_fn(Signal) !=
        HSA_STATUS_SUCCESS) {
      llvm::report_fatal_error(
          "Failed to destroy the completed kernel's signal");
    }
  }

  /// If we are exiting a kernel launch:
  /// 1. Wait on the signal of the kernel being launched
  /// 2. Get the number of wave fronts in the kernel
  /// 3. Print opcode hist for this kernel
  /// 4. Release the lock

  // Get the kernel that was launched

  auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                          SubmittedDispatchPacket.kernel_object)
                          ->getLoadedCodeObjectKernelSymbol();

  LUTHIER_REPORT_FATAL_ON_ERROR(KernelSymbol.takeError());

  auto KernelName = KernelSymbol->getName();
  LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());
  std::string KernelNameToBePrinted = DemangleKernelNames
                                          ? llvm::demangle(*KernelName)
                                          : std::string(*KernelName);

  // Copy the histogram to the global histogram
  for (int i = 0; i < MAX_OPCODES; i++){
      GlobalHistogram[i] += KernelHistogram[i];
  }
  // Note: Grid size in HSA contains all the work items in the kernel. This
  // is different from the number of blocks (grid dim) in CUDA or HIP
  // For example, if a kernel is launched from HIP with 16 blocks, each with
  // 64 threads, this will be indicated as a grid size dim of (1024, 1, 1) and
  // workgroup size of (64, 1, 1), since 1024 = 64 x 16.
  // As is evident from the example, workgroup size in HSA has the same meaning
  // as CUDA/HIP
  
  uint64_t counter = 0;
  for (auto opcode = 0; opcode <= llvm::AMDGPU::INSTRUCTION_LIST_END; ++opcode) {
    if (GlobalHistogram[opcode] != 0) {
        counter += GlobalHistogram[opcode];
    }
  }
  TotalNumInstructions += counter;
  
  llvm::outs() << llvm::formatv(
      "Kernel {0} - {1}: Total Grid Size Dims: ({2}, {3}, {4}) Workgroup Dims: "
      "({5}, {6}, {7}) Kernel Instruction "
      "Count: {8}, Total Instructions counted so far: {9}\n",
      NumKernelLaunched++, KernelNameToBePrinted,
      SubmittedDispatchPacket.grid_size_x, SubmittedDispatchPacket.grid_size_y,
      SubmittedDispatchPacket.grid_size_z,
      SubmittedDispatchPacket.workgroup_size_x,
      SubmittedDispatchPacket.workgroup_size_y,
      SubmittedDispatchPacket.workgroup_size_z, counter, TotalNumInstructions);

  llvm::outs() << "Histogram for kernel " << KernelNameToBePrinted << ":\n";
  /*
  To obtain the Opcode name:
    1. From the dispatch packet, get the kernel symbol
    2. From the kernel symbol, get the executable
    3. Call lift on the executable
    4. Get the TII from one of the functions in the LiftedRepresentation
  */
  // auto LR = luthier::lift(*KernelSymbol);
  // LUTHIER_REPORT_FATAL_ON_ERROR(LR.takeError());
  // auto funcIt = LR->functions().begin();
  // // TODO:funcIt != LR.functions().function_end()
  // llvm::MachineFunction *First_MF = funcIt->second;
  // const llvm::TargetInstrInfo *TII = First_MF->getSubtarget().getInstrInfo();
  for (auto opcode = 0; opcode <= llvm::AMDGPU::INSTRUCTION_LIST_END; ++opcode) {
    if (KernelHistogram[opcode] != 0){
          //std::string OpcodeName = TII -> getName(opcode).str();
          llvm::outs() << llvm::formatv(
              "  {0} = {1}\n", 
              opcode, // index
              KernelHistogram[opcode]);
      }
  }
  Mutex.unlock();
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
      if (Phase == luthier::API_EVT_PHASE_BEFORE) {
        if (PacketType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
          /// If we are entering to a kernel launch:
          /// 1. Lock the mutex to prevent multiple kernels to run concurrently
          /// 2. Instrument the kernel if no already instrumented
          /// 3. Select whether the instrumented kernel will run or not
          /// 4. Reset the managed Counter variable
          Mutex.lock();
          auto &DispatchPacket = Packet.asKernelDispatch();
          // Get the kernel that is about to get launched
          auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                  DispatchPacket.kernel_object)
                                  ->getLoadedCodeObjectKernelSymbol();

          LUTHIER_REPORT_FATAL_ON_ERROR(KernelSymbol.takeError());
          // Get the packet's completion signal and its value

          bool MustDestroySignalAfterLaunch = false;

          hsa_signal_value_t SignalValue{0};
          // Create a signal if the application didn't create one for this
          // packet
          if (DispatchPacket.completion_signal.handle == 0) {
            if (hsa::getHsaApiTable().amd_ext_->hsa_amd_signal_create_fn(
                    SignalValue, 0, nullptr, 0,
                    &DispatchPacket.completion_signal) != HSA_STATUS_SUCCESS)
              llvm::report_fatal_error(
                  "Failed to create a signal for the dispatch packet.");
            MustDestroySignalAfterLaunch = true;
          } else {
            MustDestroySignalAfterLaunch = false;
          }
          SignalValue =
              hsa::getHsaApiTable().core_->hsa_signal_load_scacquire_fn(
                  DispatchPacket.completion_signal);

          auto IsKernelInstrumented =
              isKernelInstrumented(*KernelSymbol, "OpcodeHistogram");
          LUTHIER_REPORT_FATAL_ON_ERROR(IsKernelInstrumented.takeError());
          if (!*IsKernelInstrumented) {
            // Get the executable associated with this kernel
            auto Exec = KernelSymbol->getExecutable();
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
                DispatchPacket, "OpcodeHistogram"));
            ToBeSubmittedPackets->push_back(Packet);
            //zero the histogram
            for (int i = 0; i < MAX_OPCODES; i++){
                KernelHistogram[i] = 0;
            }
            std::thread(atKernelCompletion, DispatchPacket.completion_signal,
                        SignalValue, MustDestroySignalAfterLaunch,
                        DispatchPacket)
                .detach();
          } else {
            ToBeSubmittedPackets->push_back(Packet);
          }
        }
        // If the packet is not of type kernel dispatch, then submit it as is
        else {
          ToBeSubmittedPackets->push_back(Packet);
        }
      }
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
    llvm::outs() << "Opcode Histogram tool is launching.\n";
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
