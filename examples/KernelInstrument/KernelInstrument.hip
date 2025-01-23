//===-- KernelInstrument.hip ----------------------------------------------===//
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
/// This file implements a sample kernel instrumentation tool.
//===----------------------------------------------------------------------===//
#include <luthier/luthier.h>
// undef ICMP_NE macro that was ill-defined in HIP headers
#undef ICMP_NE
#include <GCNSubtarget.h>
#include <llvm/Support/FormatVariadic.h>
#include <mutex>

// #define GET_REGINFO_ENUM
// #include "AMDGPUGenRegisterInfo.inc"
//
// #define GET_INSTRINFO_ENUM
// #define GET_INSTRINFO_OPERAND_ENUM
// #include "AMDGPUGenInstrInfo.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-kernel-instrument-tool"

// template <typename Integral = uint64_t>
// constexpr Integral
// bit_mask (int first, int last)
//{
//   dbgapi_assert (last >= first && "invalid argument");
//   size_t num_bits = last - first + 1;
//   return ((num_bits >= sizeof (Integral) * 8)
//               ? ~Integral{ 0 } /* num_bits exceed the size of Integral */
//               : ((Integral{ 1 } << num_bits) - 1))
//          << first;
// }
//
// static __device__ constexpr uint32_t ttmp11_wave_in_group_mask = bit_mask (0,
// 5); static __device__ constexpr int ttmp11_wave_in_group_shift = 0;

using namespace luthier;

static std::mutex Mutex;

MARK_LUTHIER_DEVICE_MODULE

__attribute__((managed)) uint64_t WaveID = 0;

__attribute__((managed)) hsa_dim3_t GroupIDs{0, 0, 0};

__attribute__((managed)) uint64_t PositionInGroup{0};

__attribute__((device, noinline)) void myFunc(const uint32_t &MyInt) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();

  // Overwrite the exec mask with one so that only a single thread is active
  luthier::writeExec(1);
  // Increment the counter by 1
  atomicAdd(&WaveID, MyInt);
  // Restore the exec mask
  luthier::writeExec(ExecMask);
}

LUTHIER_HOOK_ANNOTATE instrumentationHook(long RegVal) {
  uint32_t MyArray[20];
  for (int i = 0; i < 20; i++) {
    MyArray[i] = i;
  }
  for (int i = 0; i < 20; i++) {
    myFunc(MyArray[i]);
  }
  //     writeReg<uint64_t>(llvm::AMDGPU::SGPR4_SGPR5, 5);
  //  if (RegVal > 5)
  //    PositionInGroup += RegVal;
  //  else
  //    WaveID = RegVal + 4;
  //    writeReg<uint32_t>(llvm::AMDGPU::SGPR10, RegVal);
  //    PositionInGroup = 5;
  //   uint32_t MyReg = readReg<uint32_t>(llvm::AMDGPU::SGPR4);
  //   int MyVal;
  //   if (MyReg > 5) {
  //     MyVal = MyReg;
  //   }
  //   else if (MyReg == 2)
  //     MyVal = readReg<uint32_t>(llvm::AMDGPU::SGPR4);
  //   else
  //     MyVal = readReg<uint32_t>(llvm::AMDGPU::SGPR4);;
  ////   PositionInGroup = readReg<uint32_t>(llvm::AMDGPU::SGPR5);
  //   PositionInGroup = MyVal + 5;

  //  uint32_t idy = readReg<uint32_t>(llvm::AMDGPU::TTMP9);
  //  uint32_t v0 =  readReg<uint32_t>(llvm::AMDGPU::VGPR0);
  //  uint32_t v1 = readReg<uint32_t>(llvm::AMDGPU::SRC_POPS_EXITING_WAVE_ID);
  //  uint32_t groupPos =
  //      (readReg<uint32_t>(llvm::AMDGPU::TTMP11) & 63) >> 0;
  //  if (idx == 4 && v0 == 31) {
  //    WaveID = readReg<uint64_t>(llvm::AMDGPU::TTMP4_TTMP5);
  //    GroupIDs.x = readReg<uint32_t>(llvm::AMDGPU::TTMP8);
  //    GroupIDs.y = readReg<uint32_t>(llvm::AMDGPU::TTMP9);
  //    GroupIDs.z = readReg<uint32_t>(llvm::AMDGPU::TTMP10);
  ////    PositionInGroup = (readReg<uint32_t>(llvm::AMDGPU::TTMP11) & 63);
  //    PositionInGroup = v1;
  //  }
}

LUTHIER_EXPORT_HOOK_HANDLE(instrumentationHook);

static void atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                     luthier::ApiEvtPhase Phase, luthier::hsa::ApiEvtID ApiID) {
  if (ApiID == luthier::hsa::HSA_API_EVT_ID_hsa_queue_packet_submit &&
      Phase == API_EVT_PHASE_BEFORE) {
    LLVM_DEBUG(llvm::dbgs() << "In the packet submission callback\n");
    for (auto &Packet : *CBData->hsa_queue_packet_submit.packets) {
      if (auto *DispatchPacket = Packet.asKernelDispatch()) {
        auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                DispatchPacket->kernel_object)
                                ->getLoadedCodeObjectKernelSymbol();

        if (auto Err = KernelSymbol.takeError())
          llvm::report_fatal_error(std::move(Err), true);
        if (!llvm::cantFail(
                isKernelInstrumented(*KernelSymbol, "instrumented"))) {
          auto Exec = llvm::cantFail(KernelSymbol->getExecutable());

          auto LiftedKernel = luthier::lift(Exec);
          if (auto Err = LiftedKernel.takeError())
            llvm::report_fatal_error(std::move(Err), true);

          if (auto Err = luthier::instrumentAndLoad(
                  Exec, *LiftedKernel,
                  [&](InstrumentationTask &IT,
                      LiftedRepresentation &LR) -> llvm::Error {
                    // insert a hook after the first instruction of each basic
                    // block
                    for (auto &[Func, MF] : LR.functions()) {
                      auto &MBB = MF->front();
                      if (auto Err = IT.insertHookBefore(
                              *MBB.begin()
                                   ->getNextNode()
                                   ->getNextNode()
                                   ->getNextNode(),
                              LUTHIER_GET_HOOK_HANDLE(instrumentationHook),
                              {llvm::AMDGPU::SGPR4_SGPR5})) {
                        return Err;
                      }
                    }
                    return llvm::Error::success();
                  },
                  "instrumented")) {
            llvm::report_fatal_error(std::move(Err), true);
          }
          if (auto Err =
                  overrideWithInstrumented(*DispatchPacket, "instrumented"))
            llvm::report_fatal_error(std::move(Err), true);
        }
      }
    }
  }
}

static void atHsaApiTableUnload(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
  }
}

static void atHsaApiTableCaptureCallBack(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    if (auto Err = luthier::hsa::enableHsaApiEvtIDCallback(
            hsa::HSA_API_EVT_ID_hsa_queue_packet_submit))
      llvm::report_fatal_error(std::move(Err), true);
  }
}

namespace luthier {

llvm::StringRef getToolName() {
  const static std::string ToolName{"LuthierKernelInstrument"};
  return ToolName;
}

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Kernel instrument tool is launching.\n";
  } else {
    hsa::setAtApiTableCaptureEvtCallback(atHsaApiTableCaptureCallBack);
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
    setAtApiTableReleaseEvtCallback(atHsaApiTableUnload);
  }
}

void atToolFini(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    llvm::outs() << llvm::formatv("WaveID: {0}\n", WaveID);
    llvm::outs() << llvm::formatv("Group ID: ({0}, {1}, {2})\n", GroupIDs.z,
                                  GroupIDs.y, GroupIDs.x);
    llvm::outs() << llvm::formatv("Position in Group: {0}\n", PositionInGroup);
    //    llvm::outs() << llvm::formatv("Counter 4 Value: {0:x}\n",
    //    GlobalCounter4); llvm::outs() << llvm::formatv("Counter 5 Value:
    //    {0:x}\n", GlobalCounter5); llvm::outs() << "Pointer of counter at
    //    host: "
    //                 << llvm::to_address(&GlobalCounter) << "\n";
    //    llvm::outs() << "Reserved variable address: "
    //                 << llvm::to_address(
    //                        reinterpret_cast<uint64_t *>(&__luthier_reserved))
    //                 << "\n";
    llvm::outs() << "Kernel Instrument Tool is terminating!\n";
  }
}

} // namespace luthier
