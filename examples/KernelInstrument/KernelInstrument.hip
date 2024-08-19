//===-- KernelInstrument.hip ----------------------------------------------===//
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
/// This file implements a sample kernel instrumentation tool.
//===----------------------------------------------------------------------===//
#include <luthier/luthier.h>

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_OPERAND_ENUM
#include "AMDGPUGenInstrInfo.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-kernel-instrument-tool"

using namespace luthier;
// TODO: write the instruction counter tool

MARK_LUTHIER_DEVICE_MODULE

__attribute__((managed)) uint64_t GlobalCounter = 20;

LUTHIER_HOOK_ANNOTATE instrumentationHook() {
  GlobalCounter = __builtin_amdgcn_wavefrontsize();
  //  GlobalCounter = readReg<uint64_t>(llvm::AMDGPU::SGPR4_SGPR5);
};

LUTHIER_EXPORT_HOOK_HANDLE(instrumentationHook);

static void atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                     luthier::ApiEvtPhase Phase, luthier::hsa::ApiEvtID ApiID) {
  if (ApiID == luthier::hsa::HSA_API_EVT_ID_hsa_queue_packet_submit &&
      Phase == API_EVT_PHASE_BEFORE) {
    LLVM_DEBUG(llvm::dbgs() << "In the packet submission callback\n");
    for (auto &Packet : *CBData->hsa_queue_packet_submit.packets) {
      hsa_packet_type_t PacketType = Packet.getPacketType();

      if (PacketType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        auto &DispatchPacket = Packet.asKernelDispatch();
        auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                DispatchPacket.kernel_object)
                                ->getLoadedCodeObjectKernelSymbol();

        if (auto Err = KernelSymbol.takeError())
          llvm::report_fatal_error(std::move(Err), true);
        if (!llvm::cantFail(
                isKernelInstrumented(*KernelSymbol, "kernel instrument"))) {
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
                              *MBB.begin()->getNextNode(),
                              LUTHIER_GET_HOOK_HANDLE(instrumentationHook))) {
                        return Err;
                      }
                    }
                    return llvm::Error::success();
                  },
                  "instrumented")) {
            llvm::report_fatal_error(std::move(Err), true);
          }
          if (auto Err =
                  overrideWithInstrumented(DispatchPacket, "instrumented"))
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

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Kernel instrument tool is launching.\n";
  } else {
    hsa::setAtApiTableCaptureEvtCallback(atHsaApiTableCaptureCallBack);
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
    setAtApiTableReleaseEvtCallback(atHsaApiTableUnload);
  }
}

void atFinalization(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    llvm::outs() << "Counter Value: " << GlobalCounter << "\n";
    llvm::outs() << "Pointer of counter at host: "
                 << llvm::to_address(&GlobalCounter) << "\n";
    llvm::outs() << "Reserved variable address: "
                 << llvm::to_address(
                        reinterpret_cast<uint64_t *>(&__luthier_reserved))
                 << "\n";
    llvm::outs() << "Kernel Instrument Tool is terminating!\n";
  }
}

} // namespace luthier
