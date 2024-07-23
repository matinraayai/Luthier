#include <luthier/luthier.h>
#include <hsa/hsa.h>

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_OPERAND_ENUM
#include "AMDGPUGenInstrInfo.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-kernel-instrument-tool"

using namespace luthier;

MARK_LUTHIER_DEVICE_MODULE


__attribute__((managed)) uint64_t GlobalCounter = 20;

LUTHIER_HOOK_ANNOTATE instrumentationHook() {
//  __asm__ __volatile__("s_mov_b32 s2 300");
  GlobalCounter = readReg<uint64_t>(llvm::AMDGPU::SGPR4_SGPR5);
};

LUTHIER_EXPORT_HOOK_HANDLE(instrumentationHook);

static void atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                     luthier::ApiEvtPhase Phase, luthier::hsa::ApiEvtID ApiID) {
  if (ApiID == luthier::hsa::HSA_API_EVT_ID_hsa_queue_packet_submit) {
    LLVM_DEBUG(llvm::dbgs() << "In the packet submission callback\n");
    auto Packets = CBData->hsa_queue_packet_submit.packets;
    for (unsigned int I = 0; I < CBData->hsa_queue_packet_submit.pkt_count;
         I++) {
      auto &Packet = Packets[I];
      hsa_packet_type_t PacketType = Packet.getPacketType();

      if (PacketType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        auto &DispatchPacket = Packet.asKernelDispatch();
        auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                DispatchPacket.kernel_object)
                                ->getHsaExecutableSymbol();

        if (auto Err = KernelSymbol.takeError())
          llvm::report_fatal_error(std::move(Err), true);
        if (!llvm::cantFail(
                isKernelInstrumented(*KernelSymbol, "kernel instrument"))) {
          auto Exec = llvm::cantFail(hsa::getExecutableOfSymbol(*KernelSymbol));

          auto LiftedKernel = luthier::lift(Exec);
          if (auto Err = LiftedKernel.takeError())
            llvm::report_fatal_error(std::move(Err), true);

          // insert a hook after the first instruction of each basic block
          for (const auto &[Func, MF] : LiftedKernel->functions()) {
            MF->dump();
          }
          if (auto Res = luthier::instrumentAndLoad(
                  *KernelSymbol, *LiftedKernel,
                  [&](InstrumentationTask &IT,
                      LiftedRepresentation &LR) -> llvm::Error {
                    llvm::outs() << "Mutator called!\n";
                    for (auto &[Func, MF] : LR.functions()) {
                      MF->dump();
//                      if (auto Error = IT.insertHookBefore(
//                              *MF->front().front().getNextNode(),
//                              LUTHIER_GET_HOOK_HANDLE(instrumentationHook)))
//                        return Error;
//                      for (auto &MBB: *MF) {
//                        for (auto &MI : MBB) {
//                          if (MI.getOpcode() == llvm::AMDGPU::S_ADD_I32_vi) {
//                            MI.getOperand(2).setImm(200);
//
//                          }
//                        }
//                      }
                    }
                    return llvm::Error::success();
                  },
                  ""))
            llvm::report_fatal_error(std::move(Res), true);
        }
        llvm::outs() << "Pointer of counter at host: "
                     << llvm::to_address(&GlobalCounter) << "\n";
                if (auto Res = luthier::overrideWithInstrumented(
                        DispatchPacket, ""))
                  llvm::report_fatal_error(std::move(Res), true);
      }
    }
  }
}

static void atHsaApiTableUnload(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
  }
}

namespace luthier {

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Kernel instrument tool is launching.\n";
  } else {
    hsa::enableHsaApiEvtIDCallback(hsa::HSA_API_EVT_ID_hsa_queue_packet_submit);
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
