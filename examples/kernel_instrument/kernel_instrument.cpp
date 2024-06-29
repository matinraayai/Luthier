#include <hsa/hsa.h>
#include <luthier/luthier.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-kernel-instrument-tool"

using namespace luthier;

MARK_LUTHIER_DEVICE_MODULE

__attribute__((managed)) uint64_t GlobalCounter = 20;

LUTHIER_HOOK_ANNOTATE instrumentationHook() {
  GlobalCounter = reinterpret_cast<uint64_t>(&GlobalCounter);
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
        auto KernelSymbol = luthier::KernelDescriptor::fromKernelObject(
                                DispatchPacket.kernel_object)
                                ->getHsaExecutableSymbol();

        if (auto Err = KernelSymbol.takeError())
          llvm::report_fatal_error(std::move(Err), true);
        if (!llvm::cantFail(
                isKernelInstrumented(*KernelSymbol, "kernel instrument"))) {
          auto LiftedKernel = luthier::lift(*KernelSymbol);
          if (auto Err = LiftedKernel.takeError())
            llvm::report_fatal_error(std::move(Err), true);

          // insert a hook after the first instruction of each basic block
          InstrumentationTask IT(
              "kernel instrument",
              [](InstrumentationTask &IT,
                 LiftedRepresentation &LR) -> llvm::Error {
                for (auto &[FuncHSAHandle, MF] : LR.functions()) {
                  auto &MBB = *MF->begin();
                  auto &MI = *MBB.begin();
                  if (auto Error = IT.insertHookAt(
                          MI, LUTHIER_GET_HOOK_HANDLE(instrumentationHook),
                          INSTR_POINT_AFTER))
                    return Error;
                }
                return llvm::Error::success();
              });

          if (auto Res =
                  luthier::instrumentAndLoad(*KernelSymbol, *LiftedKernel, IT))
            llvm::report_fatal_error(std::move(Res), true);
        }
        if (auto Res = luthier::overrideWithInstrumented(
                DispatchPacket, "kernel instrumentAndLoad"))
          llvm::report_fatal_error(std::move(Res), true);
      }
    }
  }
}

static void atHsaApiTableUnload(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Counter Value: "
                 << llvm::to_address(reinterpret_cast<uint64_t *>(GlobalCounter))
                 << "\n";
    llvm::outs() << "Pointer of counter at host: "
                 << llvm::to_address(&GlobalCounter) << "\n";
    llvm::outs() << "Reserved variable address: "
                 << llvm::to_address(
                        reinterpret_cast<uint64_t *>(&__luthier_reserved))
                 << "\n";
  }
}

namespace luthier {

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Kernel instrument tool is launching.\n";
  } else {
    hsa::enableHsaApiEvtIDCallback(hsa::HSA_API_EVT_ID_hsa_queue_packet_submit);
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
    setAtApiTableUnloadEvtCallback(atHsaApiTableUnload);
  }
}

void atFinalization(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER)
    llvm::outs() << "Kernel Instrument Tool is terminating!\n";
}

} // namespace luthier
