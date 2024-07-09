#include <luthier/luthier.h>

using namespace luthier;

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
        auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                DispatchPacket.kernel_object)
                                ->getHsaExecutableSymbol();

        if (auto Err = KernelSymbol.takeError())
          llvm::report_fatal_error(std::move(Err), true);
        auto Exec = llvm::cantFail(hsa::getExecutableOfSymbol(*KernelSymbol));

        auto LiftedKernel = luthier::lift(Exec);
        if (auto Err = LiftedKernel.takeError())
          llvm::report_fatal_error(std::move(Err), true);
        for (const auto &[Symbol, MF] : LiftedKernel->functions()) {
          MF->dump();
          for (const auto & MBB: *MF) {
            MBB.dump();
            llvm::outs() << "Dumping MIs\n";
            for (const auto & MI: MBB) {
              MI.dump();
            }
          }
          MF->verify();
        };
        llvm::outs() << "Number of modules: " << LiftedKernel->size() << "\n";
      }
    }
  }
}

namespace luthier {

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Code Lifter Test is launching.\n";
  } else {
    hsa::enableHsaApiEvtIDCallback(hsa::HSA_API_EVT_ID_hsa_queue_packet_submit);
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
  }
}

void atFinalization(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER)
    llvm::outs() << "Code Lifter Test is terminating!\n";
}

} // namespace luthier
