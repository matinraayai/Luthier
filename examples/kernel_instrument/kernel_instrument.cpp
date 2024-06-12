#include <hsa/hsa.h>
#include <luthier/luthier.h>

MARK_LUTHIER_DEVICE_MODULE


#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-kernel-instrument-tool"

__attribute__((managed)) int GlobalCounter = 20;

LUTHIER_HOOK_CREATE(instrumentationHook, (), { GlobalCounter = 20000; });

namespace luthier {

void hsa::atHsaApiTableLoad() {
  llvm::outs() << "Kernel Instrument Tool is launching.\n";
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_queue_packet_submit);
}

void luthier::hsa::atHsaApiTableUnload() {
  llvm::outs() << "Counter Value: " << GlobalCounter << "\n";
  llvm::outs() << "Kernel Instrument Tool is terminating!\n";
}

void luthier::hsa::atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                            luthier::ApiEvtPhase Phase,
                            luthier::hsa::ApiEvtID ApiID) {
  if (ApiID == hsa::HSA_API_EVT_ID_hsa_queue_packet_submit) {
    LLVM_DEBUG(llvm::dbgs() << "In the packet submission callback\n");
    auto packets = CBData->hsa_queue_packet_submit.packets;
    for (unsigned int i = 0; i < CBData->hsa_queue_packet_submit.pkt_count;
         i++) {
      auto &packet = packets[i];
      hsa_packet_type_t packetType = packet.getPacketType();

      if (packetType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        auto &DispatchPacket = packet.asKernelDispatch();
        auto KD = luthier::KernelDescriptor::fromKernelObject(
            DispatchPacket.kernel_object);
        auto Symbol = KD->getHsaExecutableSymbol();
        if (auto Err = Symbol.takeError())
          llvm::report_fatal_error(
              "Failed to get the Symbol associated with the kernel descriptor");
        if (!llvm::cantFail(isKernelInstrumented(*Symbol))) {
          auto LiftedSymbol = luthier::liftSymbol(*Symbol);
          if (auto Err = LiftedSymbol.takeError())
            llvm::report_fatal_error("Kernel symbol lifting failed.");
          // insert a hook after the first instruction of each basic block
          auto &[Module, MMIWP, LSI] = *LiftedSymbol;
          InstrumentationTask IT;
          for (auto &F : *Module) {
            auto &MF = *(MMIWP->getMMI().getMachineFunction(F));
            auto &MBB = *MF.begin();
            auto &MI = *MBB.begin();
            IT.insertCallTo(MI, LUTHIER_GET_HOOK_HANDLE(instrumentationHook),
                            INSTR_POINT_AFTER);
          }

          if (auto Res = luthier::instrument(std::move(Module),
                                             std::move(MMIWP), LSI, IT))
            llvm::report_fatal_error(std::move(Res), true);
        }
        if (auto Res = luthier::overrideWithInstrumented(DispatchPacket))
          llvm::report_fatal_error(std::move(Res), true);
      }
    }
  }
}

} // namespace luthier
