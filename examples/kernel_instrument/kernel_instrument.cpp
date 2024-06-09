#include <fstream>
#include <functional>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include <luthier/luthier.h>
#include <string>

MARK_LUTHIER_DEVICE_MODULE

__attribute__((managed)) int globalCounter = 20;

LUTHIER_HOOK_CREATE(instrumentation_function, (int myInt, int myInt2), {
  globalCounter = 20000;
})

namespace luthier {

void hsa::atHsaApiTableLoad() {
  std::cout << "Kernel Instrument Tool is launching." << std::endl;
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_queue_create);
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_signal_store_screlease);
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_executable_freeze);
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_queue_packet_submit);
}

void luthier::hsa::atHsaApiTableUnload() {
  std::cout << "Counter Value: " << globalCounter << std::endl;
  std::cout << "Kernel Launch Intercept Tool is terminating!" << std::endl;
}

void luthier::hsa::atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                            luthier::ApiEvtPhase Phase,
                            luthier::hsa::ApiEvtID ApiID) {
  if (Phase == luthier::API_EVT_PHASE_EXIT) {
    if (ApiID == hsa::HSA_API_EVT_ID_hsa_executable_freeze) {
      auto executable = CBData->hsa_executable_freeze.executable;
      fprintf(stdout, "HSA Executable Freeze Callback\n");
      fprintf(stdout, "Executable handle: %lX\n", executable.handle);
    }
  }
  if (ApiID == hsa::HSA_API_EVT_ID_hsa_queue_packet_submit) {
    std::cout << "In packet submission callback" << std::endl;
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
            exit(-1);
          auto &[Module, MMIWP, LSI] = *LiftedSymbol;
          InstrumentationTask IT;
          for (auto &F : *Module) {
            auto &MF = *(MMIWP->getMMI().getMachineFunction(F));
            auto &MBB = *MF.begin();
            auto &MI = *MBB.begin();
            IT.insertCallTo(MI,
                            LUTHIER_GET_HOOK_HANDLE(instrumentation_function),
                            INSTR_POINT_AFTER);
          }

          if (auto Res = luthier::instrument(std::move(Module),
                                             std::move(MMIWP), LSI, IT))
            exit(-1);
        }
        if (auto Res = luthier::overrideWithInstrumented(DispatchPacket))
          exit(-1);
      }
    }
  }
}

} // namespace luthier
