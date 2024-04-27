#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <luthier/luthier.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <string>

static bool instrumented{false};

void check(bool pred) {
  if (!pred)
    std::exit(-1);
}

MARK_LUTHIER_DEVICE_MODULE

__managed__ int globalCounter = 20;

static int *globalCounterDynamic;

// instrumentation function:

// myFunc(int p1, int* p2)

// load p3, ... // Define p3
// Instrumentation starts:======================================================
// save p1, save p2, save p3 -> (v1, v2, v3)
// ADJSTACKUP
// Insert arguments: p1 -> v3, p2 -> address of output
// SI_CALL  <mask preserved: v1, v2, v3>
// ADJSTACKDOWN
// restore p1, restore p2, restore p3
// Instrumentation ends:========================================================
// add v1 v2 v3

// add v2 v1 v3
// v3 killed

LUTHIER_DECLARE_FUNC void instrumentation_function() {
  //    int i = 0;
  //    i = i + 4;
  //    i = i * 40;
  //    return i;
  //    return 1;
  //    atomicAdd(globalCounter, 1);

  globalCounter++;
  //    printf("Hello from LUTHIER!\n");
}

LUTHIER_EXPORT_FUNC(instrumentation_function)

namespace luthier {

void hsa::atHsaApiTableLoad() {
  std::cout << "Kernel Instrument Tool is launching." << std::endl;
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_queue_create);
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_signal_store_screlease);
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_executable_freeze);
  hsa::enableHsaOpCallback(hsa::HSA_API_EVT_ID_hsa_queue_packet_submit);
}

// class KernelInstrumentPass : public luthier::InstrumentationTask {
//
//   bool runOnModule(llvm::Module &M) override {
//     auto &MMI = getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI();
//     auto& F = *M.begin();
////    for (auto& F : M) {
//    llvm::outs() << F.getName() << "\n";
////    }
////    F++;
//    auto &MBB = *MMI.getMachineFunction(F)->begin();
//    auto &MI = *MBB.begin();
//    MBB.insert(MI, nullptr);
//    llvm::cantFail(
//        insertCallTo(MI, LUTHIER_GET_EXPORTED_FUNC(instrumentation_function),
//                     INSTR_POINT_BEFORE));
//
//    //    for (auto & MBB : MF) {
//    //      for (auto & MI : MBB) {
//    //        MI.getOpcode() // physical (part of app), virtual (part of inst)
//    //            // get me actual instruction
//    //        insertCallTo();
//    //
//    //      }
//    //    }
//    //    auto& MMI =
//    //    getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI(); MMI.get
//
//    return false;
//  }
//};

void luthier::hsa::atHsaApiTableUnload() {
  std::cout << "Counter Value: " << globalCounter << std::endl;
  std::cout << "Kernel Launch Intercept Tool is terminating!" << std::endl;
}

void luthier::hsa::atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                            luthier::ApiEvtPhase Phase,
                            luthier::hsa::ApiEvtID ApiID) {
  if (Phase == luthier::API_EVT_PHASE_EXIT) {
    if (ApiID == hsa::HSA_API_EVT_ID_hsa_queue_create) {
      std::cout << "Queue created called!" << std::endl;
      std::cout << "Signal handle: "
                << (*(CBData->hsa_queue_create.queue))->doorbell_signal.handle
                << " \n";
    } else if (ApiID == hsa::HSA_API_EVT_ID_hsa_signal_store_screlease) {
      std::cout << "Signal handle Store: "
                << CBData->hsa_signal_store_relaxed.signal.handle << std::endl;

    } else if (ApiID == hsa::HSA_API_EVT_ID_hsa_executable_freeze) {
      auto executable = CBData->hsa_executable_freeze.executable;
      fprintf(stdout, "HSA Executable Freeze Callback\n");
      // Get the state of the executable (frozen or not frozen)
      hsa_executable_state_t e_state;
      check(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_STATE,
                                    &e_state) == HSA_STATUS_SUCCESS);

      fprintf(stdout, "Is executable frozen: %s\n",
              (e_state == HSA_EXECUTABLE_STATE_FROZEN ? "yes" : "no"));
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
        std::cout << "Dispatch packet's kernel arg address: "
                  << DispatchPacket.kernarg_address << std::endl;
        std::cout << "Size of private segment: "
                  << DispatchPacket.private_segment_size << std::endl;
        DispatchPacket.private_segment_size = 100000;
        if (!instrumented) {
          auto KD = luthier::KernelDescriptor::fromKernelObject(
              DispatchPacket.kernel_object);
          auto Symbol = KD->getHsaExecutableSymbol();
          if (auto Err = Symbol.takeError())
            exit(-1);
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
                            LUTHIER_GET_EXPORTED_FUNC(instrumentation_function),
                            INSTR_POINT_AFTER);
          }

          auto Res =
              luthier::instrument(std::move(Module), std::move(MMIWP), LSI,
                                  IT);
          if (!Res.operator bool())
            exit(-1);
          std::cout << "Instrumented thingy works\n";
          instrumented = true;
        }
        //        auto Res = luthier::overrideWithInstrumented(DispatchPacket);
        //        if (Res)
        //          exit(-1);
      }
    }
    std::cout << "End of callback" << std::endl;
  }
}

} // namespace luthier
