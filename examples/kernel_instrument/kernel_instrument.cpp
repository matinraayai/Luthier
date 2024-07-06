#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
// #include "llvm/ADT/APInt.h"
#include <hsa/hsa.h>
#include <luthier/luthier.h>
#include <memory>

MARK_LUTHIER_DEVICE_MODULE

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-kernel-instrument-tool"

__attribute__((managed)) int GlobalCounter = 20;

// LUTHIER_HOOK_CREATE(instrumentationHook, (), { GlobalCounter = 20000; });

// extern "C" __device__ __forceinline__ int funcInternal(int myInt) {
//   globalCounter = 20;
//   return 100 + myInt;
// }

LUTHIER_HOOK_CREATE(instrumentationHook, (int myInt), 
{
  // int threadIdx_x;
  // __asm__ __volatile__("v_mov_b32 %0 v0\n" : "=v"(threadIdx_x));
  // globalCounter = 20000 + funcInternal(myInt) + myInt2;
  // GlobalCounter  += 10000;
  GlobalCounter  += myInt;
})

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

            // For testing, hard code an argument value for hook:
            auto NewArg = new llvm::GlobalVariable(*Module, 
                                llvm::Type::getInt32Ty(Module->getContext()), 
                                true, llvm::GlobalValue::ExternalLinkage,
                                llvm::ConstantInt::get(Module->getContext(),
                                                       llvm::APInt(32, 1000)), 
                                "HookArg1");
            llvm::ArrayRef<llvm::GlobalVariable*> HookArgs(NewArg);

            IT.insertCallTo(MI, LUTHIER_GET_HOOK_HANDLE(instrumentationHook),
                            HookArgs, INSTR_POINT_AFTER);
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
