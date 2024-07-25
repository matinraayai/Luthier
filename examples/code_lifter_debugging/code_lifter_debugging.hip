#include <luthier/luthier.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-lifter-debugging-tool"

using namespace luthier;

static void atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                     luthier::ApiEvtPhase Phase, luthier::hsa::ApiEvtID ApiID) {
  if (ApiID == hsa::HSA_API_EVT_ID_hsa_queue_packet_submit) {
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

        auto KernelName = hsa::getSymbolName(*KernelSymbol);
        if (auto Err = KernelName.takeError())
          llvm::report_fatal_error(std::move(Err), true);
        // Skip kernels coming from rocclr; The image store instructions
        // don't pass the machine verifier check because of missing mem operands
        // TODO: Remove this once the image instruction issue is resolved
        if (KernelName->starts_with("__amd"))
          return;

        auto Exec = llvm::cantFail(hsa::getExecutableOfSymbol(*KernelSymbol));

        auto LiftedExecutable = luthier::lift(Exec);
        if (auto Err = LiftedExecutable.takeError())
          llvm::report_fatal_error(std::move(Err), true);

        llvm::SmallVector<
            std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>, 1>
            SourceFiles;
        // Call the instrument function without any work so that the LR gets
        // cloned
        auto InstrumentedLR =
            luthier::instrument(*LiftedExecutable,
                                [&](InstrumentationTask &IT,
                                    LiftedRepresentation &LR) -> llvm::Error {
                                  return llvm::Error::success();
                                });
        if (auto Err = InstrumentedLR.takeError())
          llvm::report_fatal_error(std::move(Err), true);
        // Print the assembly files
        if (auto Err = luthier::printLiftedRepresentation(
                **InstrumentedLR, SourceFiles,
                llvm::CodeGenFileType::AssemblyFile))
          llvm::report_fatal_error(std::move(Err), true);

        for (const auto &[LCO, SourceFile] : SourceFiles) {
          llvm::outs() << "Instrumented code object for LCO "
                       << llvm::format_hex(LCO.handle, 8) << "\n";
          llvm::outs() << SourceFile;
          llvm::outs() << "\n";
        }
      }
    }
  }
}

namespace luthier {

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Luthier Code Lifter Debug Tool is launching!\n";
  } else {
    hsa::enableHsaApiEvtIDCallback(hsa::HSA_API_EVT_ID_hsa_queue_packet_submit);
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
  }
}

void atFinalization(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    llvm::outs() << "Luthier Code Lifter Debug Tool is terminating!\n";
  }
}

} // namespace luthier
