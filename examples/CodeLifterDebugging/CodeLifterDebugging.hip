//===-- CodeLifterDebugging.hip -------------------------------------------===//
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
/// This file implements a sample tool which intercepts the kernels being
/// launched, lifts their executables, and prints them in the <tt>.s</tt>
/// file format.
//===----------------------------------------------------------------------===//
#include <luthier/luthier.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-lifter-debugging-tool"

using namespace luthier;

static llvm::DenseSet<hsa_executable_t> PrintedExecutables;

static void atHsaEvt(luthier::hsa::ApiEvtArgs *CBData,
                     luthier::ApiEvtPhase Phase, luthier::hsa::ApiEvtID ApiID) {
  if (ApiID == hsa::HSA_API_EVT_ID_hsa_queue_packet_submit &&
      Phase == API_EVT_PHASE_AFTER) {
    LLVM_DEBUG(llvm::dbgs() << "In the packet submission callback\n");
    auto Packets = CBData->hsa_queue_packet_submit.packets;
    for (auto &Packet : *Packets) {
      hsa_packet_type_t PacketType = Packet.getPacketType();
      if (PacketType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        auto &DispatchPacket = Packet.asKernelDispatch();
        auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                DispatchPacket.kernel_object)
                                ->getLoadedCodeObjectKernelSymbol();
        if (auto Err = KernelSymbol.takeError())
          llvm::report_fatal_error(std::move(Err), true);

        auto KernelName = KernelSymbol->getName();
        if (auto Err = KernelName.takeError())
          llvm::report_fatal_error(std::move(Err), true);

        auto Exec = llvm::cantFail(KernelSymbol->getExecutable());
        if (!PrintedExecutables.contains(Exec)) {
          auto LiftedExecutable = luthier::lift(*KernelSymbol);
          if (auto Err = LiftedExecutable.takeError())
            llvm::report_fatal_error(std::move(Err), true);

          llvm::SmallVector<
              std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>,
              1>
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
          PrintedExecutables.insert(Exec);
        }
      }
    }
  }
}

static void atHsaApiTableCaptureCallBack(ApiEvtPhase Phase) {
  llvm::outs() << "in hsa api callback;\n";
  if (Phase == API_EVT_PHASE_AFTER) {
    llvm::outs() << "after phase\n";
    if (auto Err = luthier::hsa::enableHsaApiEvtIDCallback(
            hsa::HSA_API_EVT_ID_hsa_queue_packet_submit))
      llvm::report_fatal_error(std::move(Err), true);
  }
}

namespace luthier {

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Luthier Code Lifter Debug Tool is launching!\n";
  } else {
    hsa::setAtApiTableCaptureEvtCallback(atHsaApiTableCaptureCallBack);
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
  }
}

void atToolFini(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    llvm::outs() << "Luthier Code Lifter Debug Tool is terminating!\n";
  }
}

} // namespace luthier
