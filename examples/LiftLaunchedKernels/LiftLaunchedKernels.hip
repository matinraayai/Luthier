//===-- CodeLifterDebugging.hip -------------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// launched, lifts them, and prints the lifted representation as an assembly
/// file.
//===----------------------------------------------------------------------===//
#include <luthier/luthier.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lift-launched-kernels-tool"

using namespace luthier;

static llvm::DenseSet<const hsa::LoadedCodeObjectKernel *> PrintedKernels;

static void atHsaEvt(hsa::ApiEvtArgs *CBData,
                     ApiEvtPhase Phase, hsa::ApiEvtID ApiID) {
  if (ApiID == hsa::HSA_API_EVT_ID_hsa_queue_packet_submit &&
      Phase == API_EVT_PHASE_AFTER) {
    for (auto &Packet : *CBData->hsa_queue_packet_submit.packets) {
      if (auto *DispatchPacket = Packet.asKernelDispatch()) {
        auto KernelSymbol = hsa::KernelDescriptor::fromKernelObject(
                                DispatchPacket->kernel_object)
                                ->getLoadedCodeObjectKernelSymbol();
        LUTHIER_REPORT_FATAL_ON_ERROR(KernelSymbol.takeError());
        auto KernelName = KernelSymbol->getName();
        LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());

        if (!PrintedKernels.contains(&*KernelSymbol)) {
          auto LiftedKernel = lift(*KernelSymbol);
          LUTHIER_REPORT_FATAL_ON_ERROR(LiftedKernel.takeError());

          llvm::SmallVector<
              std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>,
              1>
              SourceFiles;
          // Call the instrument function without any work so that the LR gets
          // cloned
          auto InstrumentedLR =
              instrument(*LiftedKernel,
                                  [&](InstrumentationTask &IT,
                                      LiftedRepresentation &LR) -> llvm::Error {
                                    return llvm::Error::success();
                                  });
          LUTHIER_REPORT_FATAL_ON_ERROR(InstrumentedLR.takeError());
          // Print the assembly files
          LUTHIER_REPORT_FATAL_ON_ERROR(printLiftedRepresentation(
              **InstrumentedLR, SourceFiles,
              llvm::CodeGenFileType::AssemblyFile));

          for (const auto &[LCO, SourceFile] : SourceFiles) {
            llvm::outs() << "Instrumented code object for LCO "
                         << llvm::format_hex(LCO.handle, 8) << "\n";
            llvm::outs() << SourceFile << "\n";
          }
          PrintedKernels.insert(&*KernelSymbol);
        }
      }
    }
  }
}

static void atHsaApiTableCaptureCallBack(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_AFTER) {
    LUTHIER_REPORT_FATAL_ON_ERROR(hsa::enableHsaApiEvtIDCallback(
        hsa::HSA_API_EVT_ID_hsa_queue_packet_submit));
  }
}

namespace luthier {

llvm::StringRef getToolName() {
  static std::string ToolName = "LuthierLiftLaunchKernels";
  return ToolName;
}

void atToolInit(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Luthier lift launched kernels tool is launching!\n";
  } else {
    hsa::setAtApiTableCaptureEvtCallback(atHsaApiTableCaptureCallBack);
    hsa::setAtHsaApiEvtCallback(atHsaEvt);
  }
}

void atToolFini(ApiEvtPhase Phase) {
  if (Phase == API_EVT_PHASE_BEFORE) {
    llvm::outs() << "Luthier lift launched kernels tool is terminating!\n";
  }
}

} // namespace luthier
