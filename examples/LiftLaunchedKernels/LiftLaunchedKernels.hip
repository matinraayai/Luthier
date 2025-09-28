//===-- LiftLaunchedKernels.hip --------------------------------*- C++ -*-===//
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
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <luthier/llvm/streams.h>
#include <luthier/luthier.h>
#include <luthier/tooling/Context.h>
#include <unordered_set>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lift-launched-kernels-tool"

using namespace luthier;

static std::unordered_set<
    std::unique_ptr<hsa::LoadedCodeObjectKernel>,
    hsa::LoadedCodeObjectSymbolHash<hsa::LoadedCodeObjectKernel>,
    hsa::LoadedCodeObjectSymbolEqualTo<hsa::LoadedCodeObjectKernel>>
    *PrintedKernels;

/// Where the controller singleton is stored
static Context *C{nullptr};

static void
atPacketDispatchCallback(const hsa_queue_t &Queue, uint64_t PacketIdx,
                         llvm::ArrayRef<hsa::AqlPacket> Packets,
                         hsa_amd_queue_intercept_packet_writer PacketWriter) {
  for (auto &Packet : Packets) {
    if (auto *DispatchPacket = Packet.asKernelDispatch()) {
      const auto HsaCoreApiTable = C->getHsaCoreTable();
      const auto &LoaderTable = C->getHsaLoaderTable();
      auto KernelSymbol =
          hsa::KernelDescriptor::fromKernelObject(DispatchPacket->kernel_object)
              ->getLoadedCodeObjectKernelSymbol(HsaCoreApiTable, LoaderTable);
      LUTHIER_REPORT_FATAL_ON_ERROR(KernelSymbol.takeError());
      auto KernelName = (*KernelSymbol)->getName();
      LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());

      if (!PrintedKernels->contains(*KernelSymbol)) {
        auto LiftedKernel = lift(**KernelSymbol);
        LUTHIER_REPORT_FATAL_ON_ERROR(LiftedKernel.takeError());

        llvm::SmallVector<char> SourceFile;
        // Call the instrument function without any work so that the LR gets
        // cloned
        auto InstrumentedLR =
            instrument(*LiftedKernel,
                       [&](InstrumentationTask &IT, LiftedRepresentation &LR)
                           -> llvm::Error { return llvm::Error::success(); });
        LUTHIER_REPORT_FATAL_ON_ERROR(InstrumentedLR.takeError());
        // Print the assembly files
        LUTHIER_REPORT_FATAL_ON_ERROR(printLiftedRepresentation(
            **InstrumentedLR, SourceFile, llvm::CodeGenFileType::AssemblyFile));
        luthier::errs() << "Lifted Kernel contents for " << *KernelName
                        << ":\n";
        luthier::errs() << SourceFile << "\n";
      }
      PrintedKernels->insert(std::move(*KernelSymbol));
    }
    PacketWriter(&Packet, 1);
  }
}

namespace luthier {

  void atToolFini(void *) {
    static std::once_flag Once{};
    std::call_once(Once, []() {
      luthier::errs() << "Luthier lift launched kernels tool is terminating!\n";
      delete C;
  
      delete PrintedKernels;
    });
  }
  
  static void atToolFini() { atToolFini(nullptr); }

void atToolInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    luthier::errs() << "Luthier lift launched kernels tool is launching!\n";
    std::atexit(atToolFini);
    PrintedKernels = new std::unordered_set<
        std::unique_ptr<hsa::LoadedCodeObjectKernel>,
        hsa::LoadedCodeObjectSymbolHash<hsa::LoadedCodeObjectKernel>,
        hsa::LoadedCodeObjectSymbolEqualTo<hsa::LoadedCodeObjectKernel>>();

    const auto Argv = "";
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(0, &Argv,
                                          "Luthier Lift Lanched Kernel Tool",
                                          &luthier::errs(), "LUTHIER_ARGS"),
        "Failed to parse the command line arguments."));
    llvm::EnablePrettyStackTrace();
    llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/matinraayai/Luthier/issues/ and "
                          "include the crash error message and backtrace.\n");
    // Add the rocprofiler finalize function as a signal handler to LLVM so that
    // it executes in case of a fatal error
    llvm::sys::AddSignalHandler(atToolFini, nullptr);

    llvm::Error Err = llvm::Error::success();
    C = new Context(atPacketDispatchCallback, Err);
    if (Err)
      llvm::report_fatal_error(std::move(Err));
  });
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ClientID) {
  // Initialize the tooling library
  luthier::atToolInit(nullptr);

  LLVM_DEBUG(const uint32_t RocProfVerMajor = Version / 10000;
             const uint32_t RocProfVerMinor = (Version % 10000) / 100;
             const uint32_t RocProfVerPatch = Version % 100;
             llvm::dbgs() << "Registering Luthier tool with rocprofiler-sdk.\n";
             llvm::dbgs() << "Rocprofiler-sdk version: " << RocProfVerMajor
                          << "." << RocProfVerMinor << "." << RocProfVerPatch
                          << " (" << RuntimeVersion << ")"
                          << "\n";
             llvm::dbgs() << "Tool priority: " << Priority << ";\n";);
  ClientID->name = "Luthier Lift Launched Kernels tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
