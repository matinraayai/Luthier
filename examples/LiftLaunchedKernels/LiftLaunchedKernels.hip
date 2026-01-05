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
#include <luthier/hip/ApiTable.h>
#include <luthier/hsa/Executable.h>
#include <luthier/hsa/LoadedCodeObject.h>
#include <luthier/hsa/hsa.h>
#include <luthier/llvm/streams.h>
#include <luthier/luthier.h>
#include <luthier/rocprofiler-sdk/ApiTableWrapperInstaller.h>
#include <luthier/tooling/Context.h>
#include <unordered_set>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lift-launched-kernels-tool"

using namespace luthier;

static std::unordered_set<hsa_executable_t> *ExecutablesVisited;

/// Where the controller singleton is stored
static Context *C{nullptr};

static rocprofiler::HipRuntimeApiTableWrapperInstaller *HipWrapperInstaller{
    nullptr};

static decltype(hipMemcpy) *UnderlyingHipMemcpy{nullptr};

static hipError_t HipMemcpyWrapper(void *Dst, const void *Src, size_t SizeBytes,
                                   hipMemcpyKind CopyKind) {
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHipMemcpy != nullptr, "Underlying hipMemcpy is nullptr"));
  const auto AmdExtApiTable = C->getHsaAmdExtTable();
  hsa_amd_pointer_info_t Info{sizeof(hsa_amd_pointer_info_t)};
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExtApiTable.callFunction<hsa_amd_pointer_info>(Src, &Info, nullptr,
                                                        nullptr, nullptr),
      "Failed to get pointer info"));
  luthier::errs() << "Type: " << Info.type << "\n";
  luthier::errs() << "Queried src address: " << Src << "\n";
  luthier::errs() << "Device address: " << Info.agentBaseAddress << "\n";
  luthier::errs() << "Host accessible address: " << Info.hostBaseAddress
                  << "\n";
  if (Info.hostBaseAddress) {
    luthier::errs() << "Accessed from host: "
                    << llvm::StringRef(
                           reinterpret_cast<char *>(Info.hostBaseAddress),
                           SizeBytes)
                    << "\n";
  }
  luthier::errs() << "Type: " << Info.type << "\n";
  luthier::errs() << "Queried dst address: " << Dst << "\n";
  luthier::errs() << "Device address: " << Info.agentBaseAddress << "\n";
  luthier::errs() << "Host accessible address: " << Info.hostBaseAddress
                  << "\n";
  if (Info.hostBaseAddress) {
    luthier::errs() << "Accessed from host: "
                    << llvm::StringRef(
                           reinterpret_cast<char *>(Info.hostBaseAddress),
                           SizeBytes)
                    << "\n";
  }
  return UnderlyingHipMemcpy(Dst, Src, SizeBytes, CopyKind);
}

static void
atPacketDispatchCallback(const hsa_queue_t &Queue, uint64_t PacketIdx,
                         llvm::ArrayRef<hsa::AqlPacket> Packets,
                         hsa_amd_queue_intercept_packet_writer PacketWriter) {
  for (auto &Packet : Packets) {
    if (auto *DispatchPacket = Packet.asKernelDispatch()) {
      const auto HsaCoreApiTable = C->getHsaCoreTable();
      const auto AmdExtApiTable = C->getHsaAmdExtTable();
      const auto &LoaderTable = C->getHsaLoaderTable();

      std::vector<hsa_executable_t> Executables;
      LUTHIER_REPORT_FATAL_ON_ERROR(
          hsa::getAllExecutables(LoaderTable).moveInto(Executables));

      for (hsa_executable_t Exec : Executables) {
        if (!ExecutablesVisited->contains(Exec)) {
          llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
          LUTHIER_REPORT_FATAL_ON_ERROR(
              hsa::executableGetLoadedCodeObjects(LoaderTable, Exec, LCOs));
          for (hsa_loaded_code_object_t LCO : LCOs) {
            llvm::Expected<llvm::ArrayRef<uint8_t>> LoadedMemoryOrErr =
                hsa::loadedCodeObjectGetLoadedMemory(LoaderTable, LCO);
            LUTHIER_REPORT_FATAL_ON_ERROR(LoadedMemoryOrErr.takeError());
            hsa_amd_pointer_info_t Info{sizeof(hsa_amd_pointer_info_t)};
            LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
                AmdExtApiTable.callFunction<hsa_amd_pointer_info>(
                    LoadedMemoryOrErr->data(), &Info, nullptr, nullptr,
                    nullptr),
                "Failed to get pointer info"));
            luthier::errs() << "executable LCO\n";
            luthier::errs() << "Type: " << Info.type << "\n";
            luthier::errs()
                << "Queried address: " << LoadedMemoryOrErr->data() << "\n";
            luthier::errs()
                << "Device address: " << Info.agentBaseAddress << "\n";
            luthier::errs()
                << "Host accessible address: " << Info.hostBaseAddress << "\n";
            luthier::errs()
                << "Accessed from host: "
                << llvm::StringRef(
                       reinterpret_cast<char *>(Info.hostBaseAddress), 20)
                << "\n";
          }
          ExecutablesVisited->insert(Exec);
        }
      }
      // auto KernelSymbol =
      //     hsa::KernelDescriptor::fromKernelObject(DispatchPacket->kernel_object)
      //         ->getLoadedCodeObjectKernelSymbol(HsaCoreApiTable,
      //         LoaderTable);
      // LUTHIER_REPORT_FATAL_ON_ERROR(KernelSymbol.takeError());
      // auto KernelName = (*KernelSymbol)->getName();
      // LUTHIER_REPORT_FATAL_ON_ERROR(KernelName.takeError());

      // if (!PrintedKernels->contains(*KernelSymbol)) {
      //   auto LiftedKernel = lift(**KernelSymbol);
      //   LUTHIER_REPORT_FATAL_ON_ERROR(LiftedKernel.takeError());

      //   llvm::SmallVector<char> SourceFile;
      //   // Call the instrument function without any work so that the LR gets
      //   // cloned
      //   auto InstrumentedLR =
      //       instrument(*LiftedKernel,
      //                  [&](InstrumentationTask &IT, LiftedRepresentation &LR)
      //                      -> llvm::Error { return llvm::Error::success();
      //                      });
      //   LUTHIER_REPORT_FATAL_ON_ERROR(InstrumentedLR.takeError());
      //   // Print the assembly files
      //   LUTHIER_REPORT_FATAL_ON_ERROR(printLiftedRepresentation(
      //       **InstrumentedLR, SourceFile,
      //       llvm::CodeGenFileType::AssemblyFile));
      //   luthier::errs() << "Lifted Kernel contents for " << *KernelName
      //                   << ":\n";
      //   luthier::errs() << SourceFile << "\n";
      // }
      // PrintedKernels->insert(std::move(*KernelSymbol));
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

    delete ExecutablesVisited;
  });
}

static void atToolFini() { atToolFini(nullptr); }

void atToolInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    luthier::errs() << "Luthier lift launched kernels tool is launching!\n";
    std::atexit(atToolFini);
    ExecutablesVisited = new std::unordered_set<hsa_executable_t>();

    const auto Argv = "";
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(0, &Argv,
                                          "Luthier Lift Lanched Kernel Tool",
                                          &luthier::errs(),
                                          nullptr, "LUTHIER_ARGS"),
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

    HipWrapperInstaller = new rocprofiler::HipRuntimeApiTableWrapperInstaller(
        Err, std::make_tuple(&::HipDispatchTable::hipMemcpy_fn,
                             std::ref(UnderlyingHipMemcpy), HipMemcpyWrapper));
    if (Err)
      llvm::report_fatal_error(std::move(Err));
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
