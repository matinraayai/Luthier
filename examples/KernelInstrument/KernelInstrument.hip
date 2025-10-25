//===-- KernelInstrument.hip ----------------------------------------------===//
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
/// This file implements a sample kernel instrumentation tool.
//===----------------------------------------------------------------------===//
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <luthier/llvm/streams.h>
#include <luthier/luthier.h>
#include <luthier/tooling/Context.h>
#include <stdarg.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-kernel-instrument-tool"

// template <typename Integral = uint64_t>
// constexpr Integral
// bit_mask (int first, int last)
//{
//   dbgapi_assert (last >= first && "invalid argument");
//   size_t num_bits = last - first + 1;
//   return ((num_bits >= sizeof (Integral) * 8)
//               ? ~Integral{ 0 } /* num_bits exceed the size of Integral */
//               : ((Integral{ 1 } << num_bits) - 1))
//          << first;
// }
//
// static __device__ constexpr uint32_t ttmp11_wave_in_group_mask = bit_mask (0,
// 5); static __device__ constexpr int ttmp11_wave_in_group_shift = 0;

using namespace luthier;

static Context *C{nullptr};

static std::mutex Mutex;

MARK_LUTHIER_DEVICE_MODULE

__attribute__((managed)) uint64_t WaveID = 0;

__attribute__((managed)) hsa_dim3_t GroupIDs{0, 0, 0};

__attribute__((managed)) uint64_t PositionInGroup{0};

// __attribute__((device, noinline)) void myFunc(const uint32_t &MyInt) {
//   unsigned long long int ExecMask = __builtin_amdgcn_read_exec();

//   // Overwrite the exec mask with one so that only a single thread is active
//   luthier::writeExec(1);
//   // Increment the counter by 1
//   atomicAdd(&WaveID, MyInt);
//   // Restore the exec mask
//   luthier::writeExec(ExecMask);
// }

LUTHIER_HOOK_ANNOTATE instrumentationHookBool(const llvm::MCRegister Reg,
                                              const uint32_t Val) {
  luthier::writeReg(Reg, Val);
}

LUTHIER_EXPORT_HOOK_HANDLE(instrumentationHookBool);

LUTHIER_HOOK_ANNOTATE instrumentationHookUint32(const llvm::MCRegister Reg,
                                                const uint32_t Val) {
  luthier::writeReg(Reg, Val);
}

LUTHIER_EXPORT_HOOK_HANDLE(instrumentationHookUint32);

LUTHIER_HOOK_ANNOTATE instrumentationHookUint64(const llvm::MCRegister Reg,
                                                const uint64_t Val) {
  luthier::writeReg(Reg, Val);
}

LUTHIER_EXPORT_HOOK_HANDLE(instrumentationHookUint64);

LUTHIER_HOOK_ANNOTATE instrumentationHook(const llvm::MCRegister Reg,
                                          const uint32_t Val) {
  luthier::writeReg(Reg, Val);

  // uint32_t MyArray[20];
  // for (int i = 0; i < 20; i++) {
  //   MyArray[i] = i;
  // }
  // for (int i = 0; i < 20; i++) {
  //   myFunc(MyArray[i]);
  // }
  //     writeReg<uint64_t>(llvm::AMDGPU::SGPR4_SGPR5, 5);
  //  if (RegVal > 5)
  //    PositionInGroup += RegVal;
  //  else
  //    WaveID = RegVal + 4;
  //    writeReg<uint32_t>(llvm::AMDGPU::SGPR10, RegVal);
  //    PositionInGroup = 5;
  //   uint32_t MyReg = readReg<uint32_t>(llvm::AMDGPU::SGPR4);
  //   int MyVal;
  //   if (MyReg > 5) {
  //     MyVal = MyReg;
  //   }
  //   else if (MyReg == 2)
  //     MyVal = readReg<uint32_t>(llvm::AMDGPU::SGPR4);
  //   else
  //     MyVal = readReg<uint32_t>(llvm::AMDGPU::SGPR4);;
  ////   PositionInGroup = readReg<uint32_t>(llvm::AMDGPU::SGPR5);
  //   PositionInGroup = MyVal + 5;

  //  uint32_t idy = readReg<uint32_t>(llvm::AMDGPU::TTMP9);
  //  uint32_t v0 =  readReg<uint32_t>(llvm::AMDGPU::VGPR0);
  //  uint32_t v1 = readReg<uint32_t>(llvm::AMDGPU::SRC_POPS_EXITING_WAVE_ID);
  //  uint32_t groupPos =
  //      (readReg<uint32_t>(llvm::AMDGPU::TTMP11) & 63) >> 0;
  //  if (idx == 4 && v0 == 31) {
  //    WaveID = readReg<uint64_t>(llvm::AMDGPU::TTMP4_TTMP5);
  //    GroupIDs.x = readReg<uint32_t>(llvm::AMDGPU::TTMP8);
  //    GroupIDs.y = readReg<uint32_t>(llvm::AMDGPU::TTMP9);
  //    GroupIDs.z = readReg<uint32_t>(llvm::AMDGPU::TTMP10);
  ////    PositionInGroup = (readReg<uint32_t>(llvm::AMDGPU::TTMP11) & 63);
  //    PositionInGroup = v1;
  //  }
}

LUTHIER_EXPORT_HOOK_HANDLE(instrumentationHook);

static void
atPacketDispatchCallback(const hsa_queue_t &Queue, uint64_t PacketIdx,
                         llvm::ArrayRef<hsa::AqlPacket> Packets,
                         hsa_amd_queue_intercept_packet_writer PacketWriter) {
  std::vector<hsa::AqlPacket> AppPackets(Packets);
  for (auto &Packet : AppPackets) {
    if (auto *DispatchPacket = Packet.asKernelDispatch()) {
      Mutex.lock();
      const auto HsaCoreApiTable = C->getHsaCoreTable();
      const auto &LoaderTable = C->getHsaLoaderTable();
      auto KernelSymbol =
          hsa::KernelDescriptor::fromKernelObject(DispatchPacket->kernel_object)
              ->getLoadedCodeObjectKernelSymbol(HsaCoreApiTable, LoaderTable);

      if (auto Err = KernelSymbol.takeError())
        llvm::report_fatal_error(std::move(Err), true);
      if (!llvm::cantFail(
              isKernelInstrumented(**KernelSymbol, "instrumented"))) {

        auto LiftedKernel = luthier::lift(**KernelSymbol);
        if (auto Err = LiftedKernel.takeError())
          llvm::report_fatal_error(std::move(Err), true);

        if (auto Err = luthier::instrumentAndLoad(
                **KernelSymbol, *LiftedKernel,
                [&](InstrumentationTask &IT,
                    LiftedRepresentation &LR) -> llvm::Error {
                  // insert a hook after the first instruction of each basic
                  // block
                  return LR.iterateAllDefinedFunctionTypes(
                      [&](const hsa::LoadedCodeObjectSymbol &Sym,
                          llvm::MachineFunction &MF) -> llvm::Error {
                        auto &MBB = MF.front();
                        return IT.insertHookBefore(
                            *MBB.begin()
                                 ->getNextNode()
                                 ->getNextNode()
                                 ->getNextNode(),
                            LUTHIER_GET_HOOK_HANDLE(instrumentationHookBool),
                            {llvm::ConstantInt::get(
                                 llvm::Type::getInt32Ty(LR.getContext()),
                                 llvm::AMDGPU::SCC),
                                 llvm::ConstantInt::get(
                                  llvm::Type::getInt32Ty(LR.getContext()),
                                  1)});
                      });
                },
                "instrumented")) {
          llvm::report_fatal_error(std::move(Err), true);
        }
        if (auto Err =
                overrideWithInstrumented(*DispatchPacket, "instrumented"))
          llvm::report_fatal_error(std::move(Err), true);
      }
      PacketWriter(DispatchPacket, 1);
      Mutex.unlock();
    } else {
      PacketWriter(&Packet, 1);
    }
  }
}

namespace luthier {

void atToolFini(void *) {
  luthier::outs() << llvm::formatv("WaveID: {0}\n", WaveID);
  luthier::outs() << llvm::formatv("Group ID: ({0}, {1}, {2})\n", GroupIDs.z,
                                   GroupIDs.y, GroupIDs.x);
  luthier::outs() << llvm::formatv("Position in Group: {0}\n", PositionInGroup);
  //    llvm::outs() << llvm::formatv("Counter 4 Value: {0:x}\n",
  //    GlobalCounter4); llvm::outs() << llvm::formatv("Counter 5 Value:
  //    {0:x}\n", GlobalCounter5); llvm::outs() << "Pointer of counter at
  //    host: "
  //                 << llvm::to_address(&GlobalCounter) << "\n";
  //    llvm::outs() << "Reserved variable address: "
  //                 << llvm::to_address(
  //                        reinterpret_cast<uint64_t *>(&__luthier_reserved))
  //                 << "\n";
  luthier::outs() << "Kernel Instrument Tool is terminating!\n";
  delete C;
}

static void atToolFini() { atToolFini(nullptr); }

void atToolInit(void *) {
  static std::once_flag Once{};
  std::call_once(Once, [] {
    llvm::outs() << "Kernel instrument tool is launching.\n";
    std::atexit(atToolFini);

    const auto Argv = "";
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::cl::ParseCommandLineOptions(0, &Argv,
                                          "Luthier Kernel Instrument Tool",
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
  ClientID->name = "Luthier Kernel Instrument tool";

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::atToolFini, nullptr};
  return &Cfg;
}
