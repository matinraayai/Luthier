//===-- HostcallConsumer.cpp ----------------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file HostcallConsumer.cpp
/// Implements \c luthier::HostcallConsumer — a faithful port of the listener +
/// packet-processing half of ROCclr \c device/devhostcall.cpp, on top of
/// Luthier's HSA wrappers.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/HostcallConsumer.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/MemoryPool.h"
#include "luthier/HSATooling/HostcallABI.h"

#include <cstring>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/bit.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "luthier-hostcall-consumer"

namespace luthier {

namespace {
/// Doorbell poll interval (a hint, in implementation-defined ticks). A finite
/// timeout makes the listener poll the ready stack even if a doorbell pulse is
/// ever missed, while still waking immediately on a pulse.
constexpr uint64_t HostcallWaitTimeoutHint = uint64_t{1} << 24;
} // namespace

llvm::Expected<std::unique_ptr<HostcallConsumer>> HostcallConsumer::create(
    const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnap,
    const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtSnap,
    hsa_agent_t GpuAgent, uint32_t NumPackets) {
  hsa::ApiTableContainer<::CoreApiTable> Core = CoreApiSnap.getTable();
  hsa::ApiTableContainer<::AmdExtTable> AmdExt = AmdExtSnap.getTable();

  // Find a CPU agent — the hostcall buffer must be system memory the listener
  // thread can read/write while the GPU pushes packets into it.
  hsa_agent_t CpuAgent{};
  bool FoundCpu = false;
  LUTHIER_RETURN_ON_ERROR(
      hsa::iterateAgents(Core, [&](hsa_agent_t A) -> llvm::Error {
        if (FoundCpu)
          return llvm::Error::success();
        auto TypeOrErr = hsa::agentGetDeviceType(Core, A);
        LUTHIER_RETURN_ON_ERROR(TypeOrErr.takeError());
        if (*TypeOrErr == HSA_DEVICE_TYPE_CPU) {
          CpuAgent = A;
          FoundCpu = true;
        }
        return llvm::Error::success();
      }));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      FoundCpu, "hostcall: no CPU agent found to back the hostcall buffer"));

  // Pick the CPU agent's fine-grained (coherent, atomics-capable) pool.
  llvm::SmallVector<hsa_amd_memory_pool_t, 4> Pools;
  LUTHIER_RETURN_ON_ERROR(
      hsa::getAllMemoryPoolsOfAgent(AmdExt, CpuAgent, Pools));
  hsa_amd_memory_pool_t FinePool{};
  bool FoundPool = false;
  for (hsa_amd_memory_pool_t P : Pools) {
    auto FineOrErr = hsa::memoryPoolIsFineGrained(AmdExt, P);
    LUTHIER_RETURN_ON_ERROR(FineOrErr.takeError());
    if (*FineOrErr) {
      FinePool = P;
      FoundPool = true;
      break;
    }
  }
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      FoundPool, "hostcall: CPU agent exposes no fine-grained memory pool"));

  // Allocate + lay out the buffer, and grant the GPU agent access to it.
  const size_t Size = hostcall::getHostcallBufferSize(NumPackets);
  auto BufOrErr = hsa::memoryPoolAllocate(AmdExt, FinePool, Size, /*Flags=*/0);
  LUTHIER_RETURN_ON_ERROR(BufOrErr.takeError());
  void *Buf = *BufOrErr;

  if (auto Err = hsa::agentsAllowAccess(
          AmdExt, llvm::ArrayRef<hsa_agent_t>{CpuAgent, GpuAgent}, Buf)) {
    llvm::consumeError(hsa::memoryPoolFree(AmdExt, Buf));
    return std::move(Err);
  }
  reinterpret_cast<hostcall::HostcallBuffer *>(Buf)->initialize(NumPackets);

  // Arm the doorbell signal and publish its handle into the buffer so
  // __ockl_hostcall_internal can pulse it.
  hsa_signal_t Sig{};
  if (auto Err = LUTHIER_HSA_CALL_ERROR_CHECK(
          Core.callFunction<hsa_signal_create>(hostcall::SIGNAL_INIT,
                                               /*num_consumers=*/0,
                                               /*consumers=*/nullptr, &Sig),
          "hostcall: hsa_signal_create failed")) {
    llvm::consumeError(hsa::memoryPoolFree(AmdExt, Buf));
    return std::move(Err);
  }
  reinterpret_cast<hostcall::HostcallBuffer *>(Buf)->Doorbell =
      reinterpret_cast<void *>(Sig.handle);

  std::unique_ptr<HostcallConsumer> C(
      new HostcallConsumer(Core, AmdExt, NumPackets));
  C->Buffer = Buf;
  C->Doorbell = Sig;
  LLVM_DEBUG(llvm::dbgs() << "[HostcallConsumer] buffer=" << Buf
                          << " size=" << Size << " packets=" << NumPackets
                          << " doorbell=0x"
                          << llvm::Twine::utohexstr(Sig.handle) << "\n");
  // Start the listener only once everything is wired.
  C->Worker = std::thread([Self = C.get()] { Self->consumeLoop(); });
  return C;
}

HostcallConsumer::~HostcallConsumer() {
  Stop.store(true, std::memory_order_release);
  // Wake the listener out of its blocking wait so it observes Stop.
  if (Doorbell.handle)
    Core.callFunction<hsa_signal_store_screlease>(Doorbell,
                                                  hostcall::SIGNAL_DONE);
  if (Worker.joinable())
    Worker.join();
  if (Doorbell.handle)
    Core.callFunction<hsa_signal_destroy>(Doorbell);
  if (Buffer)
    llvm::consumeError(hsa::memoryPoolFree(AmdExt, Buffer));
}

void HostcallConsumer::consumeLoop() {
  uint64_t Prev = hostcall::SIGNAL_INIT;
  while (!Stop.load(std::memory_order_acquire)) {
    hsa_signal_value_t V = Core.callFunction<hsa_signal_wait_scacquire>(
        Doorbell, HSA_SIGNAL_CONDITION_NE,
        static_cast<hsa_signal_value_t>(Prev), HostcallWaitTimeoutHint,
        HSA_WAIT_STATE_BLOCKED);
    if (Stop.load(std::memory_order_acquire))
      break;
    Prev = static_cast<uint64_t>(V);
    // Drain on every wakeup (doorbell pulse or timeout poll); the device
    // strictly increments the doorbell, so NE never aliases a past value.
    processPackets();
  }
}

void HostcallConsumer::processPackets() {
  auto *Buf = reinterpret_cast<hostcall::HostcallBuffer *>(Buffer);
  // Grab the whole ready stack; the device keeps pushing onto a fresh stack
  // while we process this snapshot.
  uint64_t Ready = Buf->ReadyStack.exchange(0, std::memory_order_acquire);
  for (uint64_t Iter = Ready, Next = 0; Iter; Iter = Next) {
    hostcall::PacketHeader *Hdr = Buf->getHeader(Iter);
    // Capture the next link before we relinquish ownership of this packet.
    Next = Hdr->Next;
    const uint32_t Service = Hdr->Service;
    hostcall::Payload *Pl = Buf->getPayload(Iter);
    uint64_t ActiveMask = Hdr->ActiveMask;
    while (ActiveMask) {
      const unsigned Wi = llvm::countr_zero(ActiveMask);
      ActiveMask &= ActiveMask - 1; // clear least-set bit
      uint64_t *Slot = Pl->Slots[Wi];
      if (Service == hostcall::SERVICE_FUNCTION_CALL) {
        // payload[0] = host function pointer; payload[1..] = up to 7 inputs;
        // up to 2 outputs are written back into payload[0..1].
        uint64_t Output[2] = {0, 0};
        auto Fn = reinterpret_cast<hostcall::HostcallFunctionCall>(Slot[0]);
        if (Fn)
          Fn(Output, Slot + 1);
        std::memcpy(Slot, Output, sizeof(Output));
      }
      // Other service IDs (printf/devmem/sanitizer) are not provided by the
      // Luthier listener; their packets are completed without action.
    }
    // Release the packet back to the waiting wave by clearing the READY bit.
    uint32_t Ctrl = Hdr->Control.load(std::memory_order_relaxed);
    Ctrl &= ~(uint32_t{1} << hostcall::CONTROL_OFFSET_READY_FLAG);
    Hdr->Control.store(Ctrl, std::memory_order_release);
  }
}

} // namespace luthier
