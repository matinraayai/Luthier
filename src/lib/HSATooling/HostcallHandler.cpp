//===-- HostcallHandler.cpp -----------------------------------------------===//
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
#include "luthier/HSATooling/HostcallHandler.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/HSA/MemoryPool.h"
#include "luthier/HSA/Signal.h"
#include "luthier/HSATooling/DevicePrintf.h"

#include <cstring>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "luthier-hostcall-handler"

namespace luthier {

namespace {

//===----------------------------------------------------------------------===//
// Packet control field
//===----------------------------------------------------------------------===//

/// Bit 0 of a packet's control field: set while the packet awaits a response.
constexpr uint32_t ControlReadyFlag = 1U << 0;

//===----------------------------------------------------------------------===//
// Message descriptor field
//===----------------------------------------------------------------------===//

constexpr unsigned DescriptorOffsetBegin = 0;
constexpr unsigned DescriptorWidthBegin = 1;
constexpr unsigned DescriptorOffsetEnd = 1;
constexpr unsigned DescriptorWidthEnd = 1;
constexpr unsigned DescriptorOffsetLen = 5;
constexpr unsigned DescriptorWidthLen = 3;
constexpr unsigned DescriptorOffsetID = 8;
constexpr unsigned DescriptorWidthID = 56;

/// A hostcall payload is eight words wide and word 0 is the descriptor, so a
/// single packet can carry at most this much of a message.
constexpr unsigned MaxMessageWordsPerPacket = 7;

//===----------------------------------------------------------------------===//
// Device memory service
//===----------------------------------------------------------------------===//

/// Size of one slab in the device library's allocator (\c slab_t in
/// \c ockl/src/dm.cl). A request for exactly this much is a slab, and
/// \c __ockl_dm_dealloc finds a slab's base by masking a block address with
/// <tt>~(DeviceMemorySlabSize - 1)</tt>, so slabs must be aligned to it.
constexpr size_t DeviceMemorySlabSize = 2 * 1024 * 1024;

/// Alignment every device allocation must at least have.
/// \c __ockl_dm_dealloc treats an address aligned this far as a large,
/// non-slab allocation rather than a block inside a slab.
constexpr size_t DeviceMemoryMinAlignment = 4096;

uint64_t getDescriptorField(uint64_t Descriptor, unsigned Offset,
                            unsigned Width) {
  return (Descriptor >> Offset) & ((uint64_t{1} << Width) - 1);
}

uint64_t setDescriptorField(uint64_t Descriptor, uint64_t Value,
                            unsigned Offset, unsigned Width) {
  const uint64_t Mask = ((uint64_t{1} << Width) - 1) << Offset;
  return (Descriptor & ~Mask) | ((Value << Offset) & Mask);
}

/// Reports a failure the listener thread cannot propagate. The packet is
/// still released afterwards, so the offending wave resumes rather than
/// hanging on a request that will never be answered.
void reportServiceError(llvm::Error Err) {
  llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                              "Luthier hostcall service: ");
}

/// Finds a host memory pool that can back a hostcall buffer: fine-grained (so
/// the device sees the host's writes without an explicit copy) and open to
/// runtime allocation.
llvm::Expected<hsa_amd_memory_pool_t>
findHostFineGrainedPool(const hsa::ApiTableContainer<::CoreApiTable> &CoreApi,
                        const hsa::ApiTableContainer<::AmdExtTable> &AmdExt) {
  llvm::SmallVector<hsa_agent_t, 1> CpuAgents;
  LUTHIER_RETURN_ON_ERROR(
      hsa::getAllAgentsWithDeviceType<HSA_DEVICE_TYPE_CPU>(CoreApi, CpuAgents));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !CpuAgents.empty(),
      "No CPU agent available to back a hostcall buffer."));

  auto FoundOrErr = hsa::agentFindFineGrainedPool(AmdExt, CpuAgents.front());
  LUTHIER_RETURN_ON_ERROR(FoundOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      FoundOrErr->has_value(),
      "No host fine-grained memory pool available to back a hostcall "
      "buffer."));
  return **FoundOrErr;
}

} // namespace

//===----------------------------------------------------------------------===//
// HostcallMessageAssembler
//===----------------------------------------------------------------------===//

llvm::Error HostcallMessageAssembler::handlePayload(uint32_t Service,
                                                    uint64_t *Payload) {
  uint64_t Descriptor = Payload[0];
  const bool Begin =
      getDescriptorField(Descriptor, DescriptorOffsetBegin,
                         DescriptorWidthBegin) != 0;
  const bool End =
      getDescriptorField(Descriptor, DescriptorOffsetEnd, DescriptorWidthEnd) !=
      0;

  uint64_t ID;
  if (Begin) {
    // Allocate an id and hand it straight back to the device, which quotes it
    // on every follow-up packet of this message.
    if (FreeIDs.empty()) {
      ID = Messages.size();
      Messages.emplace_back();
    } else {
      ID = FreeIDs.back();
      FreeIDs.pop_back();
    }
    Messages[ID].Live = true;
    Messages[ID].Data.clear();

    Descriptor = setDescriptorField(Descriptor, 0, DescriptorOffsetBegin,
                                    DescriptorWidthBegin);
    Descriptor =
        setDescriptorField(Descriptor, ID, DescriptorOffsetID, DescriptorWidthID);
    Payload[0] = Descriptor;
  } else {
    ID = getDescriptorField(Descriptor, DescriptorOffsetID, DescriptorWidthID);
  }

  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      ID < Messages.size() && Messages[ID].Live,
      llvm::formatv("A hostcall packet continues message {0}, which is not "
                    "being assembled",
                    ID)));

  const uint64_t Len =
      getDescriptorField(Descriptor, DescriptorOffsetLen, DescriptorWidthLen);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Len <= MaxMessageWordsPerPacket,
      llvm::formatv("A hostcall packet declares {0} content words; a payload "
                    "holds at most {1}",
                    Len, MaxMessageWordsPerPacket)));
  Messages[ID].Data.insert(Messages[ID].Data.end(), Payload + 1,
                           Payload + 1 + Len);

  if (!End)
    return llvm::Error::success();

  // Take the message apart before running the service, so that a service
  // which throws or errors out cannot strand the slot.
  std::vector<uint64_t> Data = std::move(Messages[ID].Data);
  Messages[ID].Data.clear();
  Messages[ID].Live = false;
  FreeIDs.push_back(ID);

  switch (Service) {
  case HOSTCALL_SERVICE_PRINTF:
    handleDevicePrintfHostcall(Payload, Data);
    return llvm::Error::success();
  default:
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Hostcall service {0} does not accept streamed messages", Service));
  }
}

//===----------------------------------------------------------------------===//
// HostcallBuffer
//===----------------------------------------------------------------------===//

namespace {

/// Offset of the packet header array inside a hostcall buffer allocation.
size_t getHeaderArrayOffset() {
  return llvm::alignTo(sizeof(HostcallBuffer), alignof(HostcallPacketHeader));
}

/// Offset of the packet payload array inside a hostcall buffer allocation.
size_t getPayloadArrayOffset(uint32_t NumPackets) {
  return llvm::alignTo(getHeaderArrayOffset() +
                           sizeof(HostcallPacketHeader) * NumPackets,
                       alignof(HostcallPayload));
}

} // namespace

size_t HostcallBuffer::getRequiredSize(uint32_t NumPackets) {
  return getPayloadArrayOffset(NumPackets) + sizeof(HostcallPayload) * NumPackets;
}

size_t HostcallBuffer::getRequiredAlignment() { return alignof(HostcallPayload); }

llvm::Error HostcallBuffer::initialize(uint32_t NumPackets,
                                       HostcallServiceState &ServiceState) {
  // Two packets is the smallest free stack whose tagged pointers can be told
  // apart from the null pointer; see the tag comment below.
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      NumPackets >= 2 && llvm::isPowerOf2_32(NumPackets),
      llvm::formatv("A hostcall buffer needs a power-of-two packet count of "
                    "at least two; got {0}",
                    NumPackets)));

  auto *Base = reinterpret_cast<uint8_t *>(this);
  Headers =
      reinterpret_cast<HostcallPacketHeader *>(Base + getHeaderArrayOffset());
  Payloads = reinterpret_cast<HostcallPayload *>(
      Base + getPayloadArrayOffset(NumPackets));
  IndexMask = NumPackets - 1;
  Services = &ServiceState;

  // A tagged pointer whose index and tag are both zero is indistinguishable
  // from the null pointer that terminates a stack. The deepest entry of the
  // free stack points at packet 0, so give that pointer a non-zero tag.
  uint64_t Next = IndexMask + 1;
  Headers[0].Next = 0;
  Headers[0].ActiveMask = 0;
  Headers[0].Service = HOSTCALL_SERVICE_RESERVED;
  Headers[0].Control.store(0, std::memory_order_relaxed);
  for (uint32_t I = 1; I != NumPackets; ++I) {
    Headers[I].Next = Next;
    Headers[I].ActiveMask = 0;
    Headers[I].Service = HOSTCALL_SERVICE_RESERVED;
    Headers[I].Control.store(0, std::memory_order_relaxed);
    Next = I;
  }
  FreeStack = Next;
  ReadyStack.store(0, std::memory_order_release);
  return llvm::Error::success();
}

void HostcallBuffer::processPackets() {
  // Take the whole ready stack at once. The device keeps pushing onto the
  // emptied stack while these are answered; those land in the next pass.
  uint64_t Ready = ReadyStack.exchange(0, std::memory_order_acquire);

  for (uint64_t Iter = Ready, Next = 0; Iter != 0; Iter = Next) {
    HostcallPacketHeader *Header = getHeader(Iter);
    // Read the link before answering: releasing the packet hands ownership
    // back to the device, which may recycle it immediately.
    Next = Header->Next;

    const uint32_t Service = Header->Service;
    HostcallPayload *Payload = getPayload(Iter);

    // Every wave submits at most one packet at a time, so answering the stack
    // newest-first cannot reorder one wave's hostcalls against each other.
    for (uint64_t ActiveMask = Header->ActiveMask; ActiveMask != 0;
         ActiveMask &= ActiveMask - 1) {
      const unsigned Lane = llvm::countr_zero(ActiveMask);
      handlePayload(Service, Payload->Slots[Lane]);
    }

    // Clearing the READY flag is what unblocks the submitting wave, so it has
    // to be the last thing that happens, with release ordering to publish the
    // responses written above.
    const uint32_t Control = Header->Control.load(std::memory_order_relaxed);
    Header->Control.store(Control & ~ControlReadyFlag,
                          std::memory_order_release);
  }
}

void HostcallBuffer::handlePayload(uint32_t Service, uint64_t *Slot) {
  switch (Service) {
  case HOSTCALL_SERVICE_FUNCTION_CALL: {
    // Slot[0] is a host function pointer the device was handed by the host in
    // the first place; the remaining seven words are its arguments, and it
    // answers with two.
    using HostcallFunctionCall = void (*)(uint64_t *, const uint64_t *);
    auto Fn = reinterpret_cast<HostcallFunctionCall>(
        static_cast<uintptr_t>(Slot[0]));
    if (Fn == nullptr) {
      reportServiceError(LUTHIER_MAKE_GENERIC_ERROR(
          "A hostcall function-call request carried a null function pointer"));
      Slot[0] = 0;
      Slot[1] = 0;
      return;
    }
    uint64_t Output[2] = {0, 0};
    Fn(Output, Slot + 1);
    std::memcpy(Slot, Output, sizeof(Output));
    return;
  }
  case HOSTCALL_SERVICE_PRINTF:
    if (llvm::Error Err = Services->Messages.handlePayload(Service, Slot))
      reportServiceError(std::move(Err));
    return;
  case HOSTCALL_SERVICE_DEVMEM:
    handleDeviceMemoryRequest(Slot);
    return;
  default:
    reportServiceError(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Luthier does not implement hostcall service {0}", Service)));
    return;
  }
}

void HostcallBuffer::handleDeviceMemoryRequest(uint64_t *Slot) {
  const uint64_t Address = Slot[0];
  const uint64_t Size = Slot[1];

  if (Address != 0) {
    // A free. Only honour addresses this service handed out; anything else is
    // device code freeing memory it does not own, and passing it through to
    // HSA would corrupt an unrelated allocation.
    auto It = Services->DeviceAllocations.find(Address);
    if (It == Services->DeviceAllocations.end()) {
      reportServiceError(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "A hostcall device-memory request asked to free {0:x}, which "
          "Luthier's device allocator never returned",
          Address)));
      return;
    }
    // Free the block the pool handed out, which is only the same pointer when
    // the request did not have to be over-aligned.
    void *PoolAllocation = It->second;
    Services->DeviceAllocations.erase(It);
    if (llvm::Error Err =
            hsa::memoryPoolFree(Services->AmdExt, PoolAllocation))
      reportServiceError(std::move(Err));
    return;
  }

  // An allocation. A zero-sized request has no address to answer with.
  if (Size == 0) {
    Slot[0] = 0;
    return;
  }

  // The device library's allocator recovers a slab's base address from a
  // block inside it by masking off the low 21 bits, so a slab-sized request
  // has to come back 2 MiB aligned. It also tells a large allocation apart
  // from a block inside a slab by testing for 4 KiB alignment, which every
  // pool allocation already satisfies.
  const size_t Alignment = (Size == DeviceMemorySlabSize)
                               ? DeviceMemorySlabSize
                               : DeviceMemoryMinAlignment;
  // Only pay for over-allocation when the pool does not already align this
  // far; on a pool whose own alignment covers the request this is exact.
  const bool NeedsOverAllocation = Services->DeviceMemoryPoolAlignment < Alignment;
  const uint64_t RequestSize = NeedsOverAllocation ? Size + Alignment - 1 : Size;

  llvm::Expected<void *> PtrOrErr = hsa::memoryPoolAllocate(
      Services->AmdExt, Services->DeviceMemoryPool, RequestSize);
  if (!PtrOrErr) {
    // Device-side malloc reports exhaustion by returning null, so running out
    // is the kernel's problem to handle rather than Luthier's to escalate.
    const std::string Reason = llvm::toString(PtrOrErr.takeError());
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("[HostcallHandler] device allocation of {0} "
                                "bytes failed: {1}\n",
                                RequestSize, Reason));
    Slot[0] = 0;
    return;
  }

  const llvm::SmallVector<hsa_agent_t, 1> Agents{Services->Agent};
  if (llvm::Error Err =
          hsa::agentsAllowAccess(Services->AmdExt, Agents, *PtrOrErr)) {
    reportServiceError(std::move(Err));
    llvm::consumeError(hsa::memoryPoolFree(Services->AmdExt, *PtrOrErr));
    Slot[0] = 0;
    return;
  }

  const uint64_t Aligned =
      llvm::alignTo(reinterpret_cast<uint64_t>(*PtrOrErr), Alignment);
  Services->DeviceAllocations.insert({Aligned, *PtrOrErr});
  Slot[0] = Aligned;
}

//===----------------------------------------------------------------------===//
// HostcallBufferAllocation
//===----------------------------------------------------------------------===//

llvm::Expected<std::unique_ptr<HostcallBufferAllocation>>
HostcallBufferAllocation::create(
    const hsa::ApiTableContainer<::CoreApiTable> &CoreApi,
    const hsa::ApiTableContainer<::AmdExtTable> &AmdExt, hsa_agent_t Agent,
    uint32_t NumWaves) {
  // The free stack needs at least two packets, and a power-of-two count keeps
  // the index mask exact.
  const uint32_t NumPackets = static_cast<uint32_t>(
      std::max<uint64_t>(2, llvm::PowerOf2Ceil(std::max<uint32_t>(1, NumWaves))));

  auto PoolOrErr = findHostFineGrainedPool(CoreApi, AmdExt);
  LUTHIER_RETURN_ON_ERROR(PoolOrErr.takeError());

  auto AlignmentOrErr =
      hsa::memoryPoolGetRuntimeAllocAlignment(AmdExt, *PoolOrErr);
  LUTHIER_RETURN_ON_ERROR(AlignmentOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      *AlignmentOrErr >= HostcallBuffer::getRequiredAlignment(),
      llvm::formatv("The host fine-grained pool aligns allocations to {0} "
                    "bytes, but a hostcall buffer needs {1}",
                    *AlignmentOrErr, HostcallBuffer::getRequiredAlignment())));

  auto DeviceMemoryPoolOrErr = hsa::agentFindCoarseGrainedPool(AmdExt, Agent);
  LUTHIER_RETURN_ON_ERROR(DeviceMemoryPoolOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      DeviceMemoryPoolOrErr->has_value(),
      llvm::formatv("Agent {0:x} exposes no coarse-grained memory pool to "
                    "back the hostcall device-memory service",
                    Agent.handle)));
  auto DeviceMemoryAlignmentOrErr =
      hsa::memoryPoolGetRuntimeAllocAlignment(AmdExt, **DeviceMemoryPoolOrErr);
  LUTHIER_RETURN_ON_ERROR(DeviceMemoryAlignmentOrErr.takeError());

  const size_t Size = HostcallBuffer::getRequiredSize(NumPackets);
  auto AllocOrErr = hsa::memoryPoolAllocate(AmdExt, *PoolOrErr, Size);
  LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());

  auto Fail = [&](llvm::Error E) -> llvm::Error {
    return llvm::joinErrors(std::move(E),
                            hsa::memoryPoolFree(AmdExt, *AllocOrErr));
  };

  // The device pushes and pops the buffer's stacks with atomics, so it needs
  // direct access to the host allocation rather than a copy of it.
  const llvm::SmallVector<hsa_agent_t, 1> Agents{Agent};
  if (llvm::Error Err = hsa::agentsAllowAccess(AmdExt, Agents, *AllocOrErr))
    return Fail(std::move(Err));

  std::memset(*AllocOrErr, 0, Size);

  auto Services = std::make_unique<HostcallServiceState>(
      AmdExt, Agent, **DeviceMemoryPoolOrErr, *DeviceMemoryAlignmentOrErr);
  auto *Buffer = static_cast<HostcallBuffer *>(*AllocOrErr);
  if (llvm::Error Err = Buffer->initialize(NumPackets, *Services))
    return Fail(std::move(Err));

  LLVM_DEBUG(llvm::dbgs()
             << llvm::formatv("[HostcallHandler] allocated a {0}-packet "
                              "buffer ({1} bytes) at {2} for agent {3:x}\n",
                              NumPackets, Size, *AllocOrErr, Agent.handle));

  return std::unique_ptr<HostcallBufferAllocation>(
      new HostcallBufferAllocation(AmdExt, Buffer, std::move(Services)));
}

HostcallBufferAllocation::~HostcallBufferAllocation() {
  if (Buffer == nullptr)
    return;
  // Reclaim whatever the kernel allocated through the device-memory service
  // and never freed; nothing else knows about those blocks.
  if (Services) {
    for (const auto &[Address, PoolAllocation] : Services->DeviceAllocations)
      llvm::consumeError(hsa::memoryPoolFree(AmdExt, PoolAllocation));
    Services->DeviceAllocations.clear();
  }
  llvm::consumeError(hsa::memoryPoolFree(AmdExt, Buffer));
  Buffer = nullptr;
}

//===----------------------------------------------------------------------===//
// HostcallListener
//===----------------------------------------------------------------------===//

llvm::Expected<std::unique_ptr<HostcallListener>>
HostcallListener::create(const hsa::ApiTableContainer<::CoreApiTable> &CoreApi) {
  auto SignalOrErr = hsa::signalCreate(CoreApi, HOSTCALL_SIGNAL_INIT);
  LUTHIER_RETURN_ON_ERROR(SignalOrErr.takeError());

  std::unique_ptr<HostcallListener> Listener(
      new HostcallListener(CoreApi, *SignalOrErr));
  Listener->Thread = std::thread([L = Listener.get()] { L->consumePackets(); });

  LLVM_DEBUG(llvm::dbgs()
             << llvm::formatv("[HostcallHandler] listener started on doorbell "
                              "{0:x}\n",
                              SignalOrErr->handle));
  return Listener;
}

HostcallListener::~HostcallListener() { llvm::consumeError(stop()); }

void HostcallListener::addBuffer(HostcallBuffer &Buffer) {
  Buffer.setDoorbell(Doorbell);
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!llvm::is_contained(Buffers, &Buffer))
    Buffers.push_back(&Buffer);
}

void HostcallListener::removeBuffer(HostcallBuffer &Buffer) {
  std::lock_guard<std::mutex> Lock(Mutex);
  // Taking the lock also waits out a drain that is walking this buffer right
  // now, so the caller is free to release it once this returns.
  auto *It = llvm::find(Buffers, &Buffer);
  if (It != Buffers.end())
    Buffers.erase(It);
}

llvm::Error HostcallListener::stop() {
  if (Stopped)
    return llvm::Error::success();
  Stopped = true;

  // The device only ever increments the doorbell, so zero is unambiguously
  // the host's stop request.
  hsa::signalStore(CoreApi, Doorbell, HOSTCALL_SIGNAL_DONE);
  if (Thread.joinable())
    Thread.join();

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    Buffers.clear();
  }
  return hsa::signalDestroy(CoreApi, Doorbell);
}

void HostcallListener::consumePackets() {
  /// Bounds on how long a single doorbell wait blocks, in system-clock ticks.
  /// The wait backs off while the device is quiet and tightens up as soon as
  /// traffic arrives, so an idle listener costs almost nothing while a
  /// printf-heavy kernel is answered promptly.
  constexpr uint64_t TimeoutFloor = 4ULL * 1024 * 1024;
  constexpr uint64_t TimeoutCeiling = 16ULL * 1024 * 1024;

  uint64_t Timeout = TimeoutFloor;
  hsa_signal_value_t LastSeen = HOSTCALL_SIGNAL_INIT;

  while (true) {
    // Wait for the doorbell to move off the value already handled.
    while (true) {
      const hsa_signal_value_t Observed =
          hsa::signalWaitTimeout(CoreApi, Doorbell, HSA_SIGNAL_CONDITION_NE,
                                 LastSeen, Timeout, HSA_WAIT_STATE_BLOCKED);
      if (Observed != LastSeen) {
        LastSeen = Observed;
        Timeout = std::max(TimeoutFloor, Timeout >> 1);
        break;
      }
      Timeout = std::min(TimeoutCeiling, Timeout << 1);
    }

    if (LastSeen == HOSTCALL_SIGNAL_DONE)
      return;

    std::lock_guard<std::mutex> Lock(Mutex);
    for (HostcallBuffer *Buffer : Buffers)
      Buffer->processPackets();
  }
}

} // namespace luthier
