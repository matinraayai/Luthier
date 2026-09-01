//===-- AqlTestQueue.cpp - a minimal AQL queue built on libhsakmt ---------===//
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
#include "AqlTestQueue.h"

#include <cstdio>
#include <cstring>

#include <hsakmt/hsakmt.h>

// Byte-exact structures that GPU firmware reads. A standard ROCm install ships
// these, so the suite needs no source checkout to build. Headers only -- this
// file must not pull in the HSA runtime, and the build checks that it hasn't.
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>

namespace luthier::test::kfd {

namespace {

constexpr uint32_t AqlPacketBytes = 64;
/// Instructions start here, after the 64-byte descriptor, at an aligned offset.
constexpr uint32_t KernelCodeOffset = 256;

// The counters live inside the queue descriptor. Their positions are derived,
// never written as constants -- these assertions only cross-check the derived
// values against what was measured on live runtime queues, so a layout change
// fails here rather than silently misdirecting the GPU.
static_assert(offsetof(amd_queue_v2_t, write_dispatch_id) == 0x38,
              "write counter moved; the wrapper's assumptions need review");
static_assert(offsetof(amd_queue_v2_t, read_dispatch_id) == 0x80,
              "read counter moved; the wrapper's assumptions need review");

} // namespace

//===----------------------------------------------------------------------===//
// GpuBuffer
//===----------------------------------------------------------------------===//

GpuBuffer::GpuBuffer(GpuBuffer &&Other) noexcept
    : Address(Other.Address), Size(Other.Size), Node(Other.Node) {
  Other.Address = nullptr;
  Other.Size = 0;
}

GpuBuffer::~GpuBuffer() {
  if (Address != nullptr) {
    hsaKmtUnmapMemoryToGPU(Address);
    hsaKmtFreeMemory(Address, Size);
    Address = nullptr;
  }
}

bool GpuBuffer::allocate(uint32_t NodeId, size_t Bytes, bool Executable,
                         bool Uncached) {
  HsaMemFlags Flags = {};
  Flags.ui32.HostAccess = 1;
  Flags.ui32.NonPaged = 0;
  Flags.ui32.ExecuteAccess = Executable ? 1 : 0;
  Flags.ui32.CoarseGrain = Uncached ? 0 : 1;
  Flags.ui32.Uncached = Uncached ? 1 : 0;

  void *Ptr = nullptr;
  if (hsaKmtAllocMemory(NodeId, Bytes, Flags, &Ptr) != HSAKMT_STATUS_SUCCESS)
    return false;

  // Allocating is not enough: the GPU cannot reach the memory until it has been
  // mapped into that GPU's address space.
  if (hsaKmtMapMemoryToGPU(Ptr, Bytes, nullptr) != HSAKMT_STATUS_SUCCESS) {
    hsaKmtFreeMemory(Ptr, Bytes);
    return false;
  }

  Address = Ptr;
  Size = Bytes;
  Node = NodeId;
  return true;
}

//===----------------------------------------------------------------------===//
// TestKernel
//===----------------------------------------------------------------------===//

bool TestKernel::load(uint32_t Node, const std::vector<uint8_t> &MachineCode,
                      uint32_t KernargBytes) {
  if (MachineCode.empty())
    return false;

  const size_t Bytes = KernelCodeOffset + MachineCode.size();
  if (!Storage.allocate(Node, (Bytes + 4095) & ~size_t(4095),
                        /*Executable=*/true))
    return false;

  auto *Base = Storage.as<uint8_t>();
  memset(Base, 0, KernelCodeOffset);
  memcpy(Base + KernelCodeOffset, MachineCode.data(), MachineCode.size());

  HsaNodeProperties Props = {};
  if (hsaKmtGetNodeProperties(Node, &Props) != HSAKMT_STATUS_SUCCESS)
    return false;

  auto *Kd = reinterpret_cast<amd_kernel_code_t *>(Base);
  Kd->amd_kernel_code_version_major = AMD_KERNEL_CODE_VERSION_MAJOR;
  Kd->amd_kernel_code_version_minor = AMD_KERNEL_CODE_VERSION_MINOR;
  Kd->amd_machine_kind = AMD_MACHINE_KIND_AMDGPU;
  // Chip version from the driver, so this is not tied to one architecture.
  Kd->amd_machine_version_major = Props.EngineId.ui32.Major;
  Kd->amd_machine_version_minor = Props.EngineId.ui32.Minor;
  Kd->amd_machine_version_stepping = Props.EngineId.ui32.Stepping;

  Kd->kernel_code_entry_byte_offset = KernelCodeOffset;

  // Conservative register counts, matching what the driver's own test suite
  // uses for shaders of this size: 32 VGPRs and the standard SGPR granule.
  Kd->compute_pgm_rsrc1 = (0xc0 << 12) | (0x2 << 6) | 0x4;
  // Two user SGPRs: exactly the kernarg pointer the kernel expects in s[0:1].
  Kd->compute_pgm_rsrc2 = (2 << 1) | (1 << 7);
  // Ask for the kernarg pointer to be preloaded. AQL passes a *pointer* to the
  // arguments, unlike the PM4 path which preloads the argument values.
  Kd->kernel_code_properties =
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR;

  Kd->workitem_private_segment_byte_size = 0; // no scratch
  Kd->workgroup_group_segment_byte_size = 0;  // no LDS
  Kd->kernarg_segment_byte_size = KernargBytes;
  Kd->wavefront_sgpr_count = 8;
  Kd->workitem_vgpr_count = 4;
  Kd->kernarg_segment_alignment = 4; // 2^4 = 16 bytes
  Kd->group_segment_alignment = 4;
  Kd->private_segment_alignment = 4;
  Kd->wavefront_size = 6; // 2^6 = 64
  return true;
}

uint64_t TestKernel::descriptorAddress() const {
  return reinterpret_cast<uint64_t>(Storage.address());
}

//===----------------------------------------------------------------------===//
// AqlTestQueue
//===----------------------------------------------------------------------===//

AqlTestQueue::~AqlTestQueue() { destroy(); }

bool AqlTestQueue::create(uint32_t Node, uint32_t RingBytes,
                          bool PrefillInvalid) {
  if (RingBytes % AqlPacketBytes != 0)
    return false;
  SlotCount = RingBytes / AqlPacketBytes;

  // The ring must be executable and uncached, like a real queue's ring.
  if (!Ring.allocate(Node, RingBytes, /*Executable=*/true, /*Uncached=*/true))
    return false;

  // Mark every slot empty before the queue can be scheduled. The driver
  // promises nothing about a fresh ring's contents, and the allocator hands
  // back recycled pages -- so a slot can arrive holding a previous queue's
  // packet, header and all.
  if (PrefillInvalid) {
    auto *Slots = Ring.as<uint8_t>();
    const uint16_t Invalid = HSA_PACKET_TYPE_INVALID;
    for (uint32_t I = 0; I < SlotCount; I++)
      memcpy(Slots + I * AqlPacketBytes, &Invalid, sizeof(Invalid));
  }
  if (!Descriptor.allocate(Node, 4096, /*Executable=*/false, /*Uncached=*/true))
    return false;

  auto *Q = Descriptor.as<amd_queue_v2_t>();
  memset(Q, 0, sizeof(*Q));
  Q->hsa_queue.type = HSA_QUEUE_TYPE_SINGLE;
  Q->hsa_queue.base_address = Ring.address();
  Q->hsa_queue.size = SlotCount;
  Q->queue_properties = AMD_QUEUE_PROPERTIES_IS_PTR64;
  Q->read_dispatch_id_field_base_byte_offset =
      static_cast<uint32_t>(offsetof(amd_queue_v2_t, read_dispatch_id));

  HsaNodeProperties Props = {};
  if (hsaKmtGetNodeProperties(Node, &Props) != HSAKMT_STATUS_SUCCESS)
    return false;
  Q->max_cu_id = (Props.NumFComputeCores / Props.NumSIMDPerCU) - 1;
  Q->max_wave_id = (Props.MaxWavesPerSIMD * Props.NumSIMDPerCU) - 1;

  // Off only as a control, to reproduce the queue an application that needs no
  // scratch would build -- on which an instrumented kernel cannot run.
  if (getenv("LUTHIER_KFD_TEST_NO_SCRATCH") == nullptr &&
      !setUpScratch(Node, Q, &Props))
    return false;

  auto *Res = new HsaQueueResource();
  memset(Res, 0, sizeof(*Res));
  // The counters live inside the descriptor, at their real offsets. This is the
  // part kfdtest's adjacent-pointers interface cannot express.
  // The casts drop `volatile`: the fields are declared volatile because the GPU
  // writes them, but the thunk's struct takes plain pointers.
  Res->Queue_read_ptr_aql =
      const_cast<HSAuint64 *>(reinterpret_cast<volatile HSAuint64 *>(
          &Q->read_dispatch_id));
  Res->Queue_write_ptr_aql =
      const_cast<HSAuint64 *>(reinterpret_cast<volatile HSAuint64 *>(
          &Q->write_dispatch_id));

  HSAKMT_STATUS St = hsaKmtCreateQueue(
      Node, HSA_QUEUE_COMPUTE_AQL, 100, HSA_QUEUE_PRIORITY_NORMAL,
      Ring.address(), RingBytes, nullptr, Res);
  if (St != HSAKMT_STATUS_SUCCESS) {
    delete Res;
    return false;
  }

  Resources = Res;
  QueueId = static_cast<uint32_t>(Res->QueueId);
  Created = true;
  return true;
}

bool AqlTestQueue::setUpScratch(uint32_t NodeId, void *QueueDescriptor,
                                const void *NodeProps) {
  auto *Q = static_cast<amd_queue_v2_t *>(QueueDescriptor);
  const auto &Props = *static_cast<const HsaNodeProperties *>(NodeProps);

  // gfx9 numbers, taken from ROCr: waves are 64 lanes wide and scratch is
  // measured in 1KB units (gfx10 and later use 256B).
  constexpr uint32_t LanesPerWave = 64;
  constexpr uint32_t MemAlignment = 1024;
  // How much spill room each work-item gets. Generous on purpose: an
  // instrumented kernel asks for a handful of bytes, and sizing this off any
  // one kernel would mean re-creating the queue whenever a bigger one showed
  // up -- which is the resizing machinery ROCr has and this harness does not.
  constexpr uint32_t BytesPerThread = 64;

  const uint32_t NumXcc = Props.NumXcc != 0 ? Props.NumXcc : 1;
  const uint32_t NumCus = Props.NumFComputeCores / Props.NumSIMDPerCU;
  const uint32_t MaxScratchWaves = NumCus * Props.MaxSlotsScratchCU;
  const size_t ScratchBytes = static_cast<size_t>(BytesPerThread) *
                              Props.MaxSlotsScratchCU * LanesPerWave * NumCus;

  if (!Scratch.allocate(NodeId, ScratchBytes, /*Executable=*/false,
                        /*Uncached=*/false))
    return false;
  const auto Base = reinterpret_cast<uint64_t>(Scratch.address());

  // A buffer resource descriptor: four words the shader loads into SGPRs and
  // then addresses scratch through. Field positions are SQ_BUF_RSRC_WORD0..3
  // (ROCR-Runtime core/inc/registers.h); the values are ROCr's.
  Q->scratch_resource_descriptor[0] = static_cast<uint32_t>(Base);
  Q->scratch_resource_descriptor[1] =
      (static_cast<uint32_t>(Base >> 32) & 0xFFFFu) | // BASE_ADDRESS_HI : 16
      (1u << 31);                                     // SWIZZLE_ENABLE
  // NUM_RECORDS, reported per XCC.
  Q->scratch_resource_descriptor[2] =
      static_cast<uint32_t>(ScratchBytes / NumXcc);
  // ATC is set only on an APU, where the GPU walks the CPU's page tables. ROCr
  // keys this off its profile being "full", which it decides by asking whether
  // the node has CPU cores (amd_gpu_agent.cpp:121).
  const uint32_t Atc = Props.NumCPUCores > 0 ? 1u : 0u;
  Q->scratch_resource_descriptor[3] =
      (4u << 0) | (5u << 3) | (6u << 6) | (7u << 9) | // DST_SEL X,Y,Z,W
      (4u << 12) |                                    // NUM_FORMAT   uint
      (4u << 15) |                                    // DATA_FORMAT  32
      (1u << 19) |                                    // ELEMENT_SIZE 4 bytes
      (3u << 21) |                                    // INDEX_STRIDE 64
      (1u << 23) |                                    // ADD_TID_ENABLE
      (Atc << 24);                                    // ATC
                                                      // TYPE = buffer = 0

  // WAVESIZE is per-wave scratch in MemAlignment units; WAVES is how many waves
  // may use scratch at once. WAVES == 0 is what stops a dispatch dead, so this
  // is the field that matters most here.
  const uint32_t WaveScratch =
      ((LanesPerWave * BytesPerThread) + MemAlignment - 1) / MemAlignment;
  uint32_t NumWaves =
      static_cast<uint32_t>((ScratchBytes / NumXcc) / (WaveScratch * MemAlignment));
  if (NumWaves > MaxScratchWaves)
    NumWaves = MaxScratchWaves;
  Q->compute_tmpring_size =
      (NumWaves & 0xFFFu) | ((WaveScratch & 0x1FFFu) << 12);

  Q->scratch_backing_memory_location = 0;
  Q->scratch_backing_memory_byte_size = ScratchBytes;
  // Recorded for a 64-lane wave, which is what gfx9 has.
  Q->scratch_wave64_lane_byte_size =
      static_cast<uint32_t>((static_cast<uint64_t>(BytesPerThread) * LanesPerWave) / 64);
  return true;
}

bool AqlTestQueue::destroy() {
  if (!Created)
    return true;
  Created = false;
  auto *Res = static_cast<HsaQueueResource *>(Resources);
  HSAKMT_STATUS St = hsaKmtDestroyQueue(Res->QueueId);
  delete Res;
  Resources = nullptr;
  return St == HSAKMT_STATUS_SUCCESS;
}

const uint16_t *AqlTestQueue::ringHeadersForInspection() const {
  return static_cast<const uint16_t *>(Ring.address());
}

uint64_t AqlTestQueue::submittedCount() const {
  const auto *Q = Descriptor.as<amd_queue_v2_t>();
  return Q->write_dispatch_id;
}

uint64_t AqlTestQueue::completedCount() const {
  const auto *Q = Descriptor.as<amd_queue_v2_t>();
  return Q->read_dispatch_id;
}

bool AqlTestQueue::submit(const void *Packet) {
  auto *Q = Descriptor.as<amd_queue_v2_t>();
  const uint64_t Index = Q->write_dispatch_id;

  // Wait for the slot to be free. The GPU reports how far it has got in
  // read_dispatch_id; until it passes Index - SlotCount, the slot we want still
  // holds work that has not run. Skipping this silently destroys packets.
  unsigned Spins = 0;
  while (Index - Q->read_dispatch_id >= SlotCount) {
    if (Spins == 0)
      BlockedCount++; // count submissions that had to wait, not spin iterations
    if (++Spins > 200000000) {
      fprintf(stderr, "[kfd-test] submit timed out waiting for the GPU\n");
      return false;
    }
  }

  auto *Slot = static_cast<uint8_t *>(Ring.address()) +
               (Index % SlotCount) * AqlPacketBytes;
  const auto *Src = static_cast<const uint8_t *>(Packet);

  // Body first, header last: the header is what makes a packet live, so writing
  // it last means the GPU never sees a half-written packet.
  memcpy(Slot + 2, Src + 2, AqlPacketBytes - 2);
  uint16_t Header;
  memcpy(&Header, Src, sizeof(Header));
  __atomic_store_n(reinterpret_cast<uint16_t *>(Slot), Header,
                   __ATOMIC_RELEASE);

  // Publish the count, then ring the doorbell.
  //
  // Note this is commit-then-publish, which is stricter than what a real
  // runtime does -- ROCr claims the slot first, so its counter briefly runs
  // ahead of the packet. The strict order is deliberate here: this harness is
  // the reference the wrapper is judged against, and a racy oracle is worthless.
  // The claim-first timing is covered separately by running a real runtime.
  __atomic_store_n(&Q->write_dispatch_id, Index + 1, __ATOMIC_RELEASE);
  auto *Res = static_cast<HsaQueueResource *>(Resources);
  *Res->Queue_DoorBell_aql = Index + 1;
  return true;
}

} // namespace luthier::test::kfd
