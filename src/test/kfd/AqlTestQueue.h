//===-- AqlTestQueue.h - a minimal AQL queue built on libhsakmt -----------===//
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
/// \file
/// An AQL compute queue created directly through libhsakmt, with **no HSA
/// runtime involved**. This is what makes the test suite a valid vehicle for
/// issue #85: \c kfd-nonhsa-tests must not link \c libhsa-runtime64, and a
/// CTest check enforces that against the built binary.
///
/// Deliberately not built into \c kfd-hsa-oracle, which does link the runtime.
/// Only the harness-neutral parts -- the scenario list and the verification
/// checks -- are shared between the two, so nothing here can pull hsakmt into a
/// process that already has the runtime's own copy of it.
///
/// \par Why this is hand-rolled
/// AMD's own \c kfdtest has an \c AqlQueue class, but it cannot express a real
/// AQL queue and was never exercised: it passes the read and write counters as
/// two adjacent array entries, whereas a real queue keeps them inside an
/// \c amd_queue_t at different offsets; and its submit path counts dwords, which
/// is the PM4 convention, where an AQL write index counts whole packets. The
/// thunk takes those two addresses as plain inputs, so calling it directly is
/// both simpler and correct.
///
/// \par What the GPU needs
/// Two structures that firmware reads, both filled in here:
/// \li an \c amd_queue_t, holding the packet counters and a few limits
/// \li an \c amd_kernel_code_t (the "kernel descriptor"), which tells the GPU
///     where the instructions start and how many registers they need
///
/// Field positions are taken with \c offsetof and cross-checked against values
/// observed on live runtime queues, so a layout change fails at compile time
/// rather than silently pointing the GPU at the wrong word.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_KFD_AQL_TEST_QUEUE_H
#define LUTHIER_TEST_KFD_AQL_TEST_QUEUE_H

#include <cstdint>
#include <string>
#include <vector>

namespace luthier::test::kfd {

/// \brief A GPU-visible buffer.
///
/// Ordinary process memory is not reachable by the GPU until the driver has been
/// told about it, so every buffer the GPU touches goes through here.
class GpuBuffer {
public:
  GpuBuffer() = default;
  ~GpuBuffer();
  GpuBuffer(const GpuBuffer &) = delete;
  GpuBuffer &operator=(const GpuBuffer &) = delete;
  GpuBuffer(GpuBuffer &&Other) noexcept;

  /// \param Node the topology node of the GPU that must reach this memory
  /// \param Bytes size; rounded up to a page
  /// \param Executable set for memory holding GPU instructions. This flag turned
  /// out to be load-bearing -- without it the GPU faults on a substituted ring
  /// \param Uncached bypass caches, so host writes are visible without an
  /// explicit flush
  bool allocate(uint32_t Node, size_t Bytes, bool Executable = false,
                bool Uncached = false);

  void *address() const { return Address; }
  size_t size() const { return Size; }
  template <typename T> T *as() const { return static_cast<T *>(Address); }

private:
  void *Address = nullptr;
  size_t Size = 0;
  uint32_t Node = 0;
};

/// \brief A kernel the test can dispatch: instructions plus their descriptor.
///
/// Laid out as one buffer with the 64-byte descriptor first and the instructions
/// at a fixed aligned offset after it. Choosing the layout ourselves means the
/// entry offset is ours to compute rather than something a linker decided for a
/// different arrangement.
class TestKernel {
public:
  /// \param Node the GPU that will run it
  /// \param MachineCode assembled instructions for that GPU's architecture
  /// \param KernargBytes size of the argument block the kernel expects
  bool load(uint32_t Node, const std::vector<uint8_t> &MachineCode,
            uint32_t KernargBytes);

  /// Address to put in a dispatch packet's \c kernel_object field.
  uint64_t descriptorAddress() const;

private:
  GpuBuffer Storage;
};

/// \brief A minimal AQL compute queue.
class AqlTestQueue {
public:
  AqlTestQueue() = default;
  ~AqlTestQueue();
  AqlTestQueue(const AqlTestQueue &) = delete;
  AqlTestQueue &operator=(const AqlTestQueue &) = delete;

  /// \param Node topology node of the GPU to create the queue on
  /// \param RingBytes ring size; must be a multiple of 64
  /// \param PrefillInvalid mark every slot "empty" before the queue goes live,
  /// which is what the HSA runtime does (\c amd_aql_queue.cpp:122). Leave it on
  /// unless the test is specifically about what a raw ring contains: the
  /// allocator recycles pages, so a fresh ring can arrive holding a previous
  /// queue's packets, and leaving them there intermittently wedges the queue.
  /// See \c S18-initial-ring-contents.
  bool create(uint32_t Node, uint32_t RingBytes, bool PrefillInvalid = true);
  bool destroy();

  uint32_t slotCount() const { return SlotCount; }
  uint32_t queueId() const { return QueueId; }

  /// \brief Deliberately leaves the ring exactly as the allocator provided it.
  ///
  /// A real HSA queue arrives with every slot marked "empty", because the HSA
  /// runtime writes those markers in software. The driver promises nothing of
  /// the kind, so a queue created this way starts as whatever the allocator
  /// left. That difference is the point of this harness, not an oversight.
  const uint16_t *ringHeadersForInspection() const;

  /// \brief Publish one packet.
  ///
  /// Waits for a free slot first. That wait is not optional: without it the
  /// producer overwrites packets the GPU has not run yet, which cost this
  /// project 127 of 199 dispatches the first time it was omitted.
  ///
  /// \param Packet 64 bytes, header included
  /// \return false if the GPU stopped making progress
  bool submit(const void *Packet);

  /// Number of packets published so far.
  uint64_t submittedCount() const;

  /// Number of packets the GPU reports having finished.
  ///
  /// Read straight out of the queue descriptor, so it says what the hardware
  /// thinks rather than what we hoped. Comparing it with \c submittedCount is
  /// what separates "the GPU never ran our work" from "it ran and the result
  /// did not reach us" -- two failures that look identical from the
  /// destination buffer alone.
  uint64_t completedCount() const;

  /// How many submissions had to wait for the GPU to free a slot.
  ///
  /// Lets a test prove the ring actually filled, rather than assuming it did.
  /// Submitting more packets than the ring holds does not by itself mean the
  /// producer ever blocked -- if the GPU keeps up, slots free as fast as they
  /// are used and the wait never triggers.
  uint64_t timesBlocked() const { return BlockedCount; }

private:
  /// \brief Fill in the queue descriptor's scratch fields.
  ///
  /// Scratch is where a kernel spills registers it has run out of. A kernel that
  /// needs none -- like the one this suite dispatches -- needs nothing here, so
  /// the obvious thing is to leave it alone. That is a trap: an \b instrumented
  /// version of that same kernel almost always spills, because the injected code
  /// wants registers the original was already using.
  ///
  /// Setting scratch up has two halves, and only one is per process:
  ///
  /// \li a backing address for the GPU's private aperture, set by
  ///     \c SET_SCRATCH_BACKING_VA -- which takes a \c gpu_id and no queue id,
  ///     so it is per process and per GPU;
  /// \li the fields written here, which firmware reads out of this queue's own
  ///     \c amd_queue_t.
  ///
  /// \c compute_tmpring_size is the one that decides whether a dispatch runs at
  /// all. Its \c WAVES field is how many wavefronts are permitted to use
  /// scratch, and zero permits none -- so the command processor cannot place a
  /// single wavefront, never launches the packet, and never advances the read
  /// pointer. The failure has no error and no fault attached to it; the queue
  /// simply reports nothing consumed.
  ///
  /// Mirrors ROCr's own gfx9 setup (\c AqlQueue::InitScratchSRD and
  /// \c FillComputeTmpRingSize in \c core/runtime/amd_aql_queue.cpp) rather
  /// than inventing an encoding.
  ///
  /// \param NodeId the topology node this queue belongs to
  /// \param QueueDescriptor the \c amd_queue_v2_t being filled in
  /// \param NodeProps the \c HsaNodeProperties already read for this node.
  ///        Both are \c void* to keep hsakmt's types out of this header.
  bool setUpScratch(uint32_t NodeId, void *QueueDescriptor,
                    const void *NodeProps);

  GpuBuffer Ring;
  GpuBuffer Descriptor; ///< holds the amd_queue_t
  GpuBuffer Scratch;    ///< spill space, for kernels with a private segment
  uint32_t SlotCount = 0;
  uint32_t QueueId = 0;
  uint64_t BlockedCount = 0;
  bool Created = false;
  void *Resources = nullptr; ///< opaque HsaQueueResource, kept out of the header
};

/// \brief Machine code for the small test kernel, for the given GPU.
///
/// Built at configure time for each architecture the build targets, so the suite
/// carries no runtime assembler and no dependency on a particular LLVM version.
/// \return empty when the architecture was not built for.
std::vector<uint8_t> testKernelCodeFor(const std::string &GfxArch);

} // namespace luthier::test::kfd

#endif // LUTHIER_TEST_KFD_AQL_TEST_QUEUE_H
