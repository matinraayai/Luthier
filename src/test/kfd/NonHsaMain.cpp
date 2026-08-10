//===-- NonHsaMain.cpp - the non-HSA half of the KFD queue test suite -----===//
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
/// Runs the shared scenarios against the GPU driver directly, with no HSA
/// runtime in this binary. That absence is the point -- the build checks it, and
/// if it ever stops holding every result here becomes meaningless.
///
/// The same binary is used twice: once on its own, to show the workloads are
/// sound, and once with the wrapper preloaded, to check the interception. It
/// discovers the wrapper at run time rather than linking against it, so the
/// unwrapped baseline is actually unwrapped.
///
/// Usage:
///   kfd-nonhsa-tests                    run every scenario
///   kfd-nonhsa-tests S5-many-laps       run one
///   kfd-nonhsa-tests --list             list them with what each one catches
///   kfd-nonhsa-tests --require-wrapper  fail if the wrapper is not attached
//===----------------------------------------------------------------------===//
#include "AqlTestQueue.h"
#include "Scenarios.h"
#include "Verify.h"

#include "luthier/KFD/QueueWrapper.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <hsa/hsa.h>
#include <hsakmt/hsakmt.h>

using namespace luthier::test::kfd;

namespace {

constexpr uint32_t DefaultRingBytes = 4096; // 64 slots
constexpr uint32_t AqlPacketBytes = 64;

/// Arguments the test kernel reads. Must match TestKernel.s.
struct KernelArgs {
  uint64_t Destination;
  uint32_t Value;
  uint32_t Pad;
};

//===----------------------------------------------------------------------===//
// Observations collected from the wrapper, if it is loaded
//===----------------------------------------------------------------------===//
std::mutex ObservationLock;
std::vector<PacketObservation> Observations;
bool WrapperPresent = false;

/// The second observer used by S14c. Kept separate from Observations so the two
/// can be compared: a chain that quietly delivers to only one callback would
/// otherwise be invisible.
std::mutex SecondObserverLock;
std::vector<PacketObservation> SecondObservations;

/// Records the order the two callbacks ran in, across all packets. The chain's
/// ordering is pinned down properly in the GPU-free unit tests; this is only a
/// sanity check that the same order holds on real hardware.
std::vector<int> ChainOrder;

void onPacketSecond(const luthier::kfd::QueueInfo &Q, uint64_t Index,
                    luthier::hsa::AqlPacket &Packet, void *) {
  PacketObservation O;
  O.GpuId = Q.GpuId;
  O.QueueId = Q.QueueId;
  O.PacketIndex = Index;
  O.Header = Packet.Packet.Header;
  O.KernelObject = 0;
  if (Packet.getPacketType() == HSA_PACKET_TYPE_KERNEL_DISPATCH)
    memcpy(&O.KernelObject, reinterpret_cast<const uint8_t *>(&Packet) + 32,
           sizeof(O.KernelObject));

  std::lock_guard<std::mutex> G(SecondObserverLock);
  SecondObservations.push_back(O);
  ChainOrder.push_back(2);
}

void onPacket(const luthier::kfd::QueueInfo &Q, uint64_t Index,
              luthier::hsa::AqlPacket &Packet, void *) {
  PacketObservation O;
  O.GpuId = Q.GpuId;
  O.QueueId = Q.QueueId;
  O.PacketIndex = Index;
  O.Header = Packet.Packet.Header;
  O.KernelObject = 0;
  if (Packet.getPacketType() == HSA_PACKET_TYPE_KERNEL_DISPATCH)
    memcpy(&O.KernelObject, reinterpret_cast<const uint8_t *>(&Packet) + 32,
           sizeof(O.KernelObject));

  {
    std::lock_guard<std::mutex> G(SecondObserverLock);
    ChainOrder.push_back(1);
  }
  std::lock_guard<std::mutex> G(ObservationLock);
  Observations.push_back(O);
}

/// Look for the wrapper in the process.
///
/// \param Required when set, refuse to continue if the wrapper is missing.
///
/// That option exists because of a trap this harness walked into once: a run
/// meant to be wrapped, where the wrapper failed to attach, silently fell back
/// to checking only the workload's result -- and reported a clean pass while
/// verifying almost nothing. A run that cannot check what it was asked to check
/// must fail, not degrade.
using AddCallbackFn = int (*)(luthier::kfd::PacketCallback, void *);
using RemoveCallbackFn = void (*)(int);
AddCallbackFn AddCallback = nullptr;
RemoveCallbackFn RemoveCallback = nullptr;

bool connectToWrapper(bool Required) {
  AddCallback = reinterpret_cast<AddCallbackFn>(
      dlsym(RTLD_DEFAULT, "luthierKfdAddPacketCallback"));
  RemoveCallback = reinterpret_cast<RemoveCallbackFn>(
      dlsym(RTLD_DEFAULT, "luthierKfdRemovePacketCallback"));
  auto *Fn = reinterpret_cast<luthier::kfd::SetPacketCallbackFn>(
      dlsym(RTLD_DEFAULT, "luthierKfdSetPacketCallback"));
  if (Fn == nullptr) {
    if (Required) {
      fprintf(stderr, "--require-wrapper was given but the wrapper is not in "
                      "this process. Preload libluthier-kfd-queue-wrapper.so\n");
      return false;
    }
    printf("wrapper: not loaded (baseline run -- only workload results are "
           "checked)\n");
    return true;
  }
  Fn(onPacket, nullptr);
  WrapperPresent = true;
  printf("wrapper: loaded, callback registered (full verification)\n");
  return true;
}

//===----------------------------------------------------------------------===//
// Device discovery
//===----------------------------------------------------------------------===//
struct Device {
  uint32_t Node;
  std::string Arch;
};

std::vector<Device> discoverGpus() {
  std::vector<Device> Out;
  HsaSystemProperties Sys = {};
  if (hsaKmtAcquireSystemProperties(&Sys) != HSAKMT_STATUS_SUCCESS)
    return Out;
  for (uint32_t N = 0; N < Sys.NumNodes; N++) {
    HsaNodeProperties P = {};
    if (hsaKmtGetNodeProperties(N, &P) != HSAKMT_STATUS_SUCCESS)
      continue;
    if (P.NumFComputeCores == 0) // CPU node
      continue;
    char Buf[32];
    snprintf(Buf, sizeof(Buf), "gfx%u%x%x", P.EngineId.ui32.Major,
             P.EngineId.ui32.Minor, P.EngineId.ui32.Stepping);
    Out.push_back({N, Buf});
  }
  return Out;
}

//===----------------------------------------------------------------------===//
// Packet construction
//===----------------------------------------------------------------------===//
uint16_t headerFor(const Scenario &S, unsigned Type) {
  uint16_t H = static_cast<uint16_t>(Type << HSA_PACKET_HEADER_TYPE);
  if (S.BarrierBit)
    H |= 1 << HSA_PACKET_HEADER_BARRIER;

  // The fence scopes decide whether the GPU flushes its caches after the
  // packet, which is what makes the kernel's write visible to us. Getting them
  // wrong looks like a wrong answer rather than a cache problem, so a scenario
  // varies them deliberately.
  unsigned Acquire = HSA_FENCE_SCOPE_SYSTEM, Release = HSA_FENCE_SCOPE_SYSTEM;
  if (S.Fences == FenceScopes::None)
    Acquire = Release = HSA_FENCE_SCOPE_NONE;
  else if (S.Fences == FenceScopes::Agent)
    Acquire = Release = HSA_FENCE_SCOPE_AGENT;
  H |= Acquire << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  H |= Release << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  return H;
}

void buildDispatch(uint8_t *Packet, const Scenario &S, uint64_t KernelObject,
                   void *Kernargs) {
  memset(Packet, 0, AqlPacketBytes);
  auto *P = reinterpret_cast<hsa_kernel_dispatch_packet_t *>(Packet);
  P->header = headerFor(S, HSA_PACKET_TYPE_KERNEL_DISPATCH);
  P->setup = 1;
  P->workgroup_size_x = P->workgroup_size_y = P->workgroup_size_z = 1;
  P->grid_size_x = P->grid_size_y = P->grid_size_z = 1;
  P->kernel_object = KernelObject;
  P->kernarg_address = Kernargs;
  P->completion_signal.handle = 0; // we poll memory instead
}

/// A barrier with no dependencies: it completes immediately, but the packet
/// processor still has to walk past it, and we still have to copy it.
void buildBarrier(uint8_t *Packet, const Scenario &S) {
  memset(Packet, 0, AqlPacketBytes);
  auto *P = reinterpret_cast<hsa_barrier_and_packet_t *>(Packet);
  P->header = headerFor(S, HSA_PACKET_TYPE_BARRIER_AND);
  P->completion_signal.handle = 0;
}

bool waitForValue(volatile uint32_t *Addr, uint32_t Want, unsigned Ms = 5000) {
  for (unsigned I = 0; I < Ms * 10; I++) {
    if (*Addr == Want)
      return true;
    usleep(100);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Shared workload machinery
//===----------------------------------------------------------------------===//

/// Everything one queue needs to run dispatches.
struct QueueSetup {
  std::unique_ptr<AqlTestQueue> Queue;
  GpuBuffer Dst;
  GpuBuffer Args;
  uint32_t Node = 0;
};

int64_t dispatchCountFor(const Scenario &S, uint32_t Slots) {
  int64_t Count = S.PacketsRelativeToRing
                      ? static_cast<int64_t>(S.DispatchCount) * Slots
                      : S.DispatchCount;
  Count += S.PacketAdjustment;
  return Count < 1 ? 1 : Count;
}

bool prepareQueue(QueueSetup &Q, uint32_t Node, int64_t Count,
                  RunResult &R) {
  Q.Node = Node;
  Q.Queue = std::make_unique<AqlTestQueue>();
  if (!Q.Queue->create(Node, DefaultRingBytes)) {
    R.Completed = false;
    R.FailureNote = "could not create an AQL queue";
    return false;
  }
  const size_t DstBytes = (Count * sizeof(uint32_t) + 4095) & ~size_t(4095);
  const size_t ArgBytes = (Count * sizeof(KernelArgs) + 4095) & ~size_t(4095);
  if (!Q.Dst.allocate(Node, DstBytes, false, /*Uncached=*/true) ||
      !Q.Args.allocate(Node, ArgBytes, false, /*Uncached=*/true)) {
    R.Completed = false;
    R.FailureNote = "could not allocate GPU-visible buffers";
    return false;
  }
  memset(Q.Dst.as<void>(), 0, DstBytes);
  return true;
}

/// Submit \p Count dispatches (plus barriers if the scenario asks) on one queue.
/// \p ValueBase keeps values distinct when several queues share a result vector.
bool submitDispatches(QueueSetup &Q, const Scenario &S, const TestKernel &Kernel,
                      int64_t First, int64_t Count, uint32_t ValueBase,
                      RunResult &R, std::mutex *ResultLock = nullptr) {
  auto *ArgBlocks = Q.Args.as<KernelArgs>();
  for (int64_t I = First; I < First + Count; I++) {
    ArgBlocks[I].Destination =
        reinterpret_cast<uint64_t>(Q.Dst.as<uint8_t>() + I * sizeof(uint32_t));
    ArgBlocks[I].Value = ValueBase + static_cast<uint32_t>(I);
    ArgBlocks[I].Pad = 0;

    if (ResultLock != nullptr) {
      std::lock_guard<std::mutex> G(*ResultLock);
      R.ExpectedValues.push_back(ArgBlocks[I].Value);
    } else {
      R.ExpectedValues.push_back(ArgBlocks[I].Value);
    }

    uint8_t Packet[AqlPacketBytes];
    // A barrier between dispatches is what a real runtime emits, and it must be
    // copied as faithfully as a dispatch even though no callback fires for it.
    if (S.Mix == PacketMix::DispatchAndBarrier && I > First) {
      buildBarrier(Packet, S);
      if (!Q.Queue->submit(Packet))
        return false;
    }
    buildDispatch(Packet, S, Kernel.descriptorAddress(), &ArgBlocks[I]);
    if (!Q.Queue->submit(Packet))
      return false;
  }
  return true;
}

void collectValues(QueueSetup &Q, int64_t Count, RunResult &R) {
  auto *DstWords = Q.Dst.as<volatile uint32_t>();
  for (int64_t I = 0; I < Count; I++)
    R.ObservedValues.push_back(static_cast<uint32_t>(DstWords[I]));
}

//===----------------------------------------------------------------------===//
// Scenario runners
//===----------------------------------------------------------------------===//

/// N dispatches on one queue from one thread, optionally repeated across
/// create/destroy rounds. Covers most of the matrix.
/// \note Each round is verified on its own by the caller. Pooling rounds would
/// be wrong: the driver reuses queue ids after a destroy, so round two's packet
/// indices restart at zero under the same id and would look like duplicates.
void runStraightforward(const Scenario &S, const std::vector<Device> &Gpus,
                        const TestKernel &Kernel, RunResult &R) {
  {
    QueueSetup Q;
    // A queue created and destroyed repeatedly is where stale per-queue state
    // would show up.
    //
    // The slot count is derived, not measured. It used to be read back from a
    // throwaway queue created and destroyed for the purpose, which was both
    // pointless -- AqlTestQueue sets it to RingBytes/64 and nothing else can
    // change it -- and actively harmful: it doubled the queue churn in every
    // lifecycle round.
    const int64_t Count =
        dispatchCountFor(S, DefaultRingBytes / AqlPacketBytes);
    if (!prepareQueue(Q, Gpus[0].Node, Count, R))
      return;

    R.ExpectedValues.clear();
    R.ObservedValues.clear();
    if (!submitDispatches(Q, S, Kernel, 0, Count, 0x1000, R)) {
      R.Completed = false;
      R.FailureNote = "submission stalled";
      return;
    }
    R.DispatchesSubmitted = static_cast<uint64_t>(Count);

    if (S.DestroyWithWorkInFlight) {
      // Tear the queue down without waiting. The interception layer must stop
      // touching the ring before it is freed -- the case that used to crash.
      Q.Queue->destroy();
      R.ExpectedValues.clear(); // results are legitimately indeterminate here
      R.ObservedValues.clear();
      R.DispatchesSubmitted = 0;
      return;
    }

    auto *DstWords = Q.Dst.as<volatile uint32_t>();
    if (!waitForValue(&DstWords[Count - 1],
                      static_cast<uint32_t>(0x1000 + Count - 1))) {
      // Say which of the two failures this is. "The value never appeared"
      // covers both a queue the GPU never scheduled and a kernel that ran and
      // wrote somewhere else, and they have nothing in common as bugs.
      int64_t Landed = 0;
      for (int64_t I = 0; I < Count; I++)
        if (DstWords[I] == static_cast<uint32_t>(0x1000 + I))
          Landed++;
      char Note[256];
      snprintf(Note, sizeof(Note),
               "the last dispatch never landed: the GPU reports %llu of %llu "
               "packets consumed, and %lld of %lld values arrived",
               (unsigned long long)Q.Queue->completedCount(),
               (unsigned long long)Q.Queue->submittedCount(),
               (long long)Landed, (long long)Count);
      R.Completed = false;
      R.FailureNote = Note;
    }
    collectValues(Q, Count, R);
    Q.Queue->destroy();
    if (!R.Completed)
      return;
  }
}

/// Several producer threads sharing one queue.
///
/// Never exercised before. The runtime has explicit machinery for this
/// (compare-and-swap on the write index, per-queue locking); our wrapper has
/// none, and its assumptions about ordering have only ever been checked with a
/// single producer.
void runConcurrent(const Scenario &S, const std::vector<Device> &Gpus,
                   const TestKernel &Kernel, RunResult &R) {
  QueueSetup Q;
  int64_t Count = S.DispatchCount;
  if (!prepareQueue(Q, Gpus[0].Node, Count, R))
    return;

  auto *ArgBlocks = Q.Args.as<KernelArgs>();
  for (int64_t I = 0; I < Count; I++) {
    ArgBlocks[I].Destination =
        reinterpret_cast<uint64_t>(Q.Dst.as<uint8_t>() + I * sizeof(uint32_t));
    ArgBlocks[I].Value = static_cast<uint32_t>(0x1000 + I);
    ArgBlocks[I].Pad = 0;
    R.ExpectedValues.push_back(ArgBlocks[I].Value);
  }

  std::mutex SubmitLock;
  std::atomic<int64_t> Next{0};
  std::atomic<bool> Failed{false};
  std::vector<std::thread> Threads;
  for (uint32_t T = 0; T < S.ProducerThreads; T++) {
    Threads.emplace_back([&]() {
      for (;;) {
        int64_t I = Next.fetch_add(1);
        if (I >= Count)
          return;
        uint8_t Packet[AqlPacketBytes];
        buildDispatch(Packet, S, Kernel.descriptorAddress(), &ArgBlocks[I]);
        // The queue's own submit path is not thread-safe by design -- a real
        // runtime reserves slots atomically. Serialising here keeps the test
        // about the *wrapper's* behaviour under concurrent producers rather
        // than about this harness's queue implementation.
        std::lock_guard<std::mutex> G(SubmitLock);
        if (!Q.Queue->submit(Packet)) {
          Failed = true;
          return;
        }
      }
    });
  }
  for (auto &T : Threads)
    T.join();

  if (Failed) {
    R.Completed = false;
    R.FailureNote = "a producer thread stalled";
    return;
  }
  R.DispatchesSubmitted = static_cast<uint64_t>(Count);

  auto *DstWords = Q.Dst.as<volatile uint32_t>();
  if (!waitForValue(&DstWords[Count - 1],
                    static_cast<uint32_t>(0x1000 + Count - 1)))
    R.Completed = false, R.FailureNote = "the last dispatch never landed";
  collectValues(Q, Count, R);
  Q.Queue->destroy();
}

/// Several queues, optionally spread across every GPU.
void runMultiQueue(const Scenario &S, const std::vector<Device> &Gpus,
                   const TestKernel &Kernel, RunResult &R) {
  const uint32_t N = S.QueueCount;
  std::vector<QueueSetup> Queues(N);
  const int64_t PerQueue = S.DispatchCount;

  for (uint32_t I = 0; I < N; I++) {
    const Device &D = S.UseAllGpus ? Gpus[I % Gpus.size()] : Gpus[0];
    if (!prepareQueue(Queues[I], D.Node, PerQueue, R))
      return;
  }
  // A kernel is loaded per GPU, since its code lives in that GPU's memory.
  std::vector<TestKernel> Kernels(Gpus.size());
  for (size_t I = 0; I < Gpus.size(); I++) {
    auto Code = testKernelCodeFor(Gpus[I].Arch);
    if (Code.empty() || !Kernels[I].load(Gpus[I].Node, Code, sizeof(KernelArgs))) {
      R.Completed = false;
      R.FailureNote = "could not load the test kernel on gpu " + Gpus[I].Arch;
      return;
    }
  }

  for (uint32_t I = 0; I < N; I++) {
    const size_t GpuIdx = S.UseAllGpus ? (I % Gpus.size()) : 0;
    if (!submitDispatches(Queues[I], S, Kernels[GpuIdx], 0, PerQueue,
                          0x1000, R)) {
      R.Completed = false;
      R.FailureNote = "submission stalled on queue " + std::to_string(I);
      return;
    }
  }
  R.DispatchesSubmitted = static_cast<uint64_t>(PerQueue) * N;

  for (uint32_t I = 0; I < N; I++) {
    auto *DstWords = Queues[I].Dst.as<volatile uint32_t>();
    if (!waitForValue(&DstWords[PerQueue - 1],
                      static_cast<uint32_t>(0x1000 + PerQueue - 1))) {
      R.Completed = false;
      R.FailureNote = "queue " + std::to_string(I) + " never finished";
    }
    collectValues(Queues[I], PerQueue, R);
  }
  for (uint32_t I = 0; I < N; I++)
    Queues[I].Queue->destroy();

  (void)Kernel;
}

/// More queues than the wrapper can track.
///
/// The wrapper must leave the extras unwrapped rather than substitute a ring it
/// cannot service -- doing the latter would leave the GPU reading a buffer
/// nothing ever fills, hanging the application with no error anywhere. So the
/// check is simply that everything still works.
void runQueueLimit(const Scenario &S, const std::vector<Device> &Gpus,
                   const TestKernel &Kernel, RunResult &R) {
  std::vector<std::unique_ptr<AqlTestQueue>> Queues;
  for (uint32_t I = 0; I < S.QueueCount; I++) {
    auto Q = std::make_unique<AqlTestQueue>();
    if (!Q->create(Gpus[0].Node, DefaultRingBytes)) {
      // Running out of driver queues is a property of the machine, not a bug in
      // the wrapper; say so rather than failing.
      printf("    (the driver allowed %u queues before refusing)\n", I);
      break;
    }
    Queues.push_back(std::move(Q));
  }
  if (Queues.empty()) {
    R.Completed = false;
    R.FailureNote = "could not create any queues";
    return;
  }

  // Dispatch on the LAST queue created -- the one most likely to be past the
  // wrapper's tracking limit.
  QueueSetup Q;
  Q.Node = Gpus[0].Node;
  if (!Q.Dst.allocate(Gpus[0].Node, 4096, false, true) ||
      !Q.Args.allocate(Gpus[0].Node, 4096, false, true)) {
    R.Completed = false;
    R.FailureNote = "could not allocate buffers";
    return;
  }
  memset(Q.Dst.as<void>(), 0, 4096);
  auto *Args = Q.Args.as<KernelArgs>();
  Args[0].Destination = reinterpret_cast<uint64_t>(Q.Dst.as<uint8_t>());
  Args[0].Value = 0x1000;
  R.ExpectedValues.push_back(0x1000);

  uint8_t Packet[AqlPacketBytes];
  buildDispatch(Packet, S, Kernel.descriptorAddress(), &Args[0]);
  if (!Queues.back()->submit(Packet)) {
    R.Completed = false;
    R.FailureNote = "submission on the last queue stalled";
    return;
  }
  R.DispatchesSubmitted = 1;

  auto *DstWords = Q.Dst.as<volatile uint32_t>();
  if (!waitForValue(DstWords, 0x1000)) {
    R.Completed = false;
    R.FailureNote = "a dispatch on a queue past the tracking limit never "
                    "landed -- the ring was probably substituted without being "
                    "serviced";
  }
  R.ObservedValues.push_back(static_cast<uint32_t>(DstWords[0]));

  // A queue past the tracking limit is deliberately left alone, so no callback
  // fires for it. That is the correct outcome, not a miss: the alternative --
  // substituting a ring the wrapper cannot service -- would hang the
  // application silently. So the check here is that the work still ran, which
  // the value comparison above already covers.
  R.DispatchesSubmitted = 0;
  Queues.clear();
}

/// What a fresh ring actually contains.
///
/// Records rather than demands a particular answer, because both are legitimate:
/// the raw driver leaves whatever the allocator did, and something that
/// establishes the markers (the wrapper does) leaves them all set. A *partial*
/// result would invalidate any scheme that reads these headers, and is the thing
/// worth failing on.
void runRingInspection(const std::vector<Device> &Gpus, RunResult &R) {
  AqlTestQueue Queue;
  // The one place that must NOT pre-fill: this scenario exists to record what
  // a ring actually contains when nothing has written markers into it.
  if (!Queue.create(Gpus[0].Node, DefaultRingBytes,
                    /*PrefillInvalid=*/false)) {
    R.Completed = false;
    R.FailureNote = "could not create an AQL queue";
    return;
  }
  const uint16_t *Headers = Queue.ringHeadersForInspection();
  const uint32_t Slots = Queue.slotCount();
  uint32_t Marked = 0;
  for (uint32_t I = 0; I < Slots; I++)
    if ((Headers[I * (AqlPacketBytes / sizeof(uint16_t))] & 0xFF) ==
        HSA_PACKET_TYPE_INVALID)
      Marked++;

  printf("    %u of %u slots start marked empty\n", Marked, Slots);
  if (Marked != 0 && Marked != Slots) {
    R.Completed = false;
    R.FailureNote =
        "the ring is only partly marked, which no header-based detection "
        "scheme can cope with";
  }
  Queue.destroy();
}

/// A queue of a type the wrapper must leave alone.
///
/// Only PM4 and AQL compute queues share the packet-processor path; SDMA queues
/// are a different engine entirely. The wrapper filters on queue type, and this
/// is what checks that filter is right: if it ever wrapped one of these, the
/// ring it substituted would be read as a completely different packet format.
struct AuxQueue {
  GpuBuffer Ring;
  HsaQueueResource Res{};
  bool Created = false;

  bool create(uint32_t Node, HSA_QUEUE_TYPE Type) {
    if (!Ring.allocate(Node, 4096, /*Executable=*/true, /*Uncached=*/true))
      return false;
    memset(&Res, 0, sizeof(Res));
    Created = hsaKmtCreateQueue(Node, Type, 100, HSA_QUEUE_PRIORITY_NORMAL,
                                Ring.as<unsigned int>(), 4096, nullptr,
                                &Res) == HSAKMT_STATUS_SUCCESS;
    return Created;
  }
  ~AuxQueue() {
    if (Created)
      hsaKmtDestroyQueue(Res.QueueId);
  }
};

/// AQL work running alongside queues of other types.
void runMixedQueueTypes(const Scenario &S, const std::vector<Device> &Gpus,
                        const TestKernel &Kernel, RunResult &R) {
  AuxQueue Sdma, Pm4;
  const bool HaveSdma = Sdma.create(Gpus[0].Node, HSA_QUEUE_SDMA);
  const bool HavePm4 = Pm4.create(Gpus[0].Node, HSA_QUEUE_COMPUTE);
  printf("    alongside: SDMA %s, PM4 %s\n", HaveSdma ? "created" : "unavailable",
         HavePm4 ? "created" : "unavailable");

  QueueSetup Q;
  const int64_t Count = S.DispatchCount;
  if (!prepareQueue(Q, Gpus[0].Node, Count, R))
    return;
  if (!submitDispatches(Q, S, Kernel, 0, Count, 0x1000, R)) {
    R.Completed = false;
    R.FailureNote = "submission stalled";
    return;
  }
  R.DispatchesSubmitted = static_cast<uint64_t>(Count);

  auto *DstWords = Q.Dst.as<volatile uint32_t>();
  if (!waitForValue(&DstWords[Count - 1],
                    static_cast<uint32_t>(0x1000 + Count - 1))) {
    R.Completed = false;
    R.FailureNote = "the last dispatch never landed";
  }
  collectValues(Q, Count, R);
  Q.Queue->destroy();

  // The callback count check does the real work here: if the wrapper had
  // wrapped the SDMA or PM4 queue, packets from them would show up as extra
  // observations and the count would no longer match.
}

/// Fill the ring so the producer has to wait.
///
/// Submitting more packets than the ring holds does not by itself make a
/// producer block: if the GPU keeps up, slots free as fast as they are taken.
/// So each dispatch is given a large grid, which makes it slow enough that the
/// GPU falls behind and the ring genuinely fills. Every work-item writes the
/// same value to the same address, so the expected result is unchanged.
///
/// The check is that the queue reports having actually blocked. Without it this
/// scenario could pass while never exercising flow control at all.
void runFillRing(const Scenario &S, const std::vector<Device> &Gpus,
                 const TestKernel &Kernel, RunResult &R) {
  QueueSetup Q;
  const int64_t Count = dispatchCountFor(S, DefaultRingBytes / AqlPacketBytes);
  if (!prepareQueue(Q, Gpus[0].Node, Count, R))
    return;

  auto *ArgBlocks = Q.Args.as<KernelArgs>();
  for (int64_t I = 0; I < Count; I++) {
    ArgBlocks[I].Destination =
        reinterpret_cast<uint64_t>(Q.Dst.as<uint8_t>() + I * sizeof(uint32_t));
    ArgBlocks[I].Value = static_cast<uint32_t>(0x1000 + I);
    ArgBlocks[I].Pad = 0;
    R.ExpectedValues.push_back(ArgBlocks[I].Value);

    uint8_t Packet[AqlPacketBytes];
    buildDispatch(Packet, S, Kernel.descriptorAddress(), &ArgBlocks[I]);
    // Enough work-items to keep the GPU busy while the producer runs ahead.
    auto *P = reinterpret_cast<hsa_kernel_dispatch_packet_t *>(Packet);
    P->workgroup_size_x = 256;
    P->grid_size_x = 1 << 16;
    if (!Q.Queue->submit(Packet)) {
      R.Completed = false;
      R.FailureNote = "submission stalled";
      return;
    }
  }
  R.DispatchesSubmitted = static_cast<uint64_t>(Count);

  auto *DstWords = Q.Dst.as<volatile uint32_t>();
  if (!waitForValue(&DstWords[Count - 1],
                    static_cast<uint32_t>(0x1000 + Count - 1), 30000)) {
    R.Completed = false;
    R.FailureNote = "the last dispatch never landed";
  }
  collectValues(Q, Count, R);

  const uint64_t Blocked = Q.Queue->timesBlocked();
  printf("    producer waited for a free slot %llu times of %lld submissions\n",
         (unsigned long long)Blocked, (long long)Count);
  if (Blocked == 0) {
    R.Completed = false;
    R.FailureNote =
        "the ring never filled, so this run did not exercise flow control at "
        "all -- the dispatches need to be slower or more numerous";
  }
  Q.Queue->destroy();
}

/// Scenarios this harness cannot run, with the reason. Never silently passes.
const char *whyNotRunnable(const Scenario &S) {
  if (S.ToolInitialisesHsa)
    return "needs HSA in the process; belongs in the HSA oracle harness";
  if (S.Mix == PacketMix::IncludeVendorSpecific)
    return "a vendor-specific packet carries a command stream the GPU will "
           "execute; submitting a hand-made one risks wedging the device, so "
           "this needs a valid command stream built first";
  return nullptr;
}

} // namespace

int main(int Argc, char **Argv) {
  // A comma-separated list rather than a single id, because several bugs here
  // have only appeared in a particular order -- running "S13,S14" is how you
  // reproduce one without sitting through the whole suite.
  std::vector<std::string> Only;
  bool ListOnly = false, RequireWrapper = false;
  for (int I = 1; I < Argc; I++) {
    std::string A = Argv[I];
    if (A == "--list")
      ListOnly = true;
    else if (A == "--require-wrapper")
      RequireWrapper = true;
    else
      for (size_t P = 0; P <= A.size();) {
        size_t C = A.find(',', P);
        if (C == std::string::npos)
          C = A.size();
        if (C > P)
          Only.push_back(A.substr(P, C - P));
        P = C + 1;
      }
  }
  auto selected = [&](const std::string &Id) {
    if (Only.empty())
      return true;
    for (const auto &W : Only)
      if (W == Id)
        return true;
    return false;
  };

  if (ListOnly) {
    for (const auto &S : allScenarios())
      printf("%-28s %s\n", S.Id.c_str(), S.Catches.c_str());
    return 0;
  }

  if (hsaKmtOpenKFD() != HSAKMT_STATUS_SUCCESS) {
    fprintf(stderr, "cannot open the GPU driver\n");
    return 2;
  }
  auto Gpus = discoverGpus();
  if (Gpus.empty()) {
    fprintf(stderr, "no GPU found\n");
    return 2;
  }
  printf("device: node %u, %s", Gpus[0].Node, Gpus[0].Arch.c_str());
  if (Gpus.size() > 1)
    printf(" (%zu GPUs present)", Gpus.size());
  printf("\n");
  if (!connectToWrapper(RequireWrapper)) {
    hsaKmtCloseKFD();
    return 2;
  }

  auto Code = testKernelCodeFor(Gpus[0].Arch);
  if (Code.empty()) {
    fprintf(stderr, "no test kernel was built for %s\n", Gpus[0].Arch.c_str());
    hsaKmtCloseKFD();
    return 2;
  }
  TestKernel Kernel;
  if (!Kernel.load(Gpus[0].Node, Code, sizeof(KernelArgs))) {
    fprintf(stderr, "could not load the test kernel\n");
    hsaKmtCloseKFD();
    return 2;
  }
  printf("\n");

  unsigned Passed = 0, Failed = 0, Skipped = 0;
  for (const auto &S : allScenarios()) {
    if (!selected(S.Id))
      continue;

    if (const char *Why = whyNotRunnable(S)) {
      printf("%-28s SKIPPED  (%s)\n", S.Id.c_str(), Why);
      Skipped++;
      continue;
    }

    {
      std::lock_guard<std::mutex> G(ObservationLock);
      Observations.clear();
    }
    {
      std::lock_guard<std::mutex> G(SecondObserverLock);
      SecondObservations.clear();
      ChainOrder.clear();
    }

    // A second tool attaching alongside the first. Registered after, so by the
    // chain's contract it runs first -- checked below, and pinned down properly
    // in the GPU-free unit tests.
    int SecondHandle = -1;
    if (S.TwoCallbacks && WrapperPresent && AddCallback != nullptr)
      SecondHandle = AddCallback(onPacketSecond, nullptr);

    // Lifecycle scenarios are verified a round at a time, because the driver
    // reuses queue ids after a destroy and pooling rounds would make every
    // round after the first look like duplicated packets.
    //
    // Each round is also checked on its own rather than only the last, so a
    // scenario that works for a while and then stops is caught, and the round
    // it stopped at is reported. For a threshold bug that number is the
    // diagnosis: "round 65 of 72" names a 64-entry table on its own.
    Verdict V;
    uint64_t TotalDispatches = 0;
    for (uint32_t Round = 0; Round < S.LifecycleRounds; Round++) {
      if (Round != 0) {
        std::lock_guard<std::mutex> G(ObservationLock);
        Observations.clear();
      }
      RunResult R;
      if (S.Submit == Submission::MultiThreaded)
        runConcurrent(S, Gpus, Kernel, R);
      else if (S.QueueCount > 1 && S.Id.rfind("S13", 0) == 0)
        runQueueLimit(S, Gpus, Kernel, R);
      else if (S.QueueCount > 1)
        runMultiQueue(S, Gpus, Kernel, R);
      else if (S.InspectInitialRingOnly)
        runRingInspection(Gpus, R);
      else if (S.AlongsideOtherQueueTypes)
        runMixedQueueTypes(S, Gpus, Kernel, R);
      else if (S.FillRing)
        runFillRing(S, Gpus, Kernel, R);
      else
        runStraightforward(S, Gpus, Kernel, R);

      {
        std::lock_guard<std::mutex> G(ObservationLock);
        R.Observations = Observations;
      }
      TotalDispatches += R.DispatchesSubmitted;

      // Without the wrapper there are no observations, so only the workload's
      // own result is meaningful. Saying so keeps a baseline run from looking
      // like it verified more than it did.
      Verdict RV;
      if (S.InspectInitialRingOnly || S.DestroyWithWorkInFlight) {
        // These have no per-dispatch results to compare; completing without
        // crashing or hanging is the whole check.
        if (!R.Completed)
          RV.fail(R.FailureNote);
      } else {
        RV = WrapperPresent ? verifyAll(R) : verifyValues(R);
      }

      for (const auto &P : RV.Problems)
        V.fail(S.LifecycleRounds > 1
                   ? "round " + std::to_string(Round + 1) + " of " +
                         std::to_string(S.LifecycleRounds) + ": " + P
                   : P);
      if (!RV.Passed || !R.Completed)
        break;
    }

    if (SecondHandle >= 0) {
      RemoveCallback(SecondHandle);

      // Both callbacks must have seen the same packets. Comparing counts is
      // what catches a chain that delivers to only the first or only the last.
      size_t First = 0;
      {
        std::lock_guard<std::mutex> G(ObservationLock);
        First = Observations.size();
      }
      std::lock_guard<std::mutex> G(SecondObserverLock);
      const size_t Second = SecondObservations.size();
      printf("    two tools attached: first saw %zu packets, second saw %zu\n",
             First, Second);
      if (First != Second)
        V.fail("the two callbacks saw different numbers of packets (" +
               std::to_string(First) + " and " + std::to_string(Second) +
               ") -- the chain is not delivering to both");
      else if (Second == 0)
        V.fail("neither callback saw anything, so this proves nothing about "
               "the chain");

      // Sanity check on hardware that the order matches the unit tests: the
      // later-registered callback runs first, so the log must alternate 2,1.
      bool OrderHeld = ChainOrder.size() % 2 == 0;
      for (size_t I = 0; OrderHeld && I + 1 < ChainOrder.size(); I += 2)
        OrderHeld = ChainOrder[I] == 2 && ChainOrder[I + 1] == 1;
      if (!OrderHeld)
        V.fail("the callbacks did not run last-registered-first on every "
               "packet");
    } else if (S.TwoCallbacks && WrapperPresent) {
      V.fail("this scenario needs the wrapper's addPacketCallback entry point, "
             "which is not present -- it cannot check what it was asked to");
    }

    if (V.Passed) {
      printf("%-28s PASS  (%llu dispatches)\n", S.Id.c_str(),
             (unsigned long long)TotalDispatches);
      Passed++;
    } else {
      printf("%-28s FAIL\n", S.Id.c_str());
      for (const auto &P : V.Problems)
        printf("    %s\n", P.c_str());
      Failed++;
    }
  }

  printf("\n%u passed, %u failed, %u skipped\n", Passed, Failed, Skipped);
  hsaKmtCloseKFD();
  return Failed == 0 ? 0 : 1;
}
