//===-- Scenarios.h - shared workload definitions for KFD queue tests -----===//
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
/// One list of workloads, shared by two harnesses:
///
/// \li \c kfd-nonhsa-tests drives them through libhsakmt, with no HSA runtime in
///     the binary. This is the case issue #85 exists for.
/// \li \c kfd-hsa-oracle drives the same workloads through HSA's own
///     \c InterceptQueue, giving a reference answer.
///
/// Sharing the list is the point: the two can then be compared directly, and any
/// difference is either a bug in our wrapper or a design difference worth
/// writing down.
///
/// \par Why each scenario exists
/// Every entry names the specific thing it would catch. A scenario that cannot
/// fail for a stated reason is not worth running -- and we have already been
/// caught once by a check that passed while the mechanism underneath it was
/// broken.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_KFD_SCENARIOS_H
#define LUTHIER_TEST_KFD_SCENARIOS_H

#include <cstdint>
#include <string>
#include <vector>

namespace luthier::test::kfd {

/// \brief How a workload submits packets.
enum class Submission {
  /// One producer thread, the simple case.
  SingleThreaded,
  /// Several producer threads sharing one queue. ROCr has explicit machinery
  /// for this (compare-and-swap on the write index, thread-local recursion
  /// state, a per-queue lock); our wrapper has never been run this way.
  MultiThreaded,
};

/// \brief Which packet types a workload puts on the queue.
enum class PacketMix {
  /// Kernel dispatches only.
  DispatchOnly,
  /// Dispatches with barrier packets between them, which is what a real
  /// runtime emits (SCALE produced 303 dispatches and 3 barriers).
  DispatchAndBarrier,
  /// Includes a VENDOR_SPECIFIC packet. Type 0 is the dangerous one: it is
  /// indistinguishable from a zero-filled ring, which is the root of the bug
  /// where a non-HSA queue's whole ring read as "full of packets".
  IncludeVendorSpecific,
};

/// \brief Cache-maintenance bits to put in the packet header.
///
/// These control whether the GPU invalidates caches before a packet and flushes
/// them after. We must copy them faithfully: dropping a release scope means the
/// kernel's results may never reach the host, which presents as a wrong answer
/// rather than a cache bug.
enum class FenceScopes {
  /// System scope both ways -- the safe default most runtimes use.
  SystemBoth,
  /// No cache maintenance. Only valid where nothing outside the GPU reads the
  /// result.
  None,
  /// Agent scope: coherent within the GPU, but not necessarily to the host.
  Agent,
};

/// \brief One workload.
struct Scenario {
  /// Stable identifier used in results tables and to select a single scenario
  /// from the command line.
  std::string Id;
  /// What this scenario is for -- specifically, what failure it would expose.
  std::string Catches;

  /// Number of kernel dispatches to submit. Interpreted relative to the queue's
  /// slot count where \c PacketsRelativeToRing is set.
  uint32_t DispatchCount = 1;
  /// When true, \c DispatchCount is a multiple of the ring's slot count rather
  /// than an absolute number, so the scenario means the same thing on hardware
  /// with a different default queue size.
  bool PacketsRelativeToRing = false;
  /// Added after the multiplication above. Lets a scenario sit exactly one
  /// packet past a ring boundary regardless of ring size.
  int32_t PacketAdjustment = 0;

  Submission Submit = Submission::SingleThreaded;
  uint32_t ProducerThreads = 1;
  PacketMix Mix = PacketMix::DispatchOnly;
  FenceScopes Fences = FenceScopes::SystemBoth;

  /// Set the packet header's barrier bit. With it clear the GPU may overlap a
  /// packet with earlier ones, which is a genuinely different ordering case.
  bool BarrierBit = true;

  /// Number of AQL queues to create.
  uint32_t QueueCount = 1;
  /// Spread queues across every GPU rather than using one.
  bool UseAllGpus = false;

  /// Create and destroy the queues this many times, submitting each round.
  uint32_t LifecycleRounds = 1;
  /// Destroy the queue without waiting for its work to finish.
  bool DestroyWithWorkInFlight = false;

  /// Submit without honouring the GPU's progress counter, so the ring fills and
  /// the producer must block. Exercises flow control.
  bool FillRing = false;

  /// Also create PM4 and SDMA queues alongside, which the wrapper must leave
  /// alone.
  bool AlongsideOtherQueueTypes = false;

  /// Only inspect the ring's contents at creation; submit nothing.
  bool InspectInitialRingOnly = false;

  /// Register a second packet callback, so both must see every packet.
  /// Exercises the chain on a real queue; the ordering guarantee itself is
  /// checked in the GPU-free unit tests, where it belongs.
  bool TwoCallbacks = false;

  /// Have the harness itself initialise HSA, so the runtime creates queues on
  /// the tool's behalf. The wrapper must not wrap those -- measured: one
  /// hsa_queue_create produces two AQL queues, both of which we currently wrap.
  /// Only meaningful in the HSA-linked harness.
  bool ToolInitialisesHsa = false;
};

/// \brief The full matrix.
///
/// Ordered roughly by how much of the mechanism they exercise, so a run that is
/// cut short still covers the basics.
std::vector<Scenario> allScenarios();

/// \brief Look up one scenario by \c Id, for running a single case.
/// \return nullptr when no scenario has that id.
const Scenario *findScenario(const std::string &Id);

} // namespace luthier::test::kfd

#endif // LUTHIER_TEST_KFD_SCENARIOS_H
