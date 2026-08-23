//===-- Scenarios.cpp - the KFD queue test matrix -------------------------===//
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
#include "Scenarios.h"

namespace luthier::test::kfd {

std::vector<Scenario> allScenarios() {
  std::vector<Scenario> S;

  // -- Counts around the ring boundary ------------------------------------
  // Reuse is where detection gets hard: a slot that has been used once looks
  // "written" forever unless the marker is put back.
  {
    Scenario X;
    X.Id = "S1-one-dispatch";
    X.Catches = "the basic path: one packet seen once";
    X.DispatchCount = 1;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S2-below-ring";
    X.Catches = "steady state with no slot reuse";
    X.DispatchCount = 8;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S3-exactly-ring";
    X.Catches = "the boundary: every slot used exactly once, none reused";
    X.DispatchCount = 1;
    X.PacketsRelativeToRing = true;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S4-ring-plus-one";
    X.Catches = "the first reuse of a slot -- where the marker must be re-armed";
    X.DispatchCount = 1;
    X.PacketsRelativeToRing = true;
    X.PacketAdjustment = 1; // exactly one packet past a full ring
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S5-many-laps";
    X.Catches = "sustained reuse; a runaway copier shows up as inflated counts";
    X.DispatchCount = 3;
    X.PacketsRelativeToRing = true;
    S.push_back(X);
  }

  // -- Packet content -----------------------------------------------------
  {
    Scenario X;
    X.Id = "S6-dispatch-and-barrier";
    X.Catches = "non-dispatch packets are copied faithfully, not skipped";
    X.DispatchCount = 16;
    X.Mix = PacketMix::DispatchAndBarrier;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S7-barrier-bit-clear";
    X.Catches = "packets the GPU may overlap, a different ordering case";
    X.DispatchCount = 16;
    X.BarrierBit = false;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S8-vendor-specific";
    X.Catches =
        "type 0 packets versus a zero-filled ring -- the two are identical "
        "byte-for-byte, which is what broke detection on non-HSA queues";
    X.DispatchCount = 8;
    X.Mix = PacketMix::IncludeVendorSpecific;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S9a-fences-system";
    X.Catches = "baseline for the fence-bit comparison below";
    X.DispatchCount = 4;
    X.Fences = FenceScopes::SystemBoth;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S9b-fences-agent";
    X.Catches =
        "that we copy cache-control bits faithfully. Must be run against an "
        "uncached destination AND a cached one: our earlier test set system "
        "scope and allocated uncached, so either alone would have passed and "
        "the fence bits were never actually checked";
    X.DispatchCount = 4;
    X.Fences = FenceScopes::Agent;
    S.push_back(X);
  }

  // -- Concurrency --------------------------------------------------------
  {
    Scenario X;
    X.Id = "S10a-two-producers";
    X.Catches = "concurrent submission to one queue -- never tested; ROCr has "
                "explicit machinery for it that we have no equivalent of";
    X.DispatchCount = 64;
    X.Submit = Submission::MultiThreaded;
    X.ProducerThreads = 2;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S10b-four-producers";
    X.Catches = "the same under more contention";
    X.DispatchCount = 128;
    X.Submit = Submission::MultiThreaded;
    X.ProducerThreads = 4;
    S.push_back(X);
  }

  // -- Multiple queues and devices ----------------------------------------
  {
    Scenario X;
    X.Id = "S11-two-queues";
    X.Catches = "per-queue state kept separate";
    X.DispatchCount = 32;
    X.QueueCount = 2;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S12-all-gpus";
    X.Catches = "queues on more than one device";
    X.DispatchCount = 32;
    X.QueueCount = 2;
    X.UseAllGpus = true;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S13-queue-table-limit";
    X.Catches =
        "more queues than the wrapper can track. It must leave the extras "
        "unwrapped and say so -- substituting a ring it cannot service would "
        "hang the application with no error";
    X.DispatchCount = 1;
    X.QueueCount = 72; // above the wrapper's 64-entry table
    S.push_back(X);
  }

  // -- Lifecycle ----------------------------------------------------------
  {
    Scenario X;
    X.Id = "S14-create-destroy-rounds";
    X.Catches = "repeated teardown; leaked or stale per-queue state";
    X.DispatchCount = 8;
    X.LifecycleRounds = 5;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S14b-churn-past-table";
    X.Catches =
        "a tracking table that is never reclaimed. One queue at a time, "
        "destroyed before the next is made, so nothing is ever contended -- but "
        "a wrapper that only ever appends runs out after 64 rounds and silently "
        "stops intercepting. S13 cannot catch this: it holds its queues alive, "
        "where refusing to wrap the extras is the correct answer";
    X.DispatchCount = 4;
    // Past the wrapper's 64-entry table, with margin. One queue per round, so
    // an unreclaimed table runs out at round 65 and the round number in the
    // failure message names the table size directly.
    X.LifecycleRounds = 72;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S14c-two-callbacks";
    X.Catches =
        "a chain that only really delivers to one tool. Two callbacks are "
        "registered and both must see every packet -- the case that matters "
        "when Luthier runs alongside a profiler, which the HSA runtime supports "
        "and a driver-level replacement had better not regress";
    X.DispatchCount = 16;
    X.TwoCallbacks = true;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S15-destroy-in-flight";
    X.Catches = "teardown racing the copier -- the case that segfaulted when "
                "the queue-destroyed check sat outside an unbounded loop";
    X.DispatchCount = 64;
    X.DestroyWithWorkInFlight = true;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S16-fill-ring";
    X.Catches = "flow control: the producer must wait for the GPU rather than "
                "overwrite live packets";
    X.DispatchCount = 4;
    X.PacketsRelativeToRing = true;
    X.FillRing = true;
    S.push_back(X);
  }

  // -- Coexistence --------------------------------------------------------
  {
    Scenario X;
    X.Id = "S17-mixed-queue-types";
    X.Catches = "PM4 and SDMA queues pass through untouched";
    X.DispatchCount = 8;
    X.AlongsideOtherQueueTypes = true;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S18-initial-ring-contents";
    X.Catches =
        "what a fresh ring holds. Nothing below HSA guarantees the 'empty' "
        "marker: the HSA runtime writes it in software, and a non-HSA queue "
        "arrives zero-filled";
    X.InspectInitialRingOnly = true;
    S.push_back(X);
  }
  {
    Scenario X;
    X.Id = "S19-tool-own-queues";
    X.Catches =
        "queues the runtime creates for the tool itself must not be wrapped. "
        "Measured: one hsa_queue_create makes two AQL queues, and today we "
        "wrap both -- latent now, broken as soon as the tool dispatches";
    X.DispatchCount = 4;
    X.ToolInitialisesHsa = true;
    S.push_back(X);
  }

  return S;
}

const Scenario *findScenario(const std::string &Id) {
  static const std::vector<Scenario> All = allScenarios();
  for (const auto &S : All)
    if (S.Id == Id)
      return &S;
  return nullptr;
}

} // namespace luthier::test::kfd
