//===-- Verify.h - the checks every KFD queue scenario must pass ----------===//
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
/// Four independent checks. Each exists because a weaker check let a real bug
/// through:
///
/// \li **The workload's result.** Necessary, and nowhere near sufficient -- see
///     below.
/// \li **Callbacks equal dispatches.** At 250 launches a workload produced the
///     right answer five runs out of five while the callback had fired 903
///     times for 253 dispatches. Adding numbers is idempotent, so re-running a
///     stale packet changes nothing; anything that counts, traces or rewrites
///     was already broken.
/// \li **The index sequence.** Even matching counts are not enough: 199 out of
///     199 was equally consistent with the ring never having been reused at
///     all. Only "every index exactly once, no gaps, no repeats" shows each
///     packet was handled once.
/// \li **Agreement with the HSA oracle.** The same workload through HSA's own
///     interception, as a reference answer.
///
/// Every workload therefore gives each dispatch its own destination word and its
/// own distinct value, so a dropped, duplicated or reordered packet is visible
/// rather than averaged away.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_KFD_VERIFY_H
#define LUTHIER_TEST_KFD_VERIFY_H

#include <cstdint>
#include <string>
#include <vector>

namespace luthier::test::kfd {

/// \brief One packet as seen by an interception callback.
struct PacketObservation {
  uint32_t GpuId;
  uint32_t QueueId;
  /// Position in that queue's stream, as reported by the interception layer.
  uint64_t PacketIndex;
  /// The packet's real header. A callback that is handed the interception
  /// layer's own gate value instead of this will misreport every packet type --
  /// which has happened, so the harness records what it was actually given.
  uint16_t Header;
  /// Kernel entry address for dispatch packets; zero otherwise.
  uint64_t KernelObject;
};

/// \brief What one scenario run produced.
struct RunResult {
  /// Packets the interception layer showed us, in the order it showed them.
  std::vector<PacketObservation> Observations;
  /// Dispatches the workload actually submitted. Known by construction, not
  /// derived from the observations -- otherwise the comparison is circular.
  uint64_t DispatchesSubmitted = 0;
  /// One entry per dispatch: the value that dispatch was supposed to write.
  std::vector<uint32_t> ExpectedValues;
  /// What was actually found in memory afterwards.
  std::vector<uint32_t> ObservedValues;
  /// Set when the workload could not complete (a hang, a crash, a driver
  /// error). Distinguished from a wrong answer because the causes differ.
  bool Completed = true;
  std::string FailureNote;
};

/// \brief Outcome of checking one scenario.
struct Verdict {
  bool Passed = true;
  /// One line per problem, in plain language, suitable for a results table.
  std::vector<std::string> Problems;

  void fail(std::string Why) {
    Passed = false;
    Problems.push_back(std::move(Why));
  }
};

/// \brief Check 1: each dispatch's own effect landed.
///
/// Compares per-dispatch values rather than an aggregate, so a lost or repeated
/// packet cannot hide behind a correct total.
Verdict verifyValues(const RunResult &R);

/// \brief Check 2: the interception layer saw exactly the dispatches submitted.
///
/// Counts dispatch-type packets among the observations and compares against what
/// the workload submitted. More is a duplicate; fewer is a drop.
Verdict verifyCallbackCount(const RunResult &R);

/// \brief Check 3: within each queue, indices form a clean sequence.
///
/// Every index from 0 to N-1 exactly once: no repeats (a packet copied twice),
/// no gaps (a packet skipped). Indices restart per queue, so they are grouped by
/// queue before checking.
///
/// \warning Queue ids are unique only among *live* queues -- the driver reuses
/// them after a queue is destroyed. So a run that creates and destroys queues in
/// sequence will legitimately produce several independent sequences under one
/// id, and must verify each round separately rather than pooling them. Pooling
/// them reports every round after the first as duplicates, which cost some time
/// to diagnose the first time.
Verdict verifyIndexSequence(const RunResult &R);

/// \brief Check 4: the non-HSA run agrees with the HSA reference.
///
/// Compares packet types and kernel addresses in order. Timing and absolute
/// counts may legitimately differ between the two paths -- the runtime emits its
/// own packets -- so only the dispatch stream is compared.
Verdict compareWithOracle(const RunResult &NonHsa, const RunResult &Oracle);

/// \brief Run every applicable check and merge the verdicts.
Verdict verifyAll(const RunResult &R);

} // namespace luthier::test::kfd

#endif // LUTHIER_TEST_KFD_VERIFY_H
