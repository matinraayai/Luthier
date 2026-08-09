//===-- Verify.cpp - the checks every KFD queue scenario must pass --------===//
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
#include "Verify.h"

#include <algorithm>
#include <map>
#include <sstream>

namespace luthier::test::kfd {

namespace {

/// AQL packet type lives in the header's low 8 bits.
constexpr uint16_t PacketTypeKernelDispatch = 2;
unsigned packetType(uint16_t Header) { return Header & 0xFF; }

std::string num(uint64_t V) { return std::to_string(V); }

/// Keep failure lists readable when something goes badly wrong: a run that drops
/// every packet should not print thousands of lines.
constexpr size_t MaxReported = 5;

} // namespace

Verdict verifyValues(const RunResult &R) {
  Verdict V;
  if (!R.Completed) {
    V.fail("workload did not complete: " + R.FailureNote);
    return V;
  }
  if (R.ExpectedValues.size() != R.ObservedValues.size()) {
    V.fail("expected " + num(R.ExpectedValues.size()) + " result slots but "
           "found " + num(R.ObservedValues.size()));
    return V;
  }
  size_t Wrong = 0;
  std::ostringstream First;
  for (size_t I = 0; I < R.ExpectedValues.size(); I++) {
    if (R.ExpectedValues[I] == R.ObservedValues[I])
      continue;
    if (Wrong < MaxReported)
      First << (Wrong ? ", " : "") << "[" << I << "] wanted "
            << R.ExpectedValues[I] << " got " << R.ObservedValues[I];
    Wrong++;
  }
  if (Wrong != 0)
    V.fail(num(Wrong) + " of " + num(R.ExpectedValues.size()) +
           " dispatch results wrong: " + First.str() +
           (Wrong > MaxReported ? ", ..." : ""));
  return V;
}

Verdict verifyCallbackCount(const RunResult &R) {
  Verdict V;
  uint64_t Seen = 0;
  for (const auto &O : R.Observations)
    if (packetType(O.Header) == PacketTypeKernelDispatch)
      Seen++;

  if (Seen == R.DispatchesSubmitted)
    return V;

  // The direction matters for diagnosis, so say which it is rather than just
  // reporting a mismatch.
  if (Seen > R.DispatchesSubmitted)
    V.fail("interception saw " + num(Seen) + " dispatches but only " +
           num(R.DispatchesSubmitted) +
           " were submitted -- packets are being processed more than once");
  else
    V.fail("interception saw only " + num(Seen) + " of " +
           num(R.DispatchesSubmitted) +
           " submitted dispatches -- packets are being missed");
  return V;
}

Verdict verifyIndexSequence(const RunResult &R) {
  Verdict V;

  // Indices restart per queue, so group before checking.
  std::map<std::pair<uint32_t, uint32_t>, std::vector<uint64_t>> ByQueue;
  for (const auto &O : R.Observations)
    ByQueue[{O.GpuId, O.QueueId}].push_back(O.PacketIndex);

  for (auto &[Key, Indices] : ByQueue) {
    const auto &[Gpu, Queue] = Key;
    std::string Where =
        "gpu " + num(Gpu) + " queue " + num(Queue) + ": ";

    std::vector<uint64_t> Sorted = Indices;
    std::sort(Sorted.begin(), Sorted.end());

    size_t Duplicates = 0;
    for (size_t I = 1; I < Sorted.size(); I++)
      if (Sorted[I] == Sorted[I - 1])
        Duplicates++;
    if (Duplicates != 0)
      V.fail(Where + num(Duplicates) +
             " packet indices seen more than once -- the same packet was "
             "copied twice, which a matching count would not reveal");

    // With duplicates removed the set must be exactly 0..N-1.
    Sorted.erase(std::unique(Sorted.begin(), Sorted.end()), Sorted.end());
    if (!Sorted.empty()) {
      if (Sorted.front() != 0)
        V.fail(Where + "sequence starts at " + num(Sorted.front()) +
               " rather than 0");
      uint64_t Expected = Sorted.back() + 1;
      if (Sorted.size() != Expected)
        V.fail(Where + "sequence has gaps: " + num(Sorted.size()) +
               " distinct indices spanning 0.." + num(Sorted.back()) +
               " -- some packets were never seen");
    }
  }
  return V;
}

Verdict compareWithOracle(const RunResult &NonHsa, const RunResult &Oracle) {
  Verdict V;

  // Only the dispatch stream is comparable: each runtime emits its own
  // bookkeeping packets, and their counts legitimately differ.
  auto dispatches = [](const RunResult &R) {
    std::vector<uint64_t> Out;
    for (const auto &O : R.Observations)
      if (packetType(O.Header) == PacketTypeKernelDispatch)
        Out.push_back(O.KernelObject);
    return Out;
  };

  const auto A = dispatches(NonHsa);
  const auto B = dispatches(Oracle);

  if (A.size() != B.size()) {
    V.fail("dispatch counts differ: non-HSA saw " + num(A.size()) +
           ", the HSA reference saw " + num(B.size()));
    return V;
  }
  // Kernel addresses differ between the two runs (separate loads), so only the
  // shape is compared: the same kernel repeated in the same pattern.
  std::map<uint64_t, uint64_t> AFirstSeen, BFirstSeen;
  for (size_t I = 0; I < A.size(); I++) {
    AFirstSeen.emplace(A[I], AFirstSeen.size());
    BFirstSeen.emplace(B[I], BFirstSeen.size());
    if (AFirstSeen[A[I]] != BFirstSeen[B[I]]) {
      V.fail("dispatch " + num(I) +
             " refers to a different kernel than the HSA reference does at the "
             "same position");
      break;
    }
  }
  return V;
}

Verdict verifyAll(const RunResult &R) {
  Verdict V;
  for (const Verdict &Part :
       {verifyValues(R), verifyCallbackCount(R), verifyIndexSequence(R)}) {
    if (!Part.Passed) {
      V.Passed = false;
      V.Problems.insert(V.Problems.end(), Part.Problems.begin(),
                        Part.Problems.end());
    }
  }
  return V;
}

} // namespace luthier::test::kfd
