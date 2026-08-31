//===-- VerifyTest.cpp - tests for the KFD scenario checks ----------------===//
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
/// Tests for the checks themselves, with no GPU involved.
///
/// This exists because the project's recurring failure has been checks that
/// could not fail. A correct-looking answer hid a broken mechanism twice; the
/// checks that replaced it are only worth anything if they actually reject the
/// cases they claim to reject. So each check gets both a passing case and the
/// specific failure it was written for.
//===----------------------------------------------------------------------===//
#include "../../kfd/Verify.h"
#include <gtest/gtest.h>

using namespace luthier::test::kfd;

namespace {

constexpr uint16_t DispatchHeader = 2;    // HSA_PACKET_TYPE_KERNEL_DISPATCH
constexpr uint16_t BarrierHeader = 3;     // HSA_PACKET_TYPE_BARRIER_AND

PacketObservation dispatch(uint32_t Queue, uint64_t Index,
                           uint64_t KernelObject = 0x1000) {
  return PacketObservation{/*GpuId=*/1, Queue, Index, DispatchHeader,
                           KernelObject};
}

/// A run where everything went right: N dispatches, N observations with clean
/// indices, and every value landed.
RunResult goodRun(uint64_t N) {
  RunResult R;
  R.DispatchesSubmitted = N;
  for (uint64_t I = 0; I < N; I++) {
    R.Observations.push_back(dispatch(/*Queue=*/7, I));
    R.ExpectedValues.push_back(static_cast<uint32_t>(0x100 + I));
    R.ObservedValues.push_back(static_cast<uint32_t>(0x100 + I));
  }
  return R;
}

} // namespace

//===----------------------------------------------------------------------===//
// Check 1: per-dispatch results
//===----------------------------------------------------------------------===//

TEST(VerifyValues, AcceptsAMatchingRun) {
  EXPECT_TRUE(verifyValues(goodRun(8)).Passed);
}

TEST(VerifyValues, RejectsASingleWrongValue) {
  RunResult R = goodRun(8);
  R.ObservedValues[3] = 0; // one dispatch never landed
  Verdict V = verifyValues(R);
  EXPECT_FALSE(V.Passed);
  ASSERT_FALSE(V.Problems.empty());
}

TEST(VerifyValues, RejectsAnIncompleteRun) {
  RunResult R = goodRun(4);
  R.Completed = false;
  R.FailureNote = "timed out";
  EXPECT_FALSE(verifyValues(R).Passed);
}

//===----------------------------------------------------------------------===//
// Check 2: callbacks equal dispatches
//
// The case that matters most: a workload whose *values* are all correct while
// the interception layer saw the packets many more times than they were
// submitted. That is the real defect this project shipped for weeks.
//===----------------------------------------------------------------------===//

TEST(VerifyCallbackCount, AcceptsExactMatch) {
  EXPECT_TRUE(verifyCallbackCount(goodRun(16)).Passed);
}

TEST(VerifyCallbackCount, RejectsDuplicatedPacketsEvenWhenValuesAreCorrect) {
  RunResult R = goodRun(4);
  // Same packets seen again -- values still perfect, mechanism broken.
  for (uint64_t I = 0; I < 4; I++)
    R.Observations.push_back(dispatch(/*Queue=*/7, I));

  EXPECT_TRUE(verifyValues(R).Passed) << "values alone must still look fine";
  Verdict V = verifyCallbackCount(R);
  EXPECT_FALSE(V.Passed) << "the count check is what has to catch this";
}

TEST(VerifyCallbackCount, RejectsMissedPackets) {
  RunResult R = goodRun(8);
  R.Observations.pop_back();
  EXPECT_FALSE(verifyCallbackCount(R).Passed);
}

TEST(VerifyCallbackCount, IgnoresNonDispatchPackets) {
  RunResult R = goodRun(4);
  // Barriers are forwarded too but are not dispatches; they must not inflate
  // the comparison.
  R.Observations.push_back(
      PacketObservation{1, 7, 99, BarrierHeader, 0});
  EXPECT_TRUE(verifyCallbackCount(R).Passed);
}

//===----------------------------------------------------------------------===//
// Check 3: index sequence
//
// Counts can match while the sequence is wrong -- one packet copied twice and
// another skipped. Only this check separates those.
//===----------------------------------------------------------------------===//

TEST(VerifyIndexSequence, AcceptsACleanSequence) {
  EXPECT_TRUE(verifyIndexSequence(goodRun(32)).Passed);
}

TEST(VerifyIndexSequence, RejectsARepeatWithACompensatingGap) {
  RunResult R = goodRun(8);
  // Same total count, but index 2 twice and index 5 never.
  R.Observations[5] = dispatch(/*Queue=*/7, 2);

  EXPECT_TRUE(verifyCallbackCount(R).Passed)
      << "the count is unchanged, so counting cannot catch this";
  Verdict V = verifyIndexSequence(R);
  EXPECT_FALSE(V.Passed) << "the sequence check is the only one that can";
}

TEST(VerifyIndexSequence, RejectsAGap) {
  RunResult R = goodRun(8);
  R.Observations.erase(R.Observations.begin() + 4);
  EXPECT_FALSE(verifyIndexSequence(R).Passed);
}

TEST(VerifyIndexSequence, TreatsQueuesIndependently) {
  RunResult R;
  R.DispatchesSubmitted = 6;
  // Two queues, each with its own 0..2 -- legitimate, not a duplicate.
  for (uint64_t I = 0; I < 3; I++) {
    R.Observations.push_back(dispatch(/*Queue=*/1, I));
    R.Observations.push_back(dispatch(/*Queue=*/2, I));
  }
  EXPECT_TRUE(verifyIndexSequence(R).Passed);
}

//===----------------------------------------------------------------------===//
// Check 4: agreement with the HSA reference
//===----------------------------------------------------------------------===//

TEST(CompareWithOracle, AcceptsTheSameDispatchShape) {
  RunResult A = goodRun(4), B = goodRun(4);
  // Kernel addresses differ between runs; only the pattern should matter.
  for (auto &O : B.Observations)
    O.KernelObject = 0x9000;
  EXPECT_TRUE(compareWithOracle(A, B).Passed);
}

TEST(CompareWithOracle, RejectsDifferentDispatchCounts) {
  EXPECT_FALSE(compareWithOracle(goodRun(4), goodRun(5)).Passed);
}

TEST(CompareWithOracle, RejectsADifferentKernelPattern) {
  RunResult A = goodRun(4), B = goodRun(4);
  // A runs one kernel four times; B alternates between two.
  B.Observations[1].KernelObject = 0x2000;
  B.Observations[3].KernelObject = 0x2000;
  EXPECT_FALSE(compareWithOracle(A, B).Passed);
}

//===----------------------------------------------------------------------===//
// The combined check
//===----------------------------------------------------------------------===//

TEST(VerifyAll, PassesACleanRun) { EXPECT_TRUE(verifyAll(goodRun(64)).Passed); }

TEST(VerifyAll, ReportsEveryProblemNotJustTheFirst) {
  RunResult R = goodRun(8);
  R.ObservedValues[0] = 0;                        // a wrong value
  R.Observations.push_back(dispatch(7, 0));       // and a duplicate
  Verdict V = verifyAll(R);
  EXPECT_FALSE(V.Passed);
  EXPECT_GE(V.Problems.size(), 2u)
      << "a run with two distinct problems should report both";
}
