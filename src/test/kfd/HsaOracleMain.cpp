//===-- HsaOracleMain.cpp - the HSA half of the KFD queue test suite ------===//
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
/// The one harness that **does** link the HSA runtime, and the only place the
/// suite can check what happens when a tool -- Luthier itself -- has HSA loaded
/// in the process it is instrumenting.
///
/// \par What this covers
/// \c S19-tool-own-queues. Luthier links the HSA runtime and will call
/// \c hsa_init even when the application under instrumentation never touches
/// HSA. The runtime then creates AQL queues for Luthier's own use, and those go
/// through the same \c ioctl the wrapper interposes. Wrapping them means
/// instrumenting ourselves: the tool's dispatches would be fed to the tool's own
/// callback. ROCr guards the equivalent case in its own interception layer
/// (\c intercept_queue.cpp:328).
///
/// \par Why it needs a separate binary
/// Its requirement is the exact opposite of \c kfd-nonhsa-tests, which must not
/// link \c libhsa-runtime64 and has a test enforcing that. The two cannot be one
/// program.
///
/// \par Why it counts queues rather than packets
/// Whether a queue was wrapped is normally only visible when packets flow
/// through it. The runtime's own queues may carry none, so a packet callback
/// would stay silent whether we wrapped them or not -- it cannot tell "correctly
/// ignored" from "wrapped, but idle". The wrapper therefore exposes a count of
/// the queues it has substituted rings for, and that is what is checked.
///
/// \par History
/// This harness was written while the gap was still open, and failed: one
/// \c hsa_queue_create produced 2 wrapped queues, matching what Phase 0.2 had
/// measured by hand. It passes now that the wrapper takes a tool region into
/// account. The failing form is what made the fix checkable rather than
/// asserted.
///
/// Usage:
///   kfd-hsa-oracle            run the scenarios that need HSA
///   kfd-hsa-oracle --list     list them
//===----------------------------------------------------------------------===//
#include "Scenarios.h"
#include "Verify.h"

#include "luthier/KFD/QueueWrapper.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#include <hsa/hsa.h>

using namespace luthier::test::kfd;

namespace {

/// The wrapper's entry points are found with \c dlsym rather than linked, for
/// the same reason the non-HSA harness does it: a link-time dependency would
/// drag the wrapper in even for runs that are supposed to be unwrapped. A
/// missing symbol is how this binary detects it is running unwrapped, which it
/// treats as a configuration error rather than a reason to check less.
using CountFn = unsigned long long (*)();
using RegionFn = void (*)();

/// The wrapper's entry points, all looked up rather than linked.
struct Wrapper {
  CountFn WrappedCount = nullptr;
  CountFn ExcludedCount = nullptr;
  RegionFn BeginToolRegion = nullptr;
  RegionFn EndToolRegion = nullptr;

  bool complete() const {
    return WrappedCount && ExcludedCount && BeginToolRegion && EndToolRegion;
  }
};

Wrapper findWrapper() {
  Wrapper W;
  W.WrappedCount = reinterpret_cast<CountFn>(
      dlsym(RTLD_DEFAULT, "luthierKfdWrappedQueueCount"));
  W.ExcludedCount = reinterpret_cast<CountFn>(
      dlsym(RTLD_DEFAULT, "luthierKfdExcludedQueueCount"));
  W.BeginToolRegion = reinterpret_cast<RegionFn>(
      dlsym(RTLD_DEFAULT, "luthierKfdBeginToolRegion"));
  W.EndToolRegion = reinterpret_cast<RegionFn>(
      dlsym(RTLD_DEFAULT, "luthierKfdEndToolRegion"));
  return W;
}

hsa_status_t findGpu(hsa_agent_t Agent, void *Out) {
  hsa_device_type_t Type;
  if (hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (Type != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;
  *static_cast<hsa_agent_t *>(Out) = Agent;
  return HSA_STATUS_INFO_BREAK; // stop at the first GPU
}

/// A queue error handler is required by hsa_queue_create. Nothing here should
/// ever reach it, so it says so rather than staying silent.
void onQueueError(hsa_status_t Status, hsa_queue_t *, void *) {
  const char *Msg = nullptr;
  hsa_status_string(Status, &Msg);
  fprintf(stderr, "    the runtime reported a queue error: %s\n",
          Msg != nullptr ? Msg : "unknown");
}

/// \brief Run S19: does the wrapper tell the tool's queues from the
/// application's?
///
/// Checks **both** directions, which is the whole point. A discriminator that
/// excludes everything would pass a test that only counts the tool's queues, and
/// it would also be catastrophic -- no application would ever be instrumented
/// again. So a queue is also created outside a tool region, and that one must
/// still be wrapped.
Verdict runToolOwnedQueues(const Wrapper &W) {
  Verdict V;

  if (hsa_init() != HSA_STATUS_SUCCESS) {
    V.fail("hsa_init failed, so this check could not run at all");
    return V;
  }

  hsa_agent_t Gpu = {0};
  hsa_iterate_agents(findGpu, &Gpu);
  if (Gpu.handle == 0) {
    V.fail("the runtime found no GPU agent");
    hsa_shut_down();
    return V;
  }

  uint32_t QueueSize = 0;
  hsa_agent_get_info(Gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &QueueSize);
  if (QueueSize == 0)
    QueueSize = 1024;

  auto makeQueue = [&](hsa_queue_t **Out) {
    return hsa_queue_create(Gpu, QueueSize, HSA_QUEUE_TYPE_MULTI, onQueueError,
                            nullptr, UINT32_MAX, UINT32_MAX, Out);
  };

  //=== Warm-up: absorb the runtime's one-time setup =======================//
  // The first hsa_queue_create in a process creates more than the caller asked
  // for -- ROCr stands up its own internal queue alongside it. Measuring the
  // tool's queue first would attribute that extra queue to the tool region and
  // report a number that has nothing to do with the discriminator. So a queue
  // is made and thrown away first, and both measurements below are steady-state.
  hsa_queue_t *WarmUp = nullptr;
  if (makeQueue(&WarmUp) != HSA_STATUS_SUCCESS) {
    V.fail("hsa_queue_create failed on the warm-up, so this check could not run "
           "at all");
    hsa_shut_down();
    return V;
  }
  printf("    warm-up queue absorbed %llu one-time queue(s)\n",
         W.WrappedCount());
  hsa_queue_destroy(WarmUp);

  //=== Direction 1: a queue the tool asks for must be left alone ===========//
  const unsigned long long WrappedBefore = W.WrappedCount();
  const unsigned long long ExcludedBefore = W.ExcludedCount();

  hsa_queue_t *ToolQueue = nullptr;
  hsa_status_t St;
  {
    // Exactly what Luthier would do around its own runtime use.
    W.BeginToolRegion();
    St = makeQueue(&ToolQueue);
    W.EndToolRegion();
  }
  if (St != HSA_STATUS_SUCCESS) {
    V.fail("hsa_queue_create failed inside a tool region, so this check could "
           "not run at all");
    hsa_shut_down();
    return V;
  }

  const unsigned long long ToolWrapped = W.WrappedCount() - WrappedBefore;
  const unsigned long long ToolExcluded = W.ExcludedCount() - ExcludedBefore;
  printf("    tool's own hsa_queue_create -> %llu wrapped, %llu deliberately "
         "left alone\n",
         ToolWrapped, ToolExcluded);

  if (ToolWrapped != 0)
    V.fail("the wrapper wrapped " + std::to_string(ToolWrapped) +
           " queue(s) that the HSA runtime created for the tool itself. Those "
           "are ours, not the application's -- instrumenting them means "
           "feeding the tool's own dispatches to the tool's own callback");
  if (ToolExcluded == 0)
    V.fail("no queue was recorded as excluded, so the tool's queues were not "
           "recognised as the tool's -- if none were created at all this check "
           "proves nothing");

  //=== Direction 2: an ordinary queue must still be wrapped ================//
  // Without this the check would pass just as well if the wrapper had stopped
  // wrapping anything, which is the more damaging failure of the two.
  const unsigned long long PlainBefore = W.WrappedCount();
  hsa_queue_t *AppQueue = nullptr;
  St = makeQueue(&AppQueue);
  if (St != HSA_STATUS_SUCCESS) {
    V.fail("hsa_queue_create failed outside a tool region");
    hsa_queue_destroy(ToolQueue);
    hsa_shut_down();
    return V;
  }
  const unsigned long long PlainWrapped = W.WrappedCount() - PlainBefore;
  printf("    the same call outside a tool region -> %llu wrapped\n",
         PlainWrapped);

  if (PlainWrapped == 0)
    V.fail("a queue created outside any tool region was not wrapped, so the "
           "exclusion is not discriminating -- it is suppressing everything, "
           "which would silently stop all instrumentation");

  hsa_queue_destroy(AppQueue);
  hsa_queue_destroy(ToolQueue);
  hsa_shut_down();
  return V;
}

} // namespace

int main(int Argc, char **Argv) {
  bool ListOnly = false;
  for (int I = 1; I < Argc; I++)
    if (std::string(Argv[I]) == "--list")
      ListOnly = true;

  if (ListOnly) {
    for (const auto &S : allScenarios())
      if (S.ToolInitialisesHsa)
        printf("%-28s %s\n", S.Id.c_str(), S.Catches.c_str());
    return 0;
  }

  // The wrapper is the subject of every check here, so its absence is a
  // configuration error, not a reason to run a weaker version. Reporting a pass
  // for a check that never happened is the failure mode this suite has already
  // been bitten by once.
  const Wrapper W = findWrapper();
  if (!W.complete()) {
    fprintf(stderr, "the wrapper is not in this process, or is too old to have "
                    "the tool-region entry points; preload "
                    "libluthier-kfd-queue-wrapper.so\n");
    return 2;
  }
  printf("wrapper: loaded\n\n");

  unsigned Passed = 0, Failed = 0;
  for (const auto &S : allScenarios()) {
    if (!S.ToolInitialisesHsa)
      continue; // the non-HSA harness owns everything else

    Verdict V = runToolOwnedQueues(W);
    if (V.Passed) {
      printf("%-28s PASS\n", S.Id.c_str());
      Passed++;
    } else {
      printf("%-28s FAIL\n", S.Id.c_str());
      for (const auto &P : V.Problems)
        printf("    %s\n", P.c_str());
      Failed++;
    }
  }

  printf("\n%u passed, %u failed\n", Passed, Failed);
  return Failed == 0 ? 0 : 1;
}
