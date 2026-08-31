//===-- FdSharing.cpp - one DRM file per GPU, shared ----------------------===//
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
/// Implements \c luthier/KFD/FdSharing.h. See that header for why the descriptor
/// that matters is the one \c ACQUIRE_VM carries rather than the first one opened.
///
/// \note Deliberately free of LLVM, like the rest of \c LuthierKFD: this ships in
/// the library preloaded into arbitrary applications.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/FdSharing.h"

#include "luthier/KFD/AllocationTracker.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace luthier::kfd {

namespace {

constexpr const char RenderNodePrefix[] = "/dev/dri/renderD";

/// Highest topology node index scanned. Matches the bound used elsewhere for the
/// same walk; the count is small and the files are tiny.
constexpr unsigned MaxTopologyNodes = 64;

std::atomic<bool> SharingEnabled{false};

} // namespace

bool isRenderNodePath(const char *Path) {
  return Path != nullptr &&
         strncmp(Path, RenderNodePrefix, sizeof(RenderNodePrefix) - 1) == 0;
}

uint32_t gpuIdForRenderNodePath(const char *Path) {
  if (!isRenderNodePath(Path))
    return 0;

  const char *MinorText = Path + sizeof(RenderNodePrefix) - 1;
  char *End = nullptr;
  const unsigned long Minor = strtoul(MinorText, &End, 10);
  // Reject trailing junk rather than accepting a prefix match: "renderD128foo"
  // is not renderD128, and treating it as such would redirect an unrelated open.
  if (End == MinorText || *End != '\0')
    return 0;

  for (unsigned Node = 0; Node < MaxTopologyNodes; Node++) {
    char PropPath[160];
    snprintf(PropPath, sizeof(PropPath),
             "/sys/class/kfd/kfd/topology/nodes/%u/properties", Node);
    FILE *F = fopen(PropPath, "r");
    if (F == nullptr)
      continue;
    char Line[256];
    unsigned long FoundMinor = 0;
    bool HaveMinor = false;
    while (fgets(Line, sizeof(Line), F) != nullptr) {
      unsigned long Value;
      if (sscanf(Line, "drm_render_minor %lu", &Value) == 1) {
        FoundMinor = Value;
        HaveMinor = true;
        break;
      }
    }
    fclose(F);
    if (!HaveMinor || FoundMinor != Minor)
      continue;

    char IdPath[160];
    snprintf(IdPath, sizeof(IdPath),
             "/sys/class/kfd/kfd/topology/nodes/%u/gpu_id", Node);
    F = fopen(IdPath, "r");
    if (F == nullptr)
      return 0;
    unsigned Id = 0;
    const int Scanned = fscanf(F, "%u", &Id);
    fclose(F);
    return Scanned == 1 ? Id : 0;
  }
  return 0;
}

int borrowBoundRenderNodeFd(const char *Path) {
  if (!SharingEnabled.load())
    return -1;

  const uint32_t GpuId = gpuIdForRenderNodePath(Path);
  if (GpuId == 0)
    return -1;

  // The descriptor ACQUIRE_VM bound for this GPU, which the tracker captured
  // from the ioctl itself. Nothing else in the process knows which of several
  // opens of the same node the driver actually bound.
  const int Bound = gpuDrmFd(GpuId);
  if (Bound < 0)
    return -1;

  // A duplicate: the caller owns and will close what it gets back.
  return dup(Bound);
}

bool isFdSharingEnabled() { return SharingEnabled.load(); }

void enableFdSharing() { SharingEnabled.store(true); }

} // namespace luthier::kfd

extern "C" {

/// \brief C-linkage form of \c luthier::kfd::enableFdSharing.
///
/// Exported for the same reason the tracker's lookups are: a tool lives in a
/// different module from the wrapper and cannot link against it, so it resolves
/// this at run time. A tool calls it once, immediately before initializing HSA.
void luthierKfdEnableFdSharing() { luthier::kfd::enableFdSharing(); }

/// \brief Whether redirection is on. For a test to assert the tool asked for it.
int luthierKfdFdSharingEnabled() {
  return luthier::kfd::isFdSharingEnabled() ? 1 : 0;
}

} // extern "C"
