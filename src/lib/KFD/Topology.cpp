//===-- Topology.cpp - KFD topology lookups -------------------------------===//
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
/// Implements \c luthier/KFD/Topology.h.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/Topology.h"

#include <cstdio>
#include <cstring>

namespace luthier::kfd {

/// Highest topology node index we scan for. The count is small and the files are
/// tiny, so a bounded scan is simpler than globbing -- which is also what hsakmt
/// does.
static constexpr unsigned MaxTopologyNodes = 64;

std::optional<uint32_t> topologyNodeForGpuId(uint32_t GpuId) {
  if (GpuId == 0)
    return std::nullopt;

  for (unsigned Node = 0; Node < MaxTopologyNodes; Node++) {
    char Path[128];
    snprintf(Path, sizeof(Path), "/sys/class/kfd/kfd/topology/nodes/%u/gpu_id",
             Node);
    FILE *F = fopen(Path, "r");
    if (F == nullptr)
      continue; // node does not exist; keep looking, indices are not dense
    unsigned FoundId = 0;
    const int Scanned = fscanf(F, "%u", &FoundId);
    fclose(F);
    if (Scanned == 1 && FoundId == GpuId)
      return Node;
  }
  return std::nullopt;
}

std::optional<uint32_t> gpuIdForTopologyNode(uint32_t Node) {
  char Path[128];
  snprintf(Path, sizeof(Path), "/sys/class/kfd/kfd/topology/nodes/%u/gpu_id",
           Node);
  FILE *F = fopen(Path, "r");
  if (F == nullptr)
    return std::nullopt;
  unsigned Id = 0;
  const int Scanned = fscanf(F, "%u", &Id);
  fclose(F);
  // 0 marks a CPU node. Returning it would let a caller treat a CPU as a
  // dispatch target, so it is reported the same way an absent node is.
  if (Scanned != 1 || Id == 0)
    return std::nullopt;
  return Id;
}

std::optional<uint64_t> readNodeProperty(uint32_t Node, const char *Name) {
  char Path[160];
  snprintf(Path, sizeof(Path),
           "/sys/class/kfd/kfd/topology/nodes/%u/properties", Node);
  FILE *F = fopen(Path, "r");
  if (F == nullptr)
    return std::nullopt;

  char Line[256];
  std::optional<uint64_t> Value;
  while (fgets(Line, sizeof(Line), F) != nullptr) {
    char Key[128];
    unsigned long long V;
    // Whole-key comparison rather than a prefix match: "capability" and
    // "capability2" are both present, and a prefix match would return whichever
    // came first in the file.
    if (sscanf(Line, "%127s %llu", Key, &V) == 2 && strcmp(Key, Name) == 0) {
      Value = V;
      break;
    }
  }
  fclose(F);
  return Value;
}

std::optional<std::string> renderNodeForGpuId(uint32_t GpuId) {
  std::optional<uint32_t> Node = topologyNodeForGpuId(GpuId);
  if (!Node)
    return std::nullopt;

  std::optional<uint64_t> Minor = readNodeProperty(*Node, "drm_render_minor");
  // Render minors start at 128, so 0 means "this node has no render node" --
  // which is what a CPU node reports, and what a display-only device reports.
  if (!Minor || *Minor == 0)
    return std::nullopt;

  return std::string("/dev/dri/renderD") + std::to_string(*Minor);
}

} // namespace luthier::kfd
