//===-- Topology.h - KFD topology lookups -----------------------*- C++ -*-===//
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
/// Reads what the KFD driver publishes about the GPUs on this machine.
///
/// \par Why sysfs rather than HSA
/// Everything here could be asked of an HSA agent instead, and originally was
/// going to be. It cannot: an application that drives KFD itself holds the DRM
/// virtual address space for its GPUs, the kernel permits only one such VM per
/// GPU per process, and \c hsa_init inside that process therefore fails. Reading
/// files cannot collide with anything, which is what makes this the route that
/// works in both kinds of process.
///
/// \par The two identifiers, which are easy to confuse
/// A \b gpu_id is a large opaque number the driver assigns (38979 on our first
/// MI100). A \b node \b index is a small dense counter, and it also covers CPU
/// nodes. sysfs is keyed by the node index; every ioctl speaks gpu_id. So
/// anything that wants a node's properties starting from an ioctl has to convert,
/// which is what \c topologyNodeForGpuId is for.
///
/// Worth stating because the two HSA attributes that look like they would help do
/// not: \c HSA_AGENT_INFO_NODE and \c HSA_AMD_AGENT_INFO_DRIVER_NODE_ID return the
/// same value in ROCR (\c amd_gpu_agent.cpp:1401 and \c :1509, both
/// \c node_id()), and neither is a gpu_id.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_TOPOLOGY_H
#define LUTHIER_KFD_TOPOLOGY_H
#include <cstdint>
#include <optional>
#include <string>

namespace luthier::kfd {

/// \brief The KFD topology node index that reports \p GpuId.
///
/// \return the node index, or \c std::nullopt when no node reports this \p GpuId.
/// A \p GpuId of 0 is always \c nullopt: that marks a CPU node, which owns no
/// device memory, so accepting it would let a caller that passed 0 by mistake
/// resolve to the first CPU node in the topology.
[[nodiscard]] std::optional<uint32_t> topologyNodeForGpuId(uint32_t GpuId);

/// \brief Read one integer field of a node's \c properties file.
///
/// The file is a flat list of \c "name value" lines, which is the format hsakmt
/// parses too (\c topology.c:1180-1256).
///
/// \return the value, or \c std::nullopt when the node or the field is absent.
[[nodiscard]] std::optional<uint64_t> readNodeProperty(uint32_t Node,
                                                       const char *Name);

/// \brief The \c gpu_id a KFD topology node reports.
///
/// The inverse of \c topologyNodeForGpuId, and needed because HSA describes an
/// agent by node index while every ioctl speaks \c gpu_id -- so bridging from an
/// agent to the driver's view of the same device goes through here.
///
/// \return the \c gpu_id, or \c std::nullopt when the node does not exist or is
/// a CPU node (which reports 0 and owns no device memory).
[[nodiscard]] std::optional<uint32_t> gpuIdForTopologyNode(uint32_t Node);

/// \brief Map a KFD \c gpu_id onto its DRM render node path.
///
/// Read from the node's \c drm_render_minor, which is where hsakmt gets it too.
///
/// \warning Opening the returned path does \b not give a descriptor an
/// \c mmap_offset can be resolved on -- see \c KfdAllocationResolver's note about
/// \c ACQUIRE_VM. Kept because it is what names the device in a diagnostic.
///
/// \return the path, or \c std::nullopt when no node reports this \p GpuId, or
/// the node has no render node at all (a CPU node, or a display-only device).
[[nodiscard]] std::optional<std::string> renderNodeForGpuId(uint32_t GpuId);

} // namespace luthier::kfd

#endif // LUTHIER_KFD_TOPOLOGY_H
