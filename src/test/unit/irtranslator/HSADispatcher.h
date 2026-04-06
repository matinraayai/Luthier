#ifndef LUTHIER_TEST_HSA_DISPATCHER_H
#define LUTHIER_TEST_HSA_DISPATCHER_H

#include "luthier/HSA/Agent.h"
#include "luthier/HSA/ApiTable.h"
#include "luthier/HSA/Memory.h"
#include "luthier/HSA/Region.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <hsa/hsa.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luthier::test {

/// Per-GPU-agent state: queue, memory regions, and ISA name.
struct GpuAgentInfo {
  hsa_agent_t Agent{};
  std::string Name;          ///< e.g. "gfx908"
  hsa_queue_t *Queue = nullptr;
  hsa_region_t KernargRegion{};
  hsa_region_t FineRegion{};
  bool HasKernarg = false;
  bool HasFine = false;
};

/// Manages HSA runtime lifecycle, multi-GPU detection, memory allocation, and
/// kernel dispatch for the instruction semantic fuzzer.
///
/// At init() time every GPU agent whose ISA is distinct from the others is
/// discovered.  The fuzzer can then iterate over all unique-ISA agents and
/// dispatch tests on each.
///
/// Uses Luthier's HSA wrapper functions (via ApiTableContainer) and the core
/// HSA region API (not AMD memory pools) for allocation.
class HSADispatcher {
public:
  HSADispatcher() = default;
  ~HSADispatcher();

  HSADispatcher(const HSADispatcher &) = delete;
  HSADispatcher &operator=(const HSADispatcher &) = delete;

  /// Initialize the HSA runtime, detect all GPU agents with distinct ISAs,
  /// create queues, and discover memory regions.
  llvm::Error init();

  /// Shut down the HSA runtime and release all resources.
  void shutdown();

  /// \returns The CoreApi table container for use with Luthier wrappers.
  const luthier::hsa::ApiTableContainer<::CoreApiTable> &getCoreApi() const {
    return *CoreApi;
  }

  /// \returns All discovered GPU agents with distinct ISAs.
  llvm::ArrayRef<GpuAgentInfo> getGpuAgents() const { return GpuAgents; }

  /// \returns The number of distinct-ISA GPU agents.
  size_t getNumGpuAgents() const { return GpuAgents.size(); }

  /// \returns Info for the GPU agent at index \p Idx.
  const GpuAgentInfo &getGpuAgent(size_t Idx) const { return GpuAgents[Idx]; }

  /// Allocate \p Size bytes from the kernarg region of \p Agent.
  llvm::Expected<void *> allocKernarg(const GpuAgentInfo &Agent, size_t Size);

  /// Allocate \p Size bytes from the fine-grained region of \p Agent.
  llvm::Expected<void *> allocFinegrained(const GpuAgentInfo &Agent,
                                          size_t Size);

  /// Free memory previously allocated via alloc* methods.
  void freeMem(void *Ptr);

  struct DispatchResult {
    hsa_signal_value_t SignalValue;
  };

  /// Load an ELF code object, look up the kernel symbol, and dispatch
  /// on the given GPU agent.
  llvm::Expected<DispatchResult>
  dispatch(const GpuAgentInfo &Agent, llvm::ArrayRef<char> ELF,
           llvm::StringRef KernelName, void *KernargPtr, uint32_t KernargSize,
           uint32_t GridSizeX = 1, uint32_t WorkgroupSizeX = 1,
           uint32_t GroupSegSize = 0, uint32_t PrivateSegSize = 0);

private:
  bool Initialized = false;
  ::CoreApiTable RawCoreTable{};
  std::unique_ptr<luthier::hsa::ApiTableContainer<::CoreApiTable>> CoreApi;

  std::vector<GpuAgentInfo> GpuAgents;

  llvm::Error discoverAgents();
  llvm::Error setupAgentRegions(GpuAgentInfo &Info);
};

} // namespace luthier::test

#endif // LUTHIER_TEST_HSA_DISPATCHER_H
