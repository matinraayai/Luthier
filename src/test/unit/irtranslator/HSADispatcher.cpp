#include "HSADispatcher.h"
#include "DirectHsaApiTable.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/ISA.h"
#include "luthier/HSA/Memory.h"
#include "luthier/HSA/Region.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include <cstring>

namespace luthier::test {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static llvm::Error checkHSA(hsa_status_t Status, const char *Call) {
  if (Status == HSA_STATUS_SUCCESS)
    return llvm::Error::success();
  const char *Msg = "unknown error";
  hsa_status_string(Status, &Msg);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 llvm::formatv("{0} failed: {1}", Call, Msg));
}

#define HSA_CHECK(call)                                                        \
  if (auto Err = checkHSA((call), #call))                                      \
    return Err;

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

HSADispatcher::~HSADispatcher() {
  if (Initialized)
    shutdown();
}

llvm::Error HSADispatcher::init() {
  HSA_CHECK(hsa_init());
  Initialized = true;

  RawCoreTable = buildDirectCoreApiTable();
  CoreApi = std::make_unique<luthier::hsa::ApiTableContainer<::CoreApiTable>>(
      RawCoreTable);

  return discoverAgents();
}

void HSADispatcher::shutdown() {
  for (auto &Info : GpuAgents) {
    if (Info.Queue) {
      hsa_queue_destroy(Info.Queue);
      Info.Queue = nullptr;
    }
  }
  GpuAgents.clear();
  CoreApi.reset();
  if (Initialized) {
    hsa_shut_down();
    Initialized = false;
  }
}

//===----------------------------------------------------------------------===//
// Agent / region discovery
//===----------------------------------------------------------------------===//

llvm::Error HSADispatcher::discoverAgents() {
  // Collect all GPU agents.
  llvm::SmallVector<hsa_agent_t, 8> AllGpuAgents;

  LUTHIER_RETURN_ON_ERROR(luthier::hsa::iterateAgents(
      *CoreApi, [&](hsa_agent_t Agent) -> llvm::Error {
        auto TypeOrErr = luthier::hsa::agentGetDeviceType(*CoreApi, Agent);
        if (!TypeOrErr)
          return TypeOrErr.takeError();
        if (*TypeOrErr == HSA_DEVICE_TYPE_GPU)
          AllGpuAgents.push_back(Agent);
        return llvm::Error::success();
      }));

  if (AllGpuAgents.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No GPU agents found");

  // De-duplicate by ISA name — keep only one agent per unique ISA.
  llvm::StringSet<> SeenISAs;

  for (hsa_agent_t Agent : AllGpuAgents) {
    // Get the first ISA of this agent.
    llvm::SmallVector<hsa_isa_t, 2> ISAs;
    LUTHIER_RETURN_ON_ERROR(
        luthier::hsa::agentGetSupportedISAs(*CoreApi, Agent, ISAs));
    if (ISAs.empty())
      continue;

    auto IsaNameOrErr = luthier::hsa::isaGetName(*CoreApi, ISAs[0]);
    if (!IsaNameOrErr)
      return IsaNameOrErr.takeError();

    if (!SeenISAs.insert(*IsaNameOrErr).second)
      continue; // Already have an agent for this ISA.

    // Get the GPU name (e.g. "gfx908") from the ISA name.
    auto GpuNameOrErr = luthier::hsa::isaGetGPUName(*CoreApi, ISAs[0]);
    if (!GpuNameOrErr)
      return GpuNameOrErr.takeError();

    GpuAgentInfo Info;
    Info.Agent = Agent;
    Info.Name = *GpuNameOrErr;

    // Create a queue.
    auto QueueSizeOrErr =
        luthier::hsa::agentGetQueueMaxSize(*CoreApi, Agent);
    if (!QueueSizeOrErr)
      return QueueSizeOrErr.takeError();
    uint32_t QueueSize = std::min<uint32_t>(*QueueSizeOrErr, 1024);

    HSA_CHECK(hsa_queue_create(Agent, QueueSize, HSA_QUEUE_TYPE_SINGLE,
                               nullptr, nullptr, UINT32_MAX, UINT32_MAX,
                               &Info.Queue));

    // Discover memory regions.
    LUTHIER_RETURN_ON_ERROR(setupAgentRegions(Info));

    llvm::errs() << "Discovered GPU: " << Info.Name << " (ISA: "
                 << *IsaNameOrErr << ")\n";

    GpuAgents.push_back(std::move(Info));
  }

  if (GpuAgents.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No GPU agents with valid ISA found");

  return llvm::Error::success();
}

llvm::Error HSADispatcher::setupAgentRegions(GpuAgentInfo &Info) {
  // Find kernarg region: iterate the agent's own regions first.
  auto KernargOrErr =
      luthier::hsa::agentFindKernargRegion(*CoreApi, Info.Agent);
  if (!KernargOrErr)
    return KernargOrErr.takeError();
  if (*KernargOrErr) {
    Info.KernargRegion = **KernargOrErr;
    Info.HasKernarg = true;
  }

  // Find fine-grained region.
  auto FineOrErr =
      luthier::hsa::agentFindFineGrainedRegion(*CoreApi, Info.Agent);
  if (!FineOrErr)
    return FineOrErr.takeError();
  if (*FineOrErr) {
    Info.FineRegion = **FineOrErr;
    Info.HasFine = true;
  }

  // If the agent itself doesn't have the needed regions, search the
  // CPU agent (system memory is typically accessible by all agents).
  if (!Info.HasKernarg || !Info.HasFine) {
    LUTHIER_RETURN_ON_ERROR(luthier::hsa::iterateAgents(
        *CoreApi, [&](hsa_agent_t Agent) -> llvm::Error {
          auto TypeOrErr = luthier::hsa::agentGetDeviceType(*CoreApi, Agent);
          if (!TypeOrErr)
            return TypeOrErr.takeError();
          if (*TypeOrErr != HSA_DEVICE_TYPE_CPU)
            return llvm::Error::success();

          if (!Info.HasKernarg) {
            auto R = luthier::hsa::agentFindKernargRegion(*CoreApi, Agent);
            if (!R)
              return R.takeError();
            if (*R) {
              Info.KernargRegion = **R;
              Info.HasKernarg = true;
            }
          }
          if (!Info.HasFine) {
            auto R = luthier::hsa::agentFindFineGrainedRegion(*CoreApi, Agent);
            if (!R)
              return R.takeError();
            if (*R) {
              Info.FineRegion = **R;
              Info.HasFine = true;
            }
          }
          return llvm::Error::success();
        }));
  }

  if (!Info.HasKernarg)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::formatv("No kernarg region found for agent {0}", Info.Name));
  if (!Info.HasFine)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::formatv("No fine-grained region found for agent {0}",
                      Info.Name));

  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Memory allocation
//===----------------------------------------------------------------------===//

llvm::Expected<void *> HSADispatcher::allocKernarg(const GpuAgentInfo &Agent,
                                                   size_t Size) {
  return luthier::hsa::memoryAllocate(*CoreApi, Agent.KernargRegion, Size);
}

llvm::Expected<void *> HSADispatcher::allocFinegrained(const GpuAgentInfo &Agent,
                                                       size_t Size) {
  return luthier::hsa::memoryAllocate(*CoreApi, Agent.FineRegion, Size);
}

void HSADispatcher::freeMem(void *Ptr) {
  if (Ptr)
    llvm::consumeError(luthier::hsa::memoryFree(*CoreApi, Ptr));
}

//===----------------------------------------------------------------------===//
// Kernel dispatch
//===----------------------------------------------------------------------===//

llvm::Expected<HSADispatcher::DispatchResult>
HSADispatcher::dispatch(const GpuAgentInfo &Agent, llvm::ArrayRef<char> ELF,
                        llvm::StringRef KernelName, void *KernargPtr,
                        uint32_t KernargSize, uint32_t GridSizeX,
                        uint32_t WorkgroupSizeX, uint32_t GroupSegSize,
                        uint32_t PrivateSegSize) {
  // --- Load code object ---
  hsa_code_object_reader_t Reader{};
  HSA_CHECK(hsa_code_object_reader_create_from_memory(ELF.data(), ELF.size(),
                                                       &Reader));

  hsa_executable_t Exec{};
  HSA_CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL,
                                       HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                       nullptr, &Exec));
  HSA_CHECK(hsa_executable_load_agent_code_object(Exec, Agent.Agent, Reader,
                                                   nullptr, nullptr));
  HSA_CHECK(hsa_executable_freeze(Exec, nullptr));

  // --- Get kernel symbol ---
  std::string SymName = (KernelName + ".kd").str();
  hsa_executable_symbol_t Symbol{};
  HSA_CHECK(hsa_executable_get_symbol_by_name(Exec, SymName.c_str(),
                                               &Agent.Agent, &Symbol));

  uint64_t KernelObject = 0;
  HSA_CHECK(hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &KernelObject));

  uint32_t SymGroupSegSize = 0;
  HSA_CHECK(hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &SymGroupSegSize));

  uint32_t SymPrivateSegSize = 0;
  HSA_CHECK(hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &SymPrivateSegSize));

  if (GroupSegSize < SymGroupSegSize)
    GroupSegSize = SymGroupSegSize;
  if (PrivateSegSize < SymPrivateSegSize)
    PrivateSegSize = SymPrivateSegSize;

  // --- Create completion signal ---
  hsa_signal_t Signal{};
  HSA_CHECK(hsa_signal_create(1, 0, nullptr, &Signal));

  // --- Write AQL dispatch packet ---
  hsa_queue_t *Queue = Agent.Queue;
  uint64_t WriteIdx = hsa_queue_add_write_index_screlease(Queue, 1);
  while (WriteIdx - hsa_queue_load_read_index_scacquire(Queue) >= Queue->size)
    ;

  auto *Packet = &reinterpret_cast<hsa_kernel_dispatch_packet_t *>(
      Queue->base_address)[WriteIdx & (Queue->size - 1)];

  std::memset(Packet, 0, sizeof(*Packet));
  Packet->setup = 1;
  Packet->workgroup_size_x = static_cast<uint16_t>(WorkgroupSizeX);
  Packet->workgroup_size_y = 1;
  Packet->workgroup_size_z = 1;
  Packet->grid_size_x = GridSizeX;
  Packet->grid_size_y = 1;
  Packet->grid_size_z = 1;
  Packet->kernel_object = KernelObject;
  Packet->kernarg_address = KernargPtr;
  Packet->group_segment_size = GroupSegSize;
  Packet->private_segment_size = PrivateSegSize;
  Packet->completion_signal = Signal;

  uint16_t Header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  __atomic_store_n(reinterpret_cast<uint16_t *>(Packet), Header,
                   __ATOMIC_RELEASE);

  hsa_signal_store_screlease(Queue->doorbell_signal,
                             static_cast<hsa_signal_value_t>(WriteIdx));

  // --- Wait ---
  hsa_signal_value_t Result = hsa_signal_wait_scacquire(
      Signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
      HSA_WAIT_STATE_BLOCKED);

  hsa_signal_destroy(Signal);
  hsa_executable_destroy(Exec);
  hsa_code_object_reader_destroy(Reader);

  return DispatchResult{Result};
}

} // namespace luthier::test
