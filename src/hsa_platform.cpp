#include "hsa_platform.hpp"
#include "hsa.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_loaded_code_object.hpp"
#include "object_utils.hpp"

namespace luthier::hsa {

std::recursive_mutex ExecutableBackedCachableItem::CacheMutex;

llvm::Error Platform::registerFrozenExecutable(const Executable &Exec) {
  // Check if executable is indeed frozen
  auto State = Exec.getState();
  LUTHIER_RETURN_ON_ERROR(State.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(*State == HSA_EXECUTABLE_STATE_FROZEN));
  LUTHIER_RETURN_ON_ERROR(
      llvm::dyn_cast<const ExecutableBackedCachableItem>(&Exec)->cache());
  // Get a list of the executable's loaded code objects
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  // Create an LLVM ELF Object for each Loaded Code Object's storage memory
  // and cache it for later use
  for (const auto &LCO : *LCOs) {
    LUTHIER_RETURN_ON_ERROR(
        llvm::dyn_cast<const ExecutableBackedCachableItem>(&LCO)->cache());
  }
  llvm::SmallVector<GpuAgent> Agents;
  LUTHIER_RETURN_ON_ERROR(getGpuAgents(Agents));
  for (const auto &Agent : Agents) {
    auto AgentSymbols = Exec.getAgentSymbols(Agent);
    LUTHIER_RETURN_ON_ERROR(AgentSymbols.takeError());
    for (const auto &Symbol : *AgentSymbols)
      LUTHIER_RETURN_ON_ERROR(
          llvm::dyn_cast<const ExecutableBackedCachableItem>(&Symbol)->cache());
  }

  return llvm::Error::success();
}
llvm::Error Platform::unregisterFrozenExecutable(const Executable &Exec) {
  // Get a list of handles to invalidate before actually starting the
  // invalidation process
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  // Actually start invalidating
  LUTHIER_RETURN_ON_ERROR(
      llvm::dyn_cast<const ExecutableBackedCachableItem>(&Exec)->invalidate());
  for (const auto &LCO : *LCOs) {
    LUTHIER_RETURN_ON_ERROR(
        llvm::dyn_cast<const ExecutableBackedCachableItem>(&LCO)->invalidate());
  }
  llvm::SmallVector<GpuAgent> Agents;
  LUTHIER_RETURN_ON_ERROR(getGpuAgents(Agents));
  for (const auto &Agent : Agents) {
    auto AgentSymbols = Exec.getAgentSymbols(Agent);
    LUTHIER_RETURN_ON_ERROR(AgentSymbols.takeError());
    for (const auto &Symbol : *AgentSymbols)
      LUTHIER_RETURN_ON_ERROR(
          llvm::dyn_cast<const ExecutableBackedCachableItem>(&Symbol)
              ->invalidate());
  }

  return llvm::Error::success();
}
} // namespace luthier::hsa
