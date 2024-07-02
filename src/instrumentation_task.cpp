#include <luthier/instrumentation_task.h>

#include "code_generator.hpp"
#include "code_lifter.hpp"
#include "tool_executable_manager.hpp"

namespace luthier {

llvm::Error luthier::InstrumentationTask::insertHookAt(
    llvm::MachineInstr &MI, const void *Hook, InstrPoint IPoint,
    llvm::ArrayRef<std::pair<ArgType, uint64_t>> Args, bool RemoveInstr) {
  const auto *SIM = llvm::dyn_cast<StaticInstrumentationModule>(&IM);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(SIM != nullptr));
  auto HookName = SIM->convertHookHandleToHookName(Hook);
  LUTHIER_RETURN_ON_ERROR(HookName.takeError());
  if (!HookInsertionTasks.contains(&MI)) {
    HookInsertionTasks.insert(
        {&MI, llvm::SmallVector<hook_insertion_task_descriptor, 1>()});
  }
  HookInsertionTasks[&MI].emplace_back(
      *HookName, &MI, IPoint,
      llvm::SmallVector<std::pair<ArgType, uint64_t>>(Args), RemoveInstr);
  return llvm::Error::success();
}

InstrumentationTask::InstrumentationTask(
    llvm::StringRef Preset, InstrumentationTask::mutator_func_t Mutator)
    : Preset(Preset), MutatorFunction(std::move(Mutator)),
      IM(ToolExecutableManager::instance().getStaticInstrumentationModule()){};

} // namespace luthier