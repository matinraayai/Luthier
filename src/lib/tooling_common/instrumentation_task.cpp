#include <luthier/instrumentation_task.h>

#include "tooling_common/code_generator.hpp"
#include "tooling_common/code_lifter.hpp"
#include "tooling_common/tool_executable_manager.hpp"

namespace luthier {

llvm::Error luthier::InstrumentationTask::insertHookAt(
    llvm::MachineInstr &MI, const void *Hook, InstrPoint IPoint,
    llvm::ArrayRef<std::variant<llvm::Constant *, llvm::MCRegister>> Args) {
  const auto *SIM = llvm::dyn_cast<StaticInstrumentationModule>(&IM);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(SIM != nullptr));
  auto HookName = SIM->convertHookHandleToHookName(Hook);
  LUTHIER_RETURN_ON_ERROR(HookName.takeError());
  if (!HookInsertionTasks.contains(&MI)) {
    HookInsertionTasks.insert({&MI, mi_hook_insertion_tasks{}});
  }
  auto &MIHookTasks = HookInsertionTasks[&MI];
  auto &HookTaskListToQueue = IPoint == INSTR_POINT_BEFORE
                                  ? MIHookTasks.BeforeIPointTasks
                                  : MIHookTasks.AfterIPointTasks;
  HookTaskListToQueue.emplace_back(
      *HookName,
      llvm::SmallVector<std::variant<llvm::Constant *, llvm::MCRegister>>(
          Args));
  return llvm::Error::success();
}

InstrumentationTask::InstrumentationTask(llvm::StringRef Preset)
    : Preset(Preset),
      IM(ToolExecutableManager::instance().getStaticInstrumentationModule()){};

} // namespace luthier