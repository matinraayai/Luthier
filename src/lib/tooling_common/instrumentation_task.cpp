#include <luthier/instrumentation_task.h>

#include "tooling_common/code_generator.hpp"
#include "tooling_common/code_lifter.hpp"
#include "tooling_common/tool_executable_manager.hpp"

namespace luthier {

llvm::Error luthier::InstrumentationTask::insertHookBefore(
    llvm::MachineInstr &MI, const void *Hook,
    llvm::ArrayRef<std::variant<llvm::Constant *, llvm::MCRegister>> Args) {
  const auto *SIM = llvm::dyn_cast<StaticInstrumentationModule>(&IM);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(SIM != nullptr));
  auto HookName = SIM->convertHookHandleToHookName(Hook);
  LUTHIER_RETURN_ON_ERROR(HookName.takeError());
  // Check if the passed MI belongs to the LiftedRepresentation being
  // worked on
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(LR.getHSAInstrOfMachineInstr(MI) != nullptr));
  if (!HookInsertionTasks.contains(&MI)) {
    HookInsertionTasks.insert({&MI, {}});
  }
  HookInsertionTasks[&MI].emplace_back(
      *HookName,
      llvm::SmallVector<std::variant<llvm::Constant *, llvm::MCRegister>>(
          Args));
  return llvm::Error::success();
}

InstrumentationTask::InstrumentationTask(LiftedRepresentation &LR)
    : LR(LR),
      IM(ToolExecutableManager::instance().getStaticInstrumentationModule()){};

} // namespace luthier