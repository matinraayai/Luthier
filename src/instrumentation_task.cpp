#include <luthier/instrumentation_task.h>

#include "code_generator.hpp"
#include "code_lifter.hpp"
#include "tool_executable_manager.hpp"

namespace luthier {

void luthier::InstrumentationTask::insertHookAt(llvm::MachineInstr &MI,
                                                const void *Hook,
                                                InstrPoint IPoint) {
  HookInsertionTasks.insert({&MI, {Hook, IPoint}});
}

} // namespace luthier