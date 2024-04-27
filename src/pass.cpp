#include <luthier/pass.h>

#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"

namespace luthier {

void luthier::InstrumentationTask::insertCallTo(llvm::MachineInstr &MI,
                                                const void *DevFunc,
                                                luthier::InstrPoint IPoint) {
  InsertCallTasks.insert({&MI, {DevFunc, IPoint}});
}

} // namespace luthier