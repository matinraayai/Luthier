#include <luthier/pass.h>
#include <optional>

#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "llvm/IR/GlobalVariable.h"

namespace luthier {

void luthier::InstrumentationTask::insertCallTo(llvm::MachineInstr &MI,
                                                const void *DevFunc,
                                                luthier::InstrPoint IPoint) {
  InsertCallTasks.insert({&MI, {DevFunc, std::nullopt, IPoint}});
}

void luthier::InstrumentationTask::insertCallTo(llvm::MachineInstr &MI, 
                                                const void *DevFunc,
                                                llvm::ArrayRef<llvm::GlobalVariable*> IFuncArgs,
                                                luthier::InstrPoint IPoint) {
  InsertCallTasks.insert({&MI, {DevFunc, IFuncArgs, IPoint}});
}

} // namespace luthier
