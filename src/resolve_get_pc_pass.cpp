#include "resolve_get_pc_pass.hpp"
#include <AMDGPU.h>

namespace luthier {

bool ResolveGetPCPass::runOnMachineFunction(llvm::MachineFunction &MF) {
  for (auto& MBB: MF) {
    for (auto& MI: MBB) {
      if (MI.getOpcode() == llvm::)
      MI.getNextNode().
    }
  }
  return false;
}

} // namespace luthier
