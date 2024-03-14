#include <llvm/CodeGen/MachineFunctionPass.h>

namespace luthier {
class ResolveGetPCPass : public llvm::MachineFunctionPass {
protected:
  bool runOnMachineFunction(llvm::MachineFunction &MF) override;
};
} // namespace luthier
