
#include "luthier/expr/MachineIRExpr.h"
#include <llvm/Support/Debug.h>

namespace luthier {

void MachineIRExpr::print(llvm::raw_ostream &OS) const {}

void MachineIRExpr::dump() const { return print(llvm::dbgs()); }
} // namespace luthier