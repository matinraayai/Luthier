//===-- lifted_representation.cpp - Lifted Representation  ----------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements Luthier's Lifted Representation, which contains the
/// \c llvm::Module and \c llvm::MachineModuleInfo of a lifted HSA primitive (a
/// kernel or an executable), as well as a mapping between the HSA primitives
/// and LLVM IR primitives involved.
//===----------------------------------------------------------------------===//
#include <llvm/IR/LegacyPassManager.h>
#include <luthier/lifted_representation.h>

namespace luthier {

void LiftedRepresentation::managePassManagerLifetime(
    std::unique_ptr<llvm::legacy::PassManager> PM) {
  PMs.push_back(std::move(PM));
}

llvm::ArrayRef<llvm::MachineInstr*>
LiftedRepresentation::getUsesOfGlobalValue(hsa_executable_symbol_t GV) const {
  if (RelatedGlobalVariables.contains(GV))
    return getUsesOfGlobalValue(*RelatedGlobalVariables.at(GV));
  else if (RelatedFunctions.contains(GV))
    return getUsesOfGlobalValue(RelatedFunctions.at(GV)->getFunction());
  else
    return {};
}

LiftedRepresentation::~LiftedRepresentation() {
  if (PMs.empty()) {
    for (auto &[LCO, LCOModule] : Modules) {
//      delete LCOModule.second;
    }
  } else {
    PMs.clear();
  }
  Modules.clear();
}

LiftedRepresentation::LiftedRepresentation() = default;
} // namespace luthier