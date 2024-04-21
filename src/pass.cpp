#include <luthier/pass.h>

#include "code_generator.hpp"
#include "code_object_manager.hpp"

llvm::Error luthier::InstrumentationPass::insertCallTo(
    llvm::MachineInstr &MI, const void *DevFunc, luthier::InstrPoint IPoint) {
  auto& LSI = getAnalysis<luthier::LiftedSymbolInfoWrapperPass>().getLSI();
  const auto& HsaInst = LSI.getHSAInstrOfMachineInstr(MI);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(HsaInst.has_value()));

  auto InstrumentationFunction =
      luthier::CodeObjectManager::instance().getInstrumentationFunction(
          DevFunc, hsa::GpuAgent((**HsaInst).getAgent()));
  LUTHIER_RETURN_ON_ERROR(InstrumentationFunction.takeError());

  return llvm::Error::success();
}
