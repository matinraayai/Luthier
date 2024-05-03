#include "hsa_platform.hpp"

namespace luthier::hsa {

llvm::Error Platform::registerFrozenExecutable(hsa_executable_t Exec) {
  hsa::Executable ExecWrap(Exec);
  // Check if executable is indeed frozen
  auto State = ExecWrap.getState();
  LUTHIER_RETURN_ON_ERROR(State.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(*State == HSA_EXECUTABLE_STATE_FROZEN));
  //

  // Get a list of the executable's loaded code objects
  auto LCOs = ExecWrap.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());

  return llvm::Error::success();
}

} // namespace luthier::hsa
