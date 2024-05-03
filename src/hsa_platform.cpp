#include "hsa_platform.hpp"
#include "object_utils.hpp"

namespace luthier::hsa {

llvm::Error Platform::registerFrozenExecutable(hsa_executable_t Exec) {
  hsa::Executable ExecWrap(Exec);
  // Check if executable is indeed frozen
  auto State = ExecWrap.getState();
  LUTHIER_RETURN_ON_ERROR(State.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(*State == HSA_EXECUTABLE_STATE_FROZEN));
  // Add the Exec to the set of frozen executables
  FrozenExecs.insert(Exec.handle);
  // Get a list of the executable's loaded code objects
  auto LCOs = ExecWrap.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  // Create an LLVM ELF Object for each Loaded Code Object's storage memory
  // and cache it for later use
  for (const auto &LCO : *LCOs) {
    auto StorageMemory = LCO.getStorageMemory();
    LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());
    auto StorageELF = getAMDGCNObjectFile(*StorageMemory);
    LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());
    StorageELFOfLCOs.insert({LCO.hsaHandle(), std::move(*StorageELF)});
  }

  return llvm::Error::success();
}
llvm::Error Platform::unregisterFrozenExecutable(hsa_executable_t Exec) {
  hsa::Executable ExecWrap(Exec);
  // Remove the executable from the frozen set
  FrozenExecs.erase(Exec.handle);

  auto LCOs = ExecWrap.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  // delete the ELF file for the storage memory of each LCO
  for (const auto &LCO : *LCOs) {
    StorageELFOfLCOs.erase(LCO.hsaHandle());
  }

  return llvm::Error::success();
}
llvm::object::ELF64LEObjectFile &
Platform::getStorgeELFofLCO(hsa_loaded_code_object_t LCO) const {
  return *StorageELFOfLCOs.at(LCO.handle);
}

} // namespace luthier::hsa
