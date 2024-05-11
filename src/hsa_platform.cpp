#include "hsa_platform.hpp"
#include "hsa.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_loaded_code_object.hpp"
#include "object_utils.hpp"

namespace luthier::hsa {

std::recursive_mutex ExecutableBackedCachable::CacheMutex;

llvm::Error Platform::registerFrozenExecutable(const Executable &Exec) {
  // Check if executable is indeed frozen
  auto State = Exec.getState();
  LUTHIER_RETURN_ON_ERROR(State.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(*State == HSA_EXECUTABLE_STATE_FROZEN));
  // Get a list of the executable's loaded code objects
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  // Create an LLVM ELF Object for each Loaded Code Object's storage memory
  // and cache it for later use
  for (const auto &LCO : *LCOs) {
    auto LCOAsCachable = llvm::dyn_cast<const ExecutableBackedCachable>(&LCO);
    if (!LCOAsCachable->isCached())
      LUTHIER_RETURN_ON_ERROR(LCOAsCachable->cache());
    auto Symbols = LCO.getExecutableSymbols();
    LUTHIER_RETURN_ON_ERROR(Symbols.takeError());
    for (const auto &Symbol : *Symbols) {
      auto SymbolAsCachable =
          llvm::dyn_cast<const ExecutableBackedCachable>(&Symbol);
      if (!SymbolAsCachable->isCached())
        LUTHIER_RETURN_ON_ERROR(SymbolAsCachable->cache());
      luthier::address_t LoadedAddress;
      if (Symbol.getType() == VARIABLE) {
        LUTHIER_RETURN_ON_ERROR(
            Symbol.getVariableAddress().moveInto(LoadedAddress));
      } else if (Symbol.getType() == KERNEL) {
        auto KD = Symbol.getKernelDescriptor();
        LUTHIER_RETURN_ON_ERROR(KD.takeError());
        LoadedAddress = reinterpret_cast<address_t>(*KD);
      } else {
        auto MachineCode = Symbol.getMachineCode();
        LUTHIER_RETURN_ON_ERROR(MachineCode.takeError());
        LoadedAddress = reinterpret_cast<address_t>(MachineCode->data());
      }
      AddressToSymbolMap.insert({LoadedAddress, Symbol.asHsaType()});
    }
  }
  return llvm::Error::success();
}
llvm::Error Platform::unregisterFrozenExecutable(const Executable &Exec) {
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  for (const auto &LCO : *LCOs) {
    auto Symbols = LCO.getExecutableSymbols();
    LUTHIER_RETURN_ON_ERROR(Symbols.takeError());
    for (const auto &Symbol : *Symbols) {
      luthier::address_t LoadedAddress;
      if (Symbol.getType() == VARIABLE) {
        LUTHIER_RETURN_ON_ERROR(
            Symbol.getVariableAddress().moveInto(LoadedAddress));
      } else if (Symbol.getType() == KERNEL) {
        auto KD = Symbol.getKernelDescriptor();
        LUTHIER_RETURN_ON_ERROR(KD.takeError());
        LoadedAddress = reinterpret_cast<address_t>(*KD);
      } else {
        auto MachineCode = Symbol.getMachineCode();
        LUTHIER_RETURN_ON_ERROR(MachineCode.takeError());
        LoadedAddress = reinterpret_cast<address_t>(MachineCode->data());
      }
      AddressToSymbolMap.erase(LoadedAddress);
      LUTHIER_RETURN_ON_ERROR(
          llvm::dyn_cast<const ExecutableBackedCachable>(&Symbol)
              ->invalidate());
    }
    LUTHIER_RETURN_ON_ERROR(
        llvm::dyn_cast<const ExecutableBackedCachable>(&LCO)->invalidate());
  }
  return llvm::Error::success();
}
llvm::Error
Platform::cacheCreatedLoadedCodeObjectOfExec(const Executable &Exec) {
  std::lock_guard Lock(ExecutableBackedCachable::getCacheMutex());
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  for (const auto &LCO : *LCOs) {
    auto LCOAsCachableItem =
        llvm::dyn_cast<const ExecutableBackedCachable>(&LCO);
    if (!LCOAsCachableItem->isCached()) {
      LUTHIER_RETURN_ON_ERROR(LCOAsCachableItem->cache());
      auto Symbols = LCO.getExecutableSymbols();
      LUTHIER_RETURN_ON_ERROR(Symbols.takeError());
      llvm::outs() << "Number of Symbols: " << Symbols->size() << "\n";
      for (const auto &Symbol : *Symbols) {
        LUTHIER_RETURN_ON_ERROR(
            llvm::dyn_cast<const ExecutableBackedCachable>(&Symbol)->cache());
      }
    }
  }
  return llvm::Error::success();
}
llvm::Expected<std::optional<hsa::ExecutableSymbol>>
Platform::getSymbolFromLoadedAddress(luthier::address_t Address) {
  if (AddressToSymbolMap.contains(Address))
    return hsa::ExecutableSymbol::fromHandle(AddressToSymbolMap.at(Address));
  return std::nullopt;
}
} // namespace luthier::hsa
