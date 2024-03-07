#include "hsa_executable_symbol.hpp"

#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/ErrorHandling.h>

#include "error.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_loaded_code_object.hpp"
#include "object_utils.hpp"

namespace luthier::hsa {

llvm::Expected<hsa_symbol_kind_t> ExecutableSymbol::getType() const {
  hsa_symbol_kind_t Out;
  if (IndirectFunctionName.has_value() && IndirectFunctionCode.has_value())
    Out = HSA_SYMBOL_KIND_INDIRECT_FUNCTION;
  else {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Out)));
  }
  return Out;
}
llvm::Expected<std::string> ExecutableSymbol::getName() const {
  if (IndirectFunctionName.has_value())
    return *IndirectFunctionName;
  else {
    uint32_t NameLength;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &NameLength)));
    std::string out(NameLength, '\0');
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME, &out.front())));
    return out;
  }
}

hsa_symbol_linkage_t ExecutableSymbol::getLinkage() const {
  return HSA_SYMBOL_LINKAGE_MODULE;
}

llvm::Expected<luthier_address_t> ExecutableSymbol::getVariableAddress() const {
  luthier_address_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &Out)));
  return Out;
}
llvm::Expected<const KernelDescriptor *>
ExecutableSymbol::getKernelDescriptor() const {
  luthier_address_t KernelObject;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          this->asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
          &KernelObject)));
  return reinterpret_cast<const KernelDescriptor *>(KernelObject);
}

llvm::Expected<ExecutableSymbol>
ExecutableSymbol::fromKernelDescriptor(const hsa::KernelDescriptor *KD) {
  hsa_executable_t Executable;
  const auto &LoaderTable =
      HsaInterceptor::instance().getHsaVenAmdLoaderTable();

  // Check which executable this kernel object (address) belongs to
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(LoaderTable.hsa_ven_amd_loader_query_executable(
          reinterpret_cast<const void *>(KD), &Executable)));
  llvm::SmallVector<GpuAgent> Agents;
  LUTHIER_RETURN_ON_ERROR(hsa::getGpuAgents(Agents));

  for (const auto &A : Agents) {
    auto Symbols = hsa::Executable(Executable).getSymbols(A);
    LUTHIER_RETURN_ON_ERROR(Symbols.takeError());
    for (const auto &S : *Symbols) {
      auto CurrSymbolKD = S.getKernelDescriptor();
      LUTHIER_RETURN_ON_ERROR(CurrSymbolKD.takeError());
      if (KD == *CurrSymbolKD)
        return S;
    }
  }

  llvm::report_fatal_error(llvm::formatv(
      "Kernel descriptor {0:x} does not have a symbol associated with it.",
      reinterpret_cast<const void *>(KD)));
}
luthier::hsa::GpuAgent luthier::hsa::ExecutableSymbol::getAgent() const {
  return luthier::hsa::GpuAgent(Agent);
}
luthier::hsa::Executable luthier::hsa::ExecutableSymbol::getExecutable() const {
  return luthier::hsa::Executable(Executable);
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
luthier::hsa::ExecutableSymbol::getMachineCode() const {
  auto SymbolType = getType();
  LUTHIER_RETURN_ON_ERROR(SymbolType.takeError());

  if (auto Err = LUTHIER_ARGUMENT_ERROR_CHECK(*SymbolType !=
                                              HSA_SYMBOL_KIND_VARIABLE)) {
    return Err;
  }

  if (*SymbolType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION)
    return *IndirectFunctionCode;
  else {
    auto LoadedCodeObjects = hsa::Executable(Executable).getLoadedCodeObjects();
    LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());
    auto KdSymbolName = getName();
    LUTHIER_RETURN_ON_ERROR(KdSymbolName.takeError());
    auto KernelSymbolName = KdSymbolName->substr(0, KdSymbolName->find(".kd"));
    for (const auto &Lco : *LoadedCodeObjects) {
      auto StorageMemoryOrError = Lco.getStorageMemory();
      LUTHIER_RETURN_ON_ERROR(StorageMemoryOrError.takeError());
      auto HostElfOrError = getAMDGCNObjectFile(*StorageMemoryOrError);
      LUTHIER_RETURN_ON_ERROR(HostElfOrError.takeError());

      auto hostElf = HostElfOrError->get();

      // TODO: Replace this with a hash lookup
      auto Syms = hostElf->symbols();
      for (llvm::object::ELFSymbolRef ElfSymbol : Syms) {
        auto NameOrError = ElfSymbol.getName();
        LUTHIER_RETURN_ON_ERROR(NameOrError.takeError());
        if (NameOrError.get() == KernelSymbolName) {
          auto addressOrError = ElfSymbol.getAddress();
          LUTHIER_RETURN_ON_ERROR(addressOrError.takeError());
          auto loadedMemory = Lco.getLoadedMemory();
          LUTHIER_RETURN_ON_ERROR(loadedMemory.takeError());
          return llvm::ArrayRef<uint8_t>{
              reinterpret_cast<const uint8_t *>(*addressOrError +
                                                loadedMemory->data()),
              ElfSymbol.getSize()};
        }
      }
    };
  }
  llvm_unreachable("No device code associated with the symbol was found");
}

} // namespace luthier::hsa
