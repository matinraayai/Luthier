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

GpuAgent ExecutableSymbol::getAgent() const {
  return luthier::hsa::GpuAgent(Agent);
}

Executable ExecutableSymbol::getExecutable() const {
  return luthier::hsa::Executable(Executable);
}

llvm::Expected<std::optional<LoadedCodeObject>>
ExecutableSymbol::getLoadedCodeObject() const {
  auto LoadedCodeObjects = hsa::Executable(Executable).getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());
  auto Name = getName();
  LUTHIER_RETURN_ON_ERROR(Name.takeError());
  for (const auto &LCO : *LoadedCodeObjects) {
    auto StorageMemory = LCO.getStorageMemory();
    LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());

    auto HostElf = getAMDGCNObjectFile(*StorageMemory);
    LUTHIER_RETURN_ON_ERROR(HostElf.takeError());

    auto ElfSymbol = getSymbolByName(**HostElf, *Name);
    LUTHIER_RETURN_ON_ERROR(ElfSymbol.takeError());

    if (ElfSymbol->has_value())
      return LCO;
  }
  return std::nullopt;
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
luthier::hsa::ExecutableSymbol::getMachineCode() const {
  auto SymbolType = getType();
  LUTHIER_RETURN_ON_ERROR(SymbolType.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(*SymbolType != HSA_SYMBOL_KIND_VARIABLE));

  if (*SymbolType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION)
    return *IndirectFunctionCode;
  else {
    auto LCO = getLoadedCodeObject();
    LUTHIER_RETURN_ON_ERROR(LCO.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(LCO->has_value()));

    auto KdSymbolName = getName();
    LUTHIER_RETURN_ON_ERROR(KdSymbolName.takeError());

    auto KernelSymbolName = KdSymbolName->substr(0, KdSymbolName->find(".kd"));

    auto StorageMemory = LCO.get()->getStorageMemory();
    LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());

    auto HostELF = getAMDGCNObjectFile(*StorageMemory);
    LUTHIER_RETURN_ON_ERROR(HostELF.takeError());

    auto ElfSymbol = getSymbolByName(**HostELF, KernelSymbolName);
    LUTHIER_RETURN_ON_ERROR(ElfSymbol.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ElfSymbol->has_value()));

    auto SymbolAddress = ElfSymbol.get()->getAddress();
    LUTHIER_RETURN_ON_ERROR(SymbolAddress.takeError());

    auto TextSection = ElfSymbol.get()->getSection();
    LUTHIER_RETURN_ON_ERROR(TextSection.takeError());

    auto SymbolLMA = getSymbolLMA(HostELF.get()->getELFFile(), **ElfSymbol);
    LUTHIER_RETURN_ON_ERROR(SymbolLMA.takeError());

    auto LoadedMemory = LCO.get()->getLoadedMemory();
    LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

    return llvm::ArrayRef<uint8_t>{
          reinterpret_cast<const uint8_t *>(*SymbolLMA +
                                            LoadedMemory->data()),
          ElfSymbol.get()->getSize()};
    }
};

} // namespace luthier::hsa
