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
  hsa_symbol_kind_t out;
  if (indirectFunctionName_.has_value() && indirectFunctionCode_.has_value())
    out = HSA_SYMBOL_KIND_INDIRECT_FUNCTION;
  else {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &out)));
  }
  return out;
}
llvm::Expected<std::string> ExecutableSymbol::getName() const {
  if (indirectFunctionName_.has_value())
    return *indirectFunctionName_;
  else {
    uint32_t nameLength;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameLength)));
    std::string out(nameLength, '\0');
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
  luthier_address_t out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &out)));
  return out;
}
llvm::Expected<const KernelDescriptor *>
ExecutableSymbol::getKernelDescriptor() const {
  luthier_address_t kernelObject;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          this->asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
          &kernelObject)));
  return reinterpret_cast<const KernelDescriptor *>(kernelObject);
}

llvm::Expected<ExecutableSymbol>
ExecutableSymbol::fromKernelDescriptor(const hsa::KernelDescriptor *kd) {
  hsa_executable_t executable;
  const auto &loaderTable =
      HsaInterceptor::instance().getHsaVenAmdLoaderTable();

  // Check which executable this kernel object (address) belongs to
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(loaderTable.hsa_ven_amd_loader_query_executable(
          reinterpret_cast<const void *>(kd), &executable)));
  llvm::SmallVector<GpuAgent> agents;
  LUTHIER_RETURN_ON_ERROR(hsa::getGpuAgents(agents));

  for (const auto &a : agents) {
    auto symbols = Executable(executable).getSymbols(a);
    LUTHIER_RETURN_ON_ERROR(symbols.takeError());
    for (const auto &s : *symbols) {
      auto CurrSymbolKD = s.getKernelDescriptor();
      LUTHIER_RETURN_ON_ERROR(CurrSymbolKD.takeError());
      if (kd == *CurrSymbolKD)
        return s;
    }
  }

  llvm::report_fatal_error(llvm::formatv(
      "Kernel descriptor {0:x} does not have a symbol associated with it.",
      reinterpret_cast<const void *>(kd)));
}
luthier::hsa::GpuAgent luthier::hsa::ExecutableSymbol::getAgent() const {
  return luthier::hsa::GpuAgent(agent_);
}
luthier::hsa::Executable luthier::hsa::ExecutableSymbol::getExecutable() const {
  return luthier::hsa::Executable(executable_);
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
luthier::hsa::ExecutableSymbol::getMachineCode() const {
  auto SymbolType = getType();
  LUTHIER_RETURN_ON_ERROR(SymbolType.takeError());

  if (auto Err = LUTHIER_ARGUMENT_ERROR_CHECK(*SymbolType != HSA_SYMBOL_KIND_VARIABLE)) {
    return Err;
  }

  if (*SymbolType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION)
    return *indirectFunctionCode_;
  else {
    auto loadedCodeObjects = Executable(executable_).getLoadedCodeObjects();
    LUTHIER_RETURN_ON_ERROR(loadedCodeObjects.takeError());
    auto kdSymbolName = getName();
    LUTHIER_RETURN_ON_ERROR(kdSymbolName.takeError());
    auto kernelSymbolName = kdSymbolName->substr(0, kdSymbolName->find(".kd"));
    for (const auto &lco : *loadedCodeObjects) {
      auto storageMemoryOrError = lco.getStorageMemory();
      LUTHIER_RETURN_ON_ERROR(storageMemoryOrError.takeError());
      auto hostElfOrError = getELFObjectFileBase(*storageMemoryOrError);
      LUTHIER_RETURN_ON_ERROR(hostElfOrError.takeError());

      auto hostElf = hostElfOrError->get();

      // TODO: Replace this with a hash lookup
      auto Syms = hostElf->symbols();
      for (llvm::object::ELFSymbolRef elfSymbol : Syms) {
        auto nameOrError = elfSymbol.getName();
        LUTHIER_RETURN_ON_ERROR(nameOrError.takeError());
        if (nameOrError.get() == kernelSymbolName) {
          auto addressOrError = elfSymbol.getAddress();
          LUTHIER_RETURN_ON_ERROR(addressOrError.takeError());
          auto loadedMemory = lco.getLoadedMemory();
          LUTHIER_RETURN_ON_ERROR(loadedMemory.takeError());
          return llvm::ArrayRef<uint8_t>{
              reinterpret_cast<const uint8_t *>(*addressOrError +
                                                loadedMemory->data()),
              elfSymbol.getSize()};
        }
      }
    };
  }
}

} // namespace luthier::hsa
