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

std::unordered_map<decltype(hsa_executable_symbol_t::handle),
                   hsa::ExecutableSymbol::DeviceFunctionInfo>
    hsa::ExecutableSymbol::DeviceFunctionHandleCache{};

ExecutableSymbol ExecutableSymbol::fromHandle(hsa_executable_symbol_t Symbol) {
  if (DeviceFunctionHandleCache.contains(Symbol.handle)) {
    auto &IndirectFunctionInfo = DeviceFunctionHandleCache[Symbol.handle];
    return {IndirectFunctionInfo.Name, IndirectFunctionInfo.Code,
            hsa::LoadedCodeObject(IndirectFunctionInfo.LCO)};
  }
  return ExecutableSymbol{Symbol};
};

llvm::Expected<hsa_symbol_kind_t> ExecutableSymbol::getType() const {
  hsa_symbol_kind_t Out;
  if (DFO.has_value())
    Out = HSA_SYMBOL_KIND_INDIRECT_FUNCTION;
  else {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Out)));
  }
  return Out;
}
llvm::Expected<std::string> ExecutableSymbol::getName() const {
  if (DFO.has_value())
    return DFO->Name;
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

llvm::Expected<hsa_symbol_linkage_t> ExecutableSymbol::getLinkage() const {
  hsa_symbol_linkage_t Out;
  if (DFO.has_value()) {
    // Indirect functions have Module linkage (AKA not STT_GLOBAL)
    // See ROCr's getLinkage() function
    Out = HSA_SYMBOL_LINKAGE_MODULE;
  } else {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE, &Out)));
  }
  return Out;
}

llvm::Expected<hsa_variable_allocation_t>
ExecutableSymbol::getVariableAllocation() const {
  hsa_variable_allocation_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION, &Out)));
  return Out;
}

llvm::Expected<luthier::address_t>
ExecutableSymbol::getVariableAddress() const {
  luthier::address_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &Out)));
  return Out;
}
llvm::Expected<const KernelDescriptor *>
ExecutableSymbol::getKernelDescriptor() const {
  luthier::address_t KernelObject;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          this->asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
          &KernelObject)));
  return reinterpret_cast<const KernelDescriptor *>(KernelObject);
}

llvm::Expected<ExecutableSymbol>
ExecutableSymbol::fromKernelDescriptor(const KernelDescriptor *KD) {
  hsa_executable_t Executable;
  const auto &LoaderTable =
      hsa::Interceptor::instance().getHsaVenAmdLoaderTable();

  // Check which executable this kernel object (address) belongs to
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(LoaderTable.hsa_ven_amd_loader_query_executable(
          reinterpret_cast<const void *>(KD), &Executable)));
  llvm::SmallVector<GpuAgent, 4> Agents;
  LUTHIER_RETURN_ON_ERROR(hsa::getGpuAgents(Agents));

  for (const auto &A : Agents) {
    auto Symbols = hsa::Executable(Executable).getAgentSymbols(A);
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

llvm::Expected<GpuAgent> ExecutableSymbol::getAgent() const {
  if (DFO.has_value()) {
    return hsa::LoadedCodeObject(DFO->LCO).getAgent();
  } else {
    hsa_agent_t Agent;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &Agent)));
    return hsa::GpuAgent(Agent);
  }
}

llvm::Expected<Executable> ExecutableSymbol::getExecutable() const {
  auto Type = getType();
  LUTHIER_RETURN_ON_ERROR(Type.takeError());
  luthier::address_t Address;
  if (*Type == HSA_SYMBOL_KIND_VARIABLE)
    LUTHIER_RETURN_ON_ERROR(getVariableAddress().moveInto(Address));
  else if (*Type == HSA_SYMBOL_KIND_KERNEL) {
    auto KD = getKernelDescriptor();
    LUTHIER_RETURN_ON_ERROR(KD.takeError());
    Address = reinterpret_cast<luthier::address_t>(*KD);
  } else {
    Address = reinterpret_cast<luthier::address_t>(DFO->Code.data());
  }
  hsa_executable_t Executable;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_query_executable(
          reinterpret_cast<const void *>(Address), &Executable)));
  return luthier::hsa::Executable(Executable);
}

llvm::Expected<std::optional<LoadedCodeObject>>
ExecutableSymbol::getLoadedCodeObject() const {
  if (DFO.has_value())
    return LoadedCodeObject(DFO->LCO);
  else {
    auto Executable = getExecutable();
    LUTHIER_RETURN_ON_ERROR(Executable.takeError());

    auto LoadedCodeObjects = Executable->getLoadedCodeObjects();
    LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());
    auto Name = getName();
    LUTHIER_RETURN_ON_ERROR(Name.takeError());
    for (const auto &LCO : *LoadedCodeObjects) {

      auto HostElf = LCO.getStorageELF();
      LUTHIER_RETURN_ON_ERROR(HostElf.takeError());

      auto ElfSymbol = getSymbolByName(*HostElf, *Name);
      LUTHIER_RETURN_ON_ERROR(ElfSymbol.takeError());
      if (ElfSymbol->has_value())
        return LCO;
    }
    return std::nullopt;
  }
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
luthier::hsa::ExecutableSymbol::getMachineCode() const {
  auto SymbolType = getType();
  LUTHIER_RETURN_ON_ERROR(SymbolType.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(*SymbolType != HSA_SYMBOL_KIND_VARIABLE));

  if (*SymbolType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION)
    return DFO->Code;
  else {
    auto LCO = getLoadedCodeObject();
    LUTHIER_RETURN_ON_ERROR(LCO.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(LCO->has_value()));

    auto KdSymbolName = getName();
    LUTHIER_RETURN_ON_ERROR(KdSymbolName.takeError());

    auto KernelSymbolName = KdSymbolName->substr(0, KdSymbolName->find(".kd"));

    auto HostELF = (*LCO)->getStorageELF();
    LUTHIER_RETURN_ON_ERROR(HostELF.takeError());

    auto ElfSymbol = getSymbolByName(*HostELF, KernelSymbolName);
    LUTHIER_RETURN_ON_ERROR(ElfSymbol.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ElfSymbol->has_value()));

    auto SymbolAddress = ElfSymbol.get()->getAddress();
    LUTHIER_RETURN_ON_ERROR(SymbolAddress.takeError());

    auto TextSection = ElfSymbol.get()->getSection();
    LUTHIER_RETURN_ON_ERROR(TextSection.takeError());

    auto SymbolLMA = getSymbolLMA(HostELF->getELFFile(), **ElfSymbol);
    LUTHIER_RETURN_ON_ERROR(SymbolLMA.takeError());

    auto LoadedMemory = LCO.get()->getLoadedMemory();
    LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

    return llvm::ArrayRef<uint8_t>{
        reinterpret_cast<const uint8_t *>(*SymbolLMA + LoadedMemory->data()),
        ElfSymbol.get()->getSize()};
  }
}
llvm::Error ExecutableSymbol::cache() const {
  std::lock_guard Lock(getMutex());
  if (DFO.has_value())
    DeviceFunctionHandleCache.insert({hsaHandle(), *DFO});
  return llvm::Error::success();
}
llvm::Error ExecutableSymbol::invalidate() const {
  std::lock_guard Lock(getMutex());
  if (DFO.has_value())
    DeviceFunctionHandleCache.erase(hsaHandle());
  return llvm::Error::success();
}

} // namespace luthier::hsa
