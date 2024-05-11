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

llvm::DenseMap<decltype(hsa_executable_symbol_t::handle),
                   ExecutableSymbol::ExecutableSymbolInfo>
    ExecutableSymbol::SymbolHandleCache{};

llvm::Expected<ExecutableSymbol> ExecutableSymbol::createDeviceFunctionSymbol(
    hsa_loaded_code_object_t LCO,
    const object::ELFSymbolRef &DeviceFunctionELFSymbol) {
  hsa::LoadedCodeObject LCOWrapper{LCO};

  auto ELF = LCOWrapper.getStorageELF();
  LUTHIER_RETURN_ON_ERROR(ELF.takeError());

  auto SymbolAddress = DeviceFunctionELFSymbol.getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolAddress.takeError());
  auto ELFBase = DeviceFunctionELFSymbol.getObject()
                     ->getMemoryBufferRef()
                     .getBufferStart();
  ExecutableSymbol Out{
      {*SymbolAddress + reinterpret_cast<luthier::address_t>(ELFBase)},
      LCO,
      DeviceFunctionELFSymbol};
  LUTHIER_RETURN_ON_ERROR(Out.cache());
  return Out;
}

llvm::Expected<ExecutableSymbol>
ExecutableSymbol::fromHandle(hsa_executable_symbol_t Symbol) {
  std::lock_guard Lock(getCacheMutex());
  llvm::outs() << "Is symbol present? "
               << SymbolHandleCache.contains(Symbol.handle) << "\n";
  llvm::outs() << "symbol to look for: " << llvm::format_hex(Symbol.handle, 8)
               << "\n";
  for (const auto &[H, I] : SymbolHandleCache) {
    llvm::outs() << "Symbol in the map: " << llvm::format_hex(H, 8) << "\n";
  }
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(SymbolHandleCache.contains(Symbol.handle)));
  const auto &SymbolInfo = SymbolHandleCache.at(Symbol.handle);
  return ExecutableSymbol{Symbol, SymbolInfo.LCO, SymbolInfo.Symbol,
                          SymbolInfo.KernelFunctionSymbol};
};

SymbolKind ExecutableSymbol::getType() const {
  if (SymbolInfo.KernelFunctionSymbol.has_value())
    return KERNEL;
  else if (SymbolInfo.Symbol.getELFType() == llvm::ELF::STT_FUNC)
    return DEVICE_FUNCTION;
  else
    return VARIABLE;
}

llvm::Expected<llvm::StringRef> ExecutableSymbol::getName() const {
  return SymbolInfo.Symbol.getName();
}

llvm::Expected<hsa_symbol_linkage_t> ExecutableSymbol::getLinkage() const {
  return SymbolInfo.Symbol.getBinding() == llvm::ELF::STB_GLOBAL
      ? HSA_SYMBOL_LINKAGE_PROGRAM
      : HSA_SYMBOL_LINKAGE_MODULE;
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
  auto Symbol = Platform::instance().getSymbolFromLoadedAddress(
      reinterpret_cast<address_t>(KD));
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Symbol->has_value()));
  return **Symbol;
}

llvm::Expected<GpuAgent> ExecutableSymbol::getAgent() const {
  return LoadedCodeObject(SymbolInfo.LCO).getAgent();
}

llvm::Expected<Executable> ExecutableSymbol::getExecutable() const {
  return LoadedCodeObject(SymbolInfo.LCO).getExecutable();
}

llvm::Expected<LoadedCodeObject> ExecutableSymbol::getLoadedCodeObject() const {
  return LoadedCodeObject(SymbolInfo.LCO);
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
luthier::hsa::ExecutableSymbol::getMachineCode() const {
  auto SymbolType = getType();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(SymbolType != VARIABLE));
  auto CodeSymbol = SymbolType == KERNEL ? *SymbolInfo.KernelFunctionSymbol
                                         : SymbolInfo.Symbol;
  auto LCO = getLoadedCodeObject();
  LUTHIER_RETURN_ON_ERROR(LCO.takeError());

  auto StorageELF = LCO->getStorageELF();

  LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());

  auto LoadedMemory = LCO->getLoadedMemory();
  LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

  auto CodeAddress = CodeSymbol.getAddress();
  LUTHIER_RETURN_ON_ERROR(CodeAddress.takeError());

  auto CodeSize = CodeSymbol.getSize();

  auto SymbolLMA = getSymbolLMA(StorageELF->getELFFile(), CodeSymbol);
  LUTHIER_RETURN_ON_ERROR(SymbolLMA.takeError());

  return llvm::ArrayRef<uint8_t>{
      reinterpret_cast<const uint8_t *>(*SymbolLMA + LoadedMemory->data()),
      CodeSize};
}
llvm::Error ExecutableSymbol::cache() const {
  std::lock_guard Lock(getCacheMutex());
  llvm::outs() << "Inserting handle " << llvm::format_hex(hsaHandle(), 8) << "\n";
  SymbolHandleCache.insert({hsaHandle(), SymbolInfo});
  return llvm::Error::success();
}
llvm::Error ExecutableSymbol::invalidate() const {
  std::lock_guard Lock(getCacheMutex());
  SymbolHandleCache.erase(hsaHandle());
  return llvm::Error::success();
}

bool ExecutableSymbol::isCached() const {
  std::lock_guard Lock(getCacheMutex());
  return SymbolHandleCache.contains(this->hsaHandle());
}

} // namespace luthier::hsa
