#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "common/ObjectUtils.hpp"
#include "common/Singleton.hpp"
#include "hsa/Executable.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "hsa/hsa.hpp"
#include <llvm/Object/ELFObjectFile.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-hsa-exec-cache"

namespace object = llvm::object;

namespace luthier {
template <>
hsa::ExecutableBackedObjectsCache
    *Singleton<hsa::ExecutableBackedObjectsCache>::Instance{nullptr};

namespace hsa {

llvm::Error
hsa::ExecutableBackedObjectsCache::LoadedCodeObjectCache::cacheOnCreation(
    const hsa::LoadedCodeObject &LCO) {
  std::lock_guard Lock(ExecutableCacheMutex);
  // Cache the Storage ELF
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("Caching LCO with Handle {0:x}.\n",
                                           LCO.hsaHandle()););
  auto StorageMemory = LCO.getStorageMemory();
  LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());
  auto StorageELF = getAMDGCNObjectFile(*StorageMemory);
  LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());
  auto &CachedELF =
      *(StorageELFOfLCOs.insert({LCO.asHsaType(), std::move(*StorageELF)})
            .first->second);

  // Cache the ISA of the ELF
  auto LLVMISA = getELFObjectFileISA(CachedELF);
  LUTHIER_RETURN_ON_ERROR(LLVMISA.takeError());
  auto ISA = hsa::ISA::fromLLVM(std::get<0>(*LLVMISA), std::get<1>(*LLVMISA),
                                std::get<2>(*LLVMISA));
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());
  ISAOfLCOs.insert({LCO.asHsaType(), ISA->asHsaType()});
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("ISA of LCO {0:x}: {1}.\n",
                                           LCO.hsaHandle(),
                                           llvm::cantFail(ISA->getName())));
  // Cache the ELF Symbols
  // We can cache the variable and extern symbols right away, but we
  // need to wait until the end of the iteration to distinguish between
  // kernels and device function
  auto &VariableELFSymbolsOfThisLCO =
      VariableSymbolsOfLCOs.insert({LCO.asHsaType(), {}}).first->second;
  auto &ExternSymbolsOfThisLCO =
      ExternSymbolsOfLCOs.insert({LCO.asHsaType(), {}}).first->second;

  llvm::StringMap<const llvm::object::ELFSymbolRef> KDSymbolsOfThisLCO;
  llvm::StringMap<const llvm::object::ELFSymbolRef> FuncSymbolsOfThisLCO;

  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("Caching symbols for LCO {0:x}:\n",
                                           LCO.hsaHandle()));

  for (const object::ELFSymbolRef &Symbol : CachedELF.symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    auto Size = Symbol.getSize();
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("\tSymbol Name: {0}, Binding: {1}, Type: {2}\n",
                                *SymbolName, Binding, Type));
    if (Type == llvm::ELF::STT_FUNC)
      FuncSymbolsOfThisLCO.insert({*SymbolName, Symbol});
    else if (Type == llvm::ELF::STT_OBJECT) {
      // Kernel Descriptor Symbol
      if (SymbolName->ends_with(".kd") && Size == 64) {
        KDSymbolsOfThisLCO.insert({*SymbolName, Symbol});
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "\tSymbol {0} is a kernel descriptor.\n", *SymbolName));
      }
      // Variable Symbol
      else {
        auto VariableSymbol =
            LoadedCodeObjectVariable::create(LCO.asHsaType(), Symbol);
        LUTHIER_RETURN_ON_ERROR(VariableSymbol.takeError());
        VariableELFSymbolsOfThisLCO.insert(
            {*SymbolName, VariableSymbol->release()});
      }
    } else if (Type == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64) {
      KDSymbolsOfThisLCO.insert({*SymbolName, Symbol});
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "\tSymbol {0} is a kernel descriptor.\n", *SymbolName));
    } else if (Type == llvm::ELF::STT_NOTYPE &&
               Binding == llvm::ELF::STB_GLOBAL) {
      auto ExternSymbol =
          LoadedCodeObjectExternSymbol::create(LCO.asHsaType(), Symbol);
      LUTHIER_RETURN_ON_ERROR(ExternSymbol.takeError());

      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "\tSymbol {0} is an external symbol.\n", *SymbolName));
      ExternSymbolsOfThisLCO.insert({*SymbolName, ExternSymbol->release()});
    }
  }

  // Cache the LCO and Kernel Symbols Metadata
  auto MetaData = parseNoteMetaData(CachedELF);
  LUTHIER_RETURN_ON_ERROR(MetaData.takeError());
  auto &LCOCachedMetaData =
      MetadataOfLCOs.insert({LCO.asHsaType(), *MetaData}).first->getSecond();

  auto &KernelSymbolsOfThisLCO =
      KernelSymbolsOfLCOs.insert({LCO.asHsaType(), {}}).first->second;

  // Construct the kernel symbols and cache them
  for (auto &KernelMD : LCOCachedMetaData.Kernels) {
    auto &NameWithKDAtTheEnd = KernelMD.Symbol;
    llvm::StringRef NameWithoutKD =
        llvm::StringRef(NameWithKDAtTheEnd)
            .substr(0, NameWithKDAtTheEnd.rfind(".kd"));
    // Find the KD symbol
    auto KDSymbolIter = KDSymbolsOfThisLCO.find(NameWithKDAtTheEnd);
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(KDSymbolIter != KDSymbolsOfThisLCO.end()));
    // Find the kernel function symbol
    auto KFuncSymbolIter = FuncSymbolsOfThisLCO.find(NameWithoutKD);
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(KFuncSymbolIter != FuncSymbolsOfThisLCO.end()));

    // Construct the Kernel LCO Symbol
    auto KernelSymbol =
        LoadedCodeObjectKernel::create(LCO.asHsaType(), KFuncSymbolIter->second,
                                       KDSymbolIter->second, KernelMD);
    LUTHIER_RETURN_ON_ERROR(KernelSymbol.takeError());
    KernelSymbolsOfThisLCO.insert({NameWithoutKD, KernelSymbol->release()});

    // Remove the kernel function symbol from the map so that it doesn't
    // get counted as a device function in the later step
    FuncSymbolsOfThisLCO.erase(KFuncSymbolIter);
  }

  // Finally, construct the device function LCO symbols
  auto &DeviceFuncSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.insert({LCO.asHsaType(), {}}).first->second;
  for (const auto &[Name, FuncSymbol] : FuncSymbolsOfThisLCO) {
    auto DeviceFuncSymbol =
        LoadedCodeObjectDeviceFunction::create(LCO.asHsaType(), FuncSymbol);
    LUTHIER_RETURN_ON_ERROR(DeviceFuncSymbol.takeError());

    DeviceFuncSymbolsOfThisLCO.insert({Name, DeviceFuncSymbol->release()});
  }

  CachedLCOs.insert(LCO.asHsaType());
  return llvm::Error::success();
}

llvm::Error
ExecutableBackedObjectsCache::LoadedCodeObjectCache::invalidateOnDestruction(
    const LoadedCodeObject &LCO) {
  std::lock_guard Lock(ExecutableCacheMutex);
  ISAOfLCOs.erase(LCO.asHsaType());
  for (const auto &[Name, Symbol] : KernelSymbolsOfLCOs.at(LCO.asHsaType())) {
    delete Symbol;
  }
  for (const auto &[Name, Symbol] :
       DeviceFuncSymbolsOfLCOs.at(LCO.asHsaType())) {
    delete Symbol;
  }
  for (const auto &[Name, Symbol] : VariableSymbolsOfLCOs.at(LCO.asHsaType())) {
    delete Symbol;
  }
  for (const auto &[Name, Symbol] : ExternSymbolsOfLCOs.at(LCO.asHsaType())) {
    delete Symbol;
  }
  KernelSymbolsOfLCOs.erase(LCO.asHsaType());
  DeviceFuncSymbolsOfLCOs.erase(LCO.asHsaType());
  VariableSymbolsOfLCOs.erase(LCO.asHsaType());
  ExternSymbolsOfLCOs.erase(LCO.asHsaType());
  MetadataOfLCOs.erase(LCO.asHsaType());
  StorageELFOfLCOs.erase(LCO.asHsaType());
  return llvm::Error::success();
}

bool ExecutableBackedObjectsCache::LoadedCodeObjectCache::isCached(
    const LoadedCodeObject &LCO) {
  return CachedLCOs.contains(LCO.asHsaType());
}

llvm::Error
ExecutableBackedObjectsCache::LoadedCodeObjectSymbolCache::cacheOnCreation(
    const LoadedCodeObjectSymbol &Symbol) {
  std::lock_guard Lock(CacheMutex);
  auto ExecSymbol = Symbol.getExecutableSymbol();
  if (ExecSymbol.has_value())
    ExecToLCOSymbolMap.insert({*ExecSymbol, &Symbol});
  CachedLCOSymbols.insert(&Symbol);

  return llvm::Error::success();
}

llvm::Error ExecutableBackedObjectsCache::LoadedCodeObjectSymbolCache::
    invalidateOnDestruction(const LoadedCodeObjectSymbol &Symbol) {
  std::lock_guard Lock(CacheMutex);
  // Invalidate the address map first
  auto SymbolLoadedAddress = Symbol.getLoadedSymbolContents();
  LUTHIER_RETURN_ON_ERROR(SymbolLoadedAddress.takeError());
  LoadedAddressToSymbolMap.erase(
      reinterpret_cast<luthier::address_t>(SymbolLoadedAddress->data()));
  // If the symbol is a kernel remove the address of its KD from the map
  // as well
  if (const auto *KernelSymbol =
          llvm::dyn_cast<LoadedCodeObjectKernel>(&Symbol)) {
    auto KDAddress = KernelSymbol->getKernelDescriptor();
    LUTHIER_RETURN_ON_ERROR(KDAddress.takeError());
    LoadedAddressToSymbolMap.erase(reinterpret_cast<luthier::address_t>(*KDAddress));
  }
  // Invalidate the LCO symbol afterward
  LoadedLCOSymbols.erase(&Symbol);

  auto ExecSymbol = Symbol.getExecutableSymbol();

  if (ExecSymbol.has_value()) {
    ExecToLCOSymbolMap.erase(*ExecSymbol);
  }

  CachedLCOSymbols.erase(&Symbol);

  return llvm::Error::success();
}

llvm::Error
ExecutableBackedObjectsCache::LoadedCodeObjectSymbolCache::cacheOnDeviceLoad(
    const LoadedCodeObjectSymbol &Symbol) {
  std::lock_guard Lock(CacheMutex);
  // Record the loaded address of the symbol, and cache it
  auto SymbolLoadedAddress = Symbol.getLoadedSymbolContents();
  LUTHIER_RETURN_ON_ERROR(SymbolLoadedAddress.takeError());
  LoadedAddressToSymbolMap.insert(
      {reinterpret_cast<luthier::address_t>(SymbolLoadedAddress->data()),
       &Symbol});
  // If the symbol is a kernel, cache the address of the kernel descriptor
  // as well
  if (const auto *KernelSymbol =
          llvm::dyn_cast<LoadedCodeObjectKernel>(&Symbol)) {
    auto KDAddress = KernelSymbol->getKernelDescriptor();
    LUTHIER_RETURN_ON_ERROR(KDAddress.takeError());
    LoadedAddressToSymbolMap.insert(
        {reinterpret_cast<luthier::address_t>(*KDAddress), &Symbol});
  }
  // Record that this symbol has been loaded onto the device
  LoadedLCOSymbols.insert(&Symbol);

  return llvm::Error::success();
}

bool ExecutableBackedObjectsCache::LoadedCodeObjectSymbolCache::isCached(
    const LoadedCodeObjectSymbol &Symbol) {
  std::lock_guard Lock(CacheMutex);
  return CachedLCOSymbols.contains(&Symbol);
}

llvm::Error
ExecutableBackedObjectsCache::cacheExecutableOnLoadedCodeObjectCreation(
    const Executable &Exec) {
  // Acquire a lock during this operation
  std::lock_guard Lock(CacheMutex);
  // Get all loaded code objects of the executable
  llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(Exec.getLoadedCodeObjects(LCOs));
  // Cache all LCOs, and their symbols, if not already cached
  for (const auto &LCO : LCOs) {
    if (!LCOCache.isCached(LCO)) {
      // Cache the LCO
      LUTHIER_RETURN_ON_ERROR(LCOCache.cacheOnCreation(LCO));
      // Get all symbols of the LCO
      llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *> Symbols;
      LUTHIER_RETURN_ON_ERROR(LCO.getLoadedCodeObjectSymbols(Symbols));
      LLVM_DEBUG(llvm::dbgs()
                 << "Number of Symbols: " << Symbols.size() << "\n");

      // Cache all the LCO symbols
      for (const auto &Symbol : Symbols)
        LUTHIER_RETURN_ON_ERROR(LCOSymbolCache.cacheOnCreation(*Symbol));
    }
  }
  return llvm::Error::success();
}

llvm::Error ExecutableBackedObjectsCache::cacheExecutableOnExecutableFreeze(
    const Executable &Exec) {
  // Check if executable is indeed frozen
  auto State = Exec.getState();
  LUTHIER_RETURN_ON_ERROR(State.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(*State == HSA_EXECUTABLE_STATE_FROZEN));
  // Get a list of the executable's loaded code objects
  llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(Exec.getLoadedCodeObjects(LCOs));

  // Iterate over all LCOs in the executable
  for (const auto &LCO : LCOs) {
    // If the LCO isn't cached already, cache it
    if (!LCOCache.isCached(LCO))
      LUTHIER_RETURN_ON_ERROR(LCOCache.cacheOnCreation(LCO));
    // Get all LCO symbols
    llvm::SmallVector<const LoadedCodeObjectSymbol *, 4> Symbols;
    LUTHIER_RETURN_ON_ERROR(LCO.getLoadedCodeObjectSymbols(Symbols));
    // Iterate over all LCO symbols
    for (const auto Symbol : Symbols) {
      // Cache the symbol if not already cached
      if (!LCOSymbolCache.isCached(*Symbol))
        LUTHIER_RETURN_ON_ERROR(LCOSymbolCache.cacheOnCreation(*Symbol));
      LUTHIER_RETURN_ON_ERROR(LCOSymbolCache.cacheOnDeviceLoad(*Symbol));
    }
  }
  return llvm::Error::success();
}

llvm::Error
ExecutableBackedObjectsCache::invalidateExecutableOnExecutableDestroy(
    const Executable &Exec) {
  std::lock_guard Lock(CacheMutex);
  // Get all the LCOs in the executable
  llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(Exec.getLoadedCodeObjects(LCOs));

  // Iterate over all LCOs
  for (const auto &LCO : LCOs) {
    // Get all LCO symbols
    llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *> Symbols;
    LUTHIER_RETURN_ON_ERROR(LCO.getLoadedCodeObjectSymbols(Symbols));

    for (const auto &Symbol : Symbols) {
      LUTHIER_RETURN_ON_ERROR(LCOSymbolCache.invalidateOnDestruction(*Symbol));
    }
    // Finally, invalidate the LCO
    LUTHIER_RETURN_ON_ERROR(LCOCache.invalidateOnDestruction(LCO));
  }
  return llvm::Error::success();
}

} // namespace hsa
} // namespace luthier
