//===-- LoadedCodeObjectCache.h ---------------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file LoadedCodeObjectCache.h
/// Defines a cache of parsed storage ELFs for each \c hsa_loaded_code_object_t
/// the application has seen.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_CODE_OBJECT_CACHE_H
#define LUTHIER_HSA_TOOLING_CODE_OBJECT_CACHE_H
#include "luthier/Common/Singleton.h"
#include "luthier/HSA/ApiTable.h"
#include "luthier/HSA/hsa.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CrashRecoveryContext.h>
#include <mutex>

namespace luthier {

class LoadedCodeObjectCache {
public:
  /// Info regarding each cached loaded code object.
  struct LCOCacheEntry {
    std::unique_ptr<llvm::SmallVector<uint8_t>> CodeObject;
    std::unique_ptr<object::AMDGCNObjectFile> ParsedELF;
  };

protected:
  /// Mutex protecting the cache entries
  mutable std::recursive_mutex CacheMutex;

  /// HSA Core API snapshot
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiTableSnapshot;

  /// Loader API snapshot
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &VenLoaderSnapshot;

  /// Mapping between every loaded code object and its cached entry
  mutable llvm::DenseMap<hsa_loaded_code_object_t, LCOCacheEntry> LCOCache;

  /// Mapping between the base load address of an LCO and the LCO itself
  mutable llvm::DenseMap<const uint8_t *, hsa_loaded_code_object_t>
      LoadedBaseToLCOMap;

  llvm::Expected<LCOCacheEntry &>
  getOrCreateLoadedCodeObjectEntry(hsa_loaded_code_object_t LCO) const;

public:
  LoadedCodeObjectCache(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &VenLoader);

  /// Queries whether \p LCO is cached or not
  bool isCached(hsa_loaded_code_object_t LCO) const;

  llvm::Expected<llvm::ArrayRef<uint8_t>>
  getAssociatedCodeObject(hsa_loaded_code_object_t LCO) const;

  llvm::Expected<object::AMDGCNObjectFile &>
  getAssociatedObjectFile(hsa_loaded_code_object_t LCO) const;
};

/// \brief CRTP trait that installs the HSA executable load/destroy wrappers
/// needed for the \c LoadedCodeObjectCache that keeps it coherent across the
/// application's executable lifetime
template <typename Derived>
class LoadedCodeObjectCacheTrait : public LoadedCodeObjectCache {
private:
  /// Per-Derived pointers to the wrapped HSA functions
  inline static decltype(hsa_executable_load_agent_code_object)
      *UnderlyingHsaExecutableLoadAgentCodeObjectFn{};

  inline static decltype(hsa_executable_destroy)
      *UnderlyingHsaExecutableDestroyFn{};

  std::unique_ptr<rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>
      HsaWrapperInstaller;

  static hsa_status_t hsaExecutableLoadAgentCodeObjectWrapper(
      hsa_executable_t Executable, hsa_agent_t Agent,
      hsa_code_object_reader_t CodeObjectReader, const char *Options,
      hsa_loaded_code_object_t *LoadedCodeObject) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        UnderlyingHsaExecutableLoadAgentCodeObjectFn != nullptr,
        "Underlying hsa_executable_load_agent_code_object is nullptr"));

    hsa_loaded_code_object_t LCO;
    hsa_status_t Out = UnderlyingHsaExecutableLoadAgentCodeObjectFn(
        Executable, Agent, CodeObjectReader, Options, &LCO);

    if (LoadedCodeObject != nullptr)
      *LoadedCodeObject = LCO;

    /// Short-circuit if the underlying call failed
    if (Out != HSA_STATUS_SUCCESS)
      return Out;

    (void)Singleton<Derived>::withInstance([&](Derived &Self) {
      auto &COC = static_cast<LoadedCodeObjectCacheTrait<Derived> &>(Self);

      llvm::ArrayRef<uint8_t> StorageMemory;
      LUTHIER_REPORT_FATAL_ON_ERROR(hsa::loadedCodeObjectGetStorageMemory(
                                        COC.VenLoaderSnapshot.getTable(), LCO)
                                        .moveInto(StorageMemory));

      /// Guard the copy against the host's storage buffer being unmapped
      /// underneath us — see \c LoadedCodeObjectCache.cpp for the same pattern
      /// On signal we skip the cache insert and let HSA's own load succeed; a
      /// later lookup via \c LoadedCodeObjectCache will surface the error to
      /// the caller
      auto StorageCopy = std::make_unique<llvm::SmallVector<uint8_t>>();
      StorageCopy->resize(StorageMemory.size());
      llvm::CrashRecoveryContext CRC;
      bool CopyOk = CRC.RunSafely([&] {
        std::memcpy(StorageCopy->data(), StorageMemory.data(),
                    StorageMemory.size());
      });
      if (!CopyOk) {
        llvm::errs() << llvm::formatv(
            "luthier: storage memory for loaded code object {0:x} is "
            "unreadable; skipping cache insert.\n",
            LCO.handle);
        return;
      }

      auto ParsedElfOrErr =
          object::AMDGCNObjectFile::createAMDGCNObjectFile(*StorageCopy);
      LUTHIER_REPORT_FATAL_ON_ERROR(ParsedElfOrErr.takeError());

      llvm::ArrayRef<uint8_t> LoadedMemory;
      LUTHIER_REPORT_FATAL_ON_ERROR(hsa::loadedCodeObjectGetLoadedMemory(
                                        COC.VenLoaderSnapshot.getTable(), LCO)
                                        .moveInto(LoadedMemory));

      std::lock_guard Lock(COC.CacheMutex);
      COC.LCOCache.insert({LCO, LCOCacheEntry{std::move(StorageCopy),
                                              std::move(*ParsedElfOrErr)}});
      COC.LoadedBaseToLCOMap.insert({LoadedMemory.data(), LCO});
    });

    return Out;
  }

  static hsa_status_t hsaExecutableDestroyWrapper(hsa_executable_t Executable) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        UnderlyingHsaExecutableDestroyFn != nullptr,
        "Underlying hsa_executable_destroy is nullptr"));

    Singleton<Derived>::withInstance([&](Derived &Self) {
      auto &COC = static_cast<LoadedCodeObjectCacheTrait<Derived> &>(Self);
      llvm::SmallVector<hsa_loaded_code_object_t, 2> LCOs;
      LUTHIER_REPORT_FATAL_ON_ERROR(hsa::executableGetLoadedCodeObjects(
          COC.VenLoaderSnapshot.getTable(), Executable, LCOs));
      std::lock_guard Lock(COC.CacheMutex);
      for (hsa_loaded_code_object_t LCO : LCOs) {
        COC.LCOCache.erase(LCO);

        llvm::ArrayRef<uint8_t> LoadedMemory;
        LUTHIER_REPORT_FATAL_ON_ERROR(hsa::loadedCodeObjectGetLoadedMemory(
                                          COC.VenLoaderSnapshot.getTable(), LCO)
                                          .moveInto(LoadedMemory));
        COC.LoadedBaseToLCOMap.erase(LoadedMemory.data());
      }
    });

    return UnderlyingHsaExecutableDestroyFn(Executable);
  }

public:
  LoadedCodeObjectCacheTrait(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &VenLoader,
      llvm::Error &Err)
      : LoadedCodeObjectCache(CoreApi, VenLoader) {
    llvm::ErrorAsOutParameter EAO(Err);
    HsaWrapperInstaller = std::make_unique<
        rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>(
        Err,
        std::make_tuple(
            &::CoreApiTable::hsa_executable_load_agent_code_object_fn,
            std::ref(UnderlyingHsaExecutableLoadAgentCodeObjectFn),
            hsaExecutableLoadAgentCodeObjectWrapper),
        std::make_tuple(&::CoreApiTable::hsa_executable_destroy_fn,
                        std::ref(UnderlyingHsaExecutableDestroyFn),
                        hsaExecutableDestroyWrapper));
  }

  ~LoadedCodeObjectCacheTrait() = default;

  LoadedCodeObjectCacheTrait(const LoadedCodeObjectCacheTrait &) = delete;
  LoadedCodeObjectCacheTrait &
  operator=(const LoadedCodeObjectCacheTrait &) = delete;
};

} // namespace luthier

#endif
