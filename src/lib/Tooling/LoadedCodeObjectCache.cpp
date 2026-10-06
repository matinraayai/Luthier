//===-- LoadedCodeObjectCache.cpp ----------------------------------------===//
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
/// \file
/// Implements the plain-class LoadedCodeObjectCache. HSA API-table wrappers
/// for cache invalidation are installed by LoadedCodeObjectCacheTrait, not
/// here.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/LoadedCodeObjectCache.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include <cstring>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/CrashRecoveryContext.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-hsa-exec-cache"

namespace luthier {

LoadedCodeObjectCache::LoadedCodeObjectCache(
    const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
    const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
        &VenLoader)
    : CoreApiTableSnapshot(CoreApi), VenLoaderSnapshot(VenLoader) {
  /// Idempotent process-wide install of SIGSEGV/SIGBUS handlers, so that
  /// faults inside a \c CrashRecoveryContext::RunSafely region land as a
  /// failed \c RunSafely instead of terminating the host process. Used when
  /// reading host-side LCO storage memory that may have been unmapped by the
  /// host application after registration with HSA.
  llvm::CrashRecoveryContext::Enable();
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
LoadedCodeObjectCache::getAssociatedCodeObject(
    hsa_loaded_code_object_t LCO) const {
  llvm::Expected<LCOCacheEntry &> EntryOrErr =
      getOrCreateLoadedCodeObjectEntry(LCO);
  LUTHIER_RETURN_ON_ERROR(EntryOrErr.takeError());
  return *EntryOrErr->CodeObject;
}

llvm::Expected<luthier::object::AMDGCNObjectFile &>
LoadedCodeObjectCache::getAssociatedObjectFile(
    hsa_loaded_code_object_t LCO) const {
  llvm::Expected<LCOCacheEntry &> EntryOrErr =
      getOrCreateLoadedCodeObjectEntry(LCO);
  LUTHIER_RETURN_ON_ERROR(EntryOrErr.takeError());
  return *EntryOrErr->ParsedELF;
}

bool LoadedCodeObjectCache::isCached(hsa_loaded_code_object_t LCO) const {
  std::lock_guard Lock(CacheMutex);
  return LCOCache.contains(LCO);
}

llvm::Expected<LoadedCodeObjectCache::LCOCacheEntry &>
LoadedCodeObjectCache::getOrCreateLoadedCodeObjectEntry(
    hsa_loaded_code_object_t LCO) const {
  std::lock_guard Lock(CacheMutex);
  auto LCOEntry = LCOCache.find(LCO);
  /// If not already cached, try querying the storage ELF from the loader API.
  if (LCOEntry == LCOCache.end()) {
    llvm::ArrayRef<uint8_t> LCOStorageMemory;
    LUTHIER_RETURN_ON_ERROR(
        hsa::loadedCodeObjectGetStorageMemory(VenLoaderSnapshot.getTable(), LCO)
            .moveInto(LCOStorageMemory));

    /// The host application may have unmapped or freed the buffer that backs
    /// \c LCOStorageMemory at any point after it registered the code object
    /// with HSA. \c CrashRecoveryContext catches the resulting SIGSEGV/SIGBUS
    /// and returns false from \c RunSafely instead of crashing the process
    /// The destination \c SmallVector is allocated outside the protected
    /// region so its lifetime survives a \c siglongjmp out of the handler
    auto StorageCopy = std::make_unique<llvm::SmallVector<uint8_t>>();
    StorageCopy->resize(LCOStorageMemory.size());
    llvm::CrashRecoveryContext CRC;
    bool Ok = CRC.RunSafely([&] {
      std::memcpy(StorageCopy->data(), LCOStorageMemory.data(),
                  LCOStorageMemory.size());
    });
    if (!Ok)
      return LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
          "Storage memory for loaded code object {0:x} is no longer "
          "accessible from the host (segfault while copying).",
          LCO.handle));

    auto ParsedElfOrErr =
        object::AMDGCNObjectFile::createAMDGCNObjectFile(*StorageCopy);
    LUTHIER_RETURN_ON_ERROR(ParsedElfOrErr.takeError());

    LCOEntry = LCOCache
                   .insert({LCO, LCOCacheEntry{std::move(StorageCopy),
                                               std::move(*ParsedElfOrErr)}})
                   .first;
  }
  return LCOEntry->second;
}

} // namespace luthier::hsa
