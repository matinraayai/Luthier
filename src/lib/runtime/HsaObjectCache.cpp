//===-- HsaCodeObjectCache.cpp --------------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Implements the \c HsaCodeObjectCache singleton.
//===----------------------------------------------------------------------===//
#include <luthier/runtime/HsaObjectCache.h>

namespace luthier {

llvm::Expected<llvm::ArrayRef<uint8_t>>
HsaCodeObjectCache::getAssociatedCodeObject(
    hsa_loaded_code_object_t LCO) const {
  std::lock_guard Lock(Mutex);
  /// Return the associated object we have it already
  if (auto It = LCOToCodeObjectMap.find(LCO); It != LCOToCodeObjectMap.end())
    return *It->second.ObjectStorage;
  /// Otherwise we have to cache the LCO's object by reading its storage memory
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(HsaVenAmdLoaderLCOGetInfo != nullptr,
                          "The hsa_ven_amd_loader_loaded_code_object_get_info "
                          "of ToolExecutableLoader instance is nullptr."));
  llvm::ArrayRef<uint8_t> LCOStorageMemory;
  LUTHIER_RETURN_ON_ERROR(
      hsa::getLCOStorageMemory(LCO, *HsaVenAmdLoaderLCOGetInfo)
          .moveInto(LCOStorageMemory));
  /// Calculate the hash of the LCO storage
  size_t StorageHash =
      llvm::DenseMapInfo<llvm::ArrayRef<uint8_t>>::getHashValue(
          LCOStorageMemory);
  /// Check if we already have a copy of the object in the cache; If so,
  /// increment its use count. If not, we need to make a copy of it
  auto ObjIter = HashToCodeObjectAndUsageMap.find(StorageHash);
  if (ObjIter != HashToCodeObjectAndUsageMap.end()) {
    auto ObjectStorage = std::make_unique<ObjectStorageEntry>(
        std::make_unique<std::vector<uint8_t>>(LCOStorageMemory), 1,
        StorageHash);
    (void)HashToCodeObjectAndUsageMap.insert(
        {StorageHash, std::move(ObjectStorage)});
  } else {
    ++ObjIter->getSecond()->UseCount;
  }

  LCOToCodeObjectMap.insert({LCO, *ObjIter->getSecond()});
  return *ObjIter->getSecond()->ObjectStorage;
}
} // namespace luthier
