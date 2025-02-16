//===-- ObjectCache.cpp ---------------------------------------------------===//
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
/// This file implements the \c hsa::ObjectCache class.
//===----------------------------------------------------------------------===//
#include "hsa/ObjectCache.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include <llvm/ADT/StringExtras.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/MemoryBuffer.h>

namespace luthier::hsa {

void ObjectCache::addInvalidationCallback(
    const object_invalidation_callback_t &CB) {
  std::unique_lock Lock(CallbackMutex);
  InvalidationCallbacks.push_back(CB);
}

bool ObjectCache::isObjectFileCached(
    const llvm::object::ObjectFile &ObjFile) const {
  // Check if the underlying buffer of the object file is present in the
  // ObjectFileStorage map
  llvm::MemoryBufferRef ObjFileBuffer = ObjFile.getMemoryBufferRef();
  return ObjectFileStorage.contains(
      {reinterpret_cast<const uint8_t *>(ObjFileBuffer.getBufferStart()),
       ObjFileBuffer.getBufferSize()});
}

llvm::Error
ObjectCache::registerNewlyLoadedAgentCodeObject(const hsa::Executable &Exec) {
  // Hold the lock for the entirety of the operation
  std::lock_guard Lock(CacheMutex);
  llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(Exec.getLoadedCodeObjects(LCOs));
  for (const auto &LCO : LCOs) {
    if (!HsaObjectFiles.contains(LCO)) {
      // Copy the storage ELF of the LCO to a std::string
      llvm::Expected<llvm::ArrayRef<uint8_t>> LCOObjectStorageOrErr =
          LCO.getStorageMemory();
      LUTHIER_RETURN_ON_ERROR(LCOObjectStorageOrErr.takeError());
      // Calculate the hash of the ELF
      unsigned ObjectHash =
          llvm::DenseMapInfo<llvm::ArrayRef<uint8_t>>::getHashValue(
              *LCOObjectStorageOrErr);

      if (ObjectBufferStorage.contains(ObjectHash)) {
        // If there's already a cached copy of this ELF, update the maps
        // without creating a new copy
        LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
            NumUsesOfObjectBuffer.contains(ObjectHash),
            "Number of uses of the object buffer was not recorded correctly."));

        const llvm::SmallVectorImpl<uint8_t> &ObjectStorage =
            *ObjectBufferStorage.at(ObjectHash);
        NumUsesOfObjectBuffer[ObjectHash]++;

        LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
            ObjectFileStorage.contains(
                {ObjectStorage.data(), ObjectFileStorage.size()}),
            "Failed to find the parsed ELFObjectFileBase of the already cached "
            "object "
            "buffer."));

        HsaObjectFiles.insert(
            {LCO, ObjectFileStorage
                      .at({ObjectStorage.data(), ObjectFileStorage.size()})
                      .get()});
      } else {
        // Make a copy of the storage ELF and insert it in the maps
        auto ObjectStorageCopy =
            std::make_unique<llvm::SmallVector<uint8_t, 0>>(
                *LCOObjectStorageOrErr);

        llvm::Expected<std::unique_ptr<llvm::object::ELFObjectFileBase>>
            ParsedObjOrErr = parseELFObjectFile(*ObjectStorageCopy);
        LUTHIER_RETURN_ON_ERROR(ParsedObjOrErr.takeError());

        HsaObjectFiles.insert({LCO, ParsedObjOrErr->get()});

        ObjectFileStorage.insert(
            {byte_range_t{ObjectStorageCopy->data(), ObjectStorageCopy->size()},
             std::move(*ParsedObjOrErr)});

        ObjectBufferStorage.insert({ObjectHash, std::move(ObjectStorageCopy)});

        NumUsesOfObjectBuffer[ObjectHash]++;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error ObjectCache::unregisterExecutableCodeObjectsBeforeDestruction(
    const hsa::Executable &Exec) {
  // Hold the lock for the entirety of the operation
  std::lock_guard Lock(CacheMutex);
  llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(Exec.getLoadedCodeObjects(LCOs));
  for (const auto &LCO : LCOs) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        HsaObjectFiles.contains(LCO),
        "Loaded code object {0:x} was not cached.", LCO.hsaHandle()));
    const llvm::object::ELFObjectFileBase *LCOObjFile = HsaObjectFiles.at(LCO);
    // Calculate the hash of the obj buffer
    llvm::ArrayRef<uint8_t> ObjBuffer = llvm::arrayRefFromStringRef<uint8_t>(
        LCOObjFile->getMemoryBufferRef().getBuffer());
    unsigned ObjHash =
        llvm::DenseMapInfo<llvm::ArrayRef<uint8_t>>::getHashValue(ObjBuffer);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        NumUsesOfObjectBuffer.contains(ObjHash),
        "Failed to find the number of uses of the cached object file."));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        ObjectFileStorage.contains({ObjBuffer.data(), ObjBuffer.size()}),
        "Failed to find the storage of the cached object file."));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        ObjectBufferStorage.contains(ObjHash),
        "Failed to find the storage buffer of the cached object file."));
    // Remove the obj file from the HsaObjectFile map
    HsaObjectFiles.erase(LCO);
    // Decrement the number of uses
    NumUsesOfObjectBuffer[ObjHash]--;
    // If number of uses is now zero, remove the storage ELF all together
    if (NumUsesOfObjectBuffer[ObjHash] <= 0) {
      {
        // Perform the invalidation callbacks
        std::shared_lock CBLock(CallbackMutex);
        for (const auto &CB : InvalidationCallbacks) {
          CB(*LCOObjFile);
        }
        // Remove the file and buffer storages of the object file
        ObjectBufferStorage.erase(ObjHash);
        ObjectFileStorage.erase({ObjBuffer.data(), ObjBuffer.size()});
        NumUsesOfObjectBuffer.erase(ObjHash);
      }
    }
  }
  return llvm::Error::success();
}

llvm::Expected<const llvm::object::ELFObjectFileBase &>
ObjectCache::getHsaLoadedCodeObjectELF(const hsa::LoadedCodeObject &LCO) {
  std::lock_guard Lock(CacheMutex);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      HsaObjectFiles.contains(LCO),
      "Failed to find the cached ELF associated with LCO {0:x}.",
      LCO.hsaHandle()));
  return *HsaObjectFiles.at(LCO);
}

} // namespace luthier::hsa