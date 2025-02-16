//===-- ObjectCache.hpp - HSA Object Cache --------------------------------===//
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
/// This file defines the \c hsa::ObjectCache class, in charge of caching
/// the object storage of all \c hsa::LoadedCodeObject instances created
/// by the target application and the tooling library.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_OBJECT_CACHE
#define LUTHIER_HSA_OBJECT_CACHE
#include "Executable.hpp"
#include "common/IObjectCache.hpp"
#include "common/Singleton.hpp"

namespace luthier::hsa {

class ObjectCache final : public IObjectCache, Singleton<ObjectCache> {
private:
  //===--------------------------------------------------------------------===//
  // Object Cache Fields
  //===--------------------------------------------------------------------===//
  /// A typedef to represent a byte range in the Object cache
  /// This is done for more efficient hashing: Even though
  /// \c llvm::ArrayRef<uint8_t> conceptually does the same thing as this
  /// pair, its hashing function will call \c hash_combine_range on the entire
  /// contents of the underlying \c llvm::ArrayRef which is avoided if
  /// instead we store this pair inside a map instead
  typedef std::pair<const uint8_t *, size_t> byte_range_t;

  /// Mutex to protect the cache
  /// TODO: Investigate replacing this mutex with a shared mutex, following
  /// similar design used by rocprofiler-sdk (@bwelton)
  mutable std::mutex CacheMutex;

  /// A mapping between the hash value of the object file buffer and
  /// the \c std::vector that owns the buffer
  llvm::DenseMap<unsigned, std::unique_ptr<llvm::SmallVector<uint8_t, 0>>>
      ObjectBufferStorage;

  /// A mapping between the hash value of the object file buffer and
  /// the number of uses in the \c HsaObjectFiles map
  llvm::DenseMap<unsigned, int> NumUsesOfObjectBuffer;

  /// A mapping between the byte range of an object buffer storage and the
  /// corresponding \c llvm::object::ELFObjectFileBase
  llvm::DenseMap<byte_range_t, std::unique_ptr<llvm::object::ELFObjectFileBase>>
      ObjectFileStorage;

  /// A convenience mapping between hsa loaded code objects and their parsed
  /// \c llvm::object::ELFObjectFileBase
  /// This map must be updated every time an LCO is created or every time
  /// an executable is destroyed
  llvm::DenseMap<hsa::LoadedCodeObject, const llvm::object::ELFObjectFileBase *>
      HsaObjectFiles;

  //===--------------------------------------------------------------------===//
  // Invalidation Callback Fields
  //===--------------------------------------------------------------------===//
  /// Mutex to protect the callback functions
  /// TODO: Investigate replacing this mutex with a shared mutex, following
  /// similar design used by rocprofiler-sdk (@bwelton)
  mutable std::shared_mutex CallbackMutex;

  std::vector<object_invalidation_callback_t> InvalidationCallbacks;

  void
  addInvalidationCallback(const object_invalidation_callback_t &CB) override;

  [[nodiscard]] bool
  isObjectFileCached(const llvm::object::ObjectFile &ObjFile) const override;

  /// An event handler invoked right after
  /// \c hsa_executable_load_agent_code_object is called either by the target
  /// application or by the tooling library
  /// Iterates over the loaded code objects of \p Exec and caches
  /// the storage object of the newly created \p LCO with the
  /// \c hsa::ObjectCache
  /// \param Exec the newly created \c hsa::LoadedCodeObject
  /// \return an \c llvm::Error indicating the success or failure of the
  /// operation
  llvm::Error registerNewlyLoadedAgentCodeObject(const hsa::Executable &Exec);

  /// An event handler invoked right before
  /// \c hsa_executable_destroy is called either by the target
  /// application or by the tooling library
  /// Invalidates the association of \p LCO with its cached storage object
  /// \param Exec the newly created \c hsa::LoadedCodeObject
  /// \return an \c llvm::Error indicating the success or failure of the
  /// operation
  llvm::Error
  unregisterExecutableCodeObjectsBeforeDestruction(const hsa::Executable &Exec);

  llvm::Expected<const llvm::object::ELFObjectFileBase &>
  getHsaLoadedCodeObjectELF(const hsa::LoadedCodeObject &LCO);
};

} // namespace luthier::hsa

#endif