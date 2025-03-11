//===-- ExecutableBackedObjectsCache.hpp ----------------------------------===//
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
/// This file defines the \c ExecutableBackedObjectsCache Singleton,
/// which caches information regarding objects backed by a
/// <tt>hsa::Executable</tt>. Reason behind this cache can range from being
/// required for proper functioning of the object to simply faster queries.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_EXECUTABLE_BACKED_OBJECTS_CACHE_HPP
#define LUTHIER_HSA_EXECUTABLE_BACKED_OBJECTS_CACHE_HPP
#include "common/ObjectUtils.hpp"
#include "common/Singleton.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringMap.h>
#include <luthier/hsa/DenseMapInfo.h>
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>
#include <luthier/hsa/LoadedCodeObjectExternSymbol.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>
#include <luthier/hsa/LoadedCodeObjectVariable.h>
#include <mutex>

namespace luthier::hsa {

class Executable;

class LoadedCodeObject;

/// \brief Singleton in charge of caching and invalidating information
/// regarding objects that are backed by (dependent on) an \c hsa::Executable
/// \details Besides retaining the cached state and providing thread-safety
/// over the cache, it implements event handlers that will be invoked by
/// the HSA internal callback of Luthier to notify both itself and
/// its dependent classes about changes to the state of the cache
/// \sa \c luthier::Controller
class ExecutableBackedObjectsCache
    : public Singleton<ExecutableBackedObjectsCache> {

private:
  std::recursive_mutex CacheMutex;

public:
  /// \brief in charge of caching information regarding the \c LoadedCodeObject
  /// class
  /// \details this class should be considered an extension of
  /// the \c LoadedCodeObject class, which is why \c LoadedCodeObject is a
  /// friend of this class. Besides \c ExecutableBackedObjectsCache and
  /// \c LoadedCodeObject no other class has direct access to the internals of
  /// this cache
  class LoadedCodeObjectCache {
    friend LoadedCodeObject;
    friend ExecutableBackedObjectsCache;

  private:
    /// A reference to the Executable-backed cache
    std::recursive_mutex &ExecutableCacheMutex;

    explicit LoadedCodeObjectCache(std::recursive_mutex &Mutex)
        : ExecutableCacheMutex(Mutex) {};

    struct LoadedCodeObjectCacheEntry {
      /// Object file used to create the loaded code object
      std::unique_ptr<llvm::SmallVector<uint8_t>> CodeObject;
      /// Parsed ELF representation of \c CodeObject
      std::unique_ptr<luthier::AMDGCNObjectFile> ElfObjectFile;
      ////      /// Parsed metadata of the loaded code object
      ////      hsa::md::Metadata Metadata;
      ////      /// Mapping between names of the loaded code object kernels and
      ////      /// their symbols
      ////      llvm::StringMap<const LoadedCodeObjectKernel *> KernelSymbols;
      ////
      ////      /// A mapping between the cached
      ///<tt>hsa_loaded_code_object_t</tt>s and /      /// their
      ///<tt>LoadedCodeObjectSymbol</tt>s of device function type
      ////
      ////      llvm::StringMap<const LoadedCodeObjectDeviceFunction *>
      ////          DeviceFuncSymbolsOfLCOs{};
      ////
      ////      /// A mapping between the cached
      ///<tt>hsa_loaded_code_object_t</tt>s and /      /// their
      ///<tt>LoadedCodeObjectSymbol</tt>s of variable type /
      ///llvm::DenseMap<hsa_loaded_code_object_t, / llvm::StringMap<const
      ///LoadedCodeObjectVariable *>> /          VariableSymbolsOfLCOs{};
      ////
      ////      /// A mapping between the cached
      ///<tt>hsa_loaded_code_object_t</tt>s and /      /// their external
      ///<tt>LoadedCodeObjectSymbol</tt>s /
      ///llvm::DenseMap<hsa_loaded_code_object_t, / llvm::StringMap<const
      ///LoadedCodeObjectExternSymbol *>> /          ExternSymbolsOfLCOs{};
    };

    llvm::DenseMap<hsa_loaded_code_object_t, LoadedCodeObjectCacheEntry>
        CachedLCOs;

    /// Queries whether \p LCO is cached or not
    /// \param LCO the \c LoadedCodeObject is being queried
    /// \return true if the \p LCO is cached, false otherwise
    bool isCached(const hsa::LoadedCodeObject &LCO);

    /// A mapping between the cached <tt>hsa_loaded_code_object_t</tt>s and
    /// their parsed \c luthier::AMDGCNObjectFile
    llvm::DenseMap<hsa_loaded_code_object_t,
                   std::unique_ptr<luthier::AMDGCNObjectFile>>
        StorageELFOfLCOs{};

    /// A mapping between the cached <tt>hsa_loaded_code_object_t</tt>s and
    /// their
    /// \c hsa_isa_t
    llvm::DenseMap<hsa_loaded_code_object_t, hsa_isa_t> ISAOfLCOs{};

    /// A mapping between the cached <tt>hsa_loaded_code_object_t</tt>s and
    /// their parsed metadata
    llvm::DenseMap<hsa_loaded_code_object_t, hsa::md::Metadata>
        MetadataOfLCOs{};

    /// A mapping between the cached <tt>hsa_loaded_code_object_t</tt>s and
    /// their <tt>LoadedCodeObjectSymbol</tt>s of kernel type
    llvm::DenseMap<hsa_loaded_code_object_t,
                   llvm::StringMap<const LoadedCodeObjectKernel *>>
        KernelSymbolsOfLCOs{};

    /// A mapping between the cached <tt>hsa_loaded_code_object_t</tt>s and
    /// their <tt>LoadedCodeObjectSymbol</tt>s of device function type
    llvm::DenseMap<hsa_loaded_code_object_t,
                   llvm::StringMap<const LoadedCodeObjectDeviceFunction *>>
        DeviceFuncSymbolsOfLCOs{};

    /// A mapping between the cached <tt>hsa_loaded_code_object_t</tt>s and
    /// their <tt>LoadedCodeObjectSymbol</tt>s of variable type
    llvm::DenseMap<hsa_loaded_code_object_t,
                   llvm::StringMap<const LoadedCodeObjectVariable *>>
        VariableSymbolsOfLCOs{};

    /// A mapping between the cached <tt>hsa_loaded_code_object_t</tt>s and
    /// their external <tt>LoadedCodeObjectSymbol</tt>s
    llvm::DenseMap<hsa_loaded_code_object_t,
                   llvm::StringMap<const LoadedCodeObjectExternSymbol *>>
        ExternSymbolsOfLCOs{};

    llvm::Error cacheOnCreation(const hsa::LoadedCodeObject &LCO);

    llvm::Error invalidateOnDestruction(const hsa::LoadedCodeObject &LCO);
  };

  /// \brief in charge of caching information regarding the
  /// \c LoadedCodeObjectSymbol class
  /// \details this class should be considered an extension of
  /// the \c LoadedCodeObjectSymbol class, which is why \c
  /// LoadedCodeObjectSymbol is a friend of this class. Besides \c
  /// ExecutableBackedObjectsCache and
  /// \c LoadedCodeObjectSymbol no other class has direct access to the
  /// internals of this cache
  class LoadedCodeObjectSymbolCache {
    friend LoadedCodeObjectSymbol;
    friend ExecutableBackedObjectsCache;

  private:
    std::recursive_mutex &CacheMutex;

    /// A mapping between all <tt>hsa_executable_symbol_t</tt>s and their
    /// equivalent \c LoadedCodeObjectSymbol
    llvm::DenseMap<hsa_executable_symbol_t, const LoadedCodeObjectSymbol *>
        ExecToLCOSymbolMap{};

    /// A set of <tt>LoadedCodeObjectSymbol</tt>s that have been cached
    llvm::DenseSet<const LoadedCodeObjectSymbol *> CachedLCOSymbols{};

    /// A set of <tt>LoadedCodeObjectSymbol</tt>s loaded on the device
    llvm::DenseSet<const LoadedCodeObjectSymbol *> LoadedLCOSymbols{};

    /// A mapping between a set of symbol loaded addresses and the
    /// \c LoadedCodeObjectSymbol it is loaded there
    llvm::DenseMap<luthier::address_t, const LoadedCodeObjectSymbol *>
        LoadedAddressToSymbolMap{};

    explicit LoadedCodeObjectSymbolCache(std::recursive_mutex &Mutex)
        : CacheMutex(Mutex) {};

    bool isCached(const hsa::LoadedCodeObjectSymbol &Symbol);

    llvm::Error cacheOnCreation(const hsa::LoadedCodeObjectSymbol &Symbol);

    llvm::Error cacheOnDeviceLoad(const LoadedCodeObjectSymbol &Symbol);

    llvm::Error
    invalidateOnDestruction(const hsa::LoadedCodeObjectSymbol &Symbol);
  };

private:
  LoadedCodeObjectCache LCOCache{CacheMutex};
  LoadedCodeObjectSymbolCache LCOSymbolCache{CacheMutex};

public:
  LoadedCodeObjectCache &getLoadedCodeObjectCache() { return LCOCache; }

  LoadedCodeObjectSymbolCache &getLoadedCodeObjectSymbolCache() {
    return LCOSymbolCache;
  }

  /// An event handler that caches all objects created after a code object
  /// is loaded into \p Exec and creates a \c hsa::LoadedCodeObject
  /// \param Exec the executable with a new \c hsa::LoadedCodeObject
  /// \return \c llvm::ErrorSuccess on success, or an \c llvm::Error describing
  /// the issue encountered in the process
  llvm::Error cacheExecutableOnLoadedCodeObjectCreation(const Executable &Exec);

  /// An event handler in charge of recording information about \p Exec after
  /// it gets frozen by the HSA runtime
  /// \param Exec the \c Executable that was just frozen
  /// \return \c llvm::ErrorSuccess on success, or an \c llvm::Error describing
  /// the issue encountered in the process
  llvm::Error cacheExecutableOnExecutableFreeze(const Executable &Exec);

  /// An event handler in charge of invalidating information about \p Exec
  /// before it is destroyed by the HSA runtime
  /// \param Exec the \c Executable that is about to be destroyed
  /// \return \c llvm::ErrorSuccess on success, or an \c llvm::Error describing
  /// the issue encountered in the process
  llvm::Error invalidateExecutableOnExecutableDestroy(const Executable &Exec);
};

} // namespace luthier::hsa

#endif