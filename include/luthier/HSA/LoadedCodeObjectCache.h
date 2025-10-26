//===-- LoadedCodeObjectCache.h ----------------------------------------*-
// C++-*-===//
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
/// This file defines the \c LoadedCodeObjectCache Singleton,
/// which caches the code object of each \c hsa_loaded_code_object_t created
/// by the application.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_CODE_OBJECT_CACHE_H
#define LUTHIER_HSA_CODE_OBJECT_CACHE_H
#include "luthier/Common/Singleton.h"
#include "luthier/HSA/ApiTable.h"
#include "luthier/HSA/LoadedCodeObjectDeviceFunction.h"
#include "luthier/HSA/LoadedCodeObjectExternSymbol.h"
#include "luthier/HSA/LoadedCodeObjectKernel.h"
#include "luthier/HSA/LoadedCodeObjectVariable.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringMap.h>
#include <mutex>

namespace luthier::hsa {

class LoadedCodeObjectCache final : public Singleton<LoadedCodeObjectCache> {

private:
  /// Mutex to protect the cache entries
  mutable std::recursive_mutex CacheMutex;

  const amdgpu::hsamd::MetadataParser &MDParser;

  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiTableSnapshot;

  /// Loader API snapshot
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &VenLoaderSnapshot;

  /// Wrapper installer for installing the cache's event handlers in HSA's
  /// \c ::CoreApiTable
  std::unique_ptr<rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>
      HsaWrapperInstaller;

  /// Info regarding each loaded code object cached
  struct LCOCacheEntry {
    std::unique_ptr<llvm::SmallVector<uint8_t>> CodeObject;
    std::unique_ptr<luthier::object::AMDGCNObjectFile> ParsedELF;
  };

  /// Mapping between every loaded code object and their cached entries
  mutable llvm::DenseMap<hsa_loaded_code_object_t, LCOCacheEntry> LCOCache;

  /// Mapping between the base load address of an LCO and the LCO itself
  mutable llvm::DenseMap<const uint8_t *, hsa_loaded_code_object_t>
      LoadedBaseToLCOMap;

  static decltype(hsa_executable_load_agent_code_object)
      *UnderlyingHsaExecutableLoadAgentCodeObjectFn;

  static decltype(hsa_executable_destroy) *UnderlyingHsaExecutableDestroyFn;

  static hsa_status_t hsaExecutableLoadAgentCodeObjectWrapper(
      hsa_executable_t Executable, hsa_agent_t Agent,
      hsa_code_object_reader_t CodeObjectReader, const char *Options,
      hsa_loaded_code_object_t *LoadedCodeObject);

  static hsa_status_t hsaExecutableDestroyWrapper(hsa_executable_t Executable);

  llvm::Expected<LCOCacheEntry &>
  getOrCreateLoadedCodeObjectEntry(hsa_loaded_code_object_t LCO) const;

public:
  explicit LoadedCodeObjectCache(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
          &CoreApiTableSnapshot,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &VenLoaderSnapshot,
      const amdgpu::hsamd::MetadataParser &MDParser, llvm::Error &Err);

  /// Queries whether \p LCO is cached or not
  /// \param LCO the \c LoadedCodeObject is being queried
  /// \return true if the \p LCO is cached, false otherwise
  bool isCached(hsa_loaded_code_object_t LCO);

  llvm::Expected<llvm::ArrayRef<uint8_t>>
  getAssociatedCodeObject(hsa_loaded_code_object_t LCO) const;

  llvm::Expected<luthier::object::AMDGCNObjectFile &>
  getAssociatedObjectFile(hsa_loaded_code_object_t LCO) const;

  llvm::Error getLoadedCodeObjectSymbols(
      hsa_loaded_code_object_t LCO,
      llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObjectSymbol>> &Out)
      const;

  llvm::Error getKernelSymbols(
      hsa_loaded_code_object_t LCO,
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const;

  llvm::Error getVariableSymbols(
      hsa_loaded_code_object_t LCO,
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const;

  llvm::Error getDeviceFunctionSymbols(
      hsa_loaded_code_object_t LCO,
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const;

  llvm::Error getExternalSymbols(
      hsa_loaded_code_object_t LCO,
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const;

  llvm::Expected<std::unique_ptr<LoadedCodeObjectSymbol>>
  getLoadedCodeObjectSymbolByName(hsa_loaded_code_object_t LCO,
                                  llvm::StringRef Name) const;
};

} // namespace luthier::hsa

#endif