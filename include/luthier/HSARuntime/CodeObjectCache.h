//===-- CodeObjectCache.h ---------------------------------------*- C++ -*-===//
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
/// Describes the \c CodeObjectCache singleton, in charge of caching a copy
/// of the ELF code objects of \c hsa_loaded_code_object_t right after they
/// are created by the HSA runtime.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_RUNTIME_CODE_OBJECT_CACHE_H
#define LUTHIER_HSA_RUNTIME_CODE_OBJECT_CACHE_H
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/ADT/DenseMap.h>
#include <luthier/Common/Singleton.h>
#include <luthier/HSA/Executable.h>
#include <luthier/HSA/LoadedCodeObject.h>
#include <luthier/Rocprofiler/HIPApiTable.h>
#include <luthier/Rocprofiler/HSAApiTable.h>

namespace luthier::hsa {

/// \brief Interface for a component in charge of caching HSA code objects
/// once they get loaded into HSA and invalidate them once they are unloaded
class CodeObjectCache {
protected:
  const hsa::ApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot;

  const hsa::ExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &LoaderApiSnapshot;

  std::mutex Mutex;

  /// We store the object and its use count in a unique pointer to this struct.
  /// This is so that the \c LCOToCodeObjectMap can reference it directly
  /// without storing denese map iterators that can be invalidated at any given
  /// time
  struct ObjectStorageEntry {
    /// Where the object is actually stored; We use a unique pointer to store
    /// the vector here to avoid any issues with the move constructor of
    /// the entry struct (even if we're only storing unique pointer of the
    /// entry struct anyway)
    std::unique_ptr<std::vector<uint8_t>> ObjectStorage{nullptr};
    /// Number of LCOs that have this same object; This is so that in multi-gpu
    /// systems we don't store redundant objects that were loaded on multiple
    /// devices
    size_t UseCount{};
    /// Hash of the object; Besides being the key in \c
    /// HashToCodeObjectAndUsageMap we also store it here for quick invalidation
    /// of the entry (i.e. when the \c UseCount of this object reaches zero)
    size_t Hash{};
  };

  /// Mapping between the hash of the code object to storage entry
  mutable llvm::DenseMap<size_t, std::unique_ptr<ObjectStorageEntry>>
      HashToCodeObjectAndUsageMap{};

  /// Mapping between the \c hsa_loaded_code_object_t and its associated
  /// cached object entry
  mutable llvm::DenseMap<hsa_loaded_code_object_t, ObjectStorageEntry &>
      LCOToCodeObjectMap{};

  CodeObjectCache(const hsa::ApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
                  const hsa::ExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
                      &LoaderApiSnapshot)
      : CoreApiSnapshot(CoreApiSnapshot),
        LoaderApiSnapshot(LoaderApiSnapshot) {};

  virtual ~CodeObjectCache() = default;

public:
  /// Queries and returns the associated object with the <tt>LCO</tt>. If
  /// for whatever reason, the object is not found in the cache, attempts to
  /// obtain and cache it by reading the storage memory of the \p LCO via
  /// the AMD Loader API before giving up
  /// \return Expects the code object associated with the <tt>LCO</tt>; If
  /// fails to find the associated function, an \c llvm::Error is returned
  /// instead
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getAssociatedCodeObject(hsa_loaded_code_object_t LCO) const;
};

template <size_t Idx>
class ROCPROFILER_HIDDEN_API CodeObjectCacheInstance
    : public CodeObjectCache,
      Singleton<CodeObjectCacheInstance<Idx>> {
private:
  const std::unique_ptr<ApiTableWrapperInstaller> HsaApiTableInterceptor{
      nullptr};

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_load_agent_code_object)
      *UnderlyingHsaExecutableLoadAgentCodeObjectFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_destroy)
      *UnderlyingHsaExecutableDestroyFn;

  static ROCPROFILER_HIDDEN_API hsa_status_t
  hsaExecutableLoadAgentCodeObjectWrapper(
      hsa_executable_t Executable, hsa_agent_t Agent,
      hsa_code_object_reader_t CodeObjectReader, const char *Options,
      hsa_loaded_code_object_t *LoadedCodeObject);

  static ROCPROFILER_HIDDEN_API hsa_status_t
  hsaExecutableDestroyWrapper(hsa_executable_t Executable);

  explicit CodeObjectCacheInstance(
      const hsa::ApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
      const hsa::ExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &LoaderApiSnapshot,
      llvm::Error &Err)
      : CodeObjectCache(CoreApiSnapshot, LoaderApiSnapshot),
        HsaApiTableInterceptor([&]() {
          auto HsaApiTableWrapperInstallerOrErr =
              hsa::ApiTableWrapperInstaller::requestWrapperInstallation(
                  {&::CoreApiTable::hsa_executable_load_agent_code_object_fn,
                   UnderlyingHsaExecutableLoadAgentCodeObjectFn,
                   hsaExecutableLoadAgentCodeObjectWrapper},
                  {&::CoreApiTable::hsa_executable_destroy_fn,
                   UnderlyingHsaExecutableDestroyFn,
                   hsaExecutableDestroyWrapper});
          Err = HsaApiTableWrapperInstallerOrErr.takeError();
          if (Err) {
            Err = std::move(Err);
            return nullptr;
          }
          return *HsaApiTableWrapperInstallerOrErr;
        }()) {};

public:
  static llvm::Expected<std::unique_ptr<CodeObjectCacheInstance>>
  create(const hsa::ApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
         const hsa::ExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
             &LoaderApiSnapshot) {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<CodeObjectCacheInstance>(
        CoreApiSnapshot, LoaderApiSnapshot, Err);
    if (Err)
      return std::move(Err);
    return Out;
  }

  ~CodeObjectCacheInstance() override = default;
};

template <size_t Idx>
hsa_status_t
CodeObjectCacheInstance<Idx>::hsaExecutableLoadAgentCodeObjectWrapper(
    const hsa_executable_t Executable, const hsa_agent_t Agent,
    const hsa_code_object_reader_t CodeObjectReader, const char *Options,
    hsa_loaded_code_object_t *LoadedCodeObject) {
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHsaExecutableLoadAgentCodeObjectFn != nullptr,
      llvm::formatv(
          "The underlying hsa_executable_load_agent_code_object function of "
          "HsaCodeObjectCache instance {0} is nullptr.",
          Idx)));
  hsa_loaded_code_object_t LCO;
  /// Call the underlying function
  hsa_status_t Out = UnderlyingHsaExecutableLoadAgentCodeObjectFn(
      Executable, Agent, CodeObjectReader, Options, &LCO);

  /// If the caller of the wrapper requested to get the LCO handle, return it
  if (LoadedCodeObject != nullptr)
    *LoadedCodeObject = LCO;

  /// Return if the object cache is not initialized or we encountered an error
  /// executing the underlying function
  if (!CodeObjectCacheInstance::isInitialized() || Out != HSA_STATUS_SUCCESS)
    return Out;

  /// Cache the code object of the LCO
  LUTHIER_REPORT_FATAL_ON_ERROR(CodeObjectCacheInstance::instance()
                                    .getAssociatedCodeObject(LCO)
                                    .takeError());

  /// Return the output of the original operation

  return Out;
}

template <size_t Idx>
hsa_status_t CodeObjectCacheInstance<Idx>::hsaExecutableDestroyWrapper(
    hsa_executable_t Executable) {
  /// Check if the underlying function is nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHsaExecutableDestroyFn != nullptr,
      llvm::formatv("The underlying hsa_executable_destroy function of "
                    "HsaCodeObjectCache instance {0} is nullptr.",
                    Idx)));
  /// Quick exit to the underlying function if the singleton instance is not
  /// initialized
  if (!CodeObjectCacheInstance::isInitialized())
    return UnderlyingHsaExecutableDestroyFn(Executable);

  auto &ObjectCache = CodeObjectCacheInstance::instance();

  /// Obtain the loaded code objects of the executable
  llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
  LUTHIER_REPORT_FATAL_ON_ERROR(hsa::loadedCodeObjectGetExecutable(
      Executable,
      ObjectCache.LoaderApiSnapshot.template getFunction<
          &ExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER>::TableType::
              hsa_ven_amd_loader_loaded_code_object_get_info>(),
      LCOs));
  {
    /// For each LCO, decrement its use count
    std::lock_guard Lock(ObjectCache.Mutex);
    for (hsa_loaded_code_object_t LCO : LCOs) {
      auto LCOIt = ObjectCache.LCOToCodeObjectMap.find(LCO);
      /// For now throw an error if we can't find the object associated with
      /// the LCO here to detect and debug any potential issues regarding
      /// the caching process
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
          LCOIt != ObjectCache.LCOToCodeObjectMap.end(),
          "Failed to find the associated object entry with LCO {0:x}.",
          LCO.handle));
      --LCOIt->second.UseCount;
      /// If the usecount has reached zero, then we remove the associated
      /// code object storage
      if (LCOIt->second.UseCount == 0) {
        size_t Hash = LCOIt->second.Hash;
        ObjectCache.HashToCodeObjectAndUsageMap.erase(Hash);
      }
      /// Erase the LCO object entry
      ObjectCache.LCOToCodeObjectMap.erase(LCO);
    }
  }

  /// Call the underlying function
  return UnderlyingHsaExecutableDestroyFn(Executable);
}

template <size_t Idx>
decltype(hsa_executable_load_agent_code_object) *
    CodeObjectCacheInstance<Idx>::UnderlyingHsaExecutableLoadAgentCodeObjectFn =
        nullptr;

template <size_t Idx>
decltype(hsa_executable_destroy)
    *CodeObjectCacheInstance<Idx>::UnderlyingHsaExecutableDestroyFn = nullptr;

#define LUTHIER_CREATE_NEW_HSA_CODE_OBJECT_CACHE_INSTANCE(...)                 \
  luthier::hsa::CodeObjectCacheInstance<__COUNTER__>::create(__VA_ARGS__)

} // namespace luthier::hsa

#endif
