//===-- ToolExecutableLoader.h ----------------------------------*- C++ -*-===//
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
/// Describes Luthier's HSA Tool Executable Loader and its concrete Singleton
/// class, in charge of managing all Luthier tool HSA executables.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOL_EXECUTABLE_LOADER_H
#define LUTHIER_HSA_TOOL_EXECUTABLE_LOADER_H
#include <hip/hip_runtime.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <luthier/Common/Singleton.h>
#include <luthier/HSA/Executable.h>
#include <luthier/HSA/hsa.h>
#include <luthier/HSARuntime/LoadedInstrumentationModule.h>
#include <luthier/Instrumentation/consts.h>
#include <luthier/Rocprofiler/HipApiTable.h>
#include <luthier/Rocprofiler/HsaApiTable.h>
#include <mutex>

namespace luthier::hsa {

template <size_t Idx> class ToolExecutableLoaderInstance;

/// \brief interface in charge of loading and unloading executables that belong
/// to Luthier and keeping track of them.
/// \details This interface contains all logic that doesn't rely on capturing
/// the HSA/HIP compiler API tables. The API table-related logic is instead
/// put inside its concrete implementation, the
/// \c ToolExecutableLoaderInstance class.
/// Other tool components must directly deal with this interface.
/// \sa ToolExecutableLoaderInstance
class ToolExecutableLoader {
protected:
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreTableSnapshot;

  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &LoaderApiSnapshot;

  /// Holds info regarding HIP Hook Function handles loaded into the HIP runtime
  struct HipHookFuncInfo {
    /// The Fat binary info this hook function handle belongs to
    void **FatBinaryModuleInfo;
    /// The host-accessible handle of the hook function
    void *HostHandle;
    /// The name of the hook function
    std::string Name;
  };

  /// Mutex protecting any HIP-registered fields
  std::mutex HipLoaderMutex;

  /// Mapping between the host handle of a HIP hook function and its info
  /// Populated during \c __hipRegisterFunction and unpopulated during
  /// \c __hipUnregisterFatBinary
  /// This is used to convert hook handles to obtain the associated
  /// loaded instrumentation module and the name of the hook
  llvm::SmallDenseMap<const void *, HipHookFuncInfo> HipFunctions{};

  /// Mapping between the HIP FatBinaryInfo of all HIP modules and their CUIDs.
  /// CUID is a symbol inside HIP modules used to uniquely identify it and
  /// associate it with a host binary
  llvm::SmallDenseMap<const void **, size_t> HipModuleCUIDs{};

  /// Holds a mapping between the CUID of an
  /// instrumentation module + its HSA agent ->
  /// its \c HipLoadedInstrumentationModule instance
  /// If the entry is \c nullptr then it means \c __hipRegister indicated that
  /// it has been registered with HIP, but it has not been loaded into HSA yet
  llvm::DenseMap<std::tuple<size_t, hsa_agent_t>,
                 std::unique_ptr<HipLoadedInstrumentationModule>>
      HipLoadedIMsPerAgent{};

  /// Mutex protecting the \c ApplicationToInstrumentedExecutablesMap field
  std::mutex InstrumentedExecMutex;

  /// A multimap, mapping the original executable of the application to
  /// its instrumented copies
  std::unordered_multimap<hsa_executable_t, hsa_executable_t>
      ApplicationToInstrumentedExecutablesMap{};

  /// Mutex protecting the dynamically loaded instrumentation module handles
  std::mutex DynamicModuleMutex;

  /// Set of dynamically loaded instrumentation modules. Map pointer lifetimes
  /// are managed by the loader
  llvm::SmallDenseSet<DynamicallyLoadedInstrumentationModule *> DynModules{};

  /// Checks whether the freshly loaded \p LCO by the application is loaded by
  /// HIP and is an instrumentation module; If so, registers it with the tool
  /// executable loader in \c HipLoadedIMsPerAgent to keep track of it
  /// \param LCO a loaded code object loaded by the application
  /// \note As this function peaks into the storage memory of the \p LCO
  /// it is only safe to call right after the application has loaded it
  /// via \c hsa_executable_load_agent_code_object
  /// \return \c llvm::Error indicating the success or failure of the operation
  llvm::Error
  registerIfHipLoadedInstrumentationModule(hsa_loaded_code_object_t LCO);

  /// Checks if the \p Exec that is about to be destroyed by the application is
  /// a \c HipLoadedInstrumentationModule and if so, unregisters it from its
  /// \c HipLoadedIMsPerAgent map
  /// \param Exec the HSA executable that is about to be destroyed by the
  /// application
  /// \return \c llvm::Error indicating the success or failure of the operation
  llvm::Error unregisterIfHipLoadedIModuleExec(hsa_executable_t Exec);

  /// Destroys all instrumented executable copies associated with the given
  /// \p Exec if they exist, and removes them from the
  /// \c ApplicationToInstrumentedExecutablesMap multimap
  /// \param Exec the application executable
  /// \return \c llvm::Error indicating the success or failure of the operation
  llvm::Error destroyInstrumentedCopiesOfExecutable(hsa_executable_t Exec);

  ToolExecutableLoader(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &LoaderApiSnapshot)
      : CoreTableSnapshot(CoreApiSnapshot),
        LoaderApiSnapshot(LoaderApiSnapshot) {};

public:
  virtual ~ToolExecutableLoader() {
    for (auto *DynMod : DynModules) {
      delete DynMod;
    }
  }

  /// Loads \p InstrumentedElf (the instrumented version of
  /// <tt>OriginalLoadedCodeObject</tt>) into a newly created executable and
  /// freezes it
  /// The \p InstrumentedElf is loaded on the same agent as
  /// \p OriginalLoadedCodeObject
  /// The resulting executable will be destroyed by the loader
  /// once the executable of \p OriginalLoadedCodeObject is destroyed by the
  /// application
  /// \param InstrumentedElf a code object, an instrumented version
  /// of OriginalLoadedCodeObject
  /// \param OriginalLoadedCodeObject the original loaded code object being
  /// instrumented
  /// \param ExternVariables A mapping between the name of the external
  /// variables not defined in \p InstrumentedElf and their address on the GPU.
  /// The variables must be accessible from the agent of
  /// \p OriginalLoadedCodeObject
  /// \return Expects the newly created executable containing the loaded
  /// instrumented code object.
  llvm::Expected<hsa_executable_t> loadInstrumentedCodeObject(
      llvm::ArrayRef<uint8_t> InstrumentedElf,
      hsa_loaded_code_object_t OriginalLoadedCodeObject,
      const llvm::StringMap<const void *> &ExternVariables);

  /// \return Expects the \c HipLoadedInstrumentationModule and the
  /// name of the hook associated with \p HostHandle loaded on \p Agent
  [[nodiscard]] llvm::Expected<
      std::pair<const HipLoadedInstrumentationModule &, std::string>>
  getHipLoadedHook(void *HostHandle, hsa_agent_t Agent) const;

  /// Loads the \p CodeObject containing an \c InstrumentationModule into
  /// the \p Agent
  /// The operation will fail if \p CodeObject is not a valid instrumentation
  /// module
  /// \param CodeObject a code object containing an instrumentation module
  /// \param Agent the target \c hsa_agent_t the module will be loaded on
  /// \return Expects a handle to the newly created
  /// \c DynamicallyLoadedInstrumentationModule
  llvm::Expected<DynamicallyLoadedInstrumentationModule &>
  loadInstrumentationModule(std::vector<uint8_t> CodeObject, hsa_agent_t Agent);

  /// Unloads the \p IModule handle. The operation fails if the handle is
  /// not managed by the current loader
  /// \return \c success indicating the success or failure of the operation
  llvm::Error
  unloadInstrumentationModule(DynamicallyLoadedInstrumentationModule &IModule) {
    std::lock_guard Lock(DynamicModuleMutex);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        DynModules.contains(&IModule),
        "Invalid dynamic instrumentation module handle."));
    DynModules.erase(&IModule);
    delete &IModule;
    return llvm::Error::success();
  }
};

/// \brief Concrete singleton implementation of \c ToolExecutableLoader
/// \details This singleton class adds the API table interception and wrapping
/// functionality to the \c ToolExecutableLoader interface. This class
/// is templated over an \c Idx parameter to allow multiple independent
/// instances of the same singleton to be created.
/// \c ToolExecutableLoaderInstance<0> is referred to as the "default"
/// executable loader instance.
/// The \c ToolExecutableLoader interface is separated from the indexed
/// templated singleton part so that other components use the loader
/// functionality regardless of the loader's instance index
template <size_t Idx>
class ROCPROFILER_HIDDEN_API ToolExecutableLoaderInstance
    : public ToolExecutableLoader,
      public Singleton<ToolExecutableLoaderInstance<Idx>> {
private:
  /// Provides the HSA runtime API table to the loader
  const std::unique_ptr<rocprofiler::HsaApiTableWrapperInstaller>
      HsaApiTableInterceptor;

  /// Provides the HIP Compiler API table to the loader
  const std::unique_ptr<rocprofiler::HipCompilerApiTableWrapperInstaller>
      HipCompilerApiTableInterceptor;

  //===--------------------------------------------------------------------===//
  // Underlying functions that the loader wraps over. They are static since
  // they need to remain valid even after the loader has been destroyed to
  // ensure the application's calls are forwarded to their underlying
  // functions
  //===--------------------------------------------------------------------===//

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_load_agent_code_object)
      *UnderlyingHsaExecutableLoadAgentCodeObjectFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_destroy)
      *UnderlyingHsaExecutableDestroy;

  static ROCPROFILER_HIDDEN_API t___hipRegisterFunction
      UnderlyingHipRegisterFunctionFn;

  static ROCPROFILER_HIDDEN_API t___hipRegisterVar UnderlyingHipRegisterVarFn;

  static ROCPROFILER_HIDDEN_API t___hipRegisterManagedVar
      UnderlyingHipRegisterManagedVarFn;

  static ROCPROFILER_HIDDEN_API t___hipUnregisterFatBinary
      UnderlyingHipUnregisterFatBinaryFn;

  //===--------------------------------------------------------------------===//
  // A set of wrapper functions that will be installed by the loader once
  // the API tables are intercepted. Wrapper functions need to be static
  // as they require to remain alive even once the loader is destroyed.
  //===--------------------------------------------------------------------===//

  static ROCPROFILER_HIDDEN_API hsa_status_t
  hsaExecutableLoadAgentCodeObjectWrapper(
      hsa_executable_t Executable, hsa_agent_t Agent,
      hsa_code_object_reader_t COR, const char *Options,
      hsa_loaded_code_object_t *LoadedCodeObject);

  ROCPROFILER_HIDDEN_API static hsa_status_t
  hsaExecutableDestroyWrapper(hsa_executable_t Executable);

  ROCPROFILER_HIDDEN_API static void
  hipRegisterFunctionWrapper(void **Modules, const void *HostFunction,
                             char *DeviceFunction, const char *DeviceName,
                             unsigned int ThreadLimit, uint3 *Tid, uint3 *Bid,
                             dim3 *BlockDim, dim3 *GridDim, int *WSize);

  ROCPROFILER_HIDDEN_API static void
  hipRegisterVarWrapper(void **Modules, void *Var, char *HostVar,
                        char *DeviceVar, int Ext, size_t Size, int Constant,
                        int Global);

  ROCPROFILER_HIDDEN_API static void
  hipRegisterManagedVarWrapper(void *HipModule, void **Pointer, void *InitValue,
                               const char *Name, size_t Size, unsigned Align);

  ROCPROFILER_HIDDEN_API static void
  hipUnregisterFatBinaryWrapper(void **Modules);

  ToolExecutableLoaderInstance(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &LoaderApiSnapshot,
      llvm::Error &Err)
      : ToolExecutableLoader(CoreApiSnapshot, LoaderApiSnapshot),
        HsaApiTableInterceptor([&] {
          llvm::ErrorAsOutParameter EAO(Err);
          if (Err)
            return nullptr;

          /// Install wrappers
          auto HsaApiTableInterceptorOrErr = rocprofiler::
              HsaApiTableWrapperInstaller::requestWrapperInstallation(
                  {&::CoreApiTable::hsa_executable_load_agent_code_object_fn,
                   UnderlyingHsaExecutableLoadAgentCodeObjectFn,
                   hsaExecutableLoadAgentCodeObjectWrapper},
                  {&::CoreApiTable::hsa_executable_destroy_fn,
                   UnderlyingHsaExecutableDestroy,
                   hsaExecutableDestroyWrapper});
          if (HsaApiTableInterceptorOrErr)
            return std::move(*HsaApiTableInterceptorOrErr);
          Err = std::move(HsaApiTableInterceptorOrErr.takeError());
          return nullptr;
        }()),
        HipCompilerApiTableInterceptor([&] {
          auto HipCompilerApiInterceptorOrErr = rocprofiler::
              HipCompilerApiTableWrapperInstaller::requestWrapperInstallation(
                  {&::HipCompilerDispatchTable::__hipRegisterFunction_fn,
                   UnderlyingHipRegisterFunctionFn, hipRegisterFunctionWrapper},
                  {&::HipCompilerDispatchTable::__hipRegisterVar_fn,
                   UnderlyingHipRegisterVarFn, hipRegisterVarWrapper},
                  {&::HipCompilerDispatchTable::__hipRegisterManagedVar_fn,
                   UnderlyingHipRegisterManagedVarFn,
                   hipRegisterManagedVarWrapper},
                  {&::HipCompilerDispatchTable::__hipUnregisterFatBinary_fn,
                   UnderlyingHipUnregisterFatBinaryFn,
                   hipUnregisterFatBinaryWrapper});
          Err = HipCompilerApiInterceptorOrErr.takeError();
          if (Err) {
            Err = std::move(Err);
            return nullptr;
          }
          return std::move(*HipCompilerApiInterceptorOrErr);
        }()) {};

public:
  /// Creates and returns a new \c ToolExecutableLoaderInstance
  static llvm::Expected<std::unique_ptr<ToolExecutableLoaderInstance>> create(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
      const rocprofiler::HsaExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER>
          &LoaderApiSnapshot) {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<ToolExecutableLoaderInstance>(
        CoreApiSnapshot, LoaderApiSnapshot, Err);
    return Out;
  }

  ~ToolExecutableLoaderInstance() override = default;
};

template <size_t Idx>
hsa_status_t
ToolExecutableLoaderInstance<Idx>::hsaExecutableLoadAgentCodeObjectWrapper(
    hsa_executable_t Executable, hsa_agent_t Agent,
    hsa_code_object_reader_t COR, const char *Options,
    hsa_loaded_code_object_t *LoadedCodeObject) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHsaExecutableLoadAgentCodeObjectFn != nullptr,
      llvm::formatv("Underlying hsa_executable_load_agent_code_object of "
                    "ToolExecutableLoaderInstance<{0}> is nullptr",
                    Idx)));
  hsa_loaded_code_object_t LCO;
  /// Call the underlying function
  hsa_status_t Out = UnderlyingHsaExecutableLoadAgentCodeObjectFn(
      Executable, Agent, COR, Options, &LCO);

  /// If the caller of the wrapper requested to get the LCO handle, return it
  if (LoadedCodeObject != nullptr)
    *LoadedCodeObject = LCO;

  /// Return if the loader is not initialized or we encountered an error
  /// executing the underlying function
  if (!ToolExecutableLoaderInstance::isInitialized() ||
      Out != HSA_STATUS_SUCCESS)
    return Out;

  /// Check if this LCO is a HIP loaded IModule and register it
  auto &TEL = ToolExecutableLoaderInstance::instance();

  LUTHIER_REPORT_FATAL_ON_ERROR(
      TEL.registerIfHipLoadedInstrumentationModule(LCO));

  return Out;
}

template <size_t Idx>
hsa_status_t ToolExecutableLoaderInstance<Idx>::hsaExecutableDestroyWrapper(
    hsa_executable_t Executable) {
  // Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHsaExecutableDestroy != nullptr,
      llvm::formatv("Underlying hsa_executable_destroy function of "
                    "ToolExecutableLoaderInstance<{0}> is nullptr",
                    Idx)));
  // Call the underlying function if the loader is not initialized
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHsaExecutableDestroy(Executable);

  auto &TEL = ToolExecutableLoaderInstance::instance();
  /// Check if the executable that is about to be destroyed is a HIP loaded
  /// IModule
  LUTHIER_REPORT_FATAL_ON_ERROR(
      TEL.unregisterIfHipLoadedIModuleExec(Executable));
  /// If an application executable, destroy any instrumented executable
  /// associated with it
  LUTHIER_REPORT_FATAL_ON_ERROR(
      TEL.destroyInstrumentedCopiesOfExecutable(Executable));
  /// Call the original function
  return UnderlyingHsaExecutableDestroy(Executable);
}

template <size_t Idx>
void ToolExecutableLoaderInstance<Idx>::hipRegisterFunctionWrapper(
    void **Modules, const void *HostFunction, char *DeviceFunction,
    const char *DeviceName, unsigned int ThreadLimit, uint3 *Tid, uint3 *Bid,
    dim3 *BlockDim, dim3 *GridDim, int *WSize) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHipRegisterFunctionFn != nullptr,
      "Underlying __hipRegisterFunction is nullptr"));
  /// Call the underlying function directly if the executable loader is not
  /// initialized
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHipRegisterFunctionFn(
        Modules, HostFunction, DeviceFunction, DeviceName, ThreadLimit, Tid,
        Bid, BlockDim, GridDim, WSize);

  auto &TEL = ToolExecutableLoaderInstance::instance();
  /// Store the function's info if the function is a Luthier hook handle
  if (llvm::StringRef(DeviceFunction).starts_with(HookHandlePrefix)) {
    std::lock_guard Lock(TEL.HipLoaderMutex);
    TEL.HipFunctions.insert(
        {HostFunction, {Modules, HostFunction, DeviceFunction}});
  }
  /// Call the original function
  return UnderlyingHipRegisterFunctionFn(Modules, HostFunction, DeviceFunction,
                                         DeviceName, ThreadLimit, Tid, Bid,
                                         BlockDim, GridDim, WSize);
}

template <size_t Idx>
void ToolExecutableLoaderInstance<Idx>::hipRegisterVarWrapper(
    void **Modules, void *Var, char *HostVar, char *DeviceVar, int Ext,
    size_t Size, int Constant, int Global) {
  /// Check if the underlying function not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_GENERIC_ERROR_CHECK(UnderlyingHipRegisterVarFn != nullptr,
                                  "Underlying __hipRegisterVar is nullptr"));
  /// Call the underlying function directly if the loader is not initialized
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHipRegisterVarFn(Modules, Var, HostVar, DeviceVar, Ext,
                                      Size, Constant, Global);

  /// If we have stumbled upon the CUID of the FAT binary, extract it and store
  /// it
  auto &TEL = ToolExecutableLoaderInstance::instance();
  llvm::StringRef DeviceVarStrRef(DeviceVar);
  if (DeviceVarStrRef.starts_with(HipCUIDPrefix)) {
    size_t CUID;
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        llvm::to_integer(DeviceVarStrRef.substr(strlen(HipCUIDPrefix)), CUID),
        "Failed to parse the CUID of the HIP module"));
    {
      std::lock_guard Lock(TEL.HipLoaderMutex);
      TEL.HipModuleCUIDs.insert({Modules, CUID});
    }
  }

  /// Call the underlying function
  return UnderlyingHipRegisterVarFn(Modules, Var, HostVar, DeviceVar, Ext, Size,
                                    Constant, Global);
}

template <size_t Idx>
void ToolExecutableLoaderInstance<Idx>::hipRegisterManagedVarWrapper(
    void *HipModule, void **Pointer, void *InitValue, const char *Name,
    size_t Size, unsigned Align) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHipRegisterManagedVarFn != nullptr,
      "Underlying __hipRegisterManagedVar is nullptr"));
  /// Call the underlying function directly if the loader is not initialized
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHipRegisterManagedVarFn(HipModule, Pointer, InitValue,
                                             Name, Size, Align);
  /// Check if we have stumbled upon the Luthier reserved variable; If so,
  /// reserve a spot inside the HipLoadedIMsPerAgent map to indicate that we
  /// expect a HipLoaded IModule will be loaded for this HIP module on every
  /// agent
  if (llvm::StringRef(Name) == IModuleReservedManagedVar) {
    auto &TEL = ToolExecutableLoaderInstance::instance();
    auto ModuleVoidStar = static_cast<void **>(HipModule);
    std::lock_guard Lock(TEL.HipLoaderMutex);

    typename decltype(TEL.HipModuleCUIDs)::iterator IModuleCUIDIter =
        TEL.HipModuleCUIDs.find(ModuleVoidStar);

    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        IModuleCUIDIter != TEL.HipModuleCUIDs.end(),
        llvm::formatv(
            "Failed to find the CUID of the hip fat binary info {0:x}",
            ModuleVoidStar)));

    LUTHIER_REPORT_FATAL_ON_ERROR(
        LUTHIER_ERROR_CHECK(TEL.UnderlyingHsaIterateAgentsFn != nullptr,
                            "Underlying hsa_iterate_agents is nullptr"));

    llvm::SmallVector<hsa_agent_t, 4> Agents;
    LUTHIER_REPORT_FATAL_ON_ERROR(hsa::getGpuAgents(
        TEL.TableSnapshot
            .template getFunction<&::CoreApiTable::hsa_iterate_agents_fn>(),
        Agents));

    for (const hsa_agent_t &Agent : Agents) {
      TEL.HipLoadedIMsPerAgent.insert(
          {{IModuleCUIDIter->second, Agent},
           std::unique_ptr<HipLoadedInstrumentationModule>(nullptr)});
    }
  }
  return UnderlyingHipRegisterManagedVarFn(HipModule, Pointer, InitValue, Name,
                                           Size, Align);
}

template <size_t Idx>
void ToolExecutableLoaderInstance<Idx>::hipUnregisterFatBinaryWrapper(
    void **Modules) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHipUnregisterFatBinaryFn != nullptr,
      "Underlying __hipUnregisterFatBinary is nullptr"));
  /// Execute the underlying function if the loader is not initialized
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHipUnregisterFatBinaryFn(Modules);
  /// Erase the Module from the \c HipModuleCUIDs map. Any hook handle queries
  /// that rely on this module should fail now that the module has been
  /// unloaded
  {
    auto &TEL = ToolExecutableLoaderInstance::instance();
    std::lock_guard Lock(TEL.HipLoaderMutex);
    TEL.HipModuleCUIDs.erase(Modules);
  }
  UnderlyingHipUnregisterFatBinaryFn(Modules);
}

template <size_t Idx>
decltype(hsa_executable_load_agent_code_object) *ToolExecutableLoaderInstance<
    Idx>::UnderlyingHsaExecutableLoadAgentCodeObjectFn = nullptr;

template <size_t Idx>
decltype(hsa_executable_destroy) *
    ToolExecutableLoaderInstance<Idx>::UnderlyingHsaExecutableDestroy = nullptr;

template <size_t Idx>
t___hipRegisterFunction
    ToolExecutableLoaderInstance<Idx>::UnderlyingHipRegisterFunctionFn =
        nullptr;

template <size_t Idx>
t___hipRegisterVar
    ToolExecutableLoaderInstance<Idx>::UnderlyingHipRegisterVarFn = nullptr;

template <size_t Idx>
t___hipRegisterManagedVar
    ToolExecutableLoaderInstance<Idx>::UnderlyingHipRegisterManagedVarFn =
        nullptr;

template <size_t Idx>
t___hipUnregisterFatBinary
    ToolExecutableLoaderInstance<Idx>::UnderlyingHipUnregisterFatBinaryFn =
        nullptr;

#define LUTHIER_CREATE_NEW_HSA_TOOL_EXEC_LOADER(...)                           \
  luthier::hsa::ToolExecutableLoaderInstance<__COUNTER__>::create(__VA_ARGS__)

}; // namespace luthier::hsa

#endif
