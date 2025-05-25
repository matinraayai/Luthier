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
/// Describes Luthier's Tool Executable Loader Singleton, in
/// charge of managing all HIP-loaded instrumentation modules, the lifetime of
/// the instrumented executables, and loading instrumented kernels into a
/// dispatch packet when the tool requests it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_RUNTIME_TOOL_EXECUTABLE_LOADER_H
#define LUTHIER_RUNTIME_TOOL_EXECUTABLE_LOADER_H
#include "luthier/common/Singleton.h"
#include "luthier/consts.h"
#include "luthier/hip/HipCompilerApiTableInterceptor.h"
#include "luthier/hsa/Executable.h"
#include "luthier/hsa/HsaApiTableInterceptor.h"
#include "luthier/hsa/hsa.h"
#include "luthier/runtime/LoadedInstrumentationModule.h"
#include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/hip_runtime.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>

namespace luthier {

template <size_t Idx> class ToolExecutableLoaderInstance;

class ToolExecutableLoader {
protected:
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
  /// instrumentation module and the name of the hook
  llvm::SmallDenseMap<const void *, HipHookFuncInfo> HipFunctions;

  /// Mapping between the HIP FatBinaryInfo of all HIP modules and their CUIDs.
  /// CUID is a symbol inside HIP modules used to uniquely identify it
  llvm::SmallDenseMap<const void **, size_t> HipModuleCUIDs;

  /// Holds a mapping between the CUID of an
  /// instrumentation module + its HSA agent ->
  /// its \c HipLoadedInstrumentationModule instance
  /// If the entry is \c nullptr then it means \c __hipRegister indicated that
  /// it has been registered with HIP, but it has not been loaded into HSA yet
  llvm::DenseMap<std::tuple<size_t, hsa_agent_t>,
                 std::unique_ptr<HipLoadedInstrumentationModule>>
      SIMsByAgent;

  std::mutex InstrumentedExecMutex;

  /// A multimap from the original executable of the application to
  /// its instrumented copies
  std::unordered_multimap<hsa_executable_t, hsa_executable_t>
      ApplicationToInstrumentedExecutablesMap{};

  /// Mutex protecting the dynamically loaded instrumentation module handles
  std::mutex DynamicModuleMutex;

  /// Set of dynamically loaded instrumentation modules
  llvm::SmallDenseSet<DynamicallyLoadedInstrumentationModule *> DynModules;

  llvm::Error registerIfHipLoadedInstrumentationModule(
      hsa_loaded_code_object_t LCO,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &LoadedCodeObjectGetInfoFun);

  llvm::Error unregisterIfHipLoadedIModuleExec(
      hsa_executable_t Exec,
      const decltype(hsa_executable_get_symbol_by_name) &ExecSymbolLookupFn,
      const decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
          &LCOIteratorFn,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &LCOGetInfoFn,
      const decltype(hsa_executable_iterate_agent_symbols) &SymbolIterFn,
      const decltype(hsa_executable_symbol_get_info) &SymbolInfoGetterFn);

  llvm::Error destroyInstrumentedExecutables(
      hsa_executable_t Exec,
      const decltype(hsa_executable_destroy) &ExecutableDestroyFn);

  llvm::Expected<DynamicallyLoadedInstrumentationModule &> loadDynamicIModule(
      std::vector<uint8_t> CodeObject, hsa_agent_t Agent,
      decltype(hsa_executable_create_alt) &HsaExecutableCreateAltFn,
      decltype(hsa_code_object_reader_create_from_memory)
          &HsaCodeObjectReaderCreateFromMemory,
      decltype(hsa_executable_load_agent_code_object)
          &HsaExecutableLoadAgentCodeObjectFn,
      decltype(hsa_executable_freeze) &HsaExecutableFreezeFn,
      decltype(hsa_code_object_reader_destroy) &HsaCodeObjectReaderDestroyFn,
      decltype(hsa_executable_destroy) &HsaExecutableDestroyFn);

  llvm::Expected<hsa_executable_t> loadInstrumentedCodeObject(
      llvm::ArrayRef<uint8_t> InstrumentedElf,
      hsa_loaded_code_object_t OriginalLoadedCodeObject,
      const llvm::StringMap<const void *> &ExternVariables,
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          &LCOInfoQueryFn,
      const decltype(hsa_executable_agent_global_variable_define)
          &HsaExecutableAgentGlobalVariableDefineFn,
      decltype(hsa_executable_create_alt) &HsaExecutableCreateAltFn,
      decltype(hsa_code_object_reader_create_from_memory)
          &HsaCodeObjectReaderCreateFromMemory,
      decltype(hsa_executable_load_agent_code_object)
          &HsaExecutableLoadAgentCodeObjectFn,
      decltype(hsa_executable_freeze) &HsaExecutableFreezeFn,
      decltype(hsa_code_object_reader_destroy) &HsaCodeObjectReaderDestroyFn);

  ToolExecutableLoader() = default;

public:
  virtual ~ToolExecutableLoader() {
    for (auto *DynMod : DynModules) {
      delete DynMod;
    }
  }

  [[nodiscard]] llvm::Expected<
      std::pair<const HipLoadedInstrumentationModule &, std::string>>
  getHipLoadedHook(void *HostHandle, hsa_agent_t Agent) const;

  virtual llvm::Expected<DynamicallyLoadedInstrumentationModule &>
  loadInstrumentationModule(std::vector<uint8_t> CodeObject,
                            hsa_agent_t Agent) = 0;

  llvm::Error
  unloadInstrumentationModule(DynamicallyLoadedInstrumentationModule &IModule) {
    std::lock_guard Lock(DynamicModuleMutex);
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(DynModules.contains(&IModule),
                            "Invalid dynamic instrumentation module handle."));
    DynModules.erase(&IModule);
    delete &IModule;
    return llvm::Error::success();
  }

  virtual llvm::Error loadInstrumentedExecutable(
      llvm::ArrayRef<uint8_t> InstrumentedElf,
      hsa_loaded_code_object_t OriginalLoadedCodeObject,
      const llvm::StringMap<const void *> &ExternVariables) = 0;
};

/// \brief A singleton object that keeps track of executables that belong to
/// Luthier, including instrumented executables and tool
/// instrumentation modules, plus launching instrumented kernels

template <size_t Idx>
class ROCPROFILER_HIDDEN_API ToolExecutableLoaderInstance
    : public ToolExecutableLoader,
      public Singleton<ToolExecutableLoaderInstance<Idx>> {
private:
  /// Provides the HSA API table to the loader so that it can obtain the
  /// underlying HSA functions and install wrappers over them
  const std::unique_ptr<
      hsa::HsaApiTableInterceptor<std::function<void(HsaApiTable &)>>>
      HsaApiTableInterceptor;

  /// Provides the HIP Compiler API table to the loader so that it can obtain
  /// the underlying HIP compiler functions and install wrappers over them
  const std::unique_ptr<hip::HipCompilerApiTableInterceptor<
      std::function<void(HipCompilerDispatchTable &)>>>
      HipCompilerApiTableInterceptor;

  static ROCPROFILER_HIDDEN_API decltype(hsa_iterate_agents)
      *UnderlyingHsaIterateAgentsFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_create_alt)
      *UnderlyingHsaExecutableCreateAltFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_get_info)
      *UnderlyingHsaExecutableGetInfoFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_code_object_reader_create_from_memory)
      *UnderlyingHsaCodeObjectReaderCreateFromMemoryFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_load_agent_code_object)
      *UnderlyingHsaExecutableLoadAgentCodeObjectFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_code_object_reader_destroy)
      *UnderlyingHsaCodeObjectReaderDestroyFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_agent_global_variable_define)
      *UnderlyingHsaExecutableAgentGlobalVariableDefineFn;

  /// Function pointer to the underlying \c ::hsa_executable_freeze being
  /// wrapped. We store the function pointer here to ensure this underlying
  /// function is correctly called even after the singleton is destroyed
  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_freeze)
      *UnderlyingHsaExecutableFreezeFn;

  /// Function pointer to the underlying \c ::hsa_executable_destroy being
  /// wrapped
  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_destroy)
      *UnderlyingHsaExecutableDestroy;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_symbol_get_info)
      *UnderlyingHsaSymbolGetInfoFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_iterate_agent_symbols)
      *UnderlyingHsaExecutableIterateAgentSymbolsFn;

  static ROCPROFILER_HIDDEN_API decltype(hsa_executable_get_symbol_by_name)
      *UnderlyingHsaExecutableGetSymbolByNameFn;

  /// Underlying HSA LCO iteration for LCOs
  static ROCPROFILER_HIDDEN_API decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
      *HsaVenAmdLoaderExecutableIterateLCOsFn;

  /// Underlying HSA LCO info query function
  static ROCPROFILER_HIDDEN_API decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
      *HsaVenAmdLoaderLCOGetInfoFn;

  /// Function pointer to the underlying \c __hipRegisterFunction being wrapped
  static ROCPROFILER_HIDDEN_API t___hipRegisterFunction
      UnderlyingHipRegisterFunctionFn;

  /// Function pointer to the underlying \c __hipRegisterVar being wrapped
  static ROCPROFILER_HIDDEN_API t___hipRegisterVar UnderlyingHipRegisterVarFn;

  /// Function pointer to the underlying \c __hipRegisterManagedVar being
  /// wrapped
  static ROCPROFILER_HIDDEN_API t___hipRegisterManagedVar
      UnderlyingHipRegisterManagedVarFn;

  static ROCPROFILER_HIDDEN_API t___hipUnregisterFatBinary
      UnderlyingHipUnregisterFatBinaryFn;

  static ROCPROFILER_HIDDEN_API hsa_status_t
  hsaExecutableLoadAgentCodeObjectWrapper(
      hsa_executable_t Executable, hsa_agent_t Agent,
      hsa_code_object_reader_t COR, const char *Options,
      hsa_loaded_code_object_t *LoadedCodeObject);

  /// Wrapper for monitoring \c ::hsa_executable_destroy calls by
  /// the application
  ROCPROFILER_HIDDEN_API static hsa_status_t
  hsaExecutableDestroyWrapper(hsa_executable_t Executable);

  /// Wrapper for monitoring \c __hipRegisterFunction calls by both the
  /// application and the Luthier tool
  static void
  hipRegisterFunctionWrapper(void **Modules, const void *HostFunction,
                             char *DeviceFunction, const char *DeviceName,
                             unsigned int ThreadLimit, uint3 *Tid, uint3 *Bid,
                             dim3 *BlockDim, dim3 *GridDim, int *WSize);

  static void hipRegisterVarWrapper(void **Modules, void *Var, char *HostVar,
                                    char *DeviceVar, int Ext, size_t Size,
                                    int Constant, int Global);

  static void hipRegisterManagedVarWrapper(void *HipModule, void **Pointer,
                                           void *InitValue, const char *Name,
                                           size_t Size, unsigned Align);

  static void hipUnregisterFatBinaryWrapper(void **Modules);

  ToolExecutableLoaderInstance(
      std::unique_ptr<
          hsa::HsaApiTableInterceptor<std::function<void(HsaApiTable &)>>>
          HsaApiTableInterceptor,
      std::unique_ptr<hip::HipCompilerApiTableInterceptor<
          std::function<void(HipCompilerDispatchTable &)>>>
          HipCompilerApiTableInterceptor)
      : ToolExecutableLoader(),
        HsaApiTableInterceptor(std::move(HsaApiTableInterceptor)),
        HipCompilerApiTableInterceptor(
            std::move(HipCompilerApiTableInterceptor)) {};

public:
  static llvm::Expected<std::unique_ptr<ToolExecutableLoaderInstance>>
  create() {
    auto HsaApiTableInterceptorOrErr = hsa::HsaApiTableInterceptor<
        std::function<void(HsaApiTable &)>>::requestApiTable([](::HsaApiTable
                                                                    &Table) {
      /// Save the needed underlying function
      UnderlyingHsaIterateAgentsFn = Table.core_->hsa_iterate_agents_fn;
      UnderlyingHsaExecutableGetInfoFn =
          Table.core_->hsa_executable_get_info_fn;
      UnderlyingHsaExecutableCreateAltFn =
          Table.core_->hsa_executable_create_alt_fn;
      UnderlyingHsaCodeObjectReaderCreateFromMemoryFn =
          Table.core_->hsa_code_object_reader_create_from_memory_fn;
      UnderlyingHsaExecutableLoadAgentCodeObjectFn =
          Table.core_->hsa_executable_load_agent_code_object_fn;
      UnderlyingHsaCodeObjectReaderDestroyFn =
          Table.core_->hsa_code_object_reader_destroy_fn;
      UnderlyingHsaExecutableAgentGlobalVariableDefineFn =
          Table.core_->hsa_executable_agent_global_variable_define_fn;
      UnderlyingHsaExecutableFreezeFn = Table.core_->hsa_executable_freeze_fn;
      UnderlyingHsaExecutableDestroy = Table.core_->hsa_executable_destroy_fn;
      UnderlyingHsaSymbolGetInfoFn =
          Table.core_->hsa_executable_symbol_get_info_fn;
      UnderlyingHsaExecutableIterateAgentSymbolsFn =
          Table.core_->hsa_executable_iterate_agent_symbols_fn;
      UnderlyingHsaExecutableGetSymbolByNameFn =
          Table.core_->hsa_executable_get_symbol_by_name_fn;
      /// Save all required loader functions
      hsa_ven_amd_loader_1_03_pfn_t LoaderTable;
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
          Table.core_->hsa_system_get_major_extension_table_fn(
              HSA_EXTENSION_AMD_LOADER, 1,
              sizeof(hsa_ven_amd_loader_1_03_pfn_t), &LoaderTable)));
      HsaVenAmdLoaderExecutableIterateLCOsFn =
          LoaderTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects;
      HsaVenAmdLoaderLCOGetInfoFn =
          LoaderTable.hsa_ven_amd_loader_loaded_code_object_get_info;
      /// Install wrappers
      Table.core_->hsa_executable_load_agent_code_object_fn =
          hsaExecutableLoadAgentCodeObjectWrapper;
      Table.core_->hsa_executable_destroy_fn = hsaExecutableDestroyWrapper;
    });
    LUTHIER_RETURN_ON_ERROR(HsaApiTableInterceptorOrErr.takeError());

    auto HipCompilerApiInterceptorOrErr = hip::HipCompilerApiTableInterceptor<
        std::function<void(HipCompilerDispatchTable & Table)>>::
        requestApiTable([](HipCompilerDispatchTable &Table) {
          UnderlyingHipRegisterFunctionFn = Table.__hipRegisterFunction_fn;
          UnderlyingHipRegisterVarFn = Table.__hipRegisterVar_fn;
          UnderlyingHipRegisterManagedVarFn = Table.__hipRegisterManagedVar_fn;
          UnderlyingHipUnregisterFatBinaryFn =
              Table.__hipUnregisterFatBinary_fn;
          Table.__hipRegisterFunction_fn = hipRegisterFunctionWrapper;
          Table.__hipRegisterVar_fn = hipRegisterVarWrapper;
          Table.__hipRegisterManagedVar_fn = hipRegisterManagedVarWrapper;
          Table.__hipUnregisterFatBinary_fn = hipUnregisterFatBinaryWrapper;
        });
    LUTHIER_RETURN_ON_ERROR(HipCompilerApiInterceptorOrErr.takeError());

    return std::make_unique<ToolExecutableLoaderInstance>(
        std::move(*HsaApiTableInterceptorOrErr),
        std::move(*HipCompilerApiInterceptorOrErr));
  }

  ~ToolExecutableLoaderInstance() override = default;

  llvm::Expected<DynamicallyLoadedInstrumentationModule &>
  loadInstrumentationModule(std::vector<uint8_t> CodeObject,
                            hsa_agent_t Agent) override {
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(UnderlyingHsaExecutableCreateAltFn != nullptr,
                            "Underlying hsa_executable_create_alt of Tool "
                            "Executable Loader Instance {0} is nullptr",
                            Idx));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        UnderlyingHsaCodeObjectReaderCreateFromMemoryFn != nullptr,
        "Underlying hsa_code_object_reader_create_from_memory of Tool "
        "Executable Loader Instance {0} is nullptr",
        Idx));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        UnderlyingHsaExecutableLoadAgentCodeObjectFn != nullptr,
        "Underlying hsa_executable_load_agent_code_object of Tool "
        "Executable Loader Instance {0} is nullptr",
        Idx));
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(UnderlyingHsaExecutableFreezeFn != nullptr,
                            "Underlying hsa_executable_freeze of Tool "
                            "Executable Loader Instance {0} is nullptr",
                            Idx));
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(UnderlyingHsaCodeObjectReaderDestroyFn != nullptr,
                            "Underlying hsa_code_object_reader_destroy of Tool "
                            "Executable Loader Instance {0} is nullptr",
                            Idx));
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(UnderlyingHsaExecutableDestroy != nullptr,
                            "Underlying hsa_executable_destroy of Tool "
                            "Executable Loader Instance {0} is nullptr",
                            Idx));

    return loadDynamicIModule(std::move(CodeObject), Agent,
                              *UnderlyingHsaExecutableCreateAltFn,
                              *UnderlyingHsaCodeObjectReaderCreateFromMemoryFn,
                              *UnderlyingHsaExecutableLoadAgentCodeObjectFn,
                              *UnderlyingHsaExecutableFreezeFn,
                              *UnderlyingHsaCodeObjectReaderDestroyFn,
                              *UnderlyingHsaExecutableDestroy);
  }

  llvm::Error loadInstrumentedExecutable(
      llvm::ArrayRef<uint8_t> InstrumentedElf,
      hsa_loaded_code_object_t OriginalLoadedCodeObject,
      const llvm::StringMap<const void *> &ExternVariables) override {

    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        HsaVenAmdLoaderLCOGetInfoFn != nullptr,
        "Underlying hsa_ven_amd_loader_loaded_code_object_get_info of Tool "
        "Executable Loader Instance {0} is nullptr",
        Idx));

    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        UnderlyingHsaExecutableAgentGlobalVariableDefineFn != nullptr,
        "Underlying hsa_executable_agent_global_variable_define of Tool "
        "Executable Loader Instance {0} is nullptr",
        Idx));

    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(UnderlyingHsaExecutableCreateAltFn != nullptr,
                            "Underlying hsa_executable_create_alt of Tool "
                            "Executable Loader Instance {0} is nullptr",
                            Idx));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        UnderlyingHsaCodeObjectReaderCreateFromMemoryFn != nullptr,
        "Underlying hsa_code_object_reader_create_from_memory of Tool "
        "Executable Loader Instance {0} is nullptr",
        Idx));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        UnderlyingHsaExecutableLoadAgentCodeObjectFn != nullptr,
        "Underlying hsa_executable_load_agent_code_object of Tool "
        "Executable Loader Instance {0} is nullptr",
        Idx));
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(UnderlyingHsaExecutableFreezeFn != nullptr,
                            "Underlying hsa_executable_freeze of Tool "
                            "Executable Loader Instance {0} is nullptr",
                            Idx));
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(UnderlyingHsaCodeObjectReaderDestroyFn != nullptr,
                            "Underlying hsa_code_object_reader_destroy of Tool "
                            "Executable Loader Instance {0} is nullptr",
                            Idx));

    return loadInstrumentedExecutable(
        InstrumentedElf, OriginalLoadedCodeObject, ExternVariables,
        *HsaVenAmdLoaderLCOGetInfoFn,
        *UnderlyingHsaExecutableAgentGlobalVariableDefineFn,
        *UnderlyingHsaExecutableCreateAltFn,
        *UnderlyingHsaCodeObjectReaderCreateFromMemoryFn,
        *UnderlyingHsaExecutableLoadAgentCodeObjectFn,
        *UnderlyingHsaExecutableFreezeFn,
        *UnderlyingHsaCodeObjectReaderDestroyFn);
  }
};

template <size_t Idx>
hsa_status_t
ToolExecutableLoaderInstance<Idx>::hsaExecutableLoadAgentCodeObjectWrapper(
    hsa_executable_t Executable, hsa_agent_t Agent,
    hsa_code_object_reader_t COR, const char *Options,
    hsa_loaded_code_object_t *LoadedCodeObject) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      UnderlyingHsaExecutableLoadAgentCodeObjectFn != nullptr,
      "Underlying hsa_executable_load_agent_code_object of "
      "ToolExecutableLoaderInstance<{0}> is nullptr",
      Idx));
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
      LUTHIER_ERROR_CHECK(HsaVenAmdLoaderLCOGetInfoFn != nullptr,
                          "The hsa_ven_amd_loader_loaded_code_object_get_info "
                          "of ToolExecutableLoader instance {0} is nullptr",
                          Idx));
  LUTHIER_REPORT_FATAL_ON_ERROR(TEL.registerIfHipLoadedInstrumentationModule(
      LCO, *HsaVenAmdLoaderLCOGetInfoFn));

  return Out;
}

template <size_t Idx>
hsa_status_t ToolExecutableLoaderInstance<Idx>::hsaExecutableDestroyWrapper(
    hsa_executable_t Executable) {
  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_ERROR_CHECK(UnderlyingHsaExecutableDestroy != nullptr,
                          "Underlying hsa_executable_destroy function of "
                          "ToolExecutableLoaderInstance<{0}> is nullptr",
                          Idx));
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHsaExecutableDestroy(Executable);

  /// Should report a fatal error when obtaining the instance reference
  auto &TEL = ToolExecutableLoaderInstance::instance();
  /// Unregister if it's a tool executable
  LUTHIER_REPORT_FATAL_ON_ERROR(
      TEL.unregisterIfHipLoadedIModuleExec(Executable));
  /// If an application executable, destroy any instrumented executable
  /// associated with it
  LUTHIER_REPORT_FATAL_ON_ERROR(TEL.destroyInstrumentedExecutables(
      Executable, UnderlyingHsaExecutableDestroy));
  /// Call the original function
  return UnderlyingHsaExecutableDestroy(Executable);
}

template <size_t Idx>
void ToolExecutableLoaderInstance<Idx>::hipRegisterFunctionWrapper(
    void **Modules, const void *HostFunction, char *DeviceFunction,
    const char *DeviceName, unsigned int ThreadLimit, uint3 *Tid, uint3 *Bid,
    dim3 *BlockDim, dim3 *GridDim, int *WSize) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_ERROR_CHECK(UnderlyingHipRegisterFunctionFn != nullptr,
                          "Underlying __hipRegisterFunction is nullptr"));
  /// Call the underlying function directly if the executable loader is not
  /// initialized
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHipRegisterFunctionFn(
        Modules, HostFunction, DeviceFunction, DeviceName, ThreadLimit, Tid,
        Bid, BlockDim, GridDim, WSize);

  auto &TEL = ToolExecutableLoaderInstance::instance();
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
      LUTHIER_ERROR_CHECK(UnderlyingHipRegisterVarFn != nullptr,
                          "Underlying __hipRegisterVar is nullptr"));
  /// Call the underlying function directly if the loader is not initialized
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHipRegisterVarFn(Modules, Var, HostVar, DeviceVar, Ext,
                                      Size, Constant, Global);

  /// If we have stumbled upon the CUID of the module, extract it and store it
  auto &TEL = ToolExecutableLoaderInstance::instance();
  llvm::StringRef DeviceVarStrRef(DeviceVar);

  if (DeviceVarStrRef.starts_with(HipCUIDPrefix)) {
    size_t CUID;
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
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
  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_ERROR_CHECK(UnderlyingHipRegisterManagedVarFn != nullptr,
                          "Underlying __hipRegisterManagedVar is nullptr"));
  /// Call the underlying function directly if the loader is not initialized
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHipRegisterManagedVarFn(HipModule, Pointer, InitValue,
                                             Name, Size, Align);
  /// Check if we have stumbled upon the Luthier reserved variable; If so,
  /// reserve a spot inside the SIMsByAgent map to indicate that we expect
  /// a HipLoaded IModule will be loaded for this HIP module on every agent
  if (llvm::StringRef(Name) == IModuleReservedManagedVar) {
    auto &TEL = ToolExecutableLoaderInstance::instance();
    auto ModuleVoidStar = static_cast<void **>(HipModule);
    std::lock_guard Lock(TEL.HipLoaderMutex);

    typename decltype(TEL.HipModuleCUIDs)::iterator IModuleCUIDIter =
        TEL.HipModuleCUIDs.find(ModuleVoidStar);

    LUTHIER_REPORT_FATAL_ON_ERROR(
        IModuleCUIDIter != TEL.HipModuleCUIDs.end(),
        "Failed to find the CUID of the hip fat binary info {0:x}",
        ModuleVoidStar);

    LUTHIER_REPORT_FATAL_ON_ERROR(
        LUTHIER_ERROR_CHECK(UnderlyingHsaIterateAgentsFn != nullptr,
                            "Underlying hsa_iterate_agents is nullptr"));

    llvm::SmallVector<hsa_agent_t, 4> Agents;
    LUTHIER_REPORT_FATAL_ON_ERROR(
        hsa::getGpuAgents(*UnderlyingHsaIterateAgentsFn, Agents));

    for (const hsa_agent_t &Agent : Agents) {
      TEL.SIMsByAgent.insert(
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
  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_ERROR_CHECK(UnderlyingHipUnregisterFatBinaryFn != nullptr,
                          "Underlying __hipUnregisterFatBinary is nullptr"));
  if (!ToolExecutableLoaderInstance::isInitialized())
    return UnderlyingHipUnregisterFatBinaryFn(Modules);
}

template <size_t Idx>
decltype(hsa_iterate_agents)
    *ToolExecutableLoaderInstance<Idx>::UnderlyingHsaIterateAgentsFn = nullptr;

template <size_t Idx>
decltype(hsa_executable_get_info)
    *ToolExecutableLoaderInstance<Idx>::UnderlyingHsaExecutableGetInfoFn =
        nullptr;

template <size_t Idx>
decltype(hsa_executable_create_alt)
    *ToolExecutableLoaderInstance<Idx>::UnderlyingHsaExecutableCreateAltFn =
        nullptr;

template <size_t Idx>
decltype(hsa_code_object_reader_create_from_memory)
    *ToolExecutableLoaderInstance<
        Idx>::UnderlyingHsaCodeObjectReaderCreateFromMemoryFn = nullptr;

template <size_t Idx>
decltype(hsa_executable_load_agent_code_object) *ToolExecutableLoaderInstance<
    Idx>::UnderlyingHsaExecutableLoadAgentCodeObjectFn = nullptr;

template <size_t Idx>
decltype(hsa_code_object_reader_destroy)
    *ToolExecutableLoaderInstance<Idx>::UnderlyingHsaCodeObjectReaderDestroyFn =
        nullptr;

template <size_t Idx>
decltype(hsa_executable_agent_global_variable_define)
    *ToolExecutableLoaderInstance<
        Idx>::UnderlyingHsaExecutableAgentGlobalVariableDefineFn = nullptr;

template <size_t Idx>
decltype(hsa_executable_freeze)
    *ToolExecutableLoaderInstance<Idx>::UnderlyingHsaExecutableFreezeFn =
        nullptr;

template <size_t Idx>
decltype(hsa_executable_destroy) *
    ToolExecutableLoaderInstance<Idx>::UnderlyingHsaExecutableDestroy = nullptr;

template <size_t Idx>
decltype(hsa_executable_symbol_get_info)
    *ToolExecutableLoaderInstance<Idx>::UnderlyingHsaSymbolGetInfoFn = nullptr;

template <size_t Idx>
decltype(hsa_executable_iterate_agent_symbols) *ToolExecutableLoaderInstance<
    Idx>::UnderlyingHsaExecutableIterateAgentSymbolsFn = nullptr;

template <size_t Idx>
decltype(hsa_executable_get_symbol_by_name) *ToolExecutableLoaderInstance<
    Idx>::UnderlyingHsaExecutableGetSymbolByNameFn = nullptr;

template <size_t Idx>
decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
    *ToolExecutableLoaderInstance<Idx>::HsaVenAmdLoaderExecutableIterateLCOsFn =
        nullptr;

template <size_t Idx>
decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
    *ToolExecutableLoaderInstance<Idx>::HsaVenAmdLoaderLCOGetInfoFn = nullptr;

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

}; // namespace luthier

#endif
