//===-- ToolExecutableManager.cpp - Luthier Tool Executable Manager -------===//
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
/// Implements Luthier's HSA Tool Executable Loader interface and its concrete
/// singleton.
//===----------------------------------------------------------------------===//
#include <llvm/ADT/SmallVector.h>
#include <luthier/consts.h>
#include <luthier/hsa/Agent.h>
#include <luthier/hsa/CodeObjectReader.h>
#include <luthier/hsa/ExecutableSymbol.h>
#include <luthier/hsa/HsaApiTableInterceptor.h>
#include <luthier/hsa/LoadedCodeObject.h>
#include <luthier/hsa/ToolExecutableLoader.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <vector>

namespace luthier::hsa {

llvm::Error ToolExecutableLoader::registerIfHipLoadedInstrumentationModule(
    const hsa_loaded_code_object_t LCO,
    const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &LoadedCodeObjectGetInfoFun) {
  // Get the LCO's storage memory
  llvm::ArrayRef<uint8_t> StorageMemory;
  LUTHIER_RETURN_ON_ERROR(
      hsa::getLCOStorageMemory(LCO, LoadedCodeObjectGetInfoFun)
          .moveInto(StorageMemory));

  // Check if the LCO is indeed a loaded instrumentation module
  bool IsIModule{false};
  LUTHIER_RETURN_ON_ERROR(
      InstrumentationModule::isInstrumentationModule(StorageMemory)
          .moveInto(IsIModule));

  if (IsIModule) {
    /// Construct the instrumentation module
    llvm::Expected<std::unique_ptr<InstrumentationModule>> IModuleOrErr =
        InstrumentationModule::get(StorageMemory);
    LUTHIER_RETURN_ON_ERROR(IModuleOrErr.takeError());
    /// Get the executable of the LCO
    hsa_executable_t Exec;
    LUTHIER_RETURN_ON_ERROR(
        hsa::getLCOExecutable(LCO, LoadedCodeObjectGetInfoFun).moveInto(Exec));
    /// Get the CUID of the Module
    size_t CUID = IModuleOrErr.get()->getCUID();
    /// Get the agent of the LCO
    hsa_agent_t Agent;
    LUTHIER_RETURN_ON_ERROR(
        hsa::getLCOAgent(LCO, LoadedCodeObjectGetInfoFun).moveInto(Agent));
    /// If the CUID and the agent of the module indicate that we were expecting
    /// it, register it; Otherwise, the LCO must have been a dynamically loaded
    /// instrumentation module by another tool executable loader instance
    {
      std::lock_guard Lock(HipLoaderMutex);
      if (const auto ModuleIter = HipLoadedIMsPerAgent.find({CUID, Agent});
          ModuleIter != HipLoadedIMsPerAgent.end()) {
        // Confirm that we didn't load the same module on the same agent twice.
        LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
            ModuleIter->second == nullptr,
            llvm::formatv("HIP Instrumentation Module with CUID {0} is "
                          "being registered twice on agent {1:x}.",
                          CUID, Agent.handle)));
        // Finally, register the newly constructed loaded instrumentation
        // module
        ModuleIter->second.reset(new HipLoadedInstrumentationModule(
            Exec, LCO, std::move(*IModuleOrErr)));
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error ToolExecutableLoader::unregisterIfHipLoadedIModuleExec(
    const hsa_executable_t Exec,
    const decltype(hsa_executable_get_symbol_by_name) &ExecSymbolLookupFn,
    const decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
        &LCOIteratorFn,
    const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &LCOGetInfoFn,
    const decltype(hsa_executable_iterate_agent_symbols) &SymbolIterFn,
    const decltype(hsa_executable_symbol_get_info) &SymbolInfoGetterFn) {
  /// Iterate over the LCOs of the executable and check if they were HIP-laoded
  /// IModules
  llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(
      hsa::executableGetLoadedCodeObjects(Exec, LCOIteratorFn, LCOs));
  for (const hsa_loaded_code_object_t LCO : LCOs) {
    /// Get the agent of the LCO
    hsa_agent_t Agent;
    LUTHIER_RETURN_ON_ERROR(
        hsa::getLCOAgent(LCO, LCOGetInfoFn).moveInto(Agent));

    /// Look for the IModule reserved symbol inside the executable
    std::optional<hsa_executable_symbol_t> ReservedMangedVar{std::nullopt};
    LUTHIER_RETURN_ON_ERROR(
        hsa::executableGetSymbolByName(Exec, ExecSymbolLookupFn,
                                       IModuleReservedManagedVar, Agent)
            .moveInto(ReservedMangedVar));
    /// If the LCO is an instrumentation module, obtain its CUID and remove it
    /// from the SIM map (if its entry exists)
    if (ReservedMangedVar.has_value()) {

      auto ExtractCUIDFromSymbolCB =
          [&](hsa_executable_symbol_t S) -> llvm::Expected<bool> {
        llvm::Expected<llvm::StringRef> SymbolNameOrErr =
            executableSymbolGetName(S, SymbolInfoGetterFn);
        LUTHIER_RETURN_ON_ERROR(SymbolNameOrErr.takeError());

        if (SymbolNameOrErr->starts_with(HipCUIDPrefix)) {
          size_t CUID = 0;
          LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
              llvm::to_integer(SymbolNameOrErr->substr(strlen(HipCUIDPrefix)),
                               CUID),
              "Failed to parse the CUID of the HIP module"));
          std::lock_guard Lock(HipLoaderMutex);
          if (HipLoadedIMsPerAgent.contains({CUID, Agent})) {
            HipLoadedIMsPerAgent.erase({CUID, Agent});
          }
          return true;
        }
        return false;
      };

      LUTHIER_RETURN_ON_ERROR(
          hsa::executableFindFirstAgentSymbol(Exec, SymbolIterFn, Agent,
                                              ExtractCUIDFromSymbolCB)
              .takeError());
    }
  }
  return llvm::Error::success();
}
llvm::Error ToolExecutableLoader::destroyInstrumentedCopiesOfExecutable(
    const hsa_executable_t Exec,
    decltype(hsa_executable_destroy) &ExecutableDestroyFn) {
  std::lock_guard Lock(InstrumentedExecMutex);
  const auto &[Beg, End] =
      ApplicationToInstrumentedExecutablesMap.equal_range(Exec);
  // Early exit if the executable doesn't have any instrumented copies
  if (Beg == ApplicationToInstrumentedExecutablesMap.end())
    return llvm::Error::success();
  for (auto It = Beg; It != End; ++It)
    LUTHIER_RETURN_ON_ERROR(hsa::executableDestroy(Exec, ExecutableDestroyFn));
  ApplicationToInstrumentedExecutablesMap.erase(Exec);
  return llvm::Error::success();
}

llvm::Expected<DynamicallyLoadedInstrumentationModule &>
ToolExecutableLoader::loadDynamicIModule(
    std::vector<uint8_t> CodeObject, hsa_agent_t Agent,
    decltype(hsa_executable_create_alt) &HsaExecutableCreateAltFn,
    decltype(hsa_code_object_reader_create_from_memory)
        &HsaCodeObjectReaderCreateFromMemory,
    decltype(hsa_executable_load_agent_code_object)
        &HsaExecutableLoadAgentCodeObjectFn,
    decltype(hsa_executable_freeze) &HsaExecutableFreezeFn,
    decltype(hsa_code_object_reader_destroy) &HsaCodeObjectReaderDestroyFn,
    decltype(hsa_executable_destroy) &HsaExecutableDestroyFn) {
  std::lock_guard Lock(DynamicModuleMutex);

  std::unique_ptr<InstrumentationModule> IModule;
  LUTHIER_RETURN_ON_ERROR(
      InstrumentationModule::get(std::move(CodeObject)).moveInto(IModule));

  // Create the executable
  hsa_executable_t Exec{};
  LUTHIER_RETURN_ON_ERROR(
      hsa::executableCreate(HsaExecutableCreateAltFn).moveInto(Exec));

  // Load the code objects into the executable
  hsa_code_object_reader_t Reader{};
  LUTHIER_RETURN_ON_ERROR(hsa::codeObjectReaderCreateFromMemory(
                              HsaCodeObjectReaderCreateFromMemory, CodeObject)
                              .moveInto(Reader));
  hsa_loaded_code_object_t LCO{};
  LUTHIER_RETURN_ON_ERROR(
      hsa::executableLoadAgentCodeObject(
          Exec, HsaExecutableLoadAgentCodeObjectFn, Reader, Agent)
          .moveInto(LCO));
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(hsa::executableFreeze(Exec, HsaExecutableFreezeFn));

  // Destroy the code object reader
  LUTHIER_RETURN_ON_ERROR(
      hsa::codeObjectReaderDestroy(Reader, HsaCodeObjectReaderDestroyFn));

  auto *Out = new DynamicallyLoadedInstrumentationModule(
      Exec, LCO, std::move(IModule), HsaExecutableDestroyFn);

  DynModules.insert(Out);

  return *Out;
}

llvm::Expected<hsa_executable_t>
ToolExecutableLoader::loadInstrumentedCodeObject(
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
    decltype(hsa_code_object_reader_destroy) &HsaCodeObjectReaderDestroyFn) {

  /// Get the Agent of the original LCO
  hsa_agent_t Agent;
  LUTHIER_RETURN_ON_ERROR(
      hsa::getLCOAgent(OriginalLoadedCodeObject, LCOInfoQueryFn)
          .moveInto(Agent));
  /// Get the executable of the original LCO
  hsa_executable_t OriginalExecutable;
  LUTHIER_RETURN_ON_ERROR(
      hsa::getLCOExecutable(OriginalLoadedCodeObject, LCOInfoQueryFn)
          .moveInto(OriginalExecutable));

  // Create the instrumented executable
  hsa_executable_t InstrumentedExec{};
  LUTHIER_RETURN_ON_ERROR(hsa::executableCreate(HsaExecutableCreateAltFn)
                              .moveInto(InstrumentedExec));

  // Define the external variables
  for (const auto &[EVName, EVAddress] : ExternVariables) {
    LUTHIER_RETURN_ON_ERROR(hsa::executableDefineExternalAgentGlobalVariable(
        InstrumentedExec, HsaExecutableAgentGlobalVariableDefineFn, Agent,
        EVName, EVAddress));
  }

  // Load the code objects into the executable
  hsa_code_object_reader_t Reader{};
  LUTHIER_RETURN_ON_ERROR(
      hsa::codeObjectReaderCreateFromMemory(HsaCodeObjectReaderCreateFromMemory,
                                            InstrumentedElf)
          .moveInto(Reader));
  hsa_loaded_code_object_t InstrumentedLCO{};
  LUTHIER_RETURN_ON_ERROR(
      hsa::executableLoadAgentCodeObject(
          InstrumentedExec, HsaExecutableLoadAgentCodeObjectFn, Reader, Agent)
          .moveInto(InstrumentedLCO));
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(
      hsa::executableFreeze(InstrumentedExec, HsaExecutableFreezeFn));

  // Destroy the code object reader
  LUTHIER_RETURN_ON_ERROR(
      hsa::codeObjectReaderDestroy(Reader, HsaCodeObjectReaderDestroyFn));

  std::lock_guard Lock(InstrumentedExecMutex);
  ApplicationToInstrumentedExecutablesMap.insert(
      {OriginalExecutable, InstrumentedExec});

  return InstrumentedExec;
}

llvm::Expected<std::pair<const HipLoadedInstrumentationModule &, std::string>>
ToolExecutableLoader::getHipLoadedHook(void *HostHandle,
                                       hsa_agent_t Agent) const {
  std::lock_guard Lock(HipLoaderMutex);
  // Get the binary info of the func
  auto FuncInfo = HipFunctions.find(HostHandle);
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(FuncInfo != HipFunctions.end(),
                          "Failed to find the HIP module information for HIP "
                          "function handle {0:x}",
                          HostHandle));
  // Get the cuid associated with the HIP module
  auto CUID = HipModuleCUIDs.find(FuncInfo->second.FatBinaryModuleInfo);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      CUID != HipModuleCUIDs.end(),
      "Failed to find the CUID associated with module {0:x}, hook name {1}",
      FuncInfo->second.FatBinaryModuleInfo, FuncInfo->second.Name));
  // Find the Hip Loaded module
  auto HipModule = HipLoadedIMsPerAgent.find({CUID->second, Agent});
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(HipModule != HipLoadedIMsPerAgent.end(),
                          "Failed to find the HIP Loaded module associated "
                          "with CUID {0} and Agent {1:x}.",
                          CUID->second, Agent.handle));
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(HipModule->second != nullptr,
                          "The HIP Loaded module associated with CUID {0} and "
                          "Agent {1:x} is nullptr.",
                          CUID->second, Agent.handle));

  return std::make_pair(*HipModule->second, FuncInfo->second.Name);
}

} // namespace luthier::hsa
