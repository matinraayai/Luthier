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
/// This file implements Luthier's Tool Executable Manager Singleton, and
/// instrumentation modules which are passed to the \c CodeGenerator.
//===----------------------------------------------------------------------===//
#include "luthier/runtime/ToolExecutableLoader.h"
#include "common/Log.hpp"
#include "luthier/consts.h"
#include "luthier/hsa/Agent.h"
#include "luthier/hsa/CodeObjectReader.h"
#include "luthier/hsa/ExecutableSymbol.h"
#include "luthier/hsa/HsaApiTableInterceptor.h"
#include "luthier/hsa/LoadedCodeObject.h"
#include "luthier/hsa/hsa.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <vector>

namespace luthier {

llvm::Error ToolExecutableLoader::registerIfHipLoadedInstrumentationModule(
    hsa_loaded_code_object_t LCO,
    const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &LoadedCodeObjectGetInfoFun) {
  // Get the LCO storage memory
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
    /// Get the instrumentation module
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
    /// it, register it
    {
      std::lock_guard Lock(HipLoaderMutex);
      if (auto ModuleIter = SIMsByAgent.find({CUID, Agent});
          ModuleIter != SIMsByAgent.end()) {
        LUTHIER_RETURN_ON_ERROR(
            LUTHIER_ERROR_CHECK(ModuleIter->second == nullptr,
                                "HIP Instrumentation Module with CUID {0} is "
                                "being registered twice on agent {1:x}.",
                                CUID, Agent.handle));
        ModuleIter->second.reset(new HipLoadedInstrumentationModule(
            Exec, LCO, std::move(*IModuleOrErr)));
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error ToolExecutableLoader::unregisterIfHipLoadedIModuleExec(
    hsa_executable_t Exec,
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
      hsa::getExecLoadedCodeObjects(Exec, LCOIteratorFn, LCOs));
  for (const hsa_loaded_code_object_t LCO : LCOs) {
    /// Get the agent of the LCO
    hsa_agent_t Agent;
    LUTHIER_RETURN_ON_ERROR(
        hsa::getLCOAgent(LCO, LCOGetInfoFn).moveInto(Agent));

    /// Look for the IModule reserved symbol inside the executable
    std::optional<hsa_executable_symbol_t> ReservedMangedVar{std::nullopt};
    LUTHIER_RETURN_ON_ERROR(
        hsa::lookupExecutableSymbolByName(Exec, ExecSymbolLookupFn,
                                          IModuleReservedManagedVar, Agent)
            .moveInto(ReservedMangedVar));
    /// If the LCO is an instrumentation module, obtain its CUID and remove it
    /// from the SIM map (if its entry exists)
    if (ReservedMangedVar.has_value()) {

      auto ExtractCUIDFromSymbolCB = [&](hsa_executable_symbol_t S,
                                         llvm::Error &Err) -> bool {
        llvm::Expected<llvm::StringRef> SymbolNameOrErr =
            hsa::getSymbolName(S, SymbolInfoGetterFn);
        Err = SymbolNameOrErr.takeError();
        if (Err)
          return false;

        if (SymbolNameOrErr->starts_with(HipCUIDPrefix)) {
          size_t CUID = 0;
          LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
              llvm::to_integer(SymbolNameOrErr->substr(strlen(HipCUIDPrefix)),
                               CUID),
              "Failed to parse the CUID of the HIP module"));
          std::lock_guard Lock(HipLoaderMutex);
          if (SIMsByAgent.contains({CUID, Agent})) {
            SIMsByAgent.erase({CUID, Agent});
          }
          return false;
        }
        return true;
      };

      LUTHIER_RETURN_ON_ERROR(hsa::iterateSymbolsOfExecutable(
          Exec, SymbolIterFn, Agent, ExtractCUIDFromSymbolCB));
    }
  }

  return llvm::Error::success();
}
llvm::Error ToolExecutableLoader::destroyInstrumentedExecutables(
    hsa_executable_t Exec,
    decltype(hsa_executable_destroy) &ExecutableDestroyFn) {
  std::lock_guard Lock(InstrumentedExecMutex);
  const auto &[Beg, End] =
      ApplicationToInstrumentedExecutablesMap.equal_range(Exec);
  for (auto It = Beg; It != End; ++It)
    LUTHIER_RETURN_ON_ERROR(hsa::destroyExec(Exec, ExecutableDestroyFn));
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
      hsa::createExecutable(HsaExecutableCreateAltFn).moveInto(Exec));

  // Load the code objects into the executable
  hsa_code_object_reader_t Reader{};
  LUTHIER_RETURN_ON_ERROR(hsa::createCodeObjectReaderFromMemory(
                              HsaCodeObjectReaderCreateFromMemory, CodeObject)
                              .moveInto(Reader));
  hsa_loaded_code_object_t LCO{};
  LUTHIER_RETURN_ON_ERROR(
      hsa::loadAgentCodeObjectIntoExec(Exec, HsaExecutableLoadAgentCodeObjectFn,
                                       Reader, Agent)
          .moveInto(LCO));
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(hsa::freezeExec(Exec, HsaExecutableFreezeFn));

  // Destroy the code object reader
  LUTHIER_RETURN_ON_ERROR(
      hsa::destroyCodeObjectReader(Reader, HsaCodeObjectReaderDestroyFn));

  auto *Out = new DynamicallyLoadedInstrumentationModule(
      Exec, LCO, std::move(IModule), HsaExecutableDestroyFn);

  DynModules.insert(Out);

  return *Out;
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
  auto HipModule = SIMsByAgent.find({CUID->second, Agent});
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(HipModule != SIMsByAgent.end(),
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

llvm::Error ToolExecutableLoader::loadInstrumentedKernel(
    llvm::ArrayRef<uint8_t> InstrumentedElf,
    hsa_executable_symbol_t OriginalKernel,
    const llvm::StringMap<const void *> &ExternVariables,
    llvm::StringRef Preset,
    decltype(hsa_executable_create_alt) *HsaExecutableCreateAltFn) {

  // Create the executable
  auto Executable = hsa::Executable::create(HsaExecutableCreateAltFn);

  // Define the Agent allocation external variables
  auto Agent = OriginalKernel.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  LUTHIER_RETURN_ON_ERROR(Executable.takeError());
  for (const auto &[EVName, EVAddress] : ExternVariables) {
    LUTHIER_RETURN_ON_ERROR(Executable->defineExternalAgentGlobalVariable(
        hsa::GpuAgent(*Agent), EVName, EVAddress));
  }

  // Load the code objects into the executable
  auto Reader = hsa::CodeObjectReader::createFromMemory(InstrumentedElf);
  LUTHIER_RETURN_ON_ERROR(Reader.takeError());
  auto LCO = Executable->loadAgentCodeObject(*Reader, hsa::GpuAgent(*Agent));
  LUTHIER_RETURN_ON_ERROR(LCO.takeError());
  InstrumentedLCOInfo.insert({*LCO, *Reader});
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(Executable->freeze());

  // Find the original kernel in the instrumented executable
  auto OriginalSymbolName = OriginalKernel.getName();
  LUTHIER_RETURN_ON_ERROR(OriginalSymbolName.takeError());

  std::unique_ptr<hsa::LoadedCodeObjectSymbol> InstrumentedKernel{};
  llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(Executable->getLoadedCodeObjects(LCOs));
  for (const auto &LCO : LCOs) {
    LUTHIER_RETURN_ON_ERROR(
        LCO.getLoadedCodeObjectSymbolByName(*OriginalSymbolName)
            .moveInto(InstrumentedKernel));
    if (InstrumentedKernel != nullptr)
      break;
  }
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(InstrumentedKernel != nullptr,
                          "Failed to find the corresponding kernel "
                          "to {0} inside its instrumented executable."));

  auto InstrumentedKernelType = InstrumentedKernel->getType();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      InstrumentedKernelType == hsa::LoadedCodeObjectSymbol::SK_KERNEL,
      "Found the symbol associated with kernel {0} inside the instrumented "
      "executable, but it is not of type kernel.",
      *OriginalSymbolName));

  insertInstrumentedKernelIntoMap(
      llvm::unique_dyn_cast<hsa::LoadedCodeObjectKernel>(
          OriginalKernel.clone()),
      Preset,
      llvm::unique_dyn_cast<hsa::LoadedCodeObjectKernel>(
          InstrumentedKernel->clone()));

  OriginalExecutablesWithKernelsInstrumented.insert(
      hsa::Executable(llvm::cantFail(OriginalKernel.getExecutable())));

  return llvm::Error::success();
}

llvm::Error ToolExecutableLoader::loadInstrumentedExecutable(
    llvm::ArrayRef<std::pair<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>>
        InstrumentedElfs,
    llvm::StringRef Preset,
    llvm::ArrayRef<std::tuple<hsa::GpuAgent, llvm::StringRef, const void *>>
        ExternVariables) {
  // Ensure that all LCOs belong to the same executable, and their kernels
  // were not instrumented under this profile
  hsa::Executable Exec{{0}};
  for (const auto &[LCO, InstrumentedELF] : InstrumentedElfs) {
    auto LCOExec = LCO.getExecutable();
    LUTHIER_RETURN_ON_ERROR(LCOExec.takeError());
    if (Exec.hsaHandle() == 0)
      Exec = *LCOExec;
    else
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
          *LCOExec == Exec, "Requested loading of instrumented LCOs that do "
                            "not belong to the same executable."));

    llvm::SmallVector<std::unique_ptr<hsa::LoadedCodeObjectSymbol>, 4> Kernels;
    LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
    for (const auto &Kernel : Kernels) {
      if (isKernelInstrumented(
              *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(Kernel.get()),
              Preset)) {
        auto KernelName = Kernel->getName();
        LUTHIER_RETURN_ON_ERROR(KernelName.takeError());
        return LUTHIER_CREATE_ERROR("Found kernel {0} inside LCO {1:x} which "
                                    "was already instrumented "
                                    "under the preset {2}",
                                    *KernelName, LCO.hsaHandle(), Preset);
      }
    }
  }

  // Create the executable
  auto Executable = hsa::Executable::create();

  LUTHIER_RETURN_ON_ERROR(Executable.takeError());
  for (const auto &[EVAgent, EVName, EVAddress] : ExternVariables) {
    LUTHIER_RETURN_ON_ERROR(Executable->defineExternalAgentGlobalVariable(
        EVAgent, EVName, EVAddress));
  }

  // Load the code objects into the executable
  llvm::SmallVector<hsa::LoadedCodeObject, 1> InstrumentedLCOs;
  for (const auto &[OriginalLCO, InstrumentedELF] : InstrumentedElfs) {
    auto Reader = hsa::CodeObjectReader::createFromMemory(InstrumentedELF);
    LUTHIER_RETURN_ON_ERROR(Reader.takeError());
    auto Agent = OriginalLCO.getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());
    auto InstrumentedLCO = Executable->loadAgentCodeObject(*Reader, *Agent);
    LUTHIER_RETURN_ON_ERROR(InstrumentedLCO.takeError());
    InstrumentedLCOInfo.insert({*InstrumentedLCO, *Reader});
    InstrumentedLCOs.push_back(*InstrumentedLCO);
  }
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(Executable->freeze());

  // Establish a correspondence between original kernels and the instrumented
  // kernels
  for (unsigned int I = 0; I < InstrumentedElfs.size(); ++I) {
    const auto &OriginalLCO = InstrumentedElfs[I].first;
    const auto &InstrumentedLCO = InstrumentedLCOs[I];
    llvm::SmallVector<std::unique_ptr<hsa::LoadedCodeObjectSymbol>, 4>
        KernelsOfOriginalLCO;
    LUTHIER_RETURN_ON_ERROR(OriginalLCO.getKernelSymbols(KernelsOfOriginalLCO));

    for (auto &OriginalKernel : KernelsOfOriginalLCO) {
      auto OriginalKernelName = OriginalKernel->getName();
      LUTHIER_RETURN_ON_ERROR(OriginalKernelName.takeError());

      std::unique_ptr<hsa::LoadedCodeObjectSymbol> InstrumentedKernel{nullptr};
      llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
      LUTHIER_RETURN_ON_ERROR(Executable->getLoadedCodeObjects(LCOs));
      for (const auto &LCO : LCOs) {
        LUTHIER_RETURN_ON_ERROR(
            LCO.getLoadedCodeObjectSymbolByName(*OriginalKernelName)
                .moveInto(InstrumentedKernel));
        if (InstrumentedKernel != nullptr)
          break;
      }
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ERROR_CHECK(InstrumentedKernel != nullptr,
                              "Failed to find the corresponding instrumented "
                              "kernel for {0} inside "
                              "the instrumented executable.",
                              *OriginalKernelName));

      auto InstrumentedKernelType = InstrumentedKernel->getType();
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
          InstrumentedKernelType == hsa::LoadedCodeObjectSymbol::SK_KERNEL,
          "Found the corresponding instrumented symbol for kernel {0}, but "
          "the "
          "symbol is not of type kernel.",
          *OriginalKernelName));

      insertInstrumentedKernelIntoMap(
          std::move(llvm::unique_dyn_cast<hsa::LoadedCodeObjectKernel>(
              OriginalKernel)),
          Preset,
          std::move(llvm::unique_dyn_cast<hsa::LoadedCodeObjectKernel>(
              InstrumentedKernel)));
    }
  }
  OriginalExecutablesWithKernelsInstrumented.insert(Exec);
  return llvm::Error::success();
}

bool ToolExecutableLoader::isKernelInstrumented(
    const hsa::LoadedCodeObjectKernel &Kernel, llvm::StringRef Preset) const {
  return OriginalToInstrumentedKernelsMap.contains(Kernel) &&
         OriginalToInstrumentedKernelsMap.find(Kernel)->second.contains(Preset);
}

ToolExecutableLoader::~ToolExecutableLoader() {
  // By the time the Tool Executable Manager is deleted, all instrumentation
  // kernels must have been destroyed; If not, print a warning, and clean
  // up anyway
  // TODO: Fix this, again
  //  if (!InstrumentedLCOInfo.empty()) {
  //    luthier::outs()
  //        << "Tool executable manager is being destroyed while the original
  //        "
  //           "executables of its instrumented kernels are still frozen\n";
  //    llvm::DenseSet<hsa::Executable> InstrumentedExecs;
  //    for (auto &[LCO, COR] : InstrumentedLCOInfo) {
  //      auto Exec = llvm::cantFail(LCO.getExecutable());
  //      InstrumentedExecs.insert(Exec);
  //      if (COR.destroy()) {
  //        luthier::outs() << llvm::formatv(
  //            "Code object reader {0:x} of Loaded Code Object {1:x},
  //            Executable "
  //            "{2:x} got destroyed with errors.\n",
  //            COR.hsaHandle(), LCO.hsaHandle(), Exec.hsaHandle());
  //      }
  //    }
  //    for (auto &Exec : InstrumentedExecs) {
  //      if (Exec.destroy()) {
  //        luthier::outs() << llvm::formatv(
  //            "Executable {0:x} got destroyed with errors.\n",
  //            Exec.hsaHandle());
  //      }
  //    }
  //  }
  OriginalToInstrumentedKernelsMap.clear();
}

} // namespace luthier
