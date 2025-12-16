//===-- InstrumentationModule.cpp - Luthier Instrumentation Module --------===//
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
/// This file implements Luthier's Instrumentation Module and its variants.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/InstrumentationModule.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include "luthier/HSA/LoadedCodeObjectCache.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/consts.h"
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TimeProfiler.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-instrumentation-module"

//===----------------------------------------------------------------------===//
// Instrumentation Module Pre-processing functions (Shared between all
// Instrumentation Module sub-classes)
//===----------------------------------------------------------------------===//
namespace luthier {

static constexpr const char *BCSectionName = ".llvmbc";

/// Finds the ".llvmbc" section of the <tt>LCO</tt>'s host storage ELF
/// \param LCO the \c hsa::LoadedCodeObject containing the bitcode
/// \return an \c llvm::ArrayRef to the contents of the ".llvmbc" section
/// if found, or an \c llvm::Error if the bitcode was not found
static llvm::Expected<llvm::ArrayRef<char>>
getBitcodeBufferOfLCO(hsa_loaded_code_object_t LCO) {
  llvm::Expected<luthier::object::AMDGCNObjectFile &> StorageELFOrErr =
      hsa::LoadedCodeObjectCache::instance().getAssociatedObjectFile(LCO);
  LUTHIER_RETURN_ON_ERROR(StorageELFOrErr.takeError());

  // Find the ".llvmbc" section of the ELF
  for (const llvm::object::SectionRef &Section : StorageELFOrErr->sections()) {
    auto SectionName = Section.getName();
    LUTHIER_RETURN_ON_ERROR(SectionName.takeError());
    if (*SectionName == BCSectionName) {
      auto SectionContents = Section.getContents();
      LUTHIER_RETURN_ON_ERROR(SectionContents.takeError());
      return llvm::ArrayRef(SectionContents->data(), SectionContents->size());
    }
  }
  return llvm::make_error<GenericLuthierError>(llvm::formatv(
      "Failed to find the bitcode section of LCO {0:x}.", LCO.handle));
}

/// \return the CUID of the \p LCO
static llvm::Expected<llvm::StringRef>
getCUIDOfLCO(hsa_loaded_code_object_t LCO) {
  const auto &COC = hsa::LoadedCodeObjectCache::instance();
  llvm::SmallVector<std::unique_ptr<hsa::LoadedCodeObjectSymbol>, 4> Variables;
  LUTHIER_RETURN_ON_ERROR(COC.getVariableSymbols(LCO, Variables));
  for (const auto &Var : Variables) {
    auto VarName = Var->getName();
    LUTHIER_RETURN_ON_ERROR(VarName.takeError());
    if (VarName->starts_with(luthier::HipCUIDPrefix)) {
      return VarName->substr(strlen(HipCUIDPrefix));
    }
  }
  return llvm::make_error<GenericLuthierError>(
      llvm::formatv("Could not find a CUID for the LCO {0:x}.", LCO.handle));
}

//===----------------------------------------------------------------------===//
// Instrumentation Module Implementation
//===----------------------------------------------------------------------===//

llvm::Expected<std::unique_ptr<llvm::Module>>
StaticInstrumentationModule::readBitcodeIntoContext(llvm::LLVMContext &Ctx,
                                                    hsa_agent_t Agent) const {
  llvm::TimeTraceScope Scope("Static Module LLVM Bitcode Loading");
  std::shared_lock Lock(Mutex);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      PerAgentBitcodeBufferMap.contains(Agent),
      llvm::formatv("Failed to find the static instrumentation module "
                    "bitcode for agent {0:x}",
                    Agent.handle)));
  auto BCBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::toStringRef(PerAgentBitcodeBufferMap.at(Agent)), "", false);
  return llvm::parseBitcodeFile(*BCBuffer, Ctx);
}

//===----------------------------------------------------------------------===//
// Static Instrumentation Module Implementation
//===----------------------------------------------------------------------===//

llvm::Error
StaticInstrumentationModule::registerExecutable(hsa_executable_t Exec) {
  std::unique_lock Lock(Mutex);
  // Since static instrumentation modules are generated with HIP, we can
  // safely assume each Executable has a single LCO for now. Here we assert this
  // is indeed the case
  llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
  llvm::cantFail(hsa::executableGetLoadedCodeObjects(LoaderSnapshot.getTable(),
                                                     Exec, LCOs));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      LCOs.size() == 1,
      llvm::formatv(
          "The Luthier instrumentation module executable should only have a "
          "single LCO; Number of LCOs found inside the executable: {0}.",
          LCOs.size())));
  bool FirstTimeInit = PerAgentModuleExecutables.empty();
  if (PerAgentModuleExecutables.empty()) {
    // Record the CUID of the module
    LUTHIER_RETURN_ON_ERROR(getCUIDOfLCO(LCOs[0]).moveInto(CUID));
  } else {
    // Ensure the CUID of the Executable and the Module match
    auto ExecCUID = getCUIDOfLCO(LCOs[0]);
    LUTHIER_RETURN_ON_ERROR(ExecCUID.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        *ExecCUID == CUID,
        llvm::formatv("Error registering the executable {0:x} with the "
                      "static instrumentation module; The executable "
                      "CUID {1} does not match the CUID of module {2}.",
                      Exec.handle, *ExecCUID, CUID)));
  }
  // Ensure this executable's agent doesn't already have another copy of this
  // executable loaded on it, then insert its information into the map
  auto Agent =
      hsa::loadedCodeObjectGetAgent(LoaderSnapshot.getTable(), LCOs[0]);
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !PerAgentModuleExecutables.contains(*Agent),
      llvm::formatv("Agent {0:x} already has an instrumentation module "
                    "executable associated with it.",
                    Agent->handle)));
  PerAgentModuleExecutables.insert({*Agent, Exec});
  // Record the LCO's bitcode buffer for instrumentation use on its agent
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !PerAgentBitcodeBufferMap.contains(*Agent),
      llvm::formatv("Agent {0:x} already has a static instrumentation "
                    "module bitcode associated with it.",
                    Agent->handle)));
  auto BitcodeBuffer = getBitcodeBufferOfLCO(LCOs[0]);
  LUTHIER_RETURN_ON_ERROR(BitcodeBuffer.takeError());
  PerAgentBitcodeBufferMap.insert({*Agent, *BitcodeBuffer});

  // Populate the variables of this executable on its agent as well as the
  // global variable list
  auto &SymbolMap =
      PerAgentGlobalVariables
          .try_emplace(
              *Agent,
              llvm::StringMap<std::unique_ptr<hsa::LoadedCodeObjectVariable>>{})
          .first->second;
  llvm::SmallVector<std::unique_ptr<hsa::LoadedCodeObjectSymbol>, 4>
      LCOGlobalVariables;
  const auto &COC = hsa::LoadedCodeObjectCache::instance();
  LUTHIER_RETURN_ON_ERROR(COC.getVariableSymbols(LCOs[0], LCOGlobalVariables));
  for (auto &GVSymbol : LCOGlobalVariables) {
    auto GVName = GVSymbol->getName();
    LUTHIER_RETURN_ON_ERROR(GVName.takeError());
    if (FirstTimeInit)
      GlobalVariables.push_back(std::string(*GVName));
    SymbolMap.insert(
        {*GVName,
         std::move(
             llvm::unique_dyn_cast<hsa::LoadedCodeObjectVariable>(GVSymbol))});
  }

  return llvm::Error::success();
}

llvm::Error
StaticInstrumentationModule::unregisterExecutable(hsa_executable_t Exec) {
  std::unique_lock Lock(Mutex);
  llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(hsa::executableGetLoadedCodeObjects(
      LoaderSnapshot.getTable(), Exec, LCOs));
  // There's only a single LCO in HIP FAT binaries. Get its agent.
  auto Agent =
      hsa::loadedCodeObjectGetAgent(LoaderSnapshot.getTable(), LCOs[0]);
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  // Remove the agent from the variable list and the executable list
  PerAgentGlobalVariables.erase(*Agent);
  PerAgentModuleExecutables.erase(*Agent);
  PerAgentBitcodeBufferMap.erase(*Agent);
  // If no copies of this module is present on any of the agents, then
  // the lifetime of the static module has ended. Perform a cleanup
  if (PerAgentModuleExecutables.empty()) {
    PerAgentBitcodeBufferMap.clear();
    HookHandleMap.clear();
    GlobalVariables.clear();
  }
  return llvm::Error::success();
}

llvm::Expected<const hsa::LoadedCodeObjectVariable *>
StaticInstrumentationModule::getLCOGlobalVariableOnAgentNoLock(
    llvm::StringRef GVName, hsa_agent_t Agent) const {
  auto VariableSymbolMapIt = PerAgentGlobalVariables.find(Agent);

  if (VariableSymbolMapIt != PerAgentGlobalVariables.end()) {
    auto VariableSymbolIt = VariableSymbolMapIt->second.find(GVName);
    // Ensure the variable name is indeed in the map
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        VariableSymbolIt != VariableSymbolMapIt->second.end(),
        llvm::formatv(
            "Failed to find the symbol associated with global variable {0} on "
            "agent {1:x}",
            GVName, Agent.handle)));
    return VariableSymbolIt->second.get();
  } else {
    // If the agent is not in the map, then it probably wasn't loaded on the
    // device
    return nullptr;
  }
}

llvm::Expected<const hsa::LoadedCodeObjectVariable *>
StaticInstrumentationModule::getLCOGlobalVariableOnAgent(
    llvm::StringRef GVName, hsa_agent_t Agent) const {
  std::shared_lock Lock(Mutex);
  return getLCOGlobalVariableOnAgentNoLock(GVName, Agent).get();
}

llvm::Expected<llvm::StringRef>
StaticInstrumentationModule::convertHookHandleToHookName(
    const void *Handle) const {
  std::shared_lock Lock(Mutex);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      HookHandleMap.contains(Handle),
      llvm::formatv("Failed to find the hook name for handle {0:x}.", Handle)));
  return HookHandleMap.at(Handle);
}

llvm::Expected<bool>
StaticInstrumentationModule::isStaticInstrumentationModuleExecutable(
    const hsa::ApiTableContainer<::CoreApiTable> &CoreApi,
    const hsa_ven_amd_loader_1_03_pfn_t &LoaderApi, hsa_executable_t Exec) {
  /// Get all the loaded code objects in the executable
  llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(
      hsa::executableGetLoadedCodeObjects(LoaderApi, Exec, LCOs));

  for (const auto &LCO : LCOs) {
    /// Get the agent of the loaded code object
    llvm::Expected<hsa_agent_t> AgentOrErr =
        hsa::loadedCodeObjectGetAgent(LoaderApi, LCO);
    LUTHIER_RETURN_ON_ERROR(AgentOrErr.takeError());
    auto LuthierReservedSymbolIfFoundOrErr = hsa::executableGetSymbolByName(
        CoreApi, Exec, ReservedManagedVar, *AgentOrErr);
    LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbolIfFoundOrErr.takeError());

    if (LuthierReservedSymbolIfFoundOrErr->has_value()) {
      return true;
    }
  }
  return false;
}

llvm::Expected<std::optional<luthier::address_t>>
StaticInstrumentationModule::getGlobalVariablesLoadedOnAgent(
    llvm::StringRef GVName, hsa_agent_t Agent) const {
  std::shared_lock Lock(Mutex);
  auto GVSymbol = getLCOGlobalVariableOnAgentNoLock(GVName, Agent);
  LUTHIER_RETURN_ON_ERROR(GVSymbol.takeError());
  if (*GVSymbol == nullptr) {
    return std::nullopt;
  } else
    return (*GVSymbol)->getLoadedSymbolAddress(LoaderSnapshot.getTable());
}

} // namespace luthier