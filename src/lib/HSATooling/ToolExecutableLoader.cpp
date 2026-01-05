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
#include "../../../include/luthier/HSATooling/ToolExecutableLoader.h"
#include "../../../include/luthier/HSATooling/LoadedCodeObjectCache.h"
#include "../../../include/luthier/HSATooling/TargetManager.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include "luthier/HSA/hsa.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/consts.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/Transforms/Utils/Cloning.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-tool-executable-manager"

namespace luthier {

template <>
ToolExecutableLoader *Singleton<ToolExecutableLoader>::Instance{nullptr};

t___hipRegisterFunction ToolExecutableLoader::UnderlyingHipRegisterFn{nullptr};

decltype(hsa_executable_freeze)
    *ToolExecutableLoader::UnderlyingHsaExecutableFreezeFn{nullptr};

decltype(hsa_executable_destroy)
    *ToolExecutableLoader::UnderlyingHsaExecutableDestroyFn{nullptr};

void ToolExecutableLoader::hipRegisterFunctionWrapper(
    void **modules, const void *hostFunction, char *deviceFunction,
    const char *deviceName, unsigned int threadLimit, uint3 *tid, uint3 *bid,
    dim3 *blockDim, dim3 *gridDim, int *wSize) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHipRegisterFn != nullptr, "Underlying __hipRegisterFunction of "
                                          "ToolExecutableLoader is nullptr"));

  if (isInitialized()) {
    auto &TEL = ToolExecutableLoader::instance();
    // Look for kernels that serve as handles for hooks and register them with
    // the tool executable loader
    if (llvm::StringRef(deviceFunction).find(HookHandlePrefix) !=
        llvm::StringRef::npos) {
      TEL.SIM.HookHandleMap.insert(
          {hostFunction, llvm::StringRef(deviceFunction)
                             .substr(strlen(luthier::HookHandlePrefix))});
    }
  }
  UnderlyingHipRegisterFn(modules, hostFunction, deviceFunction, deviceName,
                          threadLimit, tid, bid, blockDim, gridDim, wSize);
}

hsa_status_t
ToolExecutableLoader::hsaExecutableFreezeWrapper(hsa_executable_t Executable,
                                                 const char *Options) {
  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_GENERIC_ERROR_CHECK(UnderlyingHsaExecutableFreezeFn != nullptr,
                                  "Underlying hsa_executable_freeze of "
                                  "ToolExecutableLoader is nullptr"));
  hsa_status_t Out = UnderlyingHsaExecutableFreezeFn(Executable, Options);
  if (Out != HSA_STATUS_SUCCESS)
    return Out;
  if (isInitialized()) {
    auto &TEL = instance();
    // Check if this executable is a static instrumentation module
    auto IsSIMExec =
        StaticInstrumentationModule::isStaticInstrumentationModuleExecutable(
            TEL.CoreApiSnapshot.getTable(), TEL.LoaderApiSnapshot.getTable(),
            Executable);
    LUTHIER_REPORT_FATAL_ON_ERROR(IsSIMExec.takeError());
    if (*IsSIMExec) {
      LUTHIER_REPORT_FATAL_ON_ERROR(TEL.SIM.registerExecutable(Executable));
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "Executable with handle {0:x} was registered as a static "
                     "instrumentation module.\n",
                     Executable.handle));
    }
  }
  return Out;
}

hsa_status_t
ToolExecutableLoader::hsaExecutableDestroyWrapper(hsa_executable_t Executable) {

  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_GENERIC_ERROR_CHECK(UnderlyingHsaExecutableDestroyFn != nullptr,
                                  "Underlying hsa_executable_destroy of "
                                  "ToolExecutableLoader is nullptr"));
  if (isInitialized()) {
    auto &TEL = instance();
    // Check if this belongs to the static instrumentation module
    // If so, then unregister it from the static module
    auto IsSIMExec =
        StaticInstrumentationModule::isStaticInstrumentationModuleExecutable(
            TEL.CoreApiSnapshot.getTable(), TEL.LoaderApiSnapshot.getTable(),
            Executable);
    LUTHIER_REPORT_FATAL_ON_ERROR(IsSIMExec.takeError());
    if (*IsSIMExec) {
      LUTHIER_REPORT_FATAL_ON_ERROR(TEL.SIM.unregisterExecutable(Executable));
      return UnderlyingHsaExecutableDestroyFn(Executable);
    }
    // Check if this executable has been instrumented before. If so,
    // destroy the instrumented versions of this executable, and remove its
    // entries from the internal maps
    if (TEL.OriginalExecutablesWithKernelsInstrumented.contains(Executable)) {
      // 1. Find all instrumented versions of each kernel of Exec
      // 2. For each instrumented kernel, get its executable and insert it in
      // InstrumentedVersionsOfExecutable to be dealt with later
      // 3. Remove instrumented entries of the original kernel
      llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
      LUTHIER_REPORT_FATAL_ON_ERROR(hsa::executableGetLoadedCodeObjects(
          TEL.LoaderApiSnapshot.getTable(), Executable, LCOs));
      const auto &COC = hsa::LoadedCodeObjectCache::instance();
      for (const auto &LCO : LCOs) {
        llvm::SmallVector<std::unique_ptr<hsa::LoadedCodeObjectSymbol>, 4>
            Kernels;
        LUTHIER_REPORT_FATAL_ON_ERROR(COC.getKernelSymbols(LCO, Kernels));
        for (const auto &Symbol : Kernels) {
          const auto &Kernel =
              *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(Symbol.get());
          hsa_executable_symbol_t ExecSymbol = *Kernel.getExecutableSymbol();
          if (TEL.OriginalToInstrumentedKernelsMap.contains(ExecSymbol)) {
            TEL.OriginalToInstrumentedKernelsMap.erase(
                TEL.OriginalToInstrumentedKernelsMap.find(ExecSymbol));
          }
        }
      }
      // clean up all instrumented versions of Exec

      for (auto &InstrumentedExec :
           TEL.OriginalExecutablesWithKernelsInstrumented[Executable]) {
        LUTHIER_REPORT_FATAL_ON_ERROR(hsa::executableDestroy(
            TEL.CoreApiSnapshot.getTable(), InstrumentedExec));
      }
    }
  }
  return UnderlyingHsaExecutableDestroyFn(Executable);
}

ToolExecutableLoader::ToolExecutableLoader(
    const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot,
    const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
        &LoaderApiSnapshot,
    const hsa::LoadedCodeObjectCache &COC,
    const amdgpu::hsamd::MetadataParser &MDParser, llvm::Error &Err)
    : Singleton<ToolExecutableLoader>(), CoreApiSnapshot(CoreApiSnapshot),
      LoaderApiSnapshot(LoaderApiSnapshot), COC(COC), SIM(LoaderApiSnapshot),
      MDParser(MDParser) {

  CoreApiWrapperInstaller = std::make_unique<
      rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>(
      Err,
      std::make_tuple(&::CoreApiTable::hsa_executable_freeze_fn,
                      std::ref(UnderlyingHsaExecutableFreezeFn),
                      hsaExecutableFreezeWrapper),
      std::make_tuple(&::CoreApiTable::hsa_executable_destroy_fn,
                      std::ref(UnderlyingHsaExecutableDestroyFn),
                      hsaExecutableDestroyWrapper));
  if (Err)
    return;

  HipCompilerWrapperInstaller =
      std::make_unique<rocprofiler::HipCompilerApiTableWrapperInstaller>(
          Err,
          std::make_tuple(&::HipCompilerDispatchTable::__hipRegisterFunction_fn,
                          std::ref(UnderlyingHipRegisterFn),
                          hipRegisterFunctionWrapper));
  if (Err)
    return;
};

llvm::Expected<
    std::pair<hsa_executable_symbol_t, const amdgpu::hsamd::Kernel::Metadata &>>
ToolExecutableLoader::getInstrumentedKernel(
    hsa_executable_symbol_t OriginalKernel, llvm::StringRef Preset) const {
  const auto CoreApiTable = CoreApiSnapshot.getTable();
  llvm::Expected<hsa_symbol_kind_t> SymTypeOrErr =
      hsa::executableSymbolGetType(CoreApiTable, OriginalKernel);
  LUTHIER_RETURN_ON_ERROR(SymTypeOrErr.takeError());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      *SymTypeOrErr == HSA_SYMBOL_KIND_KERNEL, "Symbol is not a kernel"));

  // First make sure the OriginalKernel has instrumented entries
  auto InstrumentedKernelsIt =
      OriginalToInstrumentedKernelsMap.find(OriginalKernel);
  if (InstrumentedKernelsIt == OriginalToInstrumentedKernelsMap.end()) {
    auto KernelName =
        hsa::executableSymbolGetName(CoreApiTable, OriginalKernel);
    LUTHIER_RETURN_ON_ERROR(KernelName.takeError());
    return llvm::make_error<GenericLuthierError>(llvm::formatv(
        "Failed to find any instrumented version of kernel {0}.", *KernelName));
  }
  // Then make sure the original kernel was instrumented under the given Preset,
  // and then return the instrumented version
  auto InstrumentedKernelIt = InstrumentedKernelsIt->second.find(Preset);
  if (InstrumentedKernelIt == InstrumentedKernelsIt->second.end()) {
    auto KernelName =
        hsa::executableSymbolGetName(CoreApiTable, OriginalKernel);
    LUTHIER_RETURN_ON_ERROR(KernelName.takeError());
    return llvm::make_error<GenericLuthierError>(
        llvm::formatv("Failed to find any instrumented version of "
                      "kernel {0} under preset {1}.",
                      *KernelName, Preset));
  }
  hsa_executable_symbol_t Out = InstrumentedKernelIt->second;
  const auto &MD = *InstrumentedKernelMetadata.find(Out)->second;
  return std::make_pair(Out, MD);
}

llvm::Error ToolExecutableLoader::loadInstrumentedKernel(
    llvm::ArrayRef<uint8_t> InstrumentedElf,
    const hsa::LoadedCodeObjectKernel &OriginalKernel, llvm::StringRef Preset,
    const llvm::StringMap<const void *> &ExternVariables) {
  std::lock_guard Lock(Mutex);
  // Ensure this kernel was not instrumented under this preset
  if (isKernelInstrumented(OriginalKernel, Preset)) {
    auto OriginalKernelName = OriginalKernel.getName();
    LUTHIER_RETURN_ON_ERROR(OriginalKernelName.takeError());
    return llvm::make_error<GenericLuthierError>(
        llvm::formatv("Kernel {0} is already instrumented under preset {1}.",
                      *OriginalKernelName, Preset));
  }

  auto CoreApiTable = CoreApiSnapshot.getTable();

  // Create the executable
  auto Executable = hsa::executableCreate(CoreApiTable);
  LUTHIER_RETURN_ON_ERROR(Executable.takeError());

  // Define the Agent allocation external variables
  auto Agent = OriginalKernel.getAgent(LoaderApiSnapshot.getTable());
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  for (const auto &[EVName, EVAddress] : ExternVariables) {
    LUTHIER_RETURN_ON_ERROR(hsa::executableDefineExternalAgentGlobalVariable(
        CoreApiSnapshot.getTable(), *Executable, *Agent, EVName, EVAddress));
  }

  // Load the code objects into the executable
  auto Reader =
      hsa::codeObjectReaderCreateFromMemory(CoreApiTable, InstrumentedElf);
  LUTHIER_RETURN_ON_ERROR(Reader.takeError());
  auto LCO = hsa::executableLoadAgentCodeObject(CoreApiTable, *Executable,
                                                *Reader, *Agent);
  LUTHIER_RETURN_ON_ERROR(LCO.takeError());
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(hsa::executableFreeze(CoreApiTable, *Executable));

  // Find the original kernel in the instrumented executable
  std::string OriginalSymbolName;
  LUTHIER_RETURN_ON_ERROR(
      OriginalKernel.getName().moveInto(OriginalSymbolName));
  OriginalSymbolName.append(".kd");

  auto InstrumentedKernelOrErr = hsa::executableFindFirstAgentSymbol(
      CoreApiTable, *Executable, *Agent,
      [&](hsa_executable_symbol_t Symbol) -> llvm::Expected<bool> {
        llvm::Expected<std::string> SymbolNameOrErr =
            hsa::executableSymbolGetName(CoreApiTable, Symbol);
        LUTHIER_RETURN_ON_ERROR(SymbolNameOrErr.takeError());
        return llvm::StringRef(*SymbolNameOrErr) == OriginalSymbolName;
      });
  LUTHIER_RETURN_ON_ERROR(InstrumentedKernelOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      InstrumentedKernelOrErr->has_value(),
      llvm::formatv("Failed to find the corresponding kernel "
                    "to {0} inside its instrumented executable",
                    OriginalSymbolName)));

  auto InstrumentedKernelType =
      hsa::executableSymbolGetType(CoreApiTable, **InstrumentedKernelOrErr);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      *InstrumentedKernelType == HSA_SYMBOL_KIND_KERNEL,
      llvm::formatv(
          "Found the symbol associated with kernel {0} inside the instrumented "
          "executable, but it is not of type kernel.",
          OriginalSymbolName)));

  llvm::Expected<hsa_executable_t> OriginalExecutableOrErr =
      OriginalKernel.getExecutable(LoaderApiSnapshot.getTable());
  LUTHIER_RETURN_ON_ERROR(OriginalExecutableOrErr.takeError());

  /// Parse the metadata
  auto ObjFile =
      object::AMDGCNObjectFile::createAMDGCNObjectFile(InstrumentedElf);
  LUTHIER_RETURN_ON_ERROR(ObjFile.takeError());
  std::unique_ptr<llvm::msgpack::Document> InstrumentedExecMDDoc;
  LUTHIER_RETURN_ON_ERROR(
      (*ObjFile)->getMetadataDocument().moveInto(InstrumentedExecMDDoc));

  std::unique_ptr<amdgpu::hsamd::Kernel::Metadata> MD;

  LUTHIER_RETURN_ON_ERROR(
      MDParser.parseKernelMetadata(*InstrumentedExecMDDoc, OriginalSymbolName)
          .moveInto(MD));

  InstrumentedKernelMetadata.insert({**InstrumentedKernelOrErr, std::move(MD)});

  insertInstrumentedKernelIntoMap(*OriginalExecutableOrErr,
                                  *OriginalKernel.getExecutableSymbol(), Preset,
                                  *Executable, **InstrumentedKernelOrErr);
  LUTHIER_RETURN_ON_ERROR(hsa::codeObjectReaderDestroy(*Reader, CoreApiTable));
  return llvm::Error::success();
}

bool ToolExecutableLoader::isKernelInstrumented(
    const hsa::LoadedCodeObjectKernel &Kernel, llvm::StringRef Preset) const {
  return OriginalToInstrumentedKernelsMap.contains(
             *Kernel.getExecutableSymbol()) &&
         OriginalToInstrumentedKernelsMap.find(*Kernel.getExecutableSymbol())
             ->second.contains(Preset);
}

ToolExecutableLoader::~ToolExecutableLoader() {
  // By the time the Tool Executable Manager is deleted, all instrumentation
  // kernels must have been destroyed; If not, print a warning, and clean
  // up anyway
  // TODO: Fix this, again
  //  if (!InstrumentedKernelMetadata.empty()) {
  //    luthier::outs()
  //        << "Tool executable manager is being destroyed while the original "
  //           "executables of its instrumented kernels are still frozen\n";
  //    llvm::DenseSet<hsa::Executable> InstrumentedExecs;
  //    for (auto &[LCO, COR] : InstrumentedKernelMetadata) {
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
