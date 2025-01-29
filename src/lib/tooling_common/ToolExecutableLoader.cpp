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
#include "tooling_common/ToolExecutableLoader.hpp"
#include "tooling_common/TargetManager.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <luthier/Consts.h>
#include <vector>

#include "common/Log.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "hsa/hsa.hpp"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-tool-executable-manager"

namespace luthier {

template <>
ToolExecutableLoader *Singleton<ToolExecutableLoader>::Instance{nullptr};

void ToolExecutableLoader::registerInstrumentationHookWrapper(
    const void *WrapperShadowHostPtr, const char *HookWrapperName) {
  SIM.HookHandleMap.insert(
      {WrapperShadowHostPtr, llvm::StringRef(HookWrapperName)
                                 .substr(strlen(luthier::HookHandlePrefix))});
}

llvm::Error ToolExecutableLoader::registerIfLuthierToolExecutable(
    const hsa::Executable &Exec) {
  // Check if this executable is a static instrumentation module
  auto IsSIMExec =
      StaticInstrumentationModule::isStaticInstrumentationModuleExecutable(
          Exec);
  LUTHIER_RETURN_ON_ERROR(IsSIMExec.takeError());
  if (*IsSIMExec) {
    LUTHIER_RETURN_ON_ERROR(SIM.registerExecutable(Exec));
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                   "Executable with handle {0:x} was registered as a static "
                   "instrumentation module.\n",
                   Exec.hsaHandle()));
  }
  return llvm::Error::success();
}

llvm::Error ToolExecutableLoader::unregisterIfLuthierToolExecutable(
    const hsa::Executable &Exec) {
  // Check if this belongs to the static instrumentation module
  // If so, then unregister it from the static module
  auto IsSIMExec =
      StaticInstrumentationModule::isStaticInstrumentationModuleExecutable(
          Exec);
  LUTHIER_RETURN_ON_ERROR(IsSIMExec.takeError());
  if (*IsSIMExec) {
    return SIM.unregisterExecutable(Exec);
  }
  // Check if this executable has been instrumented before. If so,
  // destroy the instrumented versions of this executable, and remove its
  // entries from the internal maps
  if (OriginalExecutablesWithKernelsInstrumented.contains(Exec)) {
    llvm::SmallDenseSet<hsa::Executable, 1> InstrumentedVersionsOfExecutable;
    // 1. Find all instrumented versions of each kernel of Exec
    // 2. For each instrumented kernel, get its executable and insert it in
    // InstrumentedVersionsOfExecutable to be dealt with later
    // 3. Remove instrumented entries of the original kernel
    llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
    LUTHIER_RETURN_ON_ERROR(Exec.getLoadedCodeObjects(LCOs));
    for (const auto &LCO : LCOs) {
      llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4> Kernels;
      LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
      for (const auto &Symbol : Kernels) {
        const auto *Kernel =
            llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(Symbol);
        if (OriginalToInstrumentedKernelsMap.contains(Kernel)) {
          auto &InstrumentedKernels = OriginalToInstrumentedKernelsMap[Kernel];
          for (const auto &[Preset, InstrumentedKernel] : InstrumentedKernels) {
            auto InstrumentedExecutable = InstrumentedKernel->getExecutable();
            LUTHIER_RETURN_ON_ERROR(InstrumentedExecutable.takeError());
            InstrumentedVersionsOfExecutable.insert(
                hsa::Executable(*InstrumentedExecutable));
          }
          OriginalToInstrumentedKernelsMap.erase(Kernel);
        }
      }
    }
    // clean up all instrumented versions of Exec
    for (auto &InstrumentedExec : InstrumentedVersionsOfExecutable) {
      // For the LCOs of the instrumented executable, delete their Code Object
      // Readers
      llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
      LUTHIER_RETURN_ON_ERROR(InstrumentedExec.getLoadedCodeObjects(LCOs));
      for (const auto &LCO : LCOs) {
        auto It = InstrumentedLCOInfo.find(LCO);
        LUTHIER_RETURN_ON_ERROR(
            LUTHIER_ERROR_CHECK(It != InstrumentedLCOInfo.end(),
                                "Failed to find the instrumented LCO {0:x}'s "
                                "record inside the tool executable manager."));
        LUTHIER_RETURN_ON_ERROR(It->getSecond().destroy());
      }
      // Finally, delete the executable
      LUTHIER_RETURN_ON_ERROR(InstrumentedExec.destroy());
    }
    return llvm::Error::success();
  }
  return llvm::Error::success();
}

llvm::Expected<const hsa::LoadedCodeObjectKernel &>
ToolExecutableLoader::getInstrumentedKernel(
    const hsa::LoadedCodeObjectKernel &OriginalKernel,
    llvm::StringRef Preset) const {
  // First make sure the OriginalKernel has instrumented entries
  auto InstrumentedKernelsIt =
      OriginalToInstrumentedKernelsMap.find(&OriginalKernel);
  if (InstrumentedKernelsIt == OriginalToInstrumentedKernelsMap.end()) {
    auto KernelName = OriginalKernel.getName();
    LUTHIER_RETURN_ON_ERROR(KernelName.takeError());
    return LUTHIER_CREATE_ERROR(
        "Failed to find any instrumented version of kernel {0}.", *KernelName);
  }
  // Then make sure the original kernel was instrumented under the given Preset,
  // and then return the instrumented version
  auto InstrumentedKernelIt = InstrumentedKernelsIt->getSecond().find(Preset);
  if (InstrumentedKernelIt == InstrumentedKernelsIt->getSecond().end()) {
    auto KernelName = OriginalKernel.getName();
    LUTHIER_RETURN_ON_ERROR(KernelName.takeError());
    return LUTHIER_CREATE_ERROR("Failed to find any instrumented version of "
                                "kernel {0} under preset {1}.",
                                *KernelName, Preset);
  }
  return *InstrumentedKernelIt->second;
}

llvm::Error ToolExecutableLoader::loadInstrumentedKernel(
    llvm::ArrayRef<std::pair<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>>
        InstrumentedElfs,
    const hsa::LoadedCodeObjectKernel &OriginalKernel, llvm::StringRef Preset,
    const llvm::StringMap<const void *> &ExternVariables) {
  // Ensure this kernel was not instrumented under this preset
  auto IsInstrumented = isKernelInstrumented(OriginalKernel, Preset);
  if (IsInstrumented) {
    auto OriginalKernelName = OriginalKernel.getName();
    LUTHIER_RETURN_ON_ERROR(OriginalKernelName.takeError());
    return LUTHIER_CREATE_ERROR(
        "Kernel {0} is already instrumented under preset {1}.",
        *OriginalKernelName, Preset);
  }

  // Create the executable
  auto Executable = hsa::Executable::create();

  // Define the Agent allocation external variables
  auto Agent = OriginalKernel.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  LUTHIER_RETURN_ON_ERROR(Executable.takeError());
  for (const auto &[EVName, EVAddress] : ExternVariables) {
    LUTHIER_RETURN_ON_ERROR(Executable->defineExternalAgentGlobalVariable(
        hsa::GpuAgent(*Agent), EVName, EVAddress));
  }

  // Load the code objects into the executable
  for (const auto &[TargetLCO, InstrumentedElf] : InstrumentedElfs) {
    auto Reader = hsa::CodeObjectReader::createFromMemory(InstrumentedElf);
    LUTHIER_RETURN_ON_ERROR(Reader.takeError());
    auto LCO = Executable->loadAgentCodeObject(*Reader, hsa::GpuAgent(*Agent));
    LUTHIER_RETURN_ON_ERROR(LCO.takeError());
    InstrumentedLCOInfo.insert({*LCO, *Reader});
  }
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(Executable->freeze());

  // Find the original kernel in the instrumented executable
  auto OriginalSymbolName = OriginalKernel.getName();
  LUTHIER_RETURN_ON_ERROR(OriginalSymbolName.takeError());

  const hsa::LoadedCodeObjectSymbol *InstrumentedKernel{nullptr};
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
      OriginalKernel, Preset,
      *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(InstrumentedKernel));

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

    llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4> Kernels;
    LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
    for (const auto &Kernel : Kernels) {
      if (isKernelInstrumented(
              *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(Kernel), Preset)) {
        auto KernelName = Kernel->getName();
        LUTHIER_RETURN_ON_ERROR(KernelName.takeError());
        return LUTHIER_CREATE_ERROR(
            "Found kernel {0} inside LCO {1:x} which was already instrumented "
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
    llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4>
        KernelsOfOriginalLCO;
    LUTHIER_RETURN_ON_ERROR(OriginalLCO.getKernelSymbols(KernelsOfOriginalLCO));

    for (const auto &OriginalKernel : KernelsOfOriginalLCO) {
      auto OriginalKernelName = OriginalKernel->getName();
      LUTHIER_RETURN_ON_ERROR(OriginalKernelName.takeError());

      const hsa::LoadedCodeObjectSymbol *InstrumentedKernel{nullptr};
      llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
      LUTHIER_RETURN_ON_ERROR(Executable->getLoadedCodeObjects(LCOs));
      for (const auto &LCO : LCOs) {
        LUTHIER_RETURN_ON_ERROR(
            LCO.getLoadedCodeObjectSymbolByName(*OriginalKernelName)
                .moveInto(InstrumentedKernel));
        if (InstrumentedKernel != nullptr)
          break;
      }
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
          InstrumentedKernel != nullptr,
          "Failed to find the corresponding instrumented kernel for {0} inside "
          "the instrumented executable.",
          *OriginalKernelName));

      auto InstrumentedKernelType = InstrumentedKernel->getType();
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
          InstrumentedKernelType == hsa::LoadedCodeObjectSymbol::SK_KERNEL,
          "Found the corresponding instrumented symbol for kernel {0}, but the "
          "symbol is not of type kernel.",
          *OriginalKernelName));

      insertInstrumentedKernelIntoMap(
          *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(OriginalKernel), Preset,
          *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(InstrumentedKernel));
    }
  }
  OriginalExecutablesWithKernelsInstrumented.insert(Exec);
  return llvm::Error::success();
}

bool ToolExecutableLoader::isKernelInstrumented(
    const hsa::LoadedCodeObjectKernel &Kernel, llvm::StringRef Preset) const {
  return OriginalToInstrumentedKernelsMap.contains(&Kernel) &&
         OriginalToInstrumentedKernelsMap.at(&Kernel).contains(Preset);
}

ToolExecutableLoader::~ToolExecutableLoader() {
  // By the time the Tool Executable Manager is deleted, all instrumentation
  // kernels must have been destroyed; If not, print a warning, and clean
  // up anyway
  // TODO: Fix this, again
  //  if (!InstrumentedLCOInfo.empty()) {
  //    luthier::outs()
  //        << "Tool executable manager is being destroyed while the original "
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
