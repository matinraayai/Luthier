//===-- tool_executable_manager.cpp - Luthier Tool Executable Manager -----===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements Luthier's Tool Executable Manager Singleton, and
/// instrumentation modules which are passed to the \c CodeGenerator.
//===----------------------------------------------------------------------===//
#include "tooling_common/tool_executable_manager.hpp"
#include "tooling_common/target_manager.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <vector>

#include "common/log.hpp"
#include "hsa/hsa.hpp"
#include "hsa/hsa_agent.hpp"
#include "hsa/hsa_executable_symbol.hpp"
#include "hsa/hsa_intercept.hpp"
#include "hsa/hsa_loaded_code_object.hpp"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-tool-executable-manager"

namespace luthier {

template <>
ToolExecutableManager *Singleton<ToolExecutableManager>::Instance{nullptr};

void ToolExecutableManager::registerInstrumentationHookWrapper(
    const void *WrapperShadowHostPtr, const char *HookWrapperName) {
  SIM.HookHandleMap.insert(
      {WrapperShadowHostPtr, llvm::StringRef(HookWrapperName)
                                 .substr(strlen(luthier::HookHandlePrefix))});
}

llvm::Error ToolExecutableManager::registerIfLuthierToolExecutable(
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

llvm::Error ToolExecutableManager::unregisterIfLuthierToolExecutable(
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
    for (const auto &LCO : llvm::cantFail(Exec.getLoadedCodeObjects())) {
      llvm::SmallVector<hsa::ExecutableSymbol, 4> Kernels;
      LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
      for (const auto &Kernel : Kernels) {
        if (OriginalToInstrumentedKernelsMap.contains(Kernel)) {
          auto &InstrumentedKernels = OriginalToInstrumentedKernelsMap[Kernel];
          for (const auto &[Preset, InstrumentedKernel] : InstrumentedKernels) {
            InstrumentedVersionsOfExecutable.insert(
                llvm::cantFail(InstrumentedKernel.getExecutable()));
          }
          OriginalToInstrumentedKernelsMap.erase(Kernel);
        }
      }
    }
    // clean up all instrumented versions of Exec
    for (auto &InstrumentedExec : InstrumentedVersionsOfExecutable) {
      // For the LCOs of the instrumented executable, delete their Code Object
      // Readers
      for (const auto &LCO :
           llvm::cantFail(InstrumentedExec.getLoadedCodeObjects())) {
        auto It = InstrumentedLCOInfo.find(LCO);
        LUTHIER_RETURN_ON_ERROR(
            LUTHIER_ASSERTION(It != InstrumentedLCOInfo.end()));
        LUTHIER_RETURN_ON_ERROR(It->getSecond().destroy());
      }
      // Finally, delete the executable
      LUTHIER_RETURN_ON_ERROR(InstrumentedExec.destroy());
    }
    return llvm::Error::success();
  }
  return llvm::Error::success();
}

llvm::Expected<const hsa::ExecutableSymbol &>
ToolExecutableManager::getInstrumentedKernel(
    const hsa::ExecutableSymbol &OriginalKernel, llvm::StringRef Preset) const {
  // First make sure the OriginalKernel has instrumented entries
  auto InstrumentedKernelsIt =
      OriginalToInstrumentedKernelsMap.find(OriginalKernel);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
      InstrumentedKernelsIt != OriginalToInstrumentedKernelsMap.end()));
  // Then make sure the original kernel was instrumented under the given Preset,
  // and then return the instrumented version
  auto InstrumentedKernelIt = InstrumentedKernelsIt->getSecond().find(Preset);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
      InstrumentedKernelIt != InstrumentedKernelsIt->getSecond().end()));
  return InstrumentedKernelIt->second;
}

llvm::Error ToolExecutableManager::loadInstrumentedKernel(
    llvm::ArrayRef<std::pair<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>>
        InstrumentedElfs,
    const hsa::ExecutableSymbol &OriginalKernel, llvm::StringRef Preset,
    const llvm::StringMap<void *> &ExternVariables) {
  // Ensure this kernel was not instrumented under this profile
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(!isKernelInstrumented(OriginalKernel, Preset)));

  // Create the executable
  auto Executable = hsa::Executable::create();

  // Define the Agent allocation external variables
  auto Agent = OriginalKernel.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  LUTHIER_RETURN_ON_ERROR(Executable.takeError());
  for (const auto &[EVName, EVAddress] : ExternVariables) {
    LUTHIER_RETURN_ON_ERROR(Executable->defineExternalAgentGlobalVariable(
        *Agent, EVName, EVAddress));
  }

  // Load the code objects into the executable
  for (const auto &[TargetLCO, InstrumentedElf] : InstrumentedElfs) {
    auto Reader = hsa::CodeObjectReader::createFromMemory(InstrumentedElf);
    LUTHIER_RETURN_ON_ERROR(Reader.takeError());
    auto LCO = Executable->loadAgentCodeObject(*Reader, *Agent);
    LUTHIER_RETURN_ON_ERROR(LCO.takeError());
    InstrumentedLCOInfo.insert({*LCO, *Reader});
  }
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(Executable->freeze());

  // Find the original kernel in the instrumented executable
  auto OriginalSymbolName = OriginalKernel.getName();
  LUTHIER_RETURN_ON_ERROR(OriginalSymbolName.takeError());

  std::optional<hsa::ExecutableSymbol> InstrumentedKernel;
  for (const auto &LCO : llvm::cantFail(Executable->getLoadedCodeObjects())) {
    LUTHIER_RETURN_ON_ERROR(LCO.getExecutableSymbolByName(*OriginalSymbolName)
                                .moveInto(InstrumentedKernel));
    if (InstrumentedKernel.has_value())
      break;
  }
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(InstrumentedKernel.has_value()));

  auto InstrumentedKernelType = InstrumentedKernel->getType();
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(InstrumentedKernelType == hsa::KERNEL));

  insertInstrumentedKernelIntoMap(OriginalKernel, Preset, *InstrumentedKernel);

  OriginalExecutablesWithKernelsInstrumented.insert(
      llvm::cantFail(OriginalKernel.getExecutable()));

  return llvm::Error::success();
}

llvm::Error ToolExecutableManager::loadInstrumentedExecutable(
    llvm::ArrayRef<std::pair<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>>
        InstrumentedElfs,
    llvm::StringRef Preset,
    llvm::ArrayRef<std::tuple<hsa::GpuAgent, llvm::StringRef, void *>>
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
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(*LCOExec == Exec));

    llvm::SmallVector<hsa::ExecutableSymbol, 4> Kernels;
    LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
    for (const auto &Kernel : Kernels) {
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ASSERTION(!isKernelInstrumented(Kernel, Preset)));
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
    llvm::SmallVector<hsa::ExecutableSymbol, 4> KernelsOfOriginalLCO;
    LUTHIER_RETURN_ON_ERROR(OriginalLCO.getKernelSymbols(KernelsOfOriginalLCO));

    for (const auto &OriginalKernel : KernelsOfOriginalLCO) {
      auto OriginalKernelName = OriginalKernel.getName();
      LUTHIER_RETURN_ON_ERROR(OriginalKernelName.takeError());

      std::optional<hsa::ExecutableSymbol> InstrumentedKernel;
      for (const auto &LCO :
           llvm::cantFail(Executable->getLoadedCodeObjects())) {
        LUTHIER_RETURN_ON_ERROR(
            LCO.getExecutableSymbolByName(*OriginalKernelName)
                .moveInto(InstrumentedKernel));
        if (InstrumentedKernel.has_value())
          break;
      }
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ASSERTION(InstrumentedKernel.has_value()));

      auto InstrumentedKernelType = InstrumentedKernel->getType();
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ASSERTION(InstrumentedKernelType == hsa::KERNEL));

      insertInstrumentedKernelIntoMap(OriginalKernel, Preset,
                                      *InstrumentedKernel);
    }
  }
  OriginalExecutablesWithKernelsInstrumented.insert(Exec);
  return llvm::Error::success();
}

bool ToolExecutableManager::isKernelInstrumented(
    const hsa::ExecutableSymbol &Kernel, llvm::StringRef Preset) const {
  return OriginalToInstrumentedKernelsMap.contains(Kernel) &&
         OriginalToInstrumentedKernelsMap.at(Kernel).contains(Preset);
}

ToolExecutableManager::~ToolExecutableManager() {
  // By the time the Tool Executable Manager is deleted, all instrumentation
  // kernels must have been destroyed; If not, print a warning, and clean
  // up anyway
  // TODO: Fix this, again
//  if (!InstrumentedLCOInfo.empty()) {
//    llvm::outs()
//        << "Tool executable manager is being destroyed while the original "
//           "executables of its instrumented kernels are still frozen\n";
//    llvm::DenseSet<hsa::Executable> InstrumentedExecs;
//    for (auto &[LCO, COR] : InstrumentedLCOInfo) {
//      auto Exec = llvm::cantFail(LCO.getExecutable());
//      InstrumentedExecs.insert(Exec);
//      if (COR.destroy()) {
//        llvm::outs() << llvm::formatv(
//            "Code object reader {0:x} of Loaded Code Object {1:x}, Executable "
//            "{2:x} got destroyed with errors.\n",
//            COR.hsaHandle(), LCO.hsaHandle(), Exec.hsaHandle());
//      }
//    }
//    for (auto &Exec : InstrumentedExecs) {
//      if (Exec.destroy()) {
//        llvm::outs() << llvm::formatv(
//            "Executable {0:x} got destroyed with errors.\n", Exec.hsaHandle());
//      }
//    }
//  }
  OriginalToInstrumentedKernelsMap.clear();
}

} // namespace luthier
