#include "code_object_manager.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>

#include <vector>

#include "disassembler.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_loaded_code_object.hpp"
#include "log.hpp"

namespace luthier {

llvm::Error CodeObjectManager::registerLuthierHsaExecutables() const {
  llvm::SmallVector<hsa::Executable> Executables;
  LUTHIER_RETURN_ON_ERROR(hsa::getAllExecutables(Executables));

  llvm::SmallVector<hsa::GpuAgent> Agents;
  LUTHIER_RETURN_ON_ERROR(hsa::getGpuAgents(Agents));

  for (const auto &e : Executables) {
    for (const auto &a : Agents) {
      auto LuthierReservedSymbolWithoutManaged =
          e.getSymbolByName(a, LUTHIER_RESERVED_MANAGED_VAR);
      LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbolWithoutManaged.takeError());
      auto LuthierReservedSymbolWithManaged = e.getSymbolByName(
          a, std::string(LUTHIER_RESERVED_MANAGED_VAR) + ".managed");
      LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbolWithManaged.takeError());

      if (LuthierReservedSymbolWithManaged->has_value() ||
          LuthierReservedSymbolWithoutManaged->has_value()) {
        toolExecutables_.insert(e);
      }
    }
  }
  LuthierLogDebug("Number of executables captured: {0}",
                  toolExecutables_.size());
  return llvm::Error::success();
}

llvm::Error CodeObjectManager::processFunctions() const {
  LUTHIER_RETURN_ON_ERROR(registerLuthierHsaExecutables());

  llvm::SmallVector<hsa::GpuAgent> agents;
  LUTHIER_RETURN_ON_ERROR(hsa::getGpuAgents(agents));

  for (const auto &e : toolExecutables_) {
    for (const auto &a : agents) {
      std::unordered_map<std::string, hsa::ExecutableSymbol> instKernelSymbols;
      std::unordered_map<std::string, hsa::ExecutableSymbol>
          instFunctionSymbols;
      auto symbols = e.getSymbols(a);
      LUTHIER_RETURN_ON_ERROR(symbols.takeError());
      for (const auto &s : *symbols) {
        auto sType = s.getType();
        LUTHIER_RETURN_ON_ERROR(sType.takeError());
        auto sName = s.getName();
        LUTHIER_RETURN_ON_ERROR(sName.takeError());
        if (*sType == HSA_SYMBOL_KIND_KERNEL)
          instKernelSymbols.insert({sName->substr(0, sName->find(".kd")), s});
        else if (*sType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION)
          instFunctionSymbols.insert(
              {LUTHIER_DEVICE_FUNCTION_WRAP + *sName, s});
      }
      for (const auto &[instKerShadowPtr, instKerName] :
           unprocessedFunctions_) {
        if (instKernelSymbols.contains(instKerName)) {
          auto kernelSymbol = instKernelSymbols.at(instKerName);
          auto functionSymbol = instFunctionSymbols.at(instKerName);
          if (functions_.contains(instKerShadowPtr)) {
            functions_.at(instKerShadowPtr)
                .insert({a, {functionSymbol, kernelSymbol}});
          } else {
            functions_.insert(
                {instKerShadowPtr, {{a, {functionSymbol, kernelSymbol}}}});
          }
        }
      }
    }
  }
  LuthierLogDebug("Number of functions registered: {0}\n", functions_.size());
  unprocessedFunctions_.clear();
  return llvm::Error::success();
}

void CodeObjectManager::registerInstrumentationFunctionWrapper(
    const void *wrapperHostPtr, const char *kernelName) {
  unprocessedFunctions_.emplace_back(wrapperHostPtr, kernelName);
}

llvm::Expected<const hsa::ExecutableSymbol &>
CodeObjectManager::getInstrumentationFunction(const void *wrapperHostPtr,
                                              hsa::GpuAgent agent) const {
  if (!unprocessedFunctions_.empty()) {
    LUTHIER_RETURN_ON_ERROR(processFunctions());
  }
  return functions_.at(wrapperHostPtr).at(agent).getInstrumentationFunction();
}

const hsa::ExecutableSymbol &CodeObjectManager::getInstrumentedKernel(
    const hsa::ExecutableSymbol &originalKernel) const {
  if (!instrumentedKernels_.contains(originalKernel)) {
    auto OriginalKD = originalKernel.getKernelDescriptor();
    LUTHIER_CHECK(llvm::errorToBool(OriginalKD.takeError()) == false);
    llvm::report_fatal_error(
        llvm::formatv("The Kernel Descriptor {0:x} has not been instrumented.",
                      reinterpret_cast<luthier_address_t>(*OriginalKD)));
  }
  return std::get<hsa::ExecutableSymbol>(
      instrumentedKernels_.at(originalKernel));
}

llvm::Error CodeObjectManager::loadInstrumentedKernel(
    const llvm::ArrayRef<uint8_t> &instrumentedElf,
    const hsa::ExecutableSymbol &originalKernel) {
  if (!instrumentedKernels_.contains(originalKernel)) {
    auto executable = hsa::Executable::create();
    LUTHIER_RETURN_ON_ERROR(executable.takeError());

    auto agent = originalKernel.getAgent();

    auto reader = hsa::CodeObjectReader::createFromMemory(instrumentedElf);
    LUTHIER_RETURN_ON_ERROR(reader.takeError());
    LUTHIER_RETURN_ON_ERROR(
        executable->loadCodeObject(*reader, agent).takeError());
    LUTHIER_RETURN_ON_ERROR(executable->freeze());

    auto originalSymbolName = originalKernel.getName();
    LUTHIER_RETURN_ON_ERROR(originalSymbolName.takeError());
    auto instrumentedKernel =
        executable->getSymbolByName(agent, *originalSymbolName);
    LUTHIER_RETURN_ON_ERROR(instrumentedKernel.takeError());

    auto instrumentedKernelType = (*instrumentedKernel)->getType();
    LUTHIER_RETURN_ON_ERROR(instrumentedKernelType.takeError());

    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(*instrumentedKernelType == HSA_SYMBOL_KIND_KERNEL));

    instrumentedKernels_.insert(
        {originalKernel,
         std::make_tuple(**instrumentedKernel, *executable, *reader)});
  }
  return llvm::Error::success();
}

llvm::Expected<const hsa::ExecutableSymbol &>
CodeObjectManager::getInstrumentationKernel(const void *wrapperHostPtr,
                                            hsa::GpuAgent agent) const {
  if (!unprocessedFunctions_.empty()) {
    LUTHIER_RETURN_ON_ERROR(processFunctions());
  }
  return functions_.at(wrapperHostPtr).at(agent).getInstrumentationKernel();
}

CodeObjectManager::~CodeObjectManager() {
  // TODO: Fix the destructor
  for (auto &[origSymbol, instInfo] : instrumentedKernels_) {
    auto &[s, e, r] = instInfo;
    //        r.destroy();
    //        e.destroy();
  }
  //    instrumentedKernels_.clear();
  //    toolExecutables_.clear();
  //    functions_.clear();
}

} // namespace luthier
