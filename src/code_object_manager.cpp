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

  for (const auto &Exec : Executables) {
    for (const auto &Agent : Agents) {
      auto LuthierReservedSymbolWithoutManaged =
          Exec.getAgentSymbolByName(Agent, luthier::ReservedManagedVar);
      LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbolWithoutManaged.takeError());
      auto LuthierReservedSymbolWithManaged = Exec.getAgentSymbolByName(
          Agent, std::string(luthier::ReservedManagedVar) + ".managed");
      LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbolWithManaged.takeError());

      if (LuthierReservedSymbolWithManaged->has_value() ||
          LuthierReservedSymbolWithoutManaged->has_value()) {
        ToolExecutables.insert(Exec);
      }
    }
  }
  LuthierLogDebug("Number of executables captured: {0}",
                  ToolExecutables.size());
  return llvm::Error::success();
}

llvm::Error CodeObjectManager::processFunctions() const {
  LUTHIER_RETURN_ON_ERROR(registerLuthierHsaExecutables());

  llvm::SmallVector<hsa::GpuAgent> Agents;
  LUTHIER_RETURN_ON_ERROR(hsa::getGpuAgents(Agents));

  for (const auto &Exec : ToolExecutables) {
    for (const auto &Agent : Agents) {
      llvm::DenseMap<llvm::StringRef, hsa::ExecutableSymbol>
          WrapperKernelSymbols;
      llvm::DenseMap<llvm::StringRef, hsa::ExecutableSymbol>
          InstFunctionSymbols;
      auto Symbols = Exec.getAgentSymbols(Agent);
      LUTHIER_RETURN_ON_ERROR(Symbols.takeError());
      for (const auto &Symbol : *Symbols) {
        auto SType = Symbol.getType();
        LUTHIER_RETURN_ON_ERROR(SType.takeError());
        auto SName = Symbol.getName();
        LUTHIER_RETURN_ON_ERROR(SName.takeError());
        if (*SType == HSA_SYMBOL_KIND_KERNEL)
          WrapperKernelSymbols.insert(
              {SName->substr(0, SName->rfind(".kd")), Symbol});
        else if (*SType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION)
          InstFunctionSymbols.insert(
              {luthier::DeviceFunctionWrap + *SName, Symbol});
      }
      for (const auto &[WrapKerShadowPtr, WrapKerName] : UnprocessedFunctions) {
        LUTHIER_RETURN_ON_ERROR(
            LUTHIER_ASSERTION(WrapperKernelSymbols.contains(WrapKerName)));
        auto &KernelSymbol = WrapperKernelSymbols.at(WrapKerName);
        auto &functionSymbol = InstFunctionSymbols.at(WrapKerName);
        LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
            ToolFunctions.contains({WrapKerShadowPtr, Agent})));
        ToolFunctions.insert(
            {{WrapKerShadowPtr, Agent}, {functionSymbol, KernelSymbol}});
      }
    }
  }
  LuthierLogDebug("Number of functions registered: {0}\n",
                  ToolFunctions.size());
  UnprocessedFunctions.clear();
  return llvm::Error::success();
}

void CodeObjectManager::registerInstrumentationFunctionWrapper(
    const void *WrapperShadowHostPtr, const char *KernelName) {
  UnprocessedFunctions.emplace_back(WrapperShadowHostPtr, KernelName);
}

llvm::Expected<const hsa::ExecutableSymbol &>
CodeObjectManager::getInstrumentationFunction(
    const void *WrapperShadowHostPtr, const hsa::GpuAgent &Agent) const {
  if (!UnprocessedFunctions.empty()) {
    LUTHIER_RETURN_ON_ERROR(processFunctions());
  }
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(ToolFunctions.contains({WrapperShadowHostPtr, Agent})));
  return ToolFunctions.at({WrapperShadowHostPtr, Agent})
      .InstrumentationFunction;
}

llvm::Expected<const hsa::ExecutableSymbol &>
CodeObjectManager::getInstrumentedKernel(
    const hsa::ExecutableSymbol &OriginalKernel) const {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(InstrumentedKernels.contains(OriginalKernel)));
  return std::get<hsa::ExecutableSymbol>(
      InstrumentedKernels.at(OriginalKernel));
}

llvm::Error CodeObjectManager::loadInstrumentedKernel(
    const llvm::ArrayRef<uint8_t> &InstrumentedElf,
    const hsa::ExecutableSymbol &OriginalKernel) {
  if (!InstrumentedKernels.contains(OriginalKernel)) {
    auto executable = hsa::Executable::create();
    LUTHIER_RETURN_ON_ERROR(executable.takeError());

    auto agent = OriginalKernel.getAgent();
    LUTHIER_RETURN_ON_ERROR(agent.takeError());

    auto reader = hsa::CodeObjectReader::createFromMemory(InstrumentedElf);
    LUTHIER_RETURN_ON_ERROR(reader.takeError());
    LUTHIER_RETURN_ON_ERROR(
        executable->loadAgentCodeObject(*reader, *agent, "").takeError());
    LUTHIER_RETURN_ON_ERROR(executable->freeze());

    auto originalSymbolName = OriginalKernel.getName();
    LUTHIER_RETURN_ON_ERROR(originalSymbolName.takeError());
    auto instrumentedKernel =
        executable->getAgentSymbolByName(*agent, *originalSymbolName);
    LUTHIER_RETURN_ON_ERROR(instrumentedKernel.takeError());

    auto instrumentedKernelType = (*instrumentedKernel)->getType();
    LUTHIER_RETURN_ON_ERROR(instrumentedKernelType.takeError());

    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(*instrumentedKernelType == HSA_SYMBOL_KIND_KERNEL));

    InstrumentedKernels.insert(
        {OriginalKernel,
         std::make_tuple(**instrumentedKernel, *executable, *reader)});
  }
  return llvm::Error::success();
}

llvm::Expected<const hsa::ExecutableSymbol &>
CodeObjectManager::getInstrumentationFunctionWrapperKernel(
    const void *WrapperHostPtr, hsa::GpuAgent Agent) const {
  if (!UnprocessedFunctions.empty()) {
    LUTHIER_RETURN_ON_ERROR(processFunctions());
  }
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(ToolFunctions.contains({WrapperHostPtr, Agent})));
  return ToolFunctions.at({WrapperHostPtr, Agent}).WrapperKernel;
}

CodeObjectManager::~CodeObjectManager() {
  // TODO: Fix the destructor
  for (auto &[origSymbol, instInfo] : InstrumentedKernels) {
    auto &[s, e, r] = instInfo;
    //        r.destroy();
    //        e.destroy();
  }
  //    instrumentedKernels_.clear();
  //    toolExecutables_.clear();
  //    functions_.clear();
}

} // namespace luthier
