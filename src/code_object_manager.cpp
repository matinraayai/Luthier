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

void CodeObjectManager::registerInstrumentationFunctionWrapper(
    const void *WrapperShadowHostPtr, const char *KernelName) {
  StaticInstrumentationFunctions.insert({KernelName, WrapperShadowHostPtr});
}

llvm::Error CodeObjectManager::checkIfLuthierToolExecutableAndRegister(
    const hsa::Executable &Exec) {
  llvm::SmallVector<hsa::GpuAgent, 8> Agents;
  LUTHIER_RETURN_ON_ERROR(hsa::getGpuAgents(Agents));

  for (const auto &Agent : Agents) {
    auto LuthierReservedSymbol =
        Exec.getAgentSymbolByName(Agent, luthier::ReservedManagedVar);
    LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbol.takeError());

    if (LuthierReservedSymbol->has_value()) {

      std::unordered_map<std::string, hsa::ExecutableSymbol>
          WrapperKernelSymbols;
      std::unordered_map<std::string, hsa::ExecutableSymbol>
          InstFunctionSymbols;
      auto Symbols = Exec.getAgentSymbols(Agent);
      LUTHIER_RETURN_ON_ERROR(Symbols.takeError());
      for (const auto &Symbol : *Symbols) {
        auto SType = Symbol.getType();
        LUTHIER_RETURN_ON_ERROR(SType.takeError());
        auto SName = Symbol.getName();
        LUTHIER_RETURN_ON_ERROR(SName.takeError());
        if (*SType == HSA_SYMBOL_KIND_KERNEL) {
          WrapperKernelSymbols.insert(
              {SName->substr(0, SName->rfind(".kd")), Symbol});

        } else if (*SType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION) {
          InstFunctionSymbols.insert(
              {luthier::DeviceFunctionWrap + *SName, Symbol});
        }
      }
      for (const auto &[WrapKerName, WrapKerShadowPtr] :
           StaticInstrumentationFunctions) {
        LUTHIER_RETURN_ON_ERROR(
            LUTHIER_ASSERTION(WrapperKernelSymbols.contains(WrapKerName)));
        auto &KernelSymbol = WrapperKernelSymbols.at(WrapKerName);
        auto &functionSymbol = InstFunctionSymbols.at(WrapKerName);
        //        LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        //            ToolFunctions.contains({WrapKerShadowPtr, Agent})));
        ToolFunctions.insert(
            {{WrapKerShadowPtr, Agent}, {functionSymbol, KernelSymbol}});
      }
    }
  }
  LuthierLogDebug("Number of executables captured: {0}\n",
                  ToolExecutables.size());
  return llvm::Error::success();
}

llvm::Expected<const hsa::ExecutableSymbol &>
CodeObjectManager::getInstrumentationFunction(
    const void *WrapperShadowHostPtr, const hsa::GpuAgent &Agent) const {
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
    const hsa::ExecutableSymbol &OriginalKernel,
    const std::vector<hsa::ExecutableSymbol> &ExternVariables) {
  if (!InstrumentedKernels.contains(OriginalKernel)) {
    auto Executable = hsa::Executable::create();
    LUTHIER_RETURN_ON_ERROR(Executable.takeError());

    for (const auto &EV : ExternVariables) {
      LUTHIER_RETURN_ON_ERROR(
          Executable->defineExternalAgentGlobalVariable(EV));
    }

    auto agent = OriginalKernel.getAgent();
    LUTHIER_RETURN_ON_ERROR(agent.takeError());

    auto reader = hsa::CodeObjectReader::createFromMemory(InstrumentedElf);
    LUTHIER_RETURN_ON_ERROR(reader.takeError());
    LUTHIER_RETURN_ON_ERROR(
        Executable->loadAgentCodeObject(*reader, *agent, "").takeError());
    LUTHIER_RETURN_ON_ERROR(Executable->freeze());

    auto originalSymbolName = OriginalKernel.getName();
    LUTHIER_RETURN_ON_ERROR(originalSymbolName.takeError());
    auto instrumentedKernel =
        Executable->getAgentSymbolByName(*agent, *originalSymbolName);
    LUTHIER_RETURN_ON_ERROR(instrumentedKernel.takeError());

    auto instrumentedKernelType = (*instrumentedKernel)->getType();
    LUTHIER_RETURN_ON_ERROR(instrumentedKernelType.takeError());

    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(*instrumentedKernelType == HSA_SYMBOL_KIND_KERNEL));

    InstrumentedKernels.insert(
        {OriginalKernel,
         std::make_tuple(**instrumentedKernel, *Executable, *reader)});
  }
  return llvm::Error::success();
}

llvm::Expected<const hsa::ExecutableSymbol &>
CodeObjectManager::getInstrumentationFunctionWrapperKernel(
    const void *WrapperHostPtr, const hsa::GpuAgent& Agent) const {
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
