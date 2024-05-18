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
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  for (const auto &LCO : *LCOs) {
    auto LuthierReservedSymbol =
        LCO.getExecutableSymbolByName(luthier::ReservedManagedVar);
    LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbol.takeError());
    if (LuthierReservedSymbol->has_value()) {
      auto Agent = LCO.getAgent();
      LUTHIER_RETURN_ON_ERROR(Agent.takeError());

      llvm::StringMap<hsa::ExecutableSymbol> WrapperKernelSymbols;
      llvm::StringMap<hsa::ExecutableSymbol> InstFunctionSymbols;
      auto Symbols = LCO.getExecutableSymbols();
      LUTHIER_RETURN_ON_ERROR(Symbols.takeError());
      for (const auto &Symbol : *Symbols) {
        auto SType = Symbol.getType();
        auto SName = Symbol.getName();
        LUTHIER_RETURN_ON_ERROR(SName.takeError());
        if (SType == hsa::KERNEL) {
          WrapperKernelSymbols.insert(
              {SName->substr(0, SName->rfind(".kd")), Symbol});

        } else if (SType == hsa::DEVICE_FUNCTION) {
          InstFunctionSymbols.insert(
              {(luthier::DeviceFunctionWrap + *SName).str(), Symbol});
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
            {{WrapKerShadowPtr, *Agent}, {functionSymbol, KernelSymbol}});
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
    auto LCO = Executable->loadAgentCodeObject(*reader, *agent, "");
    LUTHIER_RETURN_ON_ERROR(LCO.takeError());
    LUTHIER_RETURN_ON_ERROR(Executable->freeze());

    auto originalSymbolName = OriginalKernel.getName();
    LUTHIER_RETURN_ON_ERROR(originalSymbolName.takeError());
    auto instrumentedKernel =
        LCO->getExecutableSymbolByName(*originalSymbolName);
    LUTHIER_RETURN_ON_ERROR(instrumentedKernel.takeError());

    auto instrumentedKernelType = (*instrumentedKernel)->getType();

    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(instrumentedKernelType == hsa::KERNEL));

    InstrumentedKernels.insert(
        {OriginalKernel,
         std::make_tuple(**instrumentedKernel, *Executable, *reader)});
  }
  return llvm::Error::success();
}

llvm::Expected<const hsa::ExecutableSymbol &>
CodeObjectManager::getInstrumentationFunctionWrapperKernel(
    const void *WrapperHostPtr, const hsa::GpuAgent &Agent) const {
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
bool CodeObjectManager::isKernelInstrumented(
    const hsa::ExecutableSymbol &Kernel) const {
  return InstrumentedKernels.contains(Kernel);
}

} // namespace luthier
