#include "code_object_manager.hpp"
#include "target_manager.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <vector>

#include "disassembler.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_loaded_code_object.hpp"
#include "log.hpp"
#include "cloning.hpp"

namespace luthier {

#define DEBUG_TYPE "code-object-manager"

template <> CodeObjectManager *Singleton<CodeObjectManager>::Instance{nullptr};

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
      llvm::SmallVector<hsa::ExecutableSymbol> Symbols;
      LUTHIER_RETURN_ON_ERROR(LCO.getExecutableSymbols(Symbols));
      for (const auto &Symbol : Symbols) {
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
  LLVM_DEBUG(llvm::dbgs() << "Number of executables captured: "
                          << ToolExecutables.size() << "\n");
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
llvm::Expected<std::unique_ptr<llvm::Module>>
CodeObjectManager::getModuleContainingInstrumentationFunctions(
    const llvm::ArrayRef<hsa::ExecutableSymbol> Symbols) const {
  // Make sure all LCOs have the same ISA + Keep track of the LCOs
  llvm::SmallDenseSet<hsa::LoadedCodeObject, 1> LCOs;
  hsa::ISA ISA{{0}};
  for (const auto &Symbol : Symbols) {
    auto LCO = Symbol.getLoadedCodeObject();
    LUTHIER_RETURN_ON_ERROR(LCO.takeError());
    auto ISAOfLCO = LCO->getISA();
    LUTHIER_RETURN_ON_ERROR(ISAOfLCO.takeError());
    if (ISA.hsaHandle() == 0)
      ISA = *ISAOfLCO;
    else
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ISA == *ISAOfLCO));
    LCOs.insert(*LCO);
  }

  auto TargetInfo = luthier::TargetManager::instance().getTargetInfo(ISA);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());
  // Ensure all the LCOs have their Modules already created by the Code Object
  // Manager
  for (const auto &LCO : LCOs) {
    if (!ToolLCOEmbeddedIRModules.contains(LCO)) {
      auto StorageELF = LCO.getStorageELF();
      LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());

      for (const auto &Section : StorageELF->sections()) {
        auto SectionName = Section.getName();
        LUTHIER_RETURN_ON_ERROR(SectionName.takeError());
        llvm::outs() << "Is section bitcode: " << Section.isBitcode() << "\n";
        llvm::outs() << "Section name: "  << *SectionName << "\n";
        if (*SectionName == ".llvmcmd") {
          llvm::outs() << llvm::cantFail(Section.getContents()) << "\n";
        }
        if (*SectionName == ".llvmbc") {
          auto SectionContents = Section.getContents();
          LUTHIER_RETURN_ON_ERROR(SectionContents.takeError());
          auto BCBuffer =
              llvm::MemoryBuffer::getMemBuffer(*SectionContents, "", false);
          auto Module =
              llvm::parseBitcodeFile(*BCBuffer, *TargetInfo->getLLVMContext());
          LUTHIER_RETURN_ON_ERROR(Module.takeError());

          ToolLCOEmbeddedIRModules.insert({LCO, std::move(*Module)});
        }
      }
      // Ensure that the tool contained LLVM bitcode and it was successfully
      // extracted
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ASSERTION(ToolLCOEmbeddedIRModules.contains(LCO)));
    }
  }

  // Create a new LLVM Module to hold the requested symbols
  std::unique_ptr<llvm::Module> ClonedModule =
      std::make_unique<llvm::Module>("", *TargetInfo->getLLVMContext());


  llvm::SmallVector<llvm::Function*> Funcs;
  for (const auto &Symbol : Symbols) {
    auto LCO = llvm::cantFail(Symbol.getLoadedCodeObject());
    const auto &LCOModule = ToolLCOEmbeddedIRModules.at(LCO);
    ClonedModule->setSourceFileName(LCOModule->getSourceFileName());
    ClonedModule->setDataLayout(LCOModule->getDataLayout());
    ClonedModule->setTargetTriple(LCOModule->getTargetTriple());
    ClonedModule->setModuleInlineAsm(LCOModule->getModuleInlineAsm());
    ClonedModule->IsNewDbgInfoFormat = LCOModule->IsNewDbgInfoFormat;
    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    Funcs.push_back(LCOModule->getFunction(*SymbolName));
    return llvm::CloneModule(*LCOModule);
  }
//  cloneFunctionIntoModule(Funcs, *ClonedModule);
//  return std::move(ClonedModule);

}

} // namespace luthier
