#include "tool_executable_manager.hpp"
#include "target_manager.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <vector>

#include "cloning.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_loaded_code_object.hpp"
#include "log.hpp"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-object-manager"

namespace luthier {
////////////////////////////////////////////////////////////////////////////////
/// Instrumentation Module Pre-processing functions (Shared between all
/// Instrumentation Module sub-classes)
////////////////////////////////////////////////////////////////////////////////

static constexpr const char *HipCUIDPrefix = "__hip_cuid_";

/// Extracts the LLVM bitcode from the ".llvmbc" section of the LCO's storage
/// ELF
/// \param LCO the \c hsa::LoadedCodeObject containing the bitcode
/// \return an owning reference to the extracted \c llvm::Module, or an
/// \c llvm::Error if the bitcode was not found, or any other error that was
/// encountered during the extraction process
static llvm::Expected<std::unique_ptr<llvm::Module>>
extractBitcodeFromLCO(const hsa::LoadedCodeObject &LCO,
                      llvm::LLVMContext &Context) {
  auto StorageELF = LCO.getStorageELF();
  LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());

  // Find the ".llvmbc" section of the ELF
  const llvm::object::SectionRef *BitcodeSection{nullptr};
  for (const llvm::object::SectionRef &Section : StorageELF->sections()) {
    auto SectionName = Section.getName();
    LUTHIER_RETURN_ON_ERROR(SectionName.takeError());
    if (*SectionName == ".llvmbc") {
      BitcodeSection = &Section;
      break;
    }
  }
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(BitcodeSection != nullptr));

  auto SectionContents = BitcodeSection->getContents();
  LUTHIER_RETURN_ON_ERROR(SectionContents.takeError());
  auto BCBuffer = llvm::MemoryBuffer::getMemBuffer(*SectionContents, "", false);
  auto Module = llvm::parseBitcodeFile(*BCBuffer, Context);
  LUTHIER_RETURN_ON_ERROR(Module.takeError());
  return std::move(*Module);
}

/// Returns a list of instrumentation hooks found in the Module \p M.
/// \param [in] M Module to inspect
/// \param [out] Hooks a list of hook functions found in \p M
/// \return any \c llvm::Error encountered during the process
static llvm::Error getHooks(const llvm::Module &M,
                            llvm::SmallVector<llvm::Function *> &Hooks) {
  const llvm::GlobalVariable *V =
      M.getGlobalVariable("llvm.global.annotations");
  const llvm::ConstantArray *CA = cast<llvm::ConstantArray>(V->getOperand(0));
  for (llvm::Value *Op : CA->operands()) {
    auto *CS = cast<llvm::ConstantStruct>(Op);
    // The first field of the struct contains a pointer to the annotated
    // variable.
    llvm::Value *AnnotatedVar = CS->getOperand(0)->stripPointerCasts();
    if (auto *Func = llvm::dyn_cast<llvm::Function>(AnnotatedVar)) {
      // The second field contains a pointer to a global annotation string.
      auto *GV =
          cast<llvm::GlobalVariable>(CS->getOperand(1)->stripPointerCasts());
      llvm::StringRef Content;
      llvm::getConstantStringInfo(GV, Content);
      if (Content == LUTHIER_HOOK_ATTRIBUTE) {
        Hooks.push_back(Func);
        LLVM_DEBUG(llvm::dbgs() << "Found hook " << Func->getName() << ".\n");
      }
    };
  }
  return llvm::Error::success();
}

llvm::Error preprocessAndSaveModuleToStream(
    llvm::Module &Module, llvm::SmallVector<std::string> StaticVariables,
    uint64_t &CUID, llvm::SmallVector<char> &BitcodeOut) {
  // Extract all the hooks
  llvm::SmallVector<llvm::Function *> Hooks;
  LUTHIER_RETURN_ON_ERROR(getHooks(Module, Hooks));

  // Remove the annotations variable from the Module
  Module.getGlobalVariable("llvm.global.annotations")->removeFromParent();

  // Give each Hook function a "hook" attribute
  for (auto &Hook : Hooks) {
    Hook->addFnAttr(LUTHIER_HOOK_ATTRIBUTE);
  }

  // Remove all kernels that are meant to serve as a host handle
  for (auto &F : llvm::make_early_inc_range(Module.functions())) {
    if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
      F.removeFromParent();
    }
  }

  // Convert all global variables to extern, remove any managed variable
  // initializers
  // Remove any unnecessary variables
  // Extract the CUID for identification
  for (auto &GV : llvm::make_early_inc_range(Module.globals())) {
    auto GVName = GV.getName();
    if (GVName.ends_with(".managed") || GVName == luthier::ReservedManagedVar) {
      GV.removeFromParent();
    } else if (GVName.starts_with(HipCUIDPrefix)) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
          llvm::to_integer(GVName.substr(strlen(HipCUIDPrefix)), CUID)));
    } else {
      GV.setInitializer(nullptr);
      GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
      StaticVariables.push_back(std::string(GVName));
    }
  }

  // Save the modified module as a bitcode
  // When the CodeGenerator asks for a copy of this Module, it should be
  // copied over to the target app's LLVMContext
  llvm::raw_svector_ostream OS(BitcodeOut);
  llvm::WriteBitcodeToFile(Module, OS);
  return llvm::Error::success();
}

llvm::Expected<uint64_t> getCUIDOfLCO(const hsa::LoadedCodeObject &LCO) {
  llvm::SmallVector<hsa::ExecutableSymbol> Variables;
  LUTHIER_RETURN_ON_ERROR(LCO.getVariableSymbols(Variables));
  for (const auto &Var : Variables) {
    auto VarName = Var.getName();
    LUTHIER_RETURN_ON_ERROR(VarName.takeError());
    if (VarName->starts_with(luthier::HipCUIDPrefix)) {
      uint64_t CUID;
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
          llvm::to_integer(VarName->substr(strlen(HipCUIDPrefix)), CUID)));
      return CUID;
    }
  }
  llvm_unreachable("Could not find a CUID for the LCO");
}

////////////////////////////////////////////////////////////////////////////////
/// Instrumentation Module Implementation
////////////////////////////////////////////////////////////////////////////////

llvm::Error
StaticInstrumentationModule::registerExecutable(const hsa::Executable &Exec) {
  // Since static instrumentation modules are generated with HIP, we can
  // safely assume each Executable has a single LCO for now. Here we assert this
  // is indeed the case
  auto LCOs = llvm::cantFail(Exec.getLoadedCodeObjects());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(LCOs.size() == 1));

  if (PerAgentModuleExecutables.empty()) {
    // Initialize the bitcode if this is the first executable to be registered
    // Make a new context for modifying the bitcode before saving it to memory
    auto Context = std::make_unique<llvm::LLVMContext>();
    auto Module = extractBitcodeFromLCO(LCOs[0], *Context);
    LUTHIER_RETURN_ON_ERROR(Module.takeError());
    // Preprocess the Module, extract all its static variable names, and its
    // CUID
    LUTHIER_RETURN_ON_ERROR(preprocessAndSaveModuleToStream(
        **Module, GlobalVariables, CUID, BitcodeBuffer));
  }
  // Ensure the CUID of the Executable and the Module match
  auto ExecCUID = getCUIDOfLCO(LCOs[0]);
  LUTHIER_RETURN_ON_ERROR(ExecCUID.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(*ExecCUID == CUID));
  // Ensure this executable's agent doesn't already have another copy of this
  // executable loaded on it, then insert its information into the map
  auto Agent = LCOs[0].getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(!PerAgentModuleExecutables.contains(*Agent)));
  PerAgentModuleExecutables.insert({*Agent, Exec});
  // Populate the variables of this executable on its agent
  auto &SymbolMap = PerAgentGlobalVariables.insert({*Agent, {}}).first->second;
  for (const auto &GVName : GlobalVariables) {
    auto VariableExecSymbol = LCOs[0].getExecutableSymbolByName(GVName);
    LUTHIER_RETURN_ON_ERROR(VariableExecSymbol.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(VariableExecSymbol->has_value()));
    SymbolMap.insert({GVName, **VariableExecSymbol});
  }

  return llvm::Error::success();
}

llvm::Error
StaticInstrumentationModule::UnregisterExecutable(const hsa::Executable &Exec) {

  return llvm::Error::success();
}

llvm::Expected<const llvm::StringMap<hsa::ExecutableSymbol> &>
StaticInstrumentationModule::getGlobalHsaVariablesOnAgent(
    hsa::GpuAgent &Agent) {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(PerAgentGlobalVariables.contains(Agent)));
  return PerAgentGlobalVariables.at(Agent);
}

llvm::Expected<llvm::StringRef>
StaticInstrumentationModule::convertHookHandleToHookName(const void *Handle) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(HookHandleMap.contains(Handle)));
  return HookHandleMap[Handle];
}

llvm::Error StaticInstrumentationModule::getGlobalVariablesOnAgent(
    hsa::GpuAgent &Agent, llvm::StringMap<void *> &Out) {
  auto GlobalHsaVariablesOnAgent = this->getGlobalHsaVariablesOnAgent(Agent);
  LUTHIER_RETURN_ON_ERROR(GlobalHsaVariablesOnAgent.takeError());

  for (const auto &[Name, ExecVar] : *GlobalHsaVariablesOnAgent) {
    LUTHIER_RETURN_ON_MOVE_INTO_FAIL(luthier::address_t, VariableAddress,
                                     ExecVar.getVariableAddress());
    Out.insert({Name, reinterpret_cast<void *>(
                          *reinterpret_cast<uint64_t *>(VariableAddress))});
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::orc::ThreadSafeModule>
InstrumentationModule::readBitcodeIntoContext(
    llvm::orc::ThreadSafeContext &Ctx) {
  auto Lock = Ctx.getLock();
  auto BCBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::toStringRef(BitcodeBuffer), "", false);
  auto Module = llvm::parseBitcodeFile(*BCBuffer, *Ctx.getContext());
  LUTHIER_RETURN_ON_ERROR(Module.takeError());
  return llvm::orc::ThreadSafeModule(std::move(*Module), Ctx);
}

////////////////////////////////////////////////////////////////////////////////
/// Tool Executable Manager Implementation
////////////////////////////////////////////////////////////////////////////////

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
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  for (const auto &LCO : *LCOs) {
    auto LuthierReservedSymbol =
        LCO.getExecutableSymbolByName(luthier::ReservedManagedVar);
    LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbol.takeError());
    if (LuthierReservedSymbol->has_value()) {
      // This LCO belongs to Luthier tool. Register it with the static
      // instrumentation module
      LUTHIER_RETURN_ON_ERROR(SIM.registerExecutable(Exec));
      break;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                 "Executable with handle {0:x} was registered as a static "
                 "instrumentation module.\n",
                 Exec.hsaHandle()));
  return llvm::Error::success();
}

llvm::Expected<const hsa::ExecutableSymbol &>
ToolExecutableManager::getInstrumentedKernel(
    const hsa::ExecutableSymbol &OriginalKernel) const {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(InstrumentedKernels.contains(OriginalKernel)));
  return std::get<hsa::ExecutableSymbol>(
      InstrumentedKernels.at(OriginalKernel));
}

llvm::Error ToolExecutableManager::loadInstrumentedKernel(
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

ToolExecutableManager::~ToolExecutableManager() {
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
bool ToolExecutableManager::isKernelInstrumented(
    const hsa::ExecutableSymbol &Kernel) const {
  return InstrumentedKernels.contains(Kernel);
}

} // namespace luthier
