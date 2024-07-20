//===-- instrumentation_module.cpp - Luthier Instrumentation Module -------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements Luthier's Instrumentation Module and its variants.
//===----------------------------------------------------------------------===//
#include "tooling_common/instrumentation_module.hpp"
#include "hsa/hsa_executable.hpp"
#include "hsa/hsa_executable_symbol.hpp"
#include "hsa/hsa_loaded_code_object.hpp"
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-instrumentation-module"

//===----------------------------------------------------------------------===//
// Instrumentation Module Pre-processing functions (Shared between all
// Instrumentation Module sub-classes)
//===----------------------------------------------------------------------===//
namespace luthier {

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
  bool FoundBitcodeSection{false};
  for (const llvm::object::SectionRef &Section : StorageELF->sections()) {
    auto SectionName = Section.getName();
    LUTHIER_RETURN_ON_ERROR(SectionName.takeError());
    if (*SectionName == ".llvmbc") {
      auto SectionContents = Section.getContents();
      LUTHIER_RETURN_ON_ERROR(SectionContents.takeError());
      auto Module = llvm::parseBitcodeFile(
          llvm::MemoryBufferRef(*SectionContents, ""), Context);
      LUTHIER_RETURN_ON_ERROR(Module.takeError());
      return std::move(*Module);
    }
  }
  return LUTHIER_ASSERTION(FoundBitcodeSection);
}

/// Returns groups the set of annotated values in \p M into instrumentation
/// hooks and intrinsics of instrumentation hooks \n
/// \note This function should get updated as Luthier's programming model
/// gets updated as well
/// \param [in] M Module to inspect
/// \param [out] Hooks a list of hook functions found in \p M
/// \param [out] Intrinsics a list of intrinsics found in \p M
/// \return any \c llvm::Error encountered during the process
static llvm::Error
getAnnotatedValues(const llvm::Module &M,
                   llvm::SmallVectorImpl<llvm::Function *> &Hooks,
                   llvm::SmallVectorImpl<llvm::Function *> &Intrinsics) {
  const llvm::GlobalVariable *V =
      M.getGlobalVariable("llvm.global.annotations");
  const llvm::ConstantArray *CA = cast<llvm::ConstantArray>(V->getOperand(0));
  for (llvm::Value *Op : CA->operands()) {
    auto *CS = cast<llvm::ConstantStruct>(Op);
    // The first field of the struct contains a pointer to the annotated
    // variable.
    llvm::Value *AnnotatedVal = CS->getOperand(0)->stripPointerCasts();
    if (auto *Func = llvm::dyn_cast<llvm::Function>(AnnotatedVal)) {
      // The second field contains a pointer to a global annotation string.
      auto *GV =
          cast<llvm::GlobalVariable>(CS->getOperand(1)->stripPointerCasts());
      llvm::StringRef Content;
      llvm::getConstantStringInfo(GV, Content);
      if (Content == LUTHIER_HOOK_ATTRIBUTE) {
        Hooks.push_back(Func);
        LLVM_DEBUG(llvm::dbgs() << "Found hook " << Func->getName() << ".\n");
      } else if (Content == LUTHIER_INTRINSIC_ATTRIBUTE) {
        Intrinsics.push_back(Func);
        LLVM_DEBUG(llvm::dbgs()
                   << "Found intrinsic " << Func->getName() << ".\n");
      }
    };
  }
  return llvm::Error::success();
}

llvm::Error preprocessAndSaveModuleToStream(
    llvm::Module &Module, llvm::SmallVector<std::string> &StaticVariables,
    std::string &CUID, llvm::SmallVector<char> &BitcodeOut) {
  // Extract all the hooks
  llvm::SmallVector<llvm::Function *, 4> Hooks;
  llvm::SmallVector<llvm::Function *, 4> Intrinsics;
  LUTHIER_RETURN_ON_ERROR(getAnnotatedValues(Module, Hooks, Intrinsics));

  // Remove the annotations variable from the Module now that it is processed
  auto AnnotationGV = Module.getGlobalVariable("llvm.global.annotations");
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(AnnotationGV != nullptr));
  AnnotationGV->dropAllReferences();
  AnnotationGV->eraseFromParent();

  // Remove the llvm.used and llvm.compiler.use variable list
  for (const auto &VarName : {"llvm.compiler.used", "llvm.used"}) {
    auto LLVMUsedVar = Module.getGlobalVariable(VarName);
    if (LLVMUsedVar != nullptr) {
      LLVMUsedVar->dropAllReferences();
      LLVMUsedVar->eraseFromParent();
    }
  }

  // Give each Hook function a "hook" attribute
  for (auto &Hook : Hooks) {
    Hook->addFnAttr(LUTHIER_HOOK_ATTRIBUTE);
    Hook->addFnAttr(llvm::Attribute::AlwaysInline);
  }

  // Remove the body of each intrinsic function and make them extern
  for (auto &Intrinsic: Intrinsics) {
    Intrinsic->deleteBody();
  }

  // Remove all kernels that are meant to serve as a host handle
  for (auto &F : llvm::make_early_inc_range(Module.functions())) {

    if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
      F.dropAllReferences();
      F.eraseFromParent();
    }
  }

  // Convert all global variables to extern, remove any managed variable
  // initializers
  // Remove any unnecessary variables (e.g. "llvm.metadata")
  // Extract the CUID for identification
  for (auto &GV : llvm::make_early_inc_range(Module.globals())) {
    auto GVName = GV.getName();
    if (GVName.ends_with(".managed") || GVName == luthier::ReservedManagedVar ||
        GV.getSection() == "llvm.metadata") {
      GV.dropAllReferences();
      GV.eraseFromParent();
    } else if (GVName.starts_with(HipCUIDPrefix)) {
      CUID = GVName.substr(strlen(HipCUIDPrefix));
    } else {
      GV.setInitializer(nullptr);
      GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
      GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
      GV.setDSOLocal(false);
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

llvm::Expected<llvm::StringRef> getCUIDOfLCO(const hsa::LoadedCodeObject &LCO) {
  llvm::SmallVector<hsa::ExecutableSymbol> Variables;
  LUTHIER_RETURN_ON_ERROR(LCO.getVariableSymbols(Variables));
  for (const auto &Var : Variables) {
    auto VarName = Var.getName();
    LUTHIER_RETURN_ON_ERROR(VarName.takeError());
    if (VarName->starts_with(luthier::HipCUIDPrefix)) {
      return VarName->substr(strlen(HipCUIDPrefix));
    }
  }
  llvm_unreachable("Could not find a CUID for the LCO");
}

//===----------------------------------------------------------------------===//
// Instrumentation Module Implementation
//===----------------------------------------------------------------------===//

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
StaticInstrumentationModule::unregisterExecutable(const hsa::Executable &Exec) {
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  // There's only a single LCO in HIP FAT binaries. Get its agent.
  auto Agent = (*LCOs)[0].getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  // Remove the agent from the variable list and the executable list
  PerAgentGlobalVariables.erase(*Agent);
  PerAgentModuleExecutables.erase(*Agent);
  // If no copies of this module is present on any of the agents, then
  // the lifetime of the static module has ended. Perform a cleanup
  if (PerAgentModuleExecutables.empty()) {
    HookHandleMap.clear();
    GlobalVariables.clear();
  }
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
StaticInstrumentationModule::convertHookHandleToHookName(
    const void *Handle) const {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(HookHandleMap.contains(Handle)));
  return HookHandleMap.at(Handle);
}

llvm::Expected<bool>
StaticInstrumentationModule::isStaticInstrumentationModuleExecutable(
    const hsa::Executable &Exec) {
  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
  for (const auto &LCO : *LCOs) {
    auto LuthierReservedSymbol =
        LCO.getExecutableSymbolByName(luthier::ReservedManagedVar);
    LUTHIER_RETURN_ON_ERROR(LuthierReservedSymbol.takeError());
    if (LuthierReservedSymbol->has_value()) {
      return true;
    }
  }
  return false;
}
llvm::Expected<std::optional<luthier::address_t>>
StaticInstrumentationModule::getGlobalVariablesLoadedOnAgent(
    llvm::StringRef GVName, const hsa::GpuAgent &Agent) const {
  auto VariableSymbolMapIt = PerAgentGlobalVariables.find(Agent);

  if (VariableSymbolMapIt != PerAgentGlobalVariables.end()) {
    auto VariableSymbolIt = VariableSymbolMapIt->second.find(GVName);
    // Ensure the variable name is indeed in the map
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        VariableSymbolIt != VariableSymbolMapIt->second.end()));
    auto VariableSymbol = VariableSymbolIt->second;
    LUTHIER_RETURN_ON_MOVE_INTO_FAIL(luthier::address_t, VariableAddress,
                                     VariableSymbol.getVariableAddress());
    return VariableAddress;
  } else {
    // If the agent is not in the map, then it probably wasn't loaded on the
    // device
    return std::nullopt;
  }
}

llvm::Expected<llvm::orc::ThreadSafeModule>
InstrumentationModule::readBitcodeIntoContext(
    llvm::orc::ThreadSafeContext &Ctx) const {
  auto Lock = Ctx.getLock();
  auto BCBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::toStringRef(BitcodeBuffer), "", false);
  auto Module = llvm::parseBitcodeFile(*BCBuffer, *Ctx.getContext());
  LUTHIER_RETURN_ON_ERROR(Module.takeError());
  return llvm::orc::ThreadSafeModule(std::move(*Module), Ctx);
}

} // namespace luthier