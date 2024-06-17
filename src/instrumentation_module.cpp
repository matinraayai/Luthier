#include "instrumentation_module.hpp"
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>

namespace luthier {

/// Extracts an LLVM bitcode from the ".llvmbc" section of the LCO's storage
/// ELF.
/// \param LCO the \c hsa::LoadedCodeObject containing the bitcode
/// \return an owning reference to the extracted \c llvm::Module, or an
/// \c llvm::Error if the bitcode was not found, or any other error was
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

static llvm::Error getHooks(const llvm::Module &M,
                            std::unordered_set<std::string> &Hooks) {
  const llvm::GlobalVariable *V =
      M.getGlobalVariable("llvm.global.annotations");
  const llvm::ConstantArray *CA = cast<llvm::ConstantArray>(V->getOperand(0));
  for (llvm::Value *Op : CA->operands()) {
    auto *CS = cast<llvm::ConstantStruct>(Op);
    // The first field of the struct contains a pointer to
    // the annotated variable.
    llvm::Value *AnnotatedVar = CS->getOperand(0)->stripPointerCasts();
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(!isa<llvm::Function>(AnnotatedVar)));

    auto *Func = cast<llvm::Function>(AnnotatedVar);
    llvm::outs() << "Function annotated: " << Func->getName() << "\n";

    // The second field contains a pointer to a global annotation string.
    auto *GV =
        cast<llvm::GlobalVariable>(CS->getOperand(1)->stripPointerCasts());
    llvm::StringRef Content;
    llvm::getConstantStringInfo(GV, Content);
    llvm::outs() << "Annotation: " << Content << "\n";
    Hooks.insert(std::string(Content));
  };
  return llvm::Error::success();
}

llvm::Expected<InstrumentationModule>
InstrumentationModule::create(const luthier::hsa::LoadedCodeObject &LCO) {
  InstrumentationModule IM;
  // Make a new context for modifying the bitcode before saving it to memory
  auto Context = std::make_unique<llvm::LLVMContext>();
  auto Module = extractBitcodeFromLCO(LCO, *Context);
  LUTHIER_RETURN_ON_ERROR(Module.takeError());
  // Extract all the hook names
  LUTHIER_RETURN_ON_ERROR(getHooks(**Module, IM.Hooks));
  (*Module)->getGlobalVariable("llvm.global.annotations")->removeFromParent();

  // Convert all global variables to extern, remove any managed variable
  // initializers
  // Remove any special llvm variables like LLVM used
  for (auto &GV : llvm::make_early_inc_range(Module.get()->globals())) {
    auto GVName = GV.getName();
    if (GVName.ends_with(".managed") || GVName == luthier::ReservedManagedVar ||
        GVName.starts_with("__hip_cuid")) {
      GV.removeFromParent();
    } else {
      auto HSASymbol = LCO.getExecutableSymbolByName(GVName);
      LUTHIER_RETURN_ON_ERROR(HSASymbol.takeError());
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(HSASymbol->has_value()));
      IM.StaticSymbols.insert({GVName, **HSASymbol});
      GV.setInitializer(nullptr);
      GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }
  }

  // Save the modified module as a bitcode
  // When the CodeGenerator asks for a copy of this Module, it should be
  // copied over to the target app's LLVMContext
  llvm::raw_svector_ostream OS(IM.BitcodeBuffer);
  llvm::WriteBitcodeToFile(**Module, OS);

  return IM;
}
llvm::Expected<llvm::orc::ThreadSafeModule>
InstrumentationModule::readBitcodeIntoModule(
    llvm::orc::ThreadSafeContext &Ctx) {
  auto Lock = Ctx.getLock();
  auto BCBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::toStringRef(BitcodeBuffer), "", false);
  auto Module = llvm::parseBitcodeFile(*BCBuffer, *Ctx.getContext());
  LUTHIER_RETURN_ON_ERROR(Module.takeError());
  return llvm::orc::ThreadSafeModule(std::move(*Module), Ctx);
}

} // namespace luthier