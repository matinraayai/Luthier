#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/TargetSelect.h>

#include <iostream>

int main() {
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTargetMCA();

    constexpr const char* targetTriple{"amdgcn-amd-amdhsa-"};
    constexpr const char* processor{"gfx908"};
    constexpr const char* featureString{",+sramecc,-xnack"};
    std::string error;

    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    assert(target && error.c_str());

    auto mri = target->createMCRegInfo(targetTriple);
    llvm::TargetOptions options;

    assert(mri);

    auto mai = std::unique_ptr<llvm::MCAsmInfo>(target->createMCAsmInfo(*mri, targetTriple, options.MCOptions));
    assert(mai);

    auto mii = std::unique_ptr<llvm::MCInstrInfo>(target->createMCInstrInfo());
    assert(mii);

    auto mia = std::unique_ptr<llvm::MCInstrAnalysis>(target->createMCInstrAnalysis(mii.get()));
    assert(mia);

    auto sti = std::unique_ptr<llvm::MCSubtargetInfo>(target->createMCSubtargetInfo(targetTriple,
                                                                                    processor, featureString));
    assert(sti);

    auto ip =
        std::unique_ptr<llvm::MCInstPrinter>
        (target->createMCInstPrinter(llvm::Triple(targetTriple), mai->getAssemblerDialect(), *mai, *mii, *mri));
    assert(ip);

    auto context = std::make_unique<llvm::LLVMContext>();
    assert(context);
    auto theTargetMachine = std::unique_ptr<llvm::LLVMTargetMachine>(
        reinterpret_cast<llvm::LLVMTargetMachine *>(target->createTargetMachine(
            targetTriple,
            processor, featureString, options,
            llvm::Reloc::Model::PIC_)));
    assert(theTargetMachine);
    std::unique_ptr<llvm::Module> module = std::make_unique<llvm::Module>("My Module", *context);
    assert(module);
    module->setDataLayout(theTargetMachine->createDataLayout());
    auto mmiwp = std::make_unique<llvm::MachineModuleInfoWrapperPass>(theTargetMachine.get());
    assert(mmiwp);
    llvm::Type *const returnType = llvm::Type::getVoidTy(module->getContext());
    assert(returnType);
    llvm::Type *const memParamType = llvm::PointerType::get(llvm::Type::getInt32Ty(module->getContext()),
                                                            1);
    assert(memParamType);
    llvm::FunctionType *FunctionType = llvm::FunctionType::get(returnType, {memParamType}, false);
    assert(FunctionType);
    llvm::Function *const F =
        llvm::Function::Create(FunctionType, llvm::GlobalValue::ExternalLinkage, "myfunc", *module);
    assert(F);
    F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

    llvm::BasicBlock *BB = llvm::BasicBlock::Create(module->getContext(), "", F);
    assert(BB);
    new llvm::UnreachableInst(module->getContext(), BB);
    auto& mmi = mmiwp->getMMI();
    mmi.getOrCreateMachineFunction(*F);
    llvm::MachineFunction &MF = mmi.getOrCreateMachineFunction(*F);
    MF.ensureAlignment(llvm::Align(4096));
    auto &properties = MF.getProperties();
    properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
    properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
    properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);

    std::cout << "Properties: ";
    std::string propertiesString;
    llvm::raw_string_ostream pOS(propertiesString);
    properties.print(pOS);
    std::cout << propertiesString << std::endl;


}
