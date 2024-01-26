#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInst.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/TargetSelect.h>
#include <fstream>
#include "code_view.hpp"
#include <iostream>
#include <sstream>

struct TargetInfo {
  const llvm::Target* target_;
  std::unique_ptr<llvm::MCRegisterInfo> MRI_;
  std::unique_ptr<llvm::TargetOptions> options_;
  std::unique_ptr<llvm::MCAsmInfo> MAI_;
  std::unique_ptr<llvm::MCInstrInfo> MII_;
  std::unique_ptr<llvm::MCInstrAnalysis> MIA_;
  std::unique_ptr<llvm::MCSubtargetInfo> STI_;
  std::unique_ptr<llvm::MCInstPrinter> IP_;
};

void initializeTargetInfo(TargetInfo& TI, const char* TT, const char* proc, const char* featureString) {
    static bool isInitialized{false};
    if (!isInitialized) {
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUDisassembler();
        LLVMInitializeAMDGPUAsmParser();
        LLVMInitializeAMDGPUAsmPrinter();
        LLVMInitializeAMDGPUTargetMCA();
    }
    std::string error;
    TI.target_ = llvm::TargetRegistry::lookupTarget(TT, error);
    assert(TI.target_ && error.c_str());

    TI.MRI_.reset(TI.target_->createMCRegInfo(TT));
    assert(TI.MRI_);

    TI.options_ = std::make_unique<llvm::TargetOptions>();
    assert(TI.options_);

    TI.MAI_.reset(TI.target_->createMCAsmInfo(*TI.MRI_, TT, TI.options_->MCOptions));
    assert(TI.MAI_);

    TI.MII_.reset(TI.target_->createMCInstrInfo());
    assert(TI.MII_);

    TI.MIA_.reset(TI.target_->createMCInstrAnalysis(TI.MII_.get()));
    assert(TI.MIA_);

    TI.STI_.reset(
        TI.target_->createMCSubtargetInfo(TT, proc, featureString)
    );
    assert(TI.STI_);

    TI.IP_.reset(
        TI.target_->createMCInstPrinter(llvm::Triple(TT), TI.MAI_->getAssemblerDialect(), *TI.MAI_, *TI.MII_, *TI.MRI_)
        );
    assert(TI.IP_);
}


std::vector<llvm::MCInst> disassembleSymbol(const llvm::Target *target,
                                            const char* TT,
                                            const llvm::MCAsmInfo* MAI,
                                            const llvm::MCRegisterInfo* MRI,
                                            const llvm::MCSubtargetInfo* STI, luthier::byte_string_view code) {
    std::unique_ptr<llvm::MCContext> ctx(
        new (std::nothrow) llvm::MCContext(llvm::Triple(TT), MAI, MRI, STI)
        );
    std::unique_ptr<llvm::MCDisassembler> disassembler(
        target->createMCDisassembler(*STI, *ctx)
        );

    size_t maxReadSize = MAI->getMaxInstLength();
    size_t idx = 0;
    auto currentAddress = reinterpret_cast<uint64_t>(code.data());
    std::vector<llvm::MCInst> instructions;

    //TODO: Check if currentAddress needs to be bundled with MCINst
    while (idx < code.size()) {
        size_t readSize = (idx + maxReadSize) < code.size() ? maxReadSize : code.size() - idx;
        size_t instSize{};
        llvm::MCInst inst;
        std::string annotations;
        llvm::raw_string_ostream annotationsStream(annotations);
        if (disassembler->getInstruction(inst, instSize, luthier::code::toArrayRef<uint8_t>(code.substr(idx, readSize)),
                                   currentAddress, annotationsStream)
            != llvm::MCDisassembler::Success) {
            break;
        }
        inst.setLoc(llvm::SMLoc::getFromPointer(reinterpret_cast<const char *>(currentAddress)));

        idx += instSize;
        currentAddress += instSize;
        instructions.push_back(inst);
    }

    return instructions;
}

luthier::byte_string_t readCodeObject(const char* path) {
    auto f = std::make_unique<std::ifstream>();
    f->open( path, std::ios::in | std::ios::binary );
    assert(f != nullptr);
    assert(f->is_open());
    std::ostringstream ss;
    ss << f->rdbuf();

    luthier::byte_string_t codeObject{reinterpret_cast<std::byte*>(ss.str().data()), ss.str().size()};
    std::cout << "Size of code object: " << codeObject.size() << std::endl;
    return codeObject;
}

// ASSUMES THE SYMBOL IS FOUND!
std::string findKernelName(std::shared_ptr<luthier::code::ElfView> elf) {
    std::string kernelName;
    for (unsigned int i = 0; i < elf->getNumSymbols(); i++) {
        auto symbol = elf->getSymbol(i);
        auto symbolName = symbol->getName();
        auto kdPos = symbolName.find(".kd");
        if (kdPos != std::string::npos) {
            kernelName = symbol->getName().substr(0, kdPos);
            break;
        }
    }
    return kernelName;
}

void printInstructions(const std::vector<llvm::MCInst>& instructions, llvm::MCInstPrinter& IP,
                       const llvm::MCSubtargetInfo& STI) {
    for (const auto& inst: instructions) {
        std::string instStr;
        llvm::raw_string_ostream instStream(instStr);
        IP.printInst(&inst, reinterpret_cast<size_t>(inst.getLoc().getPointer()),
                                                 "", STI, instStream);
        std::cout << instStr << std::endl;
    }

}


int main(int argc, char** argv) {
    // Declare and initialize target-specific things in LLVM
    constexpr const char* targetTriple{"amdgcn-amd-amdhsa-"};
    constexpr const char* processor{"gfx908"};
    constexpr const char* featureString{",+sramecc,-xnack"};
    TargetInfo TI;
    initializeTargetInfo(TI, targetTriple, processor, featureString);

    // Read in the code object
    luthier::byte_string_t codeObject = readCodeObject(argv[1]);
    auto elfView = luthier::code::ElfView::makeView(codeObject);

    // Find the KD Symbol and the kernel code symbol
    auto kernelName = findKernelName(elfView);
    auto kernelCodeSymbol = *elfView->getSymbol(kernelName);
    auto kdSymbol = *elfView->getSymbol(kernelName + ".kd");
    std::cout << "Size of the kernel: " << kernelCodeSymbol.getSize() << std::endl;

    auto kernelInstructions = disassembleSymbol(TI.target_, targetTriple, TI.MAI_.get(), TI.MRI_.get(), TI.STI_.get(),
                                                kernelCodeSymbol.getView());

    std::cout << "Number of instructions: " << kernelInstructions.size() << std::endl;
    printInstructions(kernelInstructions, *TI.IP_, *TI.STI_);

    auto context = std::make_unique<llvm::LLVMContext>();
    assert(context);
    auto theTargetMachine = std::unique_ptr<llvm::LLVMTargetMachine>(
        reinterpret_cast<llvm::LLVMTargetMachine *>(TI.target_->createTargetMachine(
            targetTriple,
            processor, featureString, *TI.options_,
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
