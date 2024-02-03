#include "disassembler.hpp"

#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <llvm/Target/TargetMachine.h>

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_isa.hpp"
#include "target_manager.hpp"

namespace luthier {

const Disassembler::DisassemblyInfo &luthier::Disassembler::getDisassemblyInfo(const luthier::hsa::Isa &isa) {
    if (!disassemblyInfoMap_.contains(isa)) {
        const auto &targetInfo = TargetManager::instance().getTargetInfo(isa);
        const auto targetTriple = llvm::Triple(isa.getLLVMTargetTriple());
        std::unique_ptr<llvm::MCContext> ctx(new (std::nothrow) llvm::MCContext(
            targetTriple, targetInfo.getMCAsmInfo(), targetInfo.getMCRegisterInfo(), targetInfo.getMCSubTargetInfo()));
        std::unique_ptr<const llvm::MCDisassembler> disAsm(
            targetInfo.getTarget()->createMCDisassembler(*(targetInfo.getMCSubTargetInfo()), *ctx));
        disassemblyInfoMap_.insert({isa, DisassemblyInfo{std::move(ctx), std::move(disAsm)}});
    }
    return disassemblyInfoMap_.at(isa);
}

std::vector<llvm::MCInst> Disassembler::disassemble(const hsa::Isa &isa, luthier::byte_string_view code) {
    const auto &disassemblyInfo = getDisassemblyInfo(isa);
    const auto &targetInfo = TargetManager::instance().getTargetInfo(isa);
    const auto &disAsm = disassemblyInfo.disAsm_;

    size_t maxReadSize = targetInfo.getMCAsmInfo()->getMaxInstLength();
    size_t idx = 0;
    auto currentAddress = reinterpret_cast<luthier_address_t>(code.data());
    std::vector<llvm::MCInst> instructions;

    //TODO: Check if currentAddress needs to be bundled with MCINst
    while (idx < code.size()) {
        size_t readSize = (idx + maxReadSize) < code.size() ? maxReadSize : code.size() - idx;
        size_t instSize{};
        llvm::MCInst inst;
        std::string annotations;
        llvm::raw_string_ostream annotationsStream(annotations);
        if (disAsm->getInstruction(inst, instSize, code::toArrayRef<uint8_t>(code.substr(idx, readSize)),
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

const std::vector<hsa::Instr> *luthier::Disassembler::disassemble(const hsa::ExecutableSymbol &symbol,
                                                                  std::optional<hsa::Isa> isa,
                                                                  std::optional<size_t> size) {
    if (!disassembledSymbols_.contains(symbol)) {
        const auto symbolType = symbol.getType();
        LUTHIER_CHECK((symbolType != HSA_SYMBOL_KIND_VARIABLE));
        if (!isa.has_value()) isa.emplace(symbol.getAgent().getIsa());

        luthier::byte_string_view code =
            symbol.getType() == HSA_SYMBOL_KIND_KERNEL ? symbol.getKernelCode() : symbol.getIndirectFunctionCode();

        if (size.has_value()) code = code.substr(0, *size > code.size() ? code.size() : *size);

        std::vector<llvm::MCInst> instructions = disassemble(*isa, code);
        disassembledSymbols_.insert({symbol, std::make_unique<std::vector<hsa::Instr>>()});
        auto &out = disassembledSymbols_.at(symbol);
        out->reserve(instructions.size());
        for (const auto &inst: instructions) { out->push_back(hsa::Instr(inst, symbol)); }
    }
    return disassembledSymbols_.at(symbol).get();
}

std::vector<llvm::MCInst> luthier::Disassembler::disassemble(const code::SymbolView &symbol, const hsa::Isa &isa,
                                                             std::optional<size_t> size) {
    const auto symbolType = symbol.getType();
    LUTHIER_CHECK((symbolType == ELFIO::STT_FUNC));

    const auto &symbolName = symbol.getName();
    // If the kd symbol was passed, get the function symbol associated with it in the parent ELF.
    luthier::byte_string_view code;

    size_t kdPos = symbolName.find(".kd");
    if (kdPos != std::string::npos) {
        code = symbol.getElfView()->getSymbol(symbolName.substr(0, kdPos))->getView();
    } else {
        code = symbol.getView();
    }

    if (size.has_value()) code = code.substr(0, *size > code.size() ? code.size() : *size);

    return disassemble(isa, code);
}

luthier::Disassembler::~Disassembler() {
    disassemblyInfoMap_.clear();
    moduleInfoMap_.clear();
    disassembledSymbols_.clear();
}
void Disassembler::liftKernelModule(const hsa::ExecutableSymbol &symbol) {
    if (!moduleInfoMap_.contains(symbol)) {
        auto agent = symbol.getAgent();
        LUTHIER_CHECK((symbol.getType() == HSA_SYMBOL_KIND_KERNEL));
        auto isa = agent.getIsa();
        auto targetInfo = luthier::TargetManager::instance().getTargetInfo(isa);

        auto target = targetInfo.getTarget();

        auto mai = targetInfo.getMCAsmInfo();
        //        auto mai = target->createMCAsmInfo(*mri, isa.getLLVMTargetTriple(), targetOptions.MCOptions);
        //        assert(mai);

        auto mii = targetInfo.getMCInstrInfo();
        //        auto mii = target->createMCInstrInfo();
        //        assert(mii);

        auto mia = targetInfo.getMCInstrAnalysis();
        //        auto mia = target->createMCInstrAnalysis(mii);
        //        assert(mia);
        auto sti = targetInfo.getMCSubTargetInfo();

//        auto triple = isa.getLLVMTargetTriple();
//        auto cpu = isa.getProcessor();
//        auto features = isa.getFeatureString();

        constexpr const char* triple{"amdgcn-amd-amdhsa-"};
        constexpr const char* cpu{"gfx908"};
        constexpr const char* features{",+sramecc,-xnack"};

//        llvm::TargetOptions options;

        try {
            std::unique_ptr<llvm::LLVMTargetMachine> theTargetMachine(static_cast<llvm::LLVMTargetMachine *>(
                target->createTargetMachine(llvm::Triple(triple).normalize(), cpu, features,
                                            targetInfo.getTargetOptions(), llvm::Reloc::PIC_)));
            LUTHIER_CHECK(theTargetMachine);


            auto context = std::make_unique<llvm::LLVMContext>();
            //        auto proc = new std::string(isa.getProcessor());
            //        auto featureString = new std::string(isa.getFeatureString());
            LUTHIER_CHECK(context);

            //        fmt::println("{}", targetOptions.StackUsageOutput);
            //        delete proc;
            //        delete featureString;

            llvm::outs() << "Context was created!\n";
            auto module = std::make_unique<llvm::Module>(symbol.getName(), *context);
            LUTHIER_CHECK(module);
            module->setDataLayout(theTargetMachine->createDataLayout());
            auto mmiwp = std::make_unique<llvm::MachineModuleInfoWrapperPass>(theTargetMachine.get());
            LUTHIER_CHECK(mmiwp);
            llvm::Type *const returnType = llvm::Type::getVoidTy(module->getContext());
            LUTHIER_CHECK(returnType);
            llvm::Type *const memParamType = llvm::PointerType::get(
                llvm::Type::getInt32Ty(module->getContext()), llvm::AMDGPUAS::GLOBAL_ADDRESS /*default address space*/);
            LUTHIER_CHECK(memParamType);
            llvm::FunctionType *FunctionType = llvm::FunctionType::get(returnType, {memParamType}, false);
            LUTHIER_CHECK(FunctionType);
            llvm::Function *F =
                llvm::Function::Create(FunctionType, llvm::GlobalValue::ExternalLinkage, symbol.getName(), *module);
            LUTHIER_CHECK(F);
            F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

            llvm::BasicBlock *BB = llvm::BasicBlock::Create(module->getContext(), "", F);
            LUTHIER_CHECK(BB);
            new llvm::UnreachableInst(module->getContext(), BB);
            auto &mmi = mmiwp->getMMI();
//                    mmi.getOrCreateMachineFunction(*F);
            auto MF = new llvm::MachineFunction(*F, *theTargetMachine, *theTargetMachine->getSubtargetImpl(*F), 1, mmi);
//            llvm::MachineFunction& MF = mmi.getOrCreateMachineFunction(*F);
            LUTHIER_CHECK(MF);
            MF->setAlignment(llvm::Align(4096));
            //        mmi.getOrCreateMachineFunction(*F);
            //
            //
            const std::vector<hsa::Instr> *targetFunction = Disassembler::instance().disassemble(symbol);

            llvm::MachineBasicBlock *MBB = MF->CreateMachineBasicBlock();
            //    MF.push_back(MBB);

            for (const auto &i: *targetFunction) {
                auto inst = i.getInstr();

                const unsigned Opcode = inst.getOpcode();

                const llvm::MCInstrDesc &MCID = mii->get(Opcode);
                llvm::outs() << MCID.getOpcode() << "\n";
                llvm::MachineInstrBuilder Builder = llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
                for (unsigned OpIndex = 0, E = i.getInstr().getNumOperands(); OpIndex < E; ++OpIndex) {
                    const llvm::MCOperand &Op = inst.getOperand(OpIndex);
                    if (Op.isReg()) {
                        const bool IsDef = OpIndex < MCID.getNumDefs();
                        unsigned Flags = 0;
                        const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
                        if (IsDef && !OpInfo.isOptionalDef()) Flags |= llvm::RegState::Define;
                        Builder.addReg(Op.getReg(), Flags);
                    } else if (Op.isImm()) {
                        Builder.addImm(Op.getImm());
                    } else if (!Op.isValid()) {
                        llvm_unreachable("Operand is not set");
                    } else {
                        llvm_unreachable("Not yet implemented");
                    }
                }
            }
            MBB->dump();
            llvm::MachineFunctionProperties &properties = MF->getProperties();
            const llvm::MachineFunctionProperties &constProperties = MF->getProperties();
            MF->dump();
            llvm::outs() << llvm::formatv("Properties address: {0}\n", reinterpret_cast<uint64_t>(&properties));
            llvm::outs() << llvm::formatv("const Properties address: {0}\n", reinterpret_cast<uint64_t>(&constProperties));
            MF->dump();
            for (const auto& bb: *MF) {
                bb.print(llvm::outs());
            }
            properties.print(llvm::outs());
            constProperties.print(llvm::outs());
//            properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
//            properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
//            properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
//            delete mri;
//            //        delete targetOptions;
//            delete mai;
//            delete mii;
//            delete mia;
//            delete sti;

        }
        catch (std::exception& e) {
            llvm::outs() << "Caught the exception\n";
            llvm::outs() << "What: " << e.what() << "\n";
        };
        llvm::outs() << "continue executing..\n";


    }
}

}// namespace luthier