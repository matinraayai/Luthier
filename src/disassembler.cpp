#include "disassembler.hpp"

#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
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
#include "llvm/IR/Instructions.h"
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
        assert(symbolType != HSA_SYMBOL_KIND_VARIABLE);
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
    assert(symbolType == ELFIO::STT_FUNC);

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
    disassembledSymbols_.clear();
}
void Disassembler::liftModule(const hsa::ExecutableSymbol &symbol) {
    auto agent = symbol.getAgent();
    auto &targetManager = luthier::TargetManager::instance();

    const auto &targetInfo = targetManager.getTargetInfo(agent.getIsa());
    auto Context = std::make_unique<llvm::LLVMContext>();
    auto triple = llvm::Triple(agent.getIsa().getLLVMTargetTriple());
    auto processor = agent.getIsa().getProcessor();
    auto featureString = agent.getIsa().getFeatureString();
//    auto targetOptions = new llvm::TargetOptions();
//    llvm::TargetOptions targetOptions;
//    targetOptions.MCOptions.IASSearchPaths.clear("");
    auto TheTargetMachine = std::unique_ptr<llvm::LLVMTargetMachine>(
        reinterpret_cast<llvm::LLVMTargetMachine *>(targetInfo.getTarget()->createTargetMachine(
            triple.str(), llvm::StringRef(processor), llvm::StringRef(featureString), targetInfo.getTargetOptions(),
            llvm::Reloc::Model::PIC_)));
    //
    fmt::println("Options: {}", targetInfo.getTargetOptions().ObjectFilenameForDebug);
    std::unique_ptr<llvm::Module> Module = std::make_unique<llvm::Module>("MyModule", *Context);
    auto DL = TheTargetMachine->createDataLayout();
    Module->setDataLayout(DL);
    auto MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(TheTargetMachine.get());
    //
    llvm::Type *const ReturnType = llvm::Type::getVoidTy(Module->getContext());
    llvm::Type *const MemParamType = llvm::PointerType::get(llvm::Type::getInt32Ty(Module->getContext()),
                                                            llvm::AMDGPUAS::GLOBAL_ADDRESS /*default address space*/);
    llvm::FunctionType *FunctionType = llvm::FunctionType::get(ReturnType, {MemParamType}, false);
    llvm::Function *const F =
        llvm::Function::Create(FunctionType, llvm::GlobalValue::ExternalLinkage, symbol.getName(), *Module);
    F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    //
    //
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(Module->getContext(), "", F);
    new llvm::UnreachableInst(Module->getContext(), BB);
    llvm::MachineFunction &MF = MMIWP->getMMI().getOrCreateMachineFunction(*F);
//    auto &Properties = MF.getProperties();
//    Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
//    Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
//    Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
    //
    //
    const std::vector<hsa::Instr> *targetFunction = Disassembler::instance().disassemble(symbol);

    llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
//    MF.push_back(MBB);

    for (const auto& i: *targetFunction) {
        auto inst = i.getInstr();
        const unsigned Opcode = inst.getOpcode();
        const llvm::MCInstrDesc &MCID = targetInfo.getMCInstrInfo()->get(Opcode);
        llvm::MachineInstrBuilder Builder = llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
        for (unsigned OpIndex = 0, E = i.getInstr().getNumOperands(); OpIndex < E;
             ++OpIndex) {
            const llvm::MCOperand &Op = inst.getOperand(OpIndex);
            if (Op.isReg()) {
                const bool IsDef = OpIndex < MCID.getNumDefs();
                unsigned Flags = 0;
                const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
                if (IsDef && !OpInfo.isOptionalDef())
                    Flags |= llvm::RegState::Define;
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

    //    new UnreachableInst(Module->getContext(), BB);
    //    return MMI->getOrCreateMachineFunction(*F);

    //        llvm::Function f;
    //    llvm::MachineFunction func;
}

}// namespace luthier