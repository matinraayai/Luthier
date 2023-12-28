#include "disassembler.hpp"

#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/TargetRegistry.h>

#include "context_manager.hpp"
#include "hsa_executable.hpp"
#include "hsa_isa.hpp"

namespace luthier {

hsa::Instr *Disassembler::createInstr(llvm::MCInst inst, luthier::hsa::ExecutableSymbol symbol) {
    auto instr = new hsa::Instr(std::move(inst), std::move(symbol));
    instrHandles_.insert(instr);
    return instr;
}

void luthier::Disassembler::destroyInstr(luthier::hsa::Instr *instr) {
    assert(instrHandles_.contains(instr));
    instrHandles_.erase(instr);
    delete instr;
}

const Disassembler::DisassemblyInfo &luthier::Disassembler::getDisassemblyInfo(const luthier::hsa::Isa &isa) {
    if (!disassemblyInfoMap_.contains(isa)) {
        const auto &targetInfo = ContextManager::instance().getLLVMTargetInfo(isa);
        const auto targetTriple = llvm::Triple(isa.getLLVMTargetTriple());
        std::unique_ptr<llvm::MCContext> ctx(new (std::nothrow) llvm::MCContext(
            targetTriple, targetInfo.MAI_.get(), targetInfo.MRI_.get(), targetInfo.STI_.get()));
        std::unique_ptr<const llvm::MCDisassembler> disAsm(
            targetInfo.target_->createMCDisassembler(*(targetInfo.STI_), *ctx));
        disassemblyInfoMap_.insert({isa, DisassemblyInfo{std::move(ctx), std::move(disAsm)}});
    }
    return disassemblyInfoMap_.at(isa);
}

std::vector<llvm::MCInst> Disassembler::disassemble(const hsa::Isa &isa, luthier::byte_string_view code) {
    const auto &disassemblyInfo = getDisassemblyInfo(isa);
    const auto &targetInfo = ContextManager::instance().getLLVMTargetInfo(isa);
    const auto &disAsm = disassemblyInfo.disAsm_;

    size_t maxReadSize = targetInfo.MAI_->getMaxInstLength();
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

        std::string instStr;
        llvm::raw_string_ostream instStream(instStr);

        targetInfo.IP_->printInst(&inst, currentAddress, annotationsStream.str(), *targetInfo.STI_, instStream);

        idx += instSize;
        currentAddress += instSize;
        instructions.push_back(inst);
        fmt::println("{}", instStream.str());
    }

    return instructions;
}

std::vector<hsa::Instr *> luthier::Disassembler::disassemble(const hsa::ExecutableSymbol &symbol,
                                                             std::optional<size_t> size) {
    const auto symbolType = symbol.getType();
    assert(symbolType != HSA_SYMBOL_KIND_VARIABLE);
    const auto isa = symbol.getAgent().getIsa()[0];

    luthier::byte_string_view code =
        symbol.getType() == HSA_SYMBOL_KIND_KERNEL ? symbol.getKernelCode() : symbol.getIndirectFunctionCode();

    if (size.has_value()) code = code.substr(0, *size > code.size() ? code.size() : *size);

    std::vector<llvm::MCInst> instructions = disassemble(isa, code);
    std::vector<hsa::Instr *> out;
    out.reserve(instructions.size());
    for (const auto &inst: instructions) { out.push_back(createInstr(inst, symbol)); }

    return out;
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
    for (auto inst: instrHandles_) { delete inst; }
    instrHandles_.clear();
}

}// namespace luthier