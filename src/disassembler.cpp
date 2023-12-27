#include "disassembler.hpp"

#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/MCAsmLexer.h>
#include <llvm/MC/MCParser/MCAsmParser.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

#include "context_manager.hpp"
#include "hsa_executable.hpp"
#include "hsa_isa.hpp"

namespace {

typedef std::pair<std::string, bool> endPgmCallbackData;
typedef std::pair<std::string, luthier_address_t> sizeCallbackData;

uint64_t endPgmReadMemoryCallback(luthier_address_t from, char *to, size_t size, void *userData) {
    bool isEndOfProgram = reinterpret_cast<endPgmCallbackData *>(userData)->second;
    if (isEndOfProgram) return 0;
    else {
        std::memcpy(reinterpret_cast<void *>(to), reinterpret_cast<const void *>(from), size);
        return size;
    }
}

auto sizeReadMemoryCallback(luthier_address_t from, char *to, size_t size, void *userData) {
    luthier_address_t progEndAddr = reinterpret_cast<sizeCallbackData *>(userData)->second;

    if ((from + size) > progEndAddr) {
        if (from < progEndAddr) {
            size_t lastChunkSize = progEndAddr - from;
            std::memcpy(reinterpret_cast<void *>(to), reinterpret_cast<const void *>(from), lastChunkSize);
            return lastChunkSize;
        } else
            return size_t{0};
    } else {
        std::memcpy(reinterpret_cast<void *>(to), reinterpret_cast<const void *>(from), size);
        return size;
    }
};

void endPgmPrintInstructionCallback(const char *instruction, void *userData) {
    auto out = reinterpret_cast<endPgmCallbackData *>(userData);
    out->first = std::string(instruction);
}

auto sizePrintInstructionCallback(const char *instruction, void *userData) {
    auto out = reinterpret_cast<sizeCallbackData *>(userData);
    out->first = std::string(instruction);
};

void printAddressCallback(uint64_t Address, void *UserData) {}

}// namespace

//std::vector<luthier::Instr> luthier::Disassembler::disassemble(luthier_address_t kernelObject) {
//    auto symbol = hsa::ExecutableSymbol::fromKernelDescriptor(reinterpret_cast<const hsa::KernelDescriptor *>(kernelObject));
//    auto isa = symbol.getAgent().getIsa()[0];
//
//    luthier_address_t kernelEntryPoint = symbol.getKernelDescriptor()->getEntryPoint();
//
//    // Disassemble using AMD_COMGR
//
//    amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
//
//    amd_comgr_disassembly_info_t disassemblyInfo = getEndPgmDisassemblyInfo(isa);
//
//    uint64_t instrAddr = kernelEntryPoint;
//    uint64_t instrSize = 0;
//    endPgmCallbackData kdDisassemblyCallbackData{{}, false};
//    std::vector<luthier::Instr> out;
//    while (status == AMD_COMGR_STATUS_SUCCESS && !kdDisassemblyCallbackData.second) {
//        status = amd_comgr_disassemble_instruction(
//            disassemblyInfo, instrAddr, (void *) &kdDisassemblyCallbackData, &instrSize);
//        out.emplace_back(kdDisassemblyCallbackData.first, symbol.getAgent().asHsaType(), symbol.getExecutable().asHsaType(), symbol.asHsaType(), instrAddr, instrSize);
//        kdDisassemblyCallbackData.second = kdDisassemblyCallbackData.first.find("s_endpgm") != std::string::npos;
//        instrAddr += instrSize;
//    }
//    return out;
//}

std::vector<luthier::Instr> luthier::Disassembler::disassemble(const hsa::Isa &isa, luthier::byte_string_view code) {

    auto disassemblyInfo = getSizeDisassemblyInfo(isa);

    auto instrAddr = reinterpret_cast<luthier_address_t>(code.data());
    uint64_t instrSize = 0;
    std::pair<std::string, luthier_address_t> disassemblyCallbackData{
        {},
        reinterpret_cast<luthier_address_t>(code.data()) + code.size()};
    std::vector<Instr> out;
    while (true) {
        auto Status = amd_comgr_disassemble_instruction(disassemblyInfo, instrAddr, (void *) &disassemblyCallbackData,
                                                        &instrSize);
        if (Status == AMD_COMGR_STATUS_SUCCESS) {
            out.emplace_back(disassemblyCallbackData.first, instrAddr, instrSize);
            instrAddr += instrSize;
        } else
            break;
    }
    return out;
}

//TODO: implement these functions
std::vector<luthier::Instr> luthier::Disassembler::disassemble(const hsa::ExecutableSymbol &symbol) {
    const auto isa = symbol.getAgent().getIsa()[0];
    const auto &targetInfo = ContextManager::instance().getLLVMTargetInfo(isa);
    const auto targetTriple = llvm::Triple(isa.getLLVMTargetTriple());
    std::unique_ptr<llvm::MCContext> ctx(new (std::nothrow) llvm::MCContext(
        targetTriple, targetInfo.MAI_.get(), targetInfo.MRI_.get(), targetInfo.STI_.get()));
    assert(ctx);

    std::unique_ptr<const llvm::MCDisassembler> disAsm(
        targetInfo.target_->createMCDisassembler(*(targetInfo.STI_), *ctx));
    assert(disAsm);

    std::unique_ptr<const llvm::MCInstrAnalysis> mia(targetInfo.target_->createMCInstrAnalysis(targetInfo.MII_.get()));

    fmt::println("MRI has value? {}", targetInfo.MRI_.operator bool());
    std::unique_ptr<llvm::MCInstPrinter> ip(targetInfo.target_->createMCInstPrinter(
        targetTriple, targetInfo.MAI_->getAssemblerDialect(), *targetInfo.MAI_, *targetInfo.MII_, *targetInfo.MRI_));
    assert(ip);

    luthier::byte_string_view code =
        symbol.getType() == HSA_SYMBOL_KIND_KERNEL ? symbol.getKernelCode() : symbol.getIndirectFunctionCode();

    fmt::println("code size: {}", code.size());
    size_t maxReadSize = targetInfo.MAI_->getMaxInstLength();
//
    size_t idx = 0;
    auto currentAddress = reinterpret_cast<luthier_address_t>(code.data());
    auto stopAddress = currentAddress + code.size();

    std::vector<llvm::MCInst> instructions;
    fmt::println("Stop address: {:#x}", stopAddress);
    while (idx < (code.size() + 20)) {
        fmt::println("Current Address: {:#x}", currentAddress);
        size_t readSize = (idx + maxReadSize) < code.size() ? maxReadSize : code.size() - idx;
        fmt::println("read size: {}", readSize);
        size_t instSize{};
        llvm::MCInst inst;
        std::string annotations;
        llvm::raw_string_ostream annotationsStream(annotations);
        if (disAsm->getInstruction(inst, instSize,
                                   code::toArrayRef<uint8_t>(code.substr(idx, readSize)),
                                   currentAddress, annotationsStream)
            != llvm::MCDisassembler::Success) {
            break;
        }

        fmt::println("inst size: {}", instSize);

        std::string instStr;
        llvm::raw_string_ostream instStream(instStr);

        ip->printInst(&inst, currentAddress, annotationsStream.str(), *targetInfo.STI_, instStream);
        idx += instSize;
        currentAddress += instSize;
        instructions.push_back(inst);
        fmt::println("{}", instStream.str());
    }

    return std::vector<luthier::Instr>();
}

std::vector<luthier::Instr> luthier::Disassembler::disassemble(const hsa::ExecutableSymbol &symbol, size_t size) {
    return std::vector<Instr>();
}

std::vector<luthier::Instr> luthier::Disassembler::disassemble(const code::SymbolView &symbol) {
    return std::vector<luthier::Instr>();
}

std::vector<luthier::Instr> luthier::Disassembler::disassemble(const code::SymbolView &symbol, size_t size) {
    return std::vector<Instr>();
}

std::vector<luthier::Instr> luthier::Disassembler::disassemble(const hsa::Isa &isa, luthier_address_t address) {
    return std::vector<Instr>();
}

amd_comgr_disassembly_info_t luthier::Disassembler::getEndPgmDisassemblyInfo(const hsa::Isa &isa) {
    if (!endPgmDisassemblyInfoMap_.contains(isa)) {
        amd_comgr_disassembly_info_t disassemblyInfo;
        LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info(isa.getName().c_str(), &endPgmReadMemoryCallback,
                                                                  &endPgmPrintInstructionCallback,
                                                                  &printAddressCallback, &disassemblyInfo));
        endPgmDisassemblyInfoMap_.insert({isa, disassemblyInfo});
    }
    return endPgmDisassemblyInfoMap_[isa];
}

amd_comgr_disassembly_info_t luthier::Disassembler::getSizeDisassemblyInfo(const hsa::Isa &isa) {
    if (!sizeDisassemblyInfoMap_.contains(isa)) {
        amd_comgr_disassembly_info_t disassemblyInfo;
        LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info(isa.getName().c_str(), &sizeReadMemoryCallback,
                                                                  &sizePrintInstructionCallback, &printAddressCallback,
                                                                  &disassemblyInfo));
        sizeDisassemblyInfoMap_.insert({isa, disassemblyInfo});
    }
    return sizeDisassemblyInfoMap_[isa];
}
