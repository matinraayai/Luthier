#include "disassembler.hpp"
#include "hsa_isa.hpp"
#include "hsa_executable.hpp"

namespace {

typedef std::pair<std::string, bool> endPgmCallbackData;
typedef std::pair<std::string, luthier_address_t> sizeCallbackData;

uint64_t endPgmReadMemoryCallback(luthier_address_t from, char *to, size_t size, void *userData) {
    bool isEndOfProgram = reinterpret_cast<endPgmCallbackData*>(userData)->second;
    if (isEndOfProgram)
        return 0;
    else {
        std::memcpy(reinterpret_cast<void *>(to), reinterpret_cast<const void *>(from), size);
        return size;
    }
}

auto sizeReadMemoryCallback(luthier_address_t from, char *to, size_t size, void *userData) {
    luthier_address_t progEndAddr = reinterpret_cast<sizeCallbackData*>(userData)->second;

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
    auto out = reinterpret_cast<endPgmCallbackData*>(userData);
    out->first = std::string(instruction);
}

auto sizePrintInstructionCallback(const char *instruction, void *userData) {
    auto out = reinterpret_cast<sizeCallbackData*>(userData);
    out->first = std::string(instruction);
};

void printAddressCallback(uint64_t Address, void *UserData) {}


luthier_address_t getKdEntryPoint(luthier_address_t kernelObject) {
    const kernel_descriptor_t *kernelDescriptor{nullptr};
    const auto& amdTable = luthier::HsaInterceptor::instance().getHsaVenAmdLoaderTable();
    LUTHIER_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(kernelObject),
                                                                   reinterpret_cast<const void **>(&kernelDescriptor)));
    return reinterpret_cast<luthier_address_t>(kernelObject) + kernelDescriptor->kernel_code_entry_byte_offset;
}

}// namespace

std::vector<luthier::Instr> luthier::Disassembler::disassemble(luthier_address_t kernelObject) {
    auto symbol = hsa::ExecutableSymbol::fromKernelDescriptor(reinterpret_cast<const hsa::KernelDescriptor*>(kernelObject));
    std::string isaName = symbol.getAgent().getIsa()[0].getName();

    luthier_address_t kernelEntryPoint = getKdEntryPoint(kernelObject);


    // Disassemble using AMD_COMGR

    amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;

    amd_comgr_disassembly_info_t disassemblyInfo = getEndPgmDisassemblyInfo(isaName);

    uint64_t instrAddr = kernelEntryPoint;
    uint64_t instrSize = 0;
    endPgmCallbackData kdDisassemblyCallbackData{{}, false};
    std::vector<luthier::Instr> out;
    while (status == AMD_COMGR_STATUS_SUCCESS && !kdDisassemblyCallbackData.second) {
        status = amd_comgr_disassemble_instruction(
            disassemblyInfo, instrAddr, (void *) &kdDisassemblyCallbackData, &instrSize);
        out.emplace_back(kdDisassemblyCallbackData.first, symbol.getAgent().asHsaType(), symbol.getExecutable().asHsaType(), symbol.asHsaType(), instrAddr, instrSize);
        kdDisassemblyCallbackData.second = kdDisassemblyCallbackData.first.find("s_endpgm") != std::string::npos;
        instrAddr += instrSize;
    }
    return out;
}

std::vector<luthier::Instr> luthier::Disassembler::disassemble(const hsa::GpuAgent& agent, luthier::byte_string_view code) {

    std::string isaName = agent.getIsa()[0].getName();

    auto disassemblyInfo = getSizeDisassemblyInfo(isaName);

    uint64_t instrAddr = reinterpret_cast<uint64_t>(code.data());
    uint64_t instrSize = 0;
    std::pair<std::string, luthier_address_t> kdDisassemblyCallbackData{{}, reinterpret_cast<luthier_address_t>(code.data()) + code.size()};
    std::vector<Instr> out;
    std::cout << "Entry + size: " << std::hex << kdDisassemblyCallbackData.second << std::dec << std::endl;
    while (true) {
        auto Status = amd_comgr_disassemble_instruction(
            disassemblyInfo, instrAddr, (void *) &kdDisassemblyCallbackData, &instrSize);
        if (Status == AMD_COMGR_STATUS_SUCCESS) {
            out.emplace_back(kdDisassemblyCallbackData.first, instrAddr, instrSize);
            instrAddr += instrSize;
        } else
            break;
    }
    return out;
}

//TODO: implement these functions
std::vector<luthier::Instr> luthier::Disassembler::disassemble(hsa_executable_symbol_t symbol) {
    return std::vector<luthier::Instr>();
}
std::vector<luthier::Instr> luthier::Disassembler::disassemble(const hsa::GpuAgent& agent, luthier_address_t address) {
    return std::vector<Instr>();
}
std::vector<luthier::Instr> luthier::Disassembler::disassemble(luthier_address_t kernelObject, size_t size) {
    return std::vector<Instr>();
}
std::vector<luthier::Instr> luthier::Disassembler::disassemble(hsa::ExecutableSymbol symbol, size_t size) {
    return std::vector<Instr>();
}

amd_comgr_disassembly_info_t luthier::Disassembler::getEndPgmDisassemblyInfo(const std::string &isa) {
    if (!endPgmDisassemblyInfoMap_.contains(isa)) {
        amd_comgr_disassembly_info_t disassemblyInfo;
        LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info(isa.c_str(),
                                                                &endPgmReadMemoryCallback,
                                                                &endPgmPrintInstructionCallback,
                                                                &printAddressCallback,
                                                                &disassemblyInfo));
        endPgmDisassemblyInfoMap_.insert({isa, disassemblyInfo});
    }
    return endPgmDisassemblyInfoMap_[isa];
}

amd_comgr_disassembly_info_t luthier::Disassembler::getSizeDisassemblyInfo(const std::string &isa) {
    if (!sizeDisassemblyInfoMap_.contains(isa)) {
        amd_comgr_disassembly_info_t disassemblyInfo;
        LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info(isa.c_str(),
                                                                &sizeReadMemoryCallback,
                                                                &sizePrintInstructionCallback,
                                                                &printAddressCallback,
                                                                &disassemblyInfo));
        sizeDisassemblyInfoMap_.insert({isa, disassemblyInfo});
    }
    return sizeDisassemblyInfoMap_[isa];
}
