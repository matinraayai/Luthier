#include "disassembler.hpp"
#include "context_manager.hpp"
#include "hsa_intercept.hpp"

namespace {

typedef std::pair<std::string, bool> endPgmCallbackData;
typedef std::pair<std::string, sibir_address_t> sizeCallbackData;

uint64_t endPgmReadMemoryCallback(sibir_address_t from, char *to, size_t size, void *userData) {
    bool isEndOfProgram = reinterpret_cast<endPgmCallbackData*>(userData)->second;
    if (isEndOfProgram)
        return 0;
    else {
        std::memcpy(reinterpret_cast<void *>(to), reinterpret_cast<const void *>(from), size);
        return size;
    }
}

auto sizeReadMemoryCallback(sibir_address_t from, char *to, size_t size, void *userData) {
    sibir_address_t progEndAddr = reinterpret_cast<sizeCallbackData*>(userData)->second;

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


hsa_symbol_kind_t getSymbolKind(hsa_executable_symbol_t symbol) {
    auto coreHsaApi = sibir::HsaInterceptor::Instance().getSavedHsaTables().core;
    hsa_symbol_kind_t symbolKind;
    SIBIR_HSA_CHECK(coreHsaApi.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbolKind));
    return symbolKind;
}

sibir_address_t getKdEntryPoint(sibir_address_t kernelObject) {
    const kernel_descriptor_t *kernelDescriptor{nullptr};
    const auto& amdTable = sibir::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    SIBIR_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(kernelObject),
                                                                   reinterpret_cast<const void **>(&kernelDescriptor)));

    return reinterpret_cast<sibir_address_t>(kernelObject) + kernelDescriptor->kernel_code_entry_byte_offset;
}

std::tuple<hsa_agent_t, hsa_executable_t, hsa_executable_symbol_t> getKernelObjectInfo(sibir_address_t kernelObject) {
    const auto& amdTable = sibir::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    const auto& coreApi = sibir::HsaInterceptor::Instance().getSavedHsaTables().core;

    // A way to backtrack from the kernel object to the symbol it belongs to (besides keeping track of a map)
    hsa_executable_t executable;

    // Check which executable this kernel object (address) belongs to
    SIBIR_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_executable(reinterpret_cast<void *>(kernelObject),
                                                                 &executable));

    struct disassemble_callback_data_t {
        hsa_agent_t agent;
        hsa_executable_t executable;
        hsa_executable_symbol_t symbol;
        sibir_address_t ko;
    } findKoAgentCallbackData{{}, {}, {}, kernelObject};

    auto findKoAgentIterator = [](hsa_executable_t e, hsa_agent_t a, hsa_executable_symbol_t s, void *data) {
        auto cbd = reinterpret_cast<disassemble_callback_data_t*>(data);
        uint64_t ko;
        auto &coreApi = sibir::HsaInterceptor::Instance().getSavedHsaTables().core;
        auto status = SIBIR_HSA_CHECK(coreApi.hsa_executable_symbol_get_info_fn(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ko));
        if (status != HSA_STATUS_SUCCESS)
            return status;
        if (ko == cbd->ko) {
            cbd->executable = e;
            cbd->agent = a;
            cbd->symbol = s;
        }
        return HSA_STATUS_SUCCESS;
    };

    const auto &contextManager = sibir::ContextManager::Instance();

    auto agents = contextManager.getHsaAgents();
    for (const auto &agent: agents)
        SIBIR_HSA_CHECK(coreApi.hsa_executable_iterate_agent_symbols_fn(executable, agent,
                                                                        findKoAgentIterator, &findKoAgentCallbackData));
    assert(findKoAgentCallbackData.agent.handle != hsa_agent_t{}.handle);
    assert(findKoAgentCallbackData.symbol.handle != hsa_executable_symbol_t{}.handle);
    return std::make_tuple(findKoAgentCallbackData.agent, findKoAgentCallbackData.executable, findKoAgentCallbackData.symbol);
}

}// namespace

std::vector<sibir::Instr> sibir::Disassembler::disassemble(sibir_address_t kernelObject) {
    hsa_agent_t symbolAgent;
    hsa_executable_t executable;
    hsa_executable_symbol_t executableSymbol;
    const auto &contextManager = sibir::ContextManager::Instance();

    sibir_address_t kernelEntryPoint = getKdEntryPoint(kernelObject);
    std::tie(symbolAgent, executable, executableSymbol) = getKernelObjectInfo(kernelObject);

    std::string isaName = contextManager.getHsaAgentInfo(symbolAgent).isa;

    // Disassemble using AMD_COMGR

    amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;

    amd_comgr_disassembly_info_t disassemblyInfo = getEndPgmDisassemblyInfo(isaName);

    uint64_t instrAddr = kernelEntryPoint;
    uint64_t instrSize = 0;
    endPgmCallbackData kdDisassemblyCallbackData{{}, false};
    std::vector<sibir::Instr> out;
    while (status == AMD_COMGR_STATUS_SUCCESS && !kdDisassemblyCallbackData.second) {
        status = amd_comgr_disassemble_instruction(
            disassemblyInfo, instrAddr, (void *) &kdDisassemblyCallbackData, &instrSize);
        out.emplace_back(kdDisassemblyCallbackData.first, symbolAgent, executable, executableSymbol, instrAddr, instrSize);
        kdDisassemblyCallbackData.second = kdDisassemblyCallbackData.first.find("s_endpgm") != std::string::npos;
        instrAddr += instrSize;
    }
    return out;
}

std::vector<sibir::Instr> sibir::Disassembler::disassemble(hsa_agent_t agent, sibir_address_t entry, size_t size) {

    std::string isaName = sibir::ContextManager::Instance().getHsaAgentInfo(agent).isa;

    auto disassemblyInfo = getSizeDisassemblyInfo(isaName);

    uint64_t instrAddr = entry;
    uint64_t instrSize = 0;
    std::pair<std::string, sibir_address_t> kdDisassemblyCallbackData{{}, entry + size};
    std::vector<Instr> out;
    std::cout << "Entry + size: " << std::hex << entry + size << std::dec << std::endl;
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
std::vector<sibir::Instr> sibir::Disassembler::disassemble(hsa_executable_symbol_t symbol) {
    return std::vector<sibir::Instr>();
}
std::vector<sibir::Instr> sibir::Disassembler::disassemble(hsa_agent_t agent, sibir_address_t address) {
    return std::vector<Instr>();
}
std::vector<sibir::Instr> sibir::Disassembler::disassemble(sibir_address_t kernelObject, size_t size) {
    return std::vector<Instr>();
}
std::vector<sibir::Instr> sibir::Disassembler::disassemble(hsa_executable_symbol_t symbol, size_t size) {
    return std::vector<Instr>();
}

amd_comgr_disassembly_info_t sibir::Disassembler::getEndPgmDisassemblyInfo(const std::string &isa) {
    if (!endPgmDisassemblyInfoMap_.contains(isa)) {
        amd_comgr_disassembly_info_t disassemblyInfo;
        SIBIR_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info(isa.c_str(),
                                                                &endPgmReadMemoryCallback,
                                                                &endPgmPrintInstructionCallback,
                                                                &printAddressCallback,
                                                                &disassemblyInfo));
        endPgmDisassemblyInfoMap_.insert({isa, disassemblyInfo});
    }
    return endPgmDisassemblyInfoMap_[isa];
}

amd_comgr_disassembly_info_t sibir::Disassembler::getSizeDisassemblyInfo(const std::string &isa) {
    if (!sizeDisassemblyInfoMap_.contains(isa)) {
        amd_comgr_disassembly_info_t disassemblyInfo;
        SIBIR_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info(isa.c_str(),
                                                                &sizeReadMemoryCallback,
                                                                &sizePrintInstructionCallback,
                                                                &printAddressCallback,
                                                                &disassemblyInfo));
        sizeDisassemblyInfoMap_.insert({isa, disassemblyInfo});
    }
    return sizeDisassemblyInfoMap_[isa];
}
