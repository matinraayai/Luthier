#include "disassembler.hpp"
#include "hsa_intercept.h"
#include "context_manager.hpp"

namespace {
uint64_t readMemoryCallback(sibir_address_t from, char *to, size_t size,
                            void *userData) {
    bool isEndOfProgram = reinterpret_cast<std::pair<std::string, bool> *>(userData)->second;
    if (isEndOfProgram)
        return 0;
    else {
        memcpy(reinterpret_cast<void *>(to),
               reinterpret_cast<const void *>(from), size);
        return size;
    }
}

void printInstructionCallback(const char *instruction, void *userData) {
    auto out = reinterpret_cast<std::pair<std::string, bool> *>(userData);
    out->first = std::string(instruction);
}

void printAddressCallback(uint64_t Address, void *UserData) {}
}



std::vector<Instr> sibir::Disassembler::disassemble(sibir_address_t kernelObject) {
    // Determine kernel's entry point
    const kernel_descriptor_t *kernelDescriptor{nullptr};
    auto amdTable = SibirHsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    auto coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
    SIBIR_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(kernelObject),
                                                                   reinterpret_cast<const void **>(&kernelDescriptor)));

    std::cout << "KD: " << reinterpret_cast<const void *>(kernelDescriptor) << std::endl;
    std::cout << "Offset in KD: " << kernelDescriptor->kernel_code_entry_byte_offset << std::endl;

    auto kernelEntryPoint =
        reinterpret_cast<sibir_address_t>(kernelObject) + kernelDescriptor->kernel_code_entry_byte_offset;

    // A way to backtrack from the kernel object to the symbol it belongs to (besides keeping track of a map)
    hsa_executable_t executable;

    // Check which executable this kernel object (address) belongs to
    SIBIR_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_executable(
        reinterpret_cast<void*>(kernelObject), &executable));

    // TODO: Maybe make this part a private method instead of lambda?

    auto findKoAgentCallbackData = std::make_pair(hsa_agent_t{}, kernelObject);

    auto findKoAgentIterator = [](hsa_executable_t e, hsa_agent_t a, hsa_executable_symbol_t s, void* data){
        auto cbd = reinterpret_cast<std::pair<hsa_agent_t, uint64_t>*>(data);
        uint64_t ko;
        auto& coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
        SIBIR_HSA_CHECK(coreApi.hsa_executable_symbol_get_info_fn(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ko));
        if (ko == cbd->second)
            cbd->first = a;
        return HSA_STATUS_SUCCESS;
    };

    auto& contextManager = sibir::ContextManager::Instance();

    auto agents = contextManager.getHsaAgents();
    for (const auto& agent: agents)
        SIBIR_HSA_CHECK(coreApi.hsa_executable_iterate_agent_symbols_fn(executable, agent,
                                                                        findKoAgentIterator, &findKoAgentCallbackData));

    hsa_agent_t symbolAgent = findKoAgentCallbackData.first;

    //TODO: add check here in case the symbol's agent was not found


    std::string isaName = contextManager.getHsaAgentInfo(symbolAgent).isa;

    // Disassemble using AMD_COMGR

    amd_comgr_status_t Status = AMD_COMGR_STATUS_SUCCESS;

    amd_comgr_disassembly_info_t disassemblyInfo;

    // Maybe caching the disassembly info for each agent is a good idea? (instead of only the isaName)
    // The destructor has to call destroy on each disassembly_info_t

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info(isaName.c_str(),
                                                            &readMemoryCallback,
                                                            &printInstructionCallback,
                                                            &printAddressCallback,
                                                            &disassemblyInfo));

    uint64_t instrAddr = kernelEntryPoint;
    uint64_t instrSize = 0;
    std::pair<std::string, bool> kdDisassemblyCallbackData{{}, false};
    std::vector<Instr> out;
    while (Status == AMD_COMGR_STATUS_SUCCESS && !kdDisassemblyCallbackData.second) {
        Status = amd_comgr_disassemble_instruction(
            disassemblyInfo, instrAddr, (void *)&kdDisassemblyCallbackData, &instrSize);
        out.push_back({instrAddr, kdDisassemblyCallbackData.first});
        kdDisassemblyCallbackData.second = kdDisassemblyCallbackData.first.find("s_endpgm") != std::string::npos;
        instrAddr += instrSize;
    }
    SIBIR_AMD_COMGR_CHECK(amd_comgr_destroy_disassembly_info(disassemblyInfo));
    return out;
}

std::vector<Instr> sibir::Disassembler::disassemble(hsa_agent_t agent, sibir_address_t entry, size_t size) {

    std::string isaName = sibir::ContextManager::Instance().getHsaAgentInfo(agent).isa;

    // Disassemble using AMD_COMGR

    amd_comgr_status_t Status = AMD_COMGR_STATUS_SUCCESS;

    amd_comgr_disassembly_info_t disassemblyInfo;

    // Maybe caching the disassembly info for each agent is a good idea? (instead of only the isaName)
    // The destructor has to call destroy on each disassembly_info_t

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info(isaName.c_str(),
                                                            &readMemoryCallback,
                                                            &printInstructionCallback,
                                                            &printAddressCallback,
                                                            &disassemblyInfo));

    uint64_t instrAddr = entry;
    uint64_t instrSize = 0;
    std::pair<std::string, bool> kdDisassemblyCallbackData{{}, false};
    std::vector<Instr> out;
    std::cout << "Entry + size" << entry + size << std::endl;
    while (Status == AMD_COMGR_STATUS_SUCCESS && !kdDisassemblyCallbackData.second) {
        Status = amd_comgr_disassemble_instruction(
            disassemblyInfo, instrAddr, (void *)&kdDisassemblyCallbackData, &instrSize);
        out.push_back({instrAddr, kdDisassemblyCallbackData.first});
        kdDisassemblyCallbackData.second = instrAddr > (entry + size);
        instrAddr += instrSize;
    }
    SIBIR_AMD_COMGR_CHECK(amd_comgr_destroy_disassembly_info(disassemblyInfo));
    return out;
}

