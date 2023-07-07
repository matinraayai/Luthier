#include "disassembler.h"
#include "hsa_intercept.h"


uint64_t readMemoryCallback(sibir_address_t from, char *to, size_t size,
                            void *userData) {
    bool isEndOfProgram = reinterpret_cast<std::pair<std::string, bool>*>(userData)->second;
    if (isEndOfProgram)
        return 0;
    else {
        memcpy(reinterpret_cast<void*>(to),
               reinterpret_cast<const void *>(from), size);
        return size;
    }
}


void printInstructionCallback(const char *instruction, void *userData) {
    auto out = reinterpret_cast<std::pair<std::string, bool>*>(userData);
    out->first = std::string(instruction);
}

void printAddressCallback(uint64_t Address, void *UserData) {
//    checkUserData(UserData);
//    size_t ActualIdx = InstructionsIdx - 1;
//    if (ActualIdx != BrInstructionIdx) {
//        fail("absolute address resolved for instruction index %zu, expected index "
//             "%zu",
//             InstructionsIdx, BrInstructionIdx);
//    }
//    if (Address != BrInstructionAddr) {
//        fail("incorrect absolute address %u resolved for instruction index %zu, "
//             "expected %u",
//             Address, ActualIdx, BrInstructionAddr);
//    }
}

hsa_status_t Disassembler::populateAgentInfo(hsa_agent_t agent, Disassembler::hsa_agent_entry_t& entry) {
    const auto& coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
    hsa_status_t status;

    // Get the name (architecture) of the agent
    std::string agentName;
    agentName.resize(64);

    coreApi.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_ISA, agentName.data());

    // Get the Isa name of the agent
    std::vector<std::string> supportedAgentIsaNames;

    auto getIsaNameCallback = [](hsa_isa_t isa, void* data){
        auto out = reinterpret_cast<std::vector<std::string>*>(data);
        auto coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
        hsa_status_t status = HSA_STATUS_ERROR;
        uint32_t isaNameSize;
        status = SIBIR_HSA_CHECK(coreApi.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &isaNameSize));
        if (status != HSA_STATUS_SUCCESS)
            return status;
        std::string isaName;
        isaName.resize(isaNameSize);
        status = SIBIR_HSA_CHECK(coreApi.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME, isaName.data()));
        if (status != HSA_STATUS_SUCCESS)
            return status;
        out->push_back(isaName);
        return HSA_STATUS_SUCCESS;
    };

    status = SIBIR_HSA_CHECK(coreApi.hsa_agent_iterate_isas_fn(agent, getIsaNameCallback, &supportedAgentIsaNames));

    if (status != HSA_STATUS_SUCCESS)
        return status;
    // Assert that there's only one supported ISA for the agent
    assert(supportedAgentIsaNames.size() == 1);

    entry.isa = supportedAgentIsaNames[0];

    return HSA_STATUS_SUCCESS;
}


hsa_status_t Disassembler::initGpuAgentsMap() {
    int i = 0;
    auto& coreTable = SibirHsaInterceptor::Instance().getSavedHsaTables().core;

    auto returnGpuAgentsCallback = [](hsa_agent_t agent, void* data) {
        auto agentMap = reinterpret_cast<std::unordered_map<decltype(hsa_agent_t::handle), hsa_agent_entry_t>*>(data);
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS)
            return stat;
        if (dev_type == HSA_DEVICE_TYPE_GPU) {
            hsa_agent_entry_t entry;
            Disassembler::populateAgentInfo(agent, entry);
            agentMap->insert({agent.handle, entry});
        }

        return stat;
    };

    return coreTable.hsa_iterate_agents_fn(returnGpuAgentsCallback, &agents_);
}

std::vector<Instr *> Disassembler::disassemble(sibir_address_t kernelObject) {
    // Determine kernel's entry point
    const kernel_descriptor_t *kernelDescriptor{nullptr};
    auto amdTable = SibirHsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    auto coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
    SIBIR_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(kernelObject),
                                                                   reinterpret_cast<const void **>(&kernelDescriptor)));
    auto kernelEntryPoint =
        reinterpret_cast<sibir_address_t>(kernelObject) + kernelDescriptor->kernel_code_entry_byte_offset;

    // A way to backtrack from the kernel object to the symbol it belongs to (besides keeping track of a map)
    hsa_executable_t executable;

    // Check which executable this kernel object (address) belongs to
    SIBIR_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_executable(
        reinterpret_cast<void*>(kernelObject), &executable));

    if (agents_.empty())
        SIBIR_HSA_CHECK(initGpuAgentsMap());

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

    for (const auto& agent: agents_)
        SIBIR_HSA_CHECK(coreApi.hsa_executable_iterate_agent_symbols_fn(executable, {agent.first},
                                                                        findKoAgentIterator, &findKoAgentCallbackData));

    hsa_agent_t symbolAgent = findKoAgentCallbackData.first;

    //TODO: add check here in case the symbol's agent was not found

    std::string isaName = agents_[symbolAgent.handle].isa;

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
    while (Status == AMD_COMGR_STATUS_SUCCESS && !kdDisassemblyCallbackData.second) {
        Status = amd_comgr_disassemble_instruction(
            disassemblyInfo, instrAddr, (void *)&kdDisassemblyCallbackData, &instrSize);
        std::cout << "Instr: " << kdDisassemblyCallbackData.first << std::endl;
        kdDisassemblyCallbackData.second = kdDisassemblyCallbackData.first.find("s_endpgm") != std::string::npos;
        instrAddr += instrSize;
    }
    SIBIR_AMD_COMGR_CHECK(amd_comgr_destroy_disassembly_info(disassemblyInfo));
    return {};
}

