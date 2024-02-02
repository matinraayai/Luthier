#include <hsa/hsa.h>
#include <luthier.h>
#include <vector>
#include <string>

void luthier_at_init() {
    fprintf(stdout, "HSA Symbol Disassembly tool is starting.\n");
}

void luthier_at_term() {
    fprintf(stdout, "HSA Symbol Disassembly tool is exiting.\n");
}

void check(bool pred) {
    if (!pred)
        std::exit(-1);
}

/**
 * Returns the list of GPU HSA agents (devices) on the system
 * @param [out] agentList a vector of GPU agents
 * @return HSA_STATUS_SUCCESS if the operation is successful, otherwise an HSA Error
 */
hsa_status_t getGpuAgents(std::vector<hsa_agent_t> &agentList) {
    auto iterateAgentCallback = [](hsa_agent_t agent, void *data) {
        auto agent_list = reinterpret_cast<std::vector<hsa_agent_t> *>(data);
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS) {
            return stat;
        }
        if (dev_type == HSA_DEVICE_TYPE_GPU) {
            agent_list->push_back(agent);
        }
        return stat;
    };
    return hsa_iterate_agents(iterateAgentCallback, &agentList);
}

hsa_status_t getAllExecutableSymbols(const hsa_executable_t &executable,
                                     const std::vector<hsa_agent_t> &agentList,
                                     std::vector<hsa_executable_symbol_t> &symbols) {
    hsa_status_t out = HSA_STATUS_ERROR;
    for (auto agent: agentList) {
        auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data) {
            auto symbolVec = reinterpret_cast<std::vector<hsa_executable_symbol_t> *>(data);
            symbolVec->push_back(symbol);
            return HSA_STATUS_SUCCESS;
        };
        out = hsa_executable_iterate_agent_symbols(executable,
                                                   agent,
                                                   iterCallback, &symbols);
        if (out != HSA_STATUS_SUCCESS)
            return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
}

std::string symbolTypeAsString(hsa_symbol_kind_t kind) {
    switch(kind) {
        case HSA_SYMBOL_KIND_VARIABLE: return "Variable";
        case HSA_SYMBOL_KIND_KERNEL: return "Kernel";
        case HSA_SYMBOL_KIND_INDIRECT_FUNCTION: return "Indirect Function";
    }
}

void printSymbolAttributes(hsa_executable_symbol_t symbol) {
    fprintf(stdout, "Symbol attributes: \n");
    uint32_t nameLength;
    assert_perror(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameLength));
    std::string name;
    name.resize(nameLength);
    check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()) == HSA_STATUS_SUCCESS);
    fprintf(stdout, "|- Symbol name: %s\n", name.c_str());

    hsa_symbol_kind_t symbolKind;
    check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbolKind) == HSA_STATUS_SUCCESS);
    fprintf(stdout, "|- Symbol type: %s\n", symbolTypeAsString(symbolKind).c_str());

    hsa_symbol_linkage_t linkage;
    check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE, &linkage) == HSA_STATUS_SUCCESS);
    fprintf(stdout, "|- Linkage type: %s\n", linkage == HSA_SYMBOL_LINKAGE_MODULE ? "Module": "Program");

    bool isDefinition;
    check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION, &isDefinition) == HSA_STATUS_SUCCESS);
    fprintf(stdout, "|- Is definition? %s\n", isDefinition ? "yes" : "no");

    if (symbolKind == HSA_SYMBOL_KIND_VARIABLE) {
        uint64_t variableAddress;
        check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &variableAddress) == HSA_STATUS_SUCCESS);
        fprintf(stdout, "|- Variable address: 0x%lx\n", variableAddress);
        hsa_variable_allocation_t allocation;
        check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION, &allocation) == HSA_STATUS_SUCCESS);
        fprintf(stdout, "|- Variable allocation type: %s\n", allocation == HSA_VARIABLE_ALLOCATION_AGENT ? "Agent" : "Program");
    }
    if (symbolKind == HSA_SYMBOL_KIND_KERNEL) {
        uint64_t kernelObject;
        check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject) == HSA_STATUS_SUCCESS);
        fprintf(stdout, "|- Kernel Object address: 0x%lx\n", kernelObject);

        uint32_t kernArgSegmentSize;
        check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernArgSegmentSize) == HSA_STATUS_SUCCESS);
        fprintf(stdout, "|- Kernel argument segment size: 0x%x\n", kernArgSegmentSize);

        uint32_t kernArgSegmentAlignment;
        check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &kernArgSegmentAlignment) == HSA_STATUS_SUCCESS);
        fprintf(stdout, "|- Kernel argument segment alignment: 0x%x\n", kernArgSegmentAlignment);

        uint32_t kernPrivateSegmentSize;
        check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kernPrivateSegmentSize) == HSA_STATUS_SUCCESS);
        fprintf(stdout, "|- Kernel argument segment alignment: 0x%x\n", kernPrivateSegmentSize);

        bool dynamicCallstack;
        check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK, &dynamicCallstack) == HSA_STATUS_SUCCESS);
        fprintf(stdout, "|- Kernel Dynamic Callstack Flag: %s\n", dynamicCallstack ? "yes" : "no");
    }
    if (symbolKind == HSA_SYMBOL_KIND_INDIRECT_FUNCTION) {
#if defined(HSA_LARGE_MODEL)
        uint64_t objectHandle;
#else
        uint32_t objectHandle;
#endif
        check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT, &objectHandle) == HSA_STATUS_SUCCESS);
#if defined(HSA_LARGE_MODEL)
        fprintf(stdout, "|- Indirect function handle: 0x%lx\n", objectHandle);
#else
        fprintf(stdout, "|- Indirect function handle: 0x%sx\n", objectHandle);
#endif
    }
    fprintf(stdout, "|__\n");
}

void luthier_at_hsa_event(hsa_api_evt_args_t *args, luthier_api_evt_phase_t phase, hsa_api_evt_id_t api_id) {
    if (phase == LUTHIER_API_EVT_PHASE_EXIT) {
        if (api_id == HSA_API_ID_hsa_executable_freeze) {
            auto executable = args->api_args.hsa_executable_freeze.executable;
            fprintf(stdout, "Frozen executable with handle 0x%lx was captured.\n", executable.handle);
            std::vector<hsa_agent_t> agentList;
            std::vector<hsa_executable_symbol_t> symbols;
            check(getGpuAgents(agentList) == HSA_STATUS_SUCCESS);
            check(getAllExecutableSymbols(executable, agentList, symbols) == HSA_STATUS_SUCCESS);
            for (const auto &s: symbols)
                printSymbolAttributes(s);
            fprintf(stdout, "");
        }
    }
}

void luthier_at_hip_event(void *args, luthier_api_evt_phase_t phase, int api_id) {
}