#include "disassembler.h"
#include "hsa_intercept.h"

//
//const int ExpectedUserData;
//
//void checkUserData(void *UserData) {
//    if (UserData != (void *)&ExpectedUserData) {
//        fail("user_data changed");
//    }
//}
//
//const char *skipspace(const char *S) {
//    while (isspace(*S)) {
//        ++S;
//    }
//    return S;
//}
//
//size_t strlenWithoutTrailingWhitespace(const char *S) {
//    size_t I = strlen(S);
//    while (I && isspace(S[--I])) {
//        ;
//    }
//    return I + 1;
//}
//
//const char Program[] = {
//    '\x02', '\x00', '\x06', '\xC0', '\x00', '\x00', '\x00', '\x00', '\x7f',
//    '\xC0', '\x8c', '\xbf', '\x00', '\x80', '\x12', '\xbf', '\x05', '\x00',
//    '\x85', '\xbf', '\x00', '\x02', '\x00', '\x7e', '\xc0', '\x02', '\x04',
//    '\x7e', '\x01', '\x02', '\x02', '\x7e', '\x00', '\x80', '\x70', '\xdc',
//    '\x00', '\x02', '\x7f', '\x00', '\x00', '\x00', '\x81', '\xbf',
//};
//
//const char *Instructions[] = {
//    "s_load_dwordx2 s[0:1], s[4:5], 0x0",
//    "s_waitcnt lgkmcnt(0)",
//    "s_cmp_eq_u64 s[0:1], 0",
//    "s_cbranch_scc1 5",
//    "v_mov_b32_e32 v0, s0",
//    "v_mov_b32_e32 v2, 64",
//    "v_mov_b32_e32 v1, s1",
//    "global_store_dword v[0:1], v2, off",
//    "s_endpgm",
//};
//const size_t InstructionsLen = sizeof(Instructions) / sizeof(*Instructions);
//size_t InstructionsIdx = 0;
//const size_t BrInstructionIdx = 3;
//const size_t BrInstructionAddr = 40;
//
//uint64_t readMemoryCallback(uint64_t From, char *To, uint64_t Size,
//                            void *UserData) {
//    checkUserData(UserData);
//    if (From >= sizeof(Program)) {
//        return 0;
//    }
//    if (From + Size > sizeof(Program)) {
//        Size = sizeof(Program) - From;
//    }
//    memcpy(To, Program + From, Size);
//    return Size;
//}
//
//void printInstructionCallback(const char *Instruction, void *UserData) {
//    checkUserData(UserData);
//    if (InstructionsIdx == InstructionsLen) {
//        fail("too many instructions");
//    }
//    const char *Expected = skipspace(Instructions[InstructionsIdx++]);
//    const char *Actual = skipspace(Instruction);
//    if (strncmp(Expected, Actual, strlenWithoutTrailingWhitespace(Actual))) {
//        fail("incorrect instruction: expected '%s', actual '%s'", Expected, Actual);
//    }
//}
//
//void printAddressCallback(uint64_t Address, void *UserData) {
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
//}


hsa_status_t Disassembler::initGpuAgents() {
    int i = 0;
    auto& coreTable = SibirHsaInterceptor::Instance().getSavedHsaTables().core;

    auto returnGpuAgentsCallback = [](hsa_agent_t agent, void* data) {
        auto agent_list = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS)
            return stat;
        if (dev_type == HSA_DEVICE_TYPE_GPU)
            agent_list->push_back(agent);

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

    fprintf(stdout, "The kernel launch belongs to executable with handle: %lX.\n", executable.handle);

    if (agents_.empty())
        SIBIR_HSA_CHECK(initGpuAgents());

    auto callbackData = std::make_pair( hsa_agent_t{}, kernelObject);

    auto findKoSymbolIterator = [](hsa_executable_t e, hsa_agent_t a, hsa_executable_symbol_t s, void* data){
        auto cbd = reinterpret_cast<std::pair<hsa_agent_t, uint64_t>*>(data);
        uint64_t ko;
        auto& coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
        SIBIR_HSA_CHECK(coreApi.hsa_executable_symbol_get_info_fn(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ko));
        if (ko == cbd->second)
            cbd->first = a;
        return HSA_STATUS_SUCCESS;
    };

    for (auto agent: agents_)
        SIBIR_HSA_CHECK(coreApi.hsa_executable_iterate_agent_symbols_fn(executable, agent,
                                                                        findKoSymbolIterator, &callbackData));

    auto symbolAgent = callbackData.first;
    std::string agentName;
    agentName.resize(64);
    // Get the name (architecture) of the agent
    coreApi.hsa_agent_get_info_fn(symbolAgent, HSA_AGENT_INFO_ISA, agentName.data());

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

    SIBIR_HSA_CHECK(coreApi.hsa_agent_iterate_isas_fn(symbolAgent, getIsaNameCallback, &supportedAgentIsaNames));

    // Assert that there's only one supported ISA for the agent
    assert(supportedAgentIsaNames.size() == 1);

    for (const auto& isaName: supportedAgentIsaNames)
        std::cout << "Isa name of the agent: " << isaName << std::endl;

//
//    amd_comgr_status_t Status;
//
//    amd_comgr_disassembly_info_t DisassemblyInfo;
//
//    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_disassembly_info("amdgcn-amd-amdhsa--gfx900", &readMemoryCallback,
//        &printInstructionCallback, &printAddressCallback, &DisassemblyInfo));
//
//    uint64_t Addr = 0;
//    uint64_t Size = 0;
//    while (Status == AMD_COMGR_STATUS_SUCCESS && Addr < sizeof(Program)) {
//        Status = amd_comgr_disassemble_instruction(
//            DisassemblyInfo, Addr, (void *)&ExpectedUserData, &Size);
//        checkError(Status, "amd_comgr_disassemble_instruction");
//        Addr += Size;
//    }
//
//    if (InstructionsIdx != InstructionsLen) {
//        fail("too few instructions\n");
//    }
//
//    Addr = sizeof(Program) - 1;
//    Size = 0;
//    Status = amd_comgr_disassemble_instruction(DisassemblyInfo, Addr,
//                                               (void *)&ExpectedUserData, &Size);
//    if (Status != AMD_COMGR_STATUS_ERROR) {
//        fail("successfully disassembled invalid instruction encoding");
//    }
//
//    Status = amd_comgr_destroy_disassembly_info(DisassemblyInfo);
//    checkError(Status, "amd_comgr_destroy_disassembly_info");
//
//    return EXIT_SUCCESS;
//
//
//    // For now assume gfx908
//    // TODO: add the architecture code from the dbgapi headers
//    amd_dbgapi_architecture_id_t arch;
//    amd_dbgapi_get_architecture(0x030, &arch);
//
//
//    amd_dbgapi_size_t maxInstrLen;
//    amd_dbgapi_architecture_get_info(arch, AMD_DBGAPI_ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE,
//                                     sizeof(amd_dbgapi_size_t),
//                                     &maxInstrLen);
//
//    bool is_end = false;
//    // The decoded instruction will be malloced by ::amd_dbgapi_disassemble_instruction
//    // It has to be copied and freed
//    char*instChar{};
//    auto curr_address = kernelEntryPoint;
//    amd_dbgapi_size_t instrSize;
//
//    std::vector<std::pair<std::string, std::vector<std::byte>>> instList;
//    while(!is_end) {
//        instrSize = maxInstrLen;
//
//        amd_dbgapi_disassemble_instruction(arch, curr_address, &instrSize,
//                                           reinterpret_cast<void*>(curr_address),
//                                           &instChar, nullptr, {});
//
//        std::vector<std::byte> instBytes(instrSize);
//        // Copy the instruction bytes
//        for (amd_dbgapi_size_t i = 0; i < instrSize; i++) {
//            instBytes[i] = reinterpret_cast<std::byte*>(curr_address)[i];
//        }
//        // Copy the decoded instruction string
//        std::string instStr(instChar);
//
//        free(instChar);
//        instList.emplace_back(instStr, instBytes);
//
//        curr_address += instrSize;
//        is_end = instStr.find("s_endpgm") != std::string::npos;
//    }
//    return instList;
    return {};
}
