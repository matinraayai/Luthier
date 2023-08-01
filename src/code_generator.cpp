#include "code_generator.hpp"
#include "amdgpu_elf.hpp"
#include "code_object_manager.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "elfio/elfio.hpp"
#include "error_check.hpp"
#include "hsa_intercept.hpp"
#include "instr.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>

amd_comgr_status_t printEntryCo(amd_comgr_metadata_node_t key,
                              amd_comgr_metadata_node_t value, void *data) {
    amd_comgr_metadata_kind_t kind;
    amd_comgr_metadata_node_t son;
    amd_comgr_status_t status;
    size_t size;
    char *keybuf;
    char *valbuf;
    int *indent = (int *)data;

    // assume key to be string in this test function
    status = amd_comgr_get_metadata_kind(key, &kind);
    if (kind != AMD_COMGR_METADATA_KIND_STRING)
        return AMD_COMGR_STATUS_ERROR;
    status = amd_comgr_get_metadata_string(key, &size, NULL);
    keybuf = (char *)calloc(size, sizeof(char));
    status = amd_comgr_get_metadata_string(key, &size, keybuf);

    status = amd_comgr_get_metadata_kind(value, &kind);
    for (int i = 0; i < *indent; i++)
        printf("  ");

    switch (kind) {
        case AMD_COMGR_METADATA_KIND_STRING: {
            printf("%s  :  ", size ? keybuf : "");
            status = amd_comgr_get_metadata_string(value, &size, NULL);
            valbuf = (char *)calloc(size, sizeof(char));
            status = amd_comgr_get_metadata_string(value, &size, valbuf);
            printf(" %s\n", valbuf);
            free(valbuf);
            break;
        }
        case AMD_COMGR_METADATA_KIND_LIST: {
            *indent += 1;
            status = amd_comgr_get_metadata_list_size(value, &size);
            printf("LIST %s %zd entries = \n", keybuf, size);
            for (size_t i = 0; i < size; i++) {
                status = amd_comgr_index_list_metadata(value, i, &son);
                status = printEntryCo(key, son, data);
                status = amd_comgr_destroy_metadata(son);
            }
            *indent = *indent > 0 ? *indent - 1 : 0;
            break;
        }
        case AMD_COMGR_METADATA_KIND_MAP: {
            *indent += 1;
            status = amd_comgr_get_metadata_map_size(value, &size);
            printf("MAP %zd entries = \n", size);
            status = amd_comgr_iterate_map_metadata(value, printEntryCo, data);
            *indent = *indent > 0 ? *indent - 1 : 0;
            break;
        }
        default:
            free(keybuf);
            return AMD_COMGR_STATUS_ERROR;
    } // switch

    free(keybuf);
    return AMD_COMGR_STATUS_SUCCESS;
};





std::string getSymbolName(hsa_executable_symbol_t symbol) {
    const auto& coreHsaApiTable = sibir::HsaInterceptor::Instance().getSavedHsaTables().core;
    uint32_t nameSize;
    SIBIR_HSA_CHECK(coreHsaApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameSize));
    std::string name;
    name.resize(nameSize);
    SIBIR_HSA_CHECK(coreHsaApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()));
    return name;
}

hsa_status_t registerSymbolWithCodeObjectManager(const hsa_executable_t& executable,
                                      const hsa_executable_symbol_t originalSymbol,
                                      hsa_agent_t agent) {

    hsa_status_t out = HSA_STATUS_ERROR;
    auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data) {
        auto originalSymbol = reinterpret_cast<hsa_executable_symbol_t *>(data);
        auto originalSymbolName = getSymbolName(*originalSymbol);

        auto& coreTable = sibir::HsaInterceptor::Instance().getSavedHsaTables().core;
        hsa_symbol_kind_t symbolKind;
        SIBIR_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbolKind));

        fmt::println(stdout, "Symbol kind: {}.", static_cast<int>(symbolKind));

        std::string symbolName = getSymbolName(symbol);

        fmt::println(stdout, "Symbol name: {}.", symbolName);

        if (symbolKind == HSA_SYMBOL_KIND_VARIABLE) {
            sibir_address_t variableAddress;
            SIBIR_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &variableAddress));
            std::cout << "Variable location: " << std::hex << variableAddress << std::dec << std::endl;
        }
        if (symbolKind == HSA_SYMBOL_KIND_KERNEL && symbolName == originalSymbolName) {
            sibir_address_t kernelObject;
            sibir_address_t originalKernelObject;
            SIBIR_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject));
            SIBIR_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(*originalSymbol,
                                                                        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &originalKernelObject));
            sibir::CodeObjectManager::Instance().registerKD(reinterpret_cast<sibir_address_t>(originalKernelObject),
                                                            reinterpret_cast<sibir_address_t>(kernelObject)
                                                            );
            std::cout << "original kernel location: " << std::hex << originalKernelObject << std::dec << std::endl;
            std::cout << "Kernel location: " << std::hex << kernelObject << std::dec << std::endl;
            std::vector<sibir::Instr> instList = sibir::Disassembler::Instance().disassemble(kernelObject);
            std::cout << "Disassembly of the KO: " << std::endl;
            for (const auto& i : instList) {
                std::cout << std::hex << i.getDeviceAddress() << std::dec << ": " << i.getInstr() << std::endl;
            }
        }

        //            symbolVec->push_back(symbol);
        return HSA_STATUS_SUCCESS;
    };
    out = hsa_executable_iterate_agent_symbols(executable,
                                               agent,
                                               iterCallback, (void *) &originalSymbol);
    if (out != HSA_STATUS_SUCCESS)
        return HSA_STATUS_ERROR;
    return HSA_STATUS_SUCCESS;
}


hsa_executable_t createExecutable(const char* codeObjectPtr, size_t codeObjectSize, hsa_agent_t agent) {
    auto coreApi = sibir::HsaInterceptor::Instance().getSavedHsaTables().core;
    hsa_code_object_reader_t coReader;
    hsa_executable_t executable;
    SIBIR_HSA_CHECK(coreApi.hsa_code_object_reader_create_from_memory_fn(codeObjectPtr,
                                                                         codeObjectSize, &coReader));

    SIBIR_HSA_CHECK(coreApi.hsa_executable_create_alt_fn(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &executable));

    SIBIR_HSA_CHECK(coreApi.hsa_executable_load_agent_code_object_fn(executable, agent, coReader, nullptr, nullptr));

    SIBIR_HSA_CHECK(coreApi.hsa_executable_freeze_fn(executable, nullptr));
    return executable;
}

std::pair<sibir_address_t, size_t> getLoadedCodeObject(hsa_executable_t executable) {
    auto amdTable = sibir::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void* data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t>*>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, & loadedCodeObjects);

    // Can be removed
    assert(loadedCodeObjects.size() == 1);


    uint64_t lcoBaseAddrDevice;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                             HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                                             &lcoBaseAddrDevice);
    // Query the size of the loaded code object
    uint64_t lcoSizeDevice;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                             HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                                                             &lcoSizeDevice);
    return {reinterpret_cast<sibir_address_t>(lcoBaseAddrDevice), static_cast<size_t>(lcoSizeDevice)};
}

std::pair<sibir_address_t, size_t> getCodeObject(hsa_executable_t executable) {
    auto amdTable = sibir::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void* data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t>*>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, & loadedCodeObjects);

    // Can be removed
    assert(loadedCodeObjects.size() == 1);


    uint64_t lcoBaseAddr;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
                                                            &lcoBaseAddr);
    // Query the size of the loaded code object
    uint64_t lcoSize;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
                                                            &lcoSize);
    return {reinterpret_cast<sibir_address_t>(lcoBaseAddr), static_cast<size_t>(lcoSize)};
}

std::string assemble(const std::string& instListStr, hsa_agent_t agent) {

    amd_comgr_data_t dataIn;
    amd_comgr_data_set_t dataSetIn, dataSetOut;
    amd_comgr_action_info_t dataAction;

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetIn));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataIn));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(dataIn, instListStr.size(), instListStr.data()));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data_name(dataIn, "my_source.s"));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_data_set_add(dataSetIn, dataIn));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetOut));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_action_info(&dataAction));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_action_info_set_isa_name(dataAction,
                                                             sibir::ContextManager::Instance().getHsaAgentInfo(agent)->getIsaName().c_str()));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_action_info_set_option_list(dataAction, nullptr, 0));
    SIBIR_AMD_COMGR_CHECK(
        amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                            dataAction, dataSetIn, dataSetOut));
    amd_comgr_data_t dataOut;
    char* dataOutPtr;
    size_t dataOutSize;
    amd_comgr_action_data_get_data(dataSetOut, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &dataOut);
    amd_comgr_get_data(dataOut, &dataOutSize, nullptr);
    std::string outElf;
    outElf.resize(dataOutSize);
    amd_comgr_get_data(dataOut, &dataOutSize, outElf.data());

    ELFIO::elfio io;
    std::stringstream elfss{outElf};
    io.load(elfss);

    return {io.sections[".text"]->get_data(), io.sections[".text"]->get_size()};
}

std::string assemble(const std::vector<std::string>& instrVector, hsa_agent_t agent) {

    std::string instString = fmt::format("{}", fmt::join(instrVector, "\n"));
    return assemble(instString, agent);
}


void iterateCodeObjectMetaData(sibir_address_t codeObjectData, size_t codeObjectSize) {
    // COMGR symbol iteration things
    amd_comgr_data_t coData;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &coData));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(coData, codeObjectSize, reinterpret_cast<const char*>(codeObjectData)));
    amd_comgr_metadata_node_t meta;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_data_metadata(coData, &meta));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data_name(coData, "my-data.s"));
    int Indent = 1;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(meta, printEntryCo, (void *)&Indent));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(meta));
}


void* allocateHsaMemory(hsa_agent_t agent, size_t size) {
    const auto& coreApi = sibir::HsaInterceptor::Instance().getSavedHsaTables().core;
    hsa_region_t region;

    auto regionIterator = [](hsa_region_t region, void* data) {
        const auto& coreApi = sibir::HsaInterceptor::Instance().getSavedHsaTables().core;
        hsa_region_segment_t segment;
        SIBIR_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment));
        uint32_t flags;
        SIBIR_HSA_CHECK(coreApi.hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags));

        auto out = reinterpret_cast<hsa_region_t*>(data);
        if (segment == HSA_REGION_SEGMENT_GLOBAL && (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED))
            *out = region;
        return HSA_STATUS_SUCCESS;
    };
    SIBIR_HSA_CHECK(coreApi.hsa_agent_iterate_regions_fn(agent, regionIterator, &region));
    void* deviceMemory;
    SIBIR_HSA_CHECK(coreApi.hsa_memory_allocate_fn(region, size, &deviceMemory));
    return deviceMemory;
}


void sibir::CodeGenerator::instrument(sibir::Instr &instr, const std::string &instrumentationFunction, sibir_ipoint_t point) {

    hsa_agent_t agent = instr.getAgent();
    // Load the instrumentation ELF into the agent, and get its location on the device
//    auto instrumentationExecutable = createExecutable(instrumentationFunction.data(), instrumentationFunction.size(), agent);

    auto instrmntLoadedCodeObject = std::make_pair(
        reinterpret_cast<sibir_address_t>(allocateHsaMemory(agent, instrumentationFunction.size())),
        instrumentationFunction.size()
        );

//    auto instrmntLoadedCodeObject = getLoadedCodeObject(instrumentationExecutable);



    // Get a pointer to the beginning of the .text section of the instrumentation executable
    sibir_address_t instrmntTextSectionStart = instrmntLoadedCodeObject.first + 0x1000;

    // The instrumentation function is inserted first
    std::string dummyInstrmnt = assemble(std::vector<std::string>
        {"s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)", "s_setpc_b64 s[0:1]"}, agent);

    // Padded with nop
    std::string nopInstr = assemble("s_nop 0", agent);

    std::memcpy(reinterpret_cast<void*>(instrmntTextSectionStart), dummyInstrmnt.data(), dummyInstrmnt.size());

    std::memcpy(reinterpret_cast<void*>(instrmntTextSectionStart + dummyInstrmnt.size()), nopInstr.data(), nopInstr.size());

    // Trampoline starts after the nop
    sibir_address_t trampolineStartAddr = instrmntTextSectionStart + dummyInstrmnt.size() + nopInstr.size();


    std::string trampoline = assemble("s_getpc_b64 s[2:3]", agent);

    // Get the PC of the instruction after the get PC instruction
    sibir_address_t trampolinePcOffset = trampolineStartAddr + trampoline.size();


    int firstAddOffset = (int) (trampolinePcOffset - instrmntTextSectionStart);

    fmt::println(stdout, "Trampoline PC offset: {:#x}", trampolinePcOffset);
    fmt::println(stdout, "Instrument Code Offset: {:#x}", instrmntTextSectionStart);
    fmt::println(stdout, "The set PC offset: {:#x}", firstAddOffset);


    trampoline += assemble({fmt::format("s_sub_u32 s2, s2, {:#x}", firstAddOffset),
                            "s_subb_u32 s3, s3, 0x0",
                            "s_swappc_b64 s[0:1], s[2:3]",
                            instr.getInstr()}, agent);


    trampolinePcOffset = trampolineStartAddr + trampoline.size() + 4;
    //    hostCodeObjectTextSection->append_data(trampoline);

    short lastBranchImm = - static_cast<short>((trampolinePcOffset - (instr.getDeviceAddress() + 4)) / 4);

    trampoline += assemble(fmt::format("s_branch {:#x}", lastBranchImm), agent);

    std::memcpy(reinterpret_cast<void*>(trampolineStartAddr), trampoline.data(), trampoline.size());

    // Overwrite the target instruction
    auto firstBranchImm = static_cast<short>((trampolineStartAddr - 4 - instr.getDeviceAddress()) / 4);

    std::string firstJump = assemble({fmt::format("s_branch {:#x}", firstBranchImm)}, agent);
    if (instr.getSize() == 8)
        firstJump += assemble({std::string("s_nop 0")}, agent);
    std::memcpy(reinterpret_cast<void *>(instr.getDeviceAddress()), firstJump.data(), firstJump.size());

    // Get the loaded code object on the device
    // It is loaded, so it's not a complete ELF anymore
    std::pair<sibir_address_t, size_t> loadedCodeObject = getLoadedCodeObject(instr.getExecutable());
    // Get the host ELF associated with the loaded code object
    // It is used to figure out offsets in the segments of the loaded code object
    std::pair<sibir_address_t, size_t> hostLoadedCodeObject = getCodeObject(instr.getExecutable());

    iterateCodeObjectMetaData(hostLoadedCodeObject.first, hostLoadedCodeObject.second);

    ELFIO::elfio hcoElfIo;
    std::istringstream hcoStrStream{std::string(
        reinterpret_cast<char*>(hostLoadedCodeObject.first), hostLoadedCodeObject.second)};
    hcoElfIo.load(hcoStrStream, true);
    ELFIO::elfio lcoElfIo;
    std::istringstream lcoStrStream{std::string(
        reinterpret_cast<char*>(loadedCodeObject.first), loadedCodeObject.second)};
    lcoElfIo.load(lcoStrStream, true);

    std::cout << "Host Code Object starts at: " << reinterpret_cast<void*>(hostLoadedCodeObject.first) << std::endl;
    std::cout << "Host Code Object ends at" << reinterpret_cast<void*>(hostLoadedCodeObject.first + hostLoadedCodeObject.second) << std::endl;
    std::cout << "Device Code Object starts at: " << reinterpret_cast<void*>(loadedCodeObject.first) << std::endl;
    std::cout << "Device Code Object ends at" << reinterpret_cast<void*>(loadedCodeObject.first + loadedCodeObject.second) << std::endl;
    std::cout << "Text section for the ELF of HCO starts at: " << reinterpret_cast<const void*>(hcoElfIo.sections[".text"]->get_address()) << std::endl;
    auto offset = (reinterpret_cast<const sibir_address_t>(hcoElfIo.sections[".text"]->get_data()) -
                   reinterpret_cast<const sibir_address_t>(hostLoadedCodeObject.first));
    std::cout << "Text section offset: " << reinterpret_cast<const void*>(offset) << std::endl;

    ELFIO::section* noteSec = hcoElfIo.sections[".note"];

    fmt::println(stdout, "Note section's address: {:#x}", noteSec->get_address());
    fmt::println(stdout, "Note section's size: {}", noteSec->get_size());
    ELFIO::note_section_accessor note_reader(hcoElfIo, noteSec);
    auto num = note_reader.get_notes_num();
    ELFIO::Elf_Word type = 0;
    char* desc = nullptr;
    ELFIO::Elf_Word descSize = 0;

    for (unsigned int i = 0; i < num; i++) {
        std::string name;
        if(note_reader.get_note(i, type, name, desc, descSize)) {
            std::cout << "Note name: " << name << std::endl;
            //            auto f = std::fstream("./note_content", std::ios::out);
            std::string content(desc, descSize);
            std::cout << "Note content" << content << std::endl;
            //            f << content;
            //            f.close();
        }
    }
    fmt::println(stdout, "Device code header:\n{:s}", std::string(reinterpret_cast<const char*>(loadedCodeObject.first),
                                                                  0x1000));
    fmt::println(stdout, "Host code header:\n{:s}", std::string(reinterpret_cast<const char*>(hostLoadedCodeObject.first) + noteSec->get_address(),
                                                                noteSec->get_size()));
    fmt::println(stdout, "Are headers the same? {}", std::string(reinterpret_cast<const char*>(loadedCodeObject.first),
                                                                 0x1000) ==  std::string(reinterpret_cast<const char*>(hostLoadedCodeObject.first),
                                    0x1000));
}
