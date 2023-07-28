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



//    fmt::println(stdout, "Using ELFIO to iterate over the sections of the ELF");
//
//    for (auto it = coElfIO.sections.begin(); it != coElfIO.sections.end(); it++) {
//        auto sec = it->get();
//        fmt::println(stdout, "Name: {}", sec->get_name()");
//        fmt::println(stdout, "Address: {}", sec->get_address());
//    }

//    fmt::println(stdout, "Using ELFIO to iterate over the segments of the ELF");
//    for (auto it = coElfIO.segments.begin(); it != coElfIO.segments.end(); it++) {
//        auto seg = it->get();
//        auto num_sections = seg->get_sections_num();
//        fmt::println(stdout, "Section numbers: {}", num_sections);
//        for (int i = 0; i < num_sections; i++) {
//            fmt::println(stdout, "Section info: ");
//            auto section_idx = seg->get_section_index_at(i);
//            auto sec = coElfIO.sections[section_idx];
//            fmt::println(stdout, "Name: {}", sec->get_name());
//            fmt::println(stdout, "Address: {}", sec->get_address());
//            fmt::println(stdout, "Size: {}", sec->get_size());
//        }
//        fmt::println(stdout, "Type: {}", seg->get_type());
//    }
//
//    std::cout << "Read the notes section: " << std::endl;
//
//    ELFIO::section* noteSec = coElfIO.sections[".note"];
//
//    ELFIO::note_section_accessor note_reader(coElfIO, noteSec);
//
//    auto num = note_reader.get_notes_num();
//    ELFIO::Elf_Word type = 0;
//    char* desc = nullptr;
//    ELFIO::Elf_Word descSize = 0;

//    for (unsigned int i = 0; i < num; i++) {
//        std::string name;
//        if(note_reader.get_note(i, type, name, desc, descSize)) {
//            std::cout << "Note name: " << name << std::endl;
//            //            auto f = std::fstream("./note_content", std::ios::out);
//            std::string content(desc, descSize);
//            std::cout << "Note content" << content << std::endl;
//            //            f << content;
//            //            f.close();
//        }
//    }

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


std::string assemble(const std::vector<sibir::Instr>& instrVector, hsa_agent_t agent) {
    std::vector<std::string> instrStringVec(instrVector.size());
    for (unsigned int i = 0; i < instrStringVec.size(); i++)
        instrStringVec[i] = instrVector[i].getInstr();

    return assemble(fmt::format("{}", fmt::join(instrStringVec, "\n")), agent);
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


void sibir::CodeGenerator::instrument(sibir::Instr &instr, const std::string &instrumentationFunction, sibir_ipoint_t point) {

    hsa_agent_t agent = instr.getAgent();
    // Load the instrumentation ELF into the agent, and get its location on the device
    auto instrumentationExecutable = createExecutable(instrumentationFunction.data(), instrumentationFunction.size(), agent);

    auto instrmntLoadedCodeObject = getLoadedCodeObject(instrumentationExecutable);

    // Get a pointer to the beginning of the .text section of the instrumentation executable
    sibir_address_t instrmntTextSectionStart = instrmntLoadedCodeObject.first + 0x1000;

    // The instrumentation function is inserted first
    std::string dummyInstrmnt = assemble({std::string("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"),
                                      std::string("s_setpc_b64 s[0:1]")}, agent);

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
    //    int firstAddOffset = 0x60;

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

//
//    // Load the host code object to ELFIO and extend the .text section with the instrumentation code section + trampoline
//    ELFIO::elfio hostCodeObjectElfIo;
//    std::istringstream temp{std::string{reinterpret_cast<char*>(originalExecHostCo.first),
//                                        originalExecHostCo.second}
//    };
//    hostCodeObjectElfIo.load(temp);
//    // Instrument Section append
//    auto hostCodeObjectTextSection = hostCodeObjectElfIo.sections[".text"];
//
//    std::string nop = assemble("s_nop 0", agent);
//    hostCodeObjectTextSection->append_data(nop);
//    size_t instrumentCodeOffset = hostCodeObjectTextSection->get_size();
//
////    hostCodeObjectTextSection->append_data(instrumentationFunction);
//    std::string dummyInst = assemble({std::string("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"),
//                                      std::string("s_setpc_b64 s[0:1]")}, agent);
//
//    hostCodeObjectTextSection->append_data(dummyInst);
//    hostCodeObjectTextSection->append_data(nop);
//
//    size_t trampolineCodeOffset = hostCodeObjectTextSection->get_size();
//
//    // Extend the original code object with trampoline code
//    std::string trampoline = assemble({std::string("s_getpc_b64 s[2:3]")}, agent);
//    size_t trampolinePcOffset = trampolineCodeOffset + trampoline.size();
//
//
//    int firstAddOffset = (int) (trampolinePcOffset - instrumentCodeOffset);
////    int firstAddOffset = 0x60;
//
//    fmt::println(stdout, "Trampoline PC offset: {:#x}", trampolinePcOffset);
//    fmt::println(stdout, "Instrument Code Offset: {:#x}", instrumentCodeOffset);
//    fmt::println(stdout, "The set PC offset: {:#x}", firstAddOffset);
//
//
//    trampoline += assemble({fmt::format("s_sub_u32 s2, s2, {:#x}", firstAddOffset),
//                            "s_subb_u32 s3, s3, 0x0",
//                            "s_swappc_b64 s[0:1], s[2:3]",
//                            instr.getInstr()}, agent);
//
//
//    //    auto trmpPC = insts[nop_inst_idx + 1].addr + 4;
//    //    short trmpBranchImm = (instr.addr - trmpPC - 4) / 4;
//
//    trampolinePcOffset = trampolineCodeOffset + trampoline.size() + 4;
////    hostCodeObjectTextSection->append_data(trampoline);
//
//    short lastBranchImm = - static_cast<short>((trampolinePcOffset - 0x04) / 4);
//    trampoline += assemble({fmt::format("s_branch {:#x}", lastBranchImm)}, agent);
//    hostCodeObjectTextSection->append_data(trampoline);
//
//    fmt::println(stdout, "Trampoline code offset {:#x}", trampolineCodeOffset);
//
//    auto firstBranchImm = static_cast<short>((trampolineCodeOffset - 4 - 0x00) / 4);
//
//    std::string firstJump = assemble({fmt::format("s_branch {:#x}", firstBranchImm)}, agent);
//    if (instr.getSize() == 8)
//        firstJump += assemble({std::string("s_nop 0")}, agent);
//    std::string newContent;
//    newContent.resize(hostCodeObjectTextSection->get_size());
//    std::memcpy(newContent.data(), hostCodeObjectTextSection->get_data(), newContent.size());
//
//    std::memcpy(newContent.data(), firstJump.data(), firstJump.size());
//    hostCodeObjectTextSection->set_data(newContent);
//
//    std::stringstream instElfSS;
//    hostCodeObjectElfIo.save(instElfSS);
//    std::string outputElf = instElfSS.str();
//
//    // Create an executable with the ELF
//    hsa_executable_t executable = createExecutable(outputElf.data(), outputElf.size(), agent);

//    registerSymbolWithCodeObjectManager(executable, instr.getSymbol(), agent);
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
    hcoElfIo.load(hcoStrStream);

    std::cout << "Host Code Object starts at: " << reinterpret_cast<void*>(hostLoadedCodeObject.first) << std::endl;
    std::cout << "Host Code Object ends at" << reinterpret_cast<void*>(hostLoadedCodeObject.first + hostLoadedCodeObject.second) << std::endl;
    std::cout << "Device Code Object starts at: " << reinterpret_cast<void*>(loadedCodeObject.first) << std::endl;
    std::cout << "Device Code Object ends at" << reinterpret_cast<void*>(loadedCodeObject.first + loadedCodeObject.second) << std::endl;
    std::cout << "Text section for the ELF of HCO starts at: " << reinterpret_cast<const void*>(hcoElfIo.sections[".text"]->get_address()) << std::endl;
    auto offset = (reinterpret_cast<const sibir_address_t>(hcoElfIo.sections[".text"]->get_data()) -
                   reinterpret_cast<const sibir_address_t>(hostLoadedCodeObject.first));
    std::cout << "Text section offset: " << reinterpret_cast<const void*>(offset) << std::endl;

//    auto instSectionOffset = hcoElfIo.sections[".trampoline"]->get_address();
//    std::cout << "The trampoline should be located at: " << reinterpret_cast<void*>(instSectionOffset) << std::endl;

    const sibir_address_t loadedCoTextSection = reinterpret_cast<const sibir_address_t>(hcoElfIo.sections[".text"]->get_address()) + reinterpret_cast<const sibir_address_t>(loadedCodeObject.first);


    std::cout << "Loaded code object is located at: " <<  reinterpret_cast<const void*>(loadedCodeObject.first) << std::endl;
    auto insts = sibir::Disassembler::Instance().disassemble(agent, reinterpret_cast<sibir_address_t>(hcoElfIo.sections[".text"]->get_address()) + reinterpret_cast<sibir_address_t>(loadedCodeObject.first), hcoElfIo.sections[".text"]->get_size());
    unsigned int nop_inst_idx = 0;

    for (unsigned int i = 0; i < insts.size(); i++) {
        auto inst = insts[i];
        std::cout << std::hex << inst.getHostAddress() << std::dec << ": " << inst.getInstr() << std::endl;
        if (inst.getInstr().find("s_nop") != std::string::npos) {
            nop_inst_idx = i;
        }
    }
    std::cout << "First instrumentation instruction located at: " << reinterpret_cast<const void*>(insts[nop_inst_idx + 1].getDeviceAddress()) << std::endl;

            for (auto it = hcoElfIo.sections.begin(); it != hcoElfIo.sections.end(); it++) {
                //        auto seg = it->get();
                //        auto num_sections = seg->get_sections_num();
                //        std::cout << "Section numbers: " << num_sections << std::endl;
                //        for (int i = 0; i < num_sections; i++) {
                std::cout << "Section info: " << std::endl;
                //            auto section_idx = seg->get_section_index_at(i);
                //            auto sec = lcoElfIo.sections[section_idx];
                auto sec = it->get();
                std::cout << "Name: " << sec->get_name() << std::endl;
                std::cout << "Address: " << sec->get_address() << std::endl;
                std::cout << "Size: " << sec->get_size() << std::endl;
            }
            std::cout << "Number of symbols: " << sibir::elf::getSymbolNum(hcoElfIo) << std::endl;
            for (int i = 0; i < sibir::elf::getSymbolNum(hcoElfIo); i++) {
                sibir::elf::SymbolInfo info;
                sibir::elf::getSymbolInfo(hcoElfIo, i, info);
                std::cout << info.sym_name << std::endl;
                std::cout << "Addr: " << reinterpret_cast<const void*>(info.address) << std::endl;
                std::cout << "Value: " << info.value << std::endl;
                std::cout << "Sec Address: " << reinterpret_cast<const void*>(info.sec_addr) << std::endl;
                std::cout << "Sec Name: " << info.sec_name << std::endl;
                if (info.sym_name.find("kd") != std::string::npos) {

                    const auto kd = reinterpret_cast<const kernel_descriptor_t*>(info.address);
//                    sibir::CodeObjectManager::Instance().registerKD(instr.getKernelDescriptor(), kd);
//                    std::cout << "KD: " << reinterpret_cast<const void *>(kd) << std::endl;
//                    std::cout << "Offset in KD: " << kd->kernel_code_entry_byte_offset << std::endl;
                }
            }



//    std::memcpy(reinterpret_cast<void *>(instr.addr), firstBranchAssembled.data(), firstBranchAssembled.size());
//    hwInsts[0].append(o_branch.str());

    //    hwInsts[6].append(t_branch.str());
//    instList.push_back(a.Assemble(hwInsts[6], instList.back()->PC + 4));



    //    std::cout << "Text section for the original elf starts at: " << reinterpret_cast<const void*>(coElfIO.sections[".text"]->get_data()) << std::endl;
    //    std::cout << "loaded code object: " << hostCodeObject.size() << std::endl;
//    std::cout << "Did elfio load correctly? " << hcoElfIo.load(hcoStrStream) << std::endl;

        //    std::cout << "Number of symbols: " << sibir::elf::getSymbolNum(lcoElfIo) << std::endl;

//        std::cout << "Using ELFIO to iterate over the segments of the ELF" << std::endl;
//        std::cout << "Number of segments: " << hcoElfIo.segments.size() << std::endl;
//        std::cout << "Number of sections: " << hcoElfIo.sections.size() << std::endl;
//        std::cout << "Machine: " << hcoElfIo.get_machine() << std::endl;
//
//        for (auto it = hcoElfIo.sections.begin(); it != hcoElfIo.sections.end(); it++) {
//            //        auto seg = it->get();
//            //        auto num_sections = seg->get_sections_num();
//            //        std::cout << "Section numbers: " << num_sections << std::endl;
//            //        for (int i = 0; i < num_sections; i++) {
//            std::cout << "Section info: " << std::endl;
//            //            auto section_idx = seg->get_section_index_at(i);
//            //            auto sec = lcoElfIo.sections[section_idx];
//            auto sec = it->get();
//            std::cout << "Name: " << sec->get_name() << std::endl;
//            std::cout << "Address: " << sec->get_address() << std::endl;
//            std::cout << "Size: " << sec->get_size() << std::endl;
//        }

//        std::cout << "Type: " << seg->get_type() << std::endl;
//    }




//    for (unsigned int i = 0; i < sibir::elf::getSymbolNum(lcoElfIo); i++) {
//        sibir::elf::SymbolInfo info;
//        sibir::elf::getSymbolInfo(lcoElfIo, i, info);
//        if (info.sec_name == ".text") {
//            std::cout << "Symbol Name: " << sibir::ContextManager::getDemangledName(info.sym_name.c_str()) << std::endl;
//            std::cout << "Address: " << reinterpret_cast<const void*>(info.address) << std::endl;
//            std::cout << "Sec address: " << reinterpret_cast<const void*>(info.sec_addr) << std::endl;
//            std::cout << "Size: " << info.size << std::endl;
//            std::cout << "Sec Size: " << info.sec_size << std::endl;
//            if (info.sym_name.find("instrumentation_kernel") != std::string::npos
//                and info.sym_name.find("kd") == std::string::npos) {
////                std::vector<Instr> writeInst{{0, "s_branch 0xffff"}};
////                std::string instContent = assemble(writeInst, agent);
////                memcpy((void *) info.address, instContent.data(), instContent.size() * sizeof(char));
////                std::cout << "Assembled instruction size: " << instContent.size() << std::endl;
//                auto insts = sibir::Disassembler::disassemble(agent, reinterpret_cast<sibir_address_t>(info.address), info.size);
//                for (const auto& inst : insts)
//                    std::cout << std::hex << inst.addr << std::dec << ": " << inst.instr << std::endl;
//            }
//            else if (info.sym_name.find("kd") != std::string::npos) {
//
//                const auto kd = reinterpret_cast<const kernel_descriptor_t*>(info.address);
//                std::cout << "KD: " << reinterpret_cast<const void *>(kd) << std::endl;
//                std::cout << "Offset in KD: " << kd->kernel_code_entry_byte_offset << std::endl;
//            }
//
//        }
//    }
}
