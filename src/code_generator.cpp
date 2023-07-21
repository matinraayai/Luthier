#include "code_generator.hpp"
#include "amdgpu_elf.hpp"
#include "elfio/elfio.hpp"
#include "disassembler.hpp"
#include "context_manager.hpp"
#include "error_check.h"
#include "hsa_intercept.h"
#include "instr.h"
#include "code_object_manager.hpp"

//    std::cout << "Using ELFIO to iterate over the sections of the ELF" << std::endl;
//
//    for (auto it = coElfIO.sections.begin(); it != coElfIO.sections.end(); it++) {
//        auto sec = it->get();
//        std::cout << "Name: " << sec->get_name() << std::endl;
//        std::cout << "Address: " << sec->get_address() << std::endl;
//    }

//    std::cout << "Using ELFIO to iterate over the segments of the ELF" << std::endl;
//    for (auto it = coElfIO.segments.begin(); it != coElfIO.segments.end(); it++) {
//        auto seg = it->get();
//        auto num_sections = seg->get_sections_num();
//        std::cout << "Section numbers: " << num_sections << std::endl;
//        for (int i = 0; i < num_sections; i++) {
//            std::cout << "Section info: " << std::endl;
//            auto section_idx = seg->get_section_index_at(i);
//            auto sec = coElfIO.sections[section_idx];
//            std::cout << "Name: " << sec->get_name() << std::endl;
//            std::cout << "Address: " << sec->get_address() << std::endl;
//            std::cout << "Size: " << sec->get_size() << std::endl;
//        }
//        std::cout << "Type: " << seg->get_type() << std::endl;
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
    const auto& coreHsaApiTable = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
    uint32_t nameSize;
    SIBIR_HSA_CHECK(coreHsaApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameSize));
    std::string name;
    name.resize(nameSize);
    SIBIR_HSA_CHECK(coreHsaApiTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()));
    return name;
}

hsa_status_t registerExecutableSymbol(const hsa_executable_t& executable,
                                      const hsa_executable_symbol_t originalSymbol,
                                      hsa_agent_t agent) {

    hsa_status_t out = HSA_STATUS_ERROR;
    auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data) {
        auto originalSymbol = reinterpret_cast<hsa_executable_symbol_t *>(data);
        auto originalSymbolName = getSymbolName(*originalSymbol);

        auto& coreTable = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
        hsa_symbol_kind_t symbolKind;
        SIBIR_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbolKind));

        std::cout << "Symbol kind: " << symbolKind << std::endl;

        std::string symbolName = getSymbolName(symbol);
        std::cout << "Symbol Name: " << symbolName << std::endl;

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
            std::vector<sibir::Instr> instList = sibir::Disassembler::disassemble(kernelObject);
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
    auto coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
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
    auto amdTable = SibirHsaInterceptor::Instance().getHsaVenAmdLoaderTable();
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
    auto amdTable = SibirHsaInterceptor::Instance().getHsaVenAmdLoaderTable();
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
                                                             sibir::ContextManager::Instance().getHsaAgentInfo(agent).isa.c_str()));
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
    std::stringstream instStrStream;

    for (const auto& i: instrVector)
        instStrStream << i.getInstr() << std::endl;
    std::string instString = instStrStream.str();
    return assemble(instString, agent);
}

std::string assemble(const std::vector<std::string>& instrVector, hsa_agent_t agent) {
    std::stringstream instStrStream;

    for (const auto& i: instrVector)
        instStrStream << i << std::endl;
    std::string instString = instStrStream.str();
    return assemble(instString, agent);
}


void sibir::CodeGenerator::instrument(sibir::Instr &instr, const std::string &instrumentationFunction, sibir_ipoint_t point) {

    hsa_agent_t agent = instr.getAgent();

    // Query the original executable associated with the instr
    std::cout << "Instruction to be instrumented: " << instr.getInstr() << std::endl;
    std::cout << "Instruction address: " << std::hex << reinterpret_cast<void*>(instr.getDeviceAddress()) << std::dec << std::endl;

    hsa_executable_t originalExecutable = instr.getExecutable();

    // Get the host code object associated with the executable
    auto originalExecHostCo = getCodeObject(originalExecutable);

    // Load the host code object to ELFIO and extend the .text section with the instrumentation code section + trampoline
    ELFIO::elfio hostCodeObjectElfIo;
    std::istringstream temp{std::string{reinterpret_cast<char*>(originalExecHostCo.first),
                                        originalExecHostCo.second}
    };
    hostCodeObjectElfIo.load(temp);
    // Instrument Section append
    auto hostCodeObjectTextSection = hostCodeObjectElfIo.sections[".text"];

    std::string nop = assemble("s_nop 0", agent);
    hostCodeObjectTextSection->append_data(nop);
    size_t instrumentCodeOffset = hostCodeObjectTextSection->get_size();

    hostCodeObjectTextSection->append_data(instrumentationFunction);

    hostCodeObjectTextSection->append_data(nop);

    size_t trampolineCodeOffset = hostCodeObjectTextSection->get_size();

    // Extend the original code object with trampoline code
    std::string trampoline = assemble({std::string("s_getpc_b64 s[2:3]")}, agent);
    size_t trampolinePcOffset = trampolineCodeOffset + trampoline.size();


    int firstAddOffset = (int) (trampolinePcOffset - instrumentCodeOffset);
//    int firstAddOffset = 0x60;

    std::cout << "Trampoline PC offset: " << std::hex << trampolinePcOffset << std::endl;
    std::cout << "Instrument Code Offset: " << std::hex << instrumentCodeOffset << std::endl;
    std::cout << "The set PC offset: " << std::hex << firstAddOffset << std::dec << std::endl;
    std::stringstream ss;
    ss << "0x" << std::hex << firstAddOffset << std::endl;
    trampoline += assemble({"s_sub_u32 s2, s2, " + ss.str(),
                            "s_subb_u32 s3, s3, 0x0",
                            "s_swappc_b64 s[30:31], s[2:3]",
                            instr.getInstr()}, agent);


    //    auto trmpPC = insts[nop_inst_idx + 1].addr + 4;
    //    short trmpBranchImm = (instr.addr - trmpPC - 4) / 4;

    trampolinePcOffset = trampolineCodeOffset + trampoline.size() + 4;
//    hostCodeObjectTextSection->append_data(trampoline);

    ss.str("");
    ss.clear();
    short lastBranchImm = - static_cast<short>((trampolinePcOffset - 0x04) / 4);
//    short lastBranchImm = -45;
    ss << "s_branch 0x" << std::hex << lastBranchImm << std::endl;
    std::cout << ss.str() << std::endl;
    trampoline += assemble({ss.str()}, agent);
    hostCodeObjectTextSection->append_data(trampoline);

    ss.str("");
    ss.clear();

    std::cout << std::hex << "trampoline code offset" << trampolineCodeOffset << std::endl;

    auto firstBranchImm = static_cast<short>((trampolineCodeOffset - 4 - 0x00) / 4);
    ss << "s_branch 0x" << std::hex << firstBranchImm << std::endl;
//    ss << "s_branch 0x12" << std::endl; // >> correct jump to endpgm
    std::cout << ss.str() << std::endl;
    std::string firstJump = assemble({ss.str()}, agent);
    if (instr.getSize() == 8)
        firstJump += assemble({std::string("s_nop 0")}, agent);
    std::string newContent;
    newContent.resize(hostCodeObjectTextSection->get_size());
    std::memcpy(newContent.data(), hostCodeObjectTextSection->get_data(), newContent.size());
//    std::string desperateJump = assemble({
//        "s_getpc_b64 s[10:11]",
//        "s_add_u32 s10, s10, 0x10",
//        "s_addc_u32 s11, s11, 0x0",
//        "s_swappc_b64 s[30:31], s[10:11]"
//    }, agent);

    std::memcpy(newContent.data(), firstJump.data(), firstJump.size());
//    std::memcpy(newContent.data(), desperateJump.data(), desperateJump.size());
    hostCodeObjectTextSection->set_data(newContent);
//    std::memcpy(hostCodeObjectTextSection->get_data(), firstJump.data(), firstJump.size());

//    hostCodeObjectTextSection->append_data(lastJump);

    std::stringstream instElfSS;
    hostCodeObjectElfIo.save(instElfSS);
    std::string outputElf = instElfSS.str();

    // Create an executable with the ELF
    hsa_executable_t executable = createExecutable(outputElf.data(), outputElf.size(), agent);

    registerExecutableSymbol(executable, instr.getSymbol(), agent);
    // Get the loaded code object on the device
    // It is loaded, so it's not a complete ELF anymore
    std::pair<sibir_address_t, size_t> loadedCodeObject = getLoadedCodeObject(executable);
    // Get the host ELF associated with the loaded code object
    // It is used to figure out offsets in the segments of the loaded code object
    std::pair<sibir_address_t, size_t> hostLoadedCodeObject = getCodeObject(executable);

//    hsa_amd_pointer_info_t ptrInfo;
//    ptrInfo.size = sizeof(hsa_amd_pointer_info_t);
//    SibirHsaInterceptor::Instance().getSavedHsaTables().amd_ext.hsa_amd_pointer_info_fn(
//        reinterpret_cast<void*>(loadedCodeObject.first),
//        &ptrInfo,
//        nullptr,
//        nullptr,
//        nullptr
//        );

//    std::cout << ptrInfo << std::endl;
//    std::cout << ptrInfo.agentBaseAddress << std::endl;
//    std::cout << ptrInfo.hostBaseAddress << std::endl;
//    std::cout << "Agent? " << (ptrInfo.agentOwner.handle == agent.handle) << std::endl;

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
    auto insts = sibir::Disassembler::disassemble(agent, reinterpret_cast<sibir_address_t>(hcoElfIo.sections[".text"]->get_address()) + reinterpret_cast<sibir_address_t>(loadedCodeObject.first), hcoElfIo.sections[".text"]->get_size());
    unsigned int nop_inst_idx = 0;

    for (unsigned int i = 0; i < insts.size(); i++) {
        auto inst = insts[i];
        std::cout << std::hex << inst.getHostAddress() << std::dec << ": " << inst.getInstr() << std::endl;
        if (inst.getInstr().find("s_nop") != std::string::npos) {
            nop_inst_idx = i;
        }
    }
    std::cout << "First instrumentation instruction located at: " << reinterpret_cast<const void*>(insts[nop_inst_idx + 1].getDeviceAddress()) << std::endl;

//    // Calculate the last unconditional branch instruction's address
//
//    std::stringstream lastBranch;
//
//    auto trmpPC = insts[nop_inst_idx + 1].addr + 4;
//    short trmpBranchImm = (instr.addr - trmpPC - 4) / 4;
//    lastBranch << "s_branch 0x" << std::hex << trmpBranchImm;
//    Instr lastBranchInstr{insts.back().addr, lastBranch.str(), 0};
//    std::cout << "Last branch instruction: " << lastBranchInstr.instr << std::endl;
//    std::string lastBranchAssembled = assemble({lastBranchInstr}, agent);
//    std::memcpy(reinterpret_cast<void *>(insts.back().addr), lastBranchAssembled.data(), lastBranchAssembled.size());
//    std::stringstream firstBranch;
//
//    trmpPC = insts[nop_inst_idx + 1].addr;
//    short origBranchImm = (trmpPC - instr.addr - 4) / 4;
//
//    firstBranch << "s_branch 0x" << std::hex << origBranchImm;
//    Instr firstBranchInstr{instr.addr, firstBranch.str(), 0};
//    std::cout << "First branch instruction: " << firstBranchInstr.instr << std::endl;
//    std::string firstBranchAssembled = assemble({firstBranchInstr}, agent);


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
