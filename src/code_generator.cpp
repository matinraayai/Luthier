#include "code_generator.hpp"
#include "amdgpu_elf.hpp"
#include "elfio/elfio.hpp"
#include "disassembler.hpp"
#include "context_manager.hpp"
#include "error_check.h"
#include "hsa_intercept.h"

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


std::string assemble(const std::vector<Instr>& instrVector, hsa_agent_t agent) {
    std::stringstream instStrStream;

    for (const auto& i: instrVector)
        instStrStream << i.instr << std::endl;
    std::string instString = instStrStream.str();

    amd_comgr_data_t dataIn;
    amd_comgr_data_set_t dataSetIn, dataSetOut;
    amd_comgr_action_info_t dataAction;

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data_set(&dataSetIn));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataIn));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(dataIn, instString.size(), instString.data()));
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

void sibir::CodeGenerator::instrument(hsa_agent_t agent, const Instr &instr, const std::string &elf, sibir_ipoint_t point) {
    ELFIO::elfio coElfIO;
    std::istringstream coStrStream{elf};
    coElfIO.load(coStrStream);

    std::cout << "Instruction to be instrumented: " << instr.instr << std::endl;
    std::cout << "Instruction address: " << std::hex << reinterpret_cast<void*>(instr.addr) << std::dec << std::endl;

    // Extend the original code object with trampoline code
    std::string trampoline = assemble({
                                          {0, "s_nop 0", 0}, // A NOP sled
                                          {0, "s_nop 0", 0},
                                          {0, "s_nop 0", 0},
                                          {0, "s_nop 0", 0},
                                          {0, "s_nop 0", 0},
                                          {0, "s_getpc_b64 s[10:11]", 0},
                                          {0, "s_add_u32 s10, s10, 0xffffffbc", 0},
                                          {0, "s_addc_u32 s11, s11, -1", 0},
                                          {0, "s_swappc_b64 s[30:31], s[10:11]", 0},
                                          instr,
                                          {0, "s_branch 0x00", 0}
                                      }, agent);

    coElfIO.sections[".text"]->append_data(trampoline);

    std::stringstream trampolineStrStream;
    coElfIO.save(trampolineStrStream);
    std::string outputElf = trampolineStrStream.str();

    // Create an executable with the ELF
    hsa_executable_t executable = createExecutable(outputElf.data(), outputElf.size(), agent);
    // Get the loaded code object on the device
    // It is loaded, so it's not a complete ELF anymore
    std::pair<sibir_address_t, size_t> loadedCodeObject = getLoadedCodeObject(executable);
    // Get the host ELF associated with the loaded code object
    // It is used to figure out offsets in the segments of the loaded code object
    std::pair<sibir_address_t, size_t> hostLoadedCodeObject = getCodeObject(executable);

    ELFIO::elfio hcoElfIo;
    std::istringstream hcoStrStream{std::string(
        reinterpret_cast<char*>(hostLoadedCodeObject.first), hostLoadedCodeObject.second)};
    hcoElfIo.load(hcoStrStream);

    std::cout << "Host Code Object starts at: " << reinterpret_cast<void*>(hostLoadedCodeObject.first) << std::endl;
    std::cout << "Host Code Object ends at" << reinterpret_cast<void*>(hostLoadedCodeObject.first + hostLoadedCodeObject.second) << std::endl;
            std::cout << "Text section for the ELF of HCO starts at: " << reinterpret_cast<const void*>(hcoElfIo.sections[".text"]->get_address()) << std::endl;
    auto offset = (reinterpret_cast<const sibir_address_t>(hcoElfIo.sections[".text"]->get_data()) -
                   reinterpret_cast<const sibir_address_t>(hostLoadedCodeObject.first));
    std::cout << "Text section offset: " << reinterpret_cast<const void*>(offset) << std::endl;
    const sibir_address_t loadedCoTextSection = reinterpret_cast<const sibir_address_t>(hcoElfIo.sections[".text"]->get_address()) + reinterpret_cast<const sibir_address_t>(loadedCodeObject.first);


    std::cout << "Loaded code object is located at: " <<  reinterpret_cast<const void*>(loadedCodeObject.first) << std::endl;
    auto insts = sibir::Disassembler::disassemble(agent, loadedCoTextSection, hcoElfIo.sections[".text"]->get_size());
    unsigned int nop_inst_idx = 0;

    for (unsigned int i = 0; i < insts.size(); i++) {
        auto inst = insts[i];
        std::cout << std::hex << inst.addr << std::dec << ": " << inst.instr << std::endl;
        if (inst.instr.find("s_nop") != std::string::npos) {
            nop_inst_idx = i;
        }
    }
    std::cout << "First instrumentation instruction located at: " << reinterpret_cast<const void*>(insts[nop_inst_idx + 1].addr) << std::endl;

    // Calculate the last unconditional branch instruction's address

    std::stringstream lastBranch;

    auto trmpPC = insts[nop_inst_idx + 1].addr + 4;
    short trmpBranchImm = (instr.addr - trmpPC - 4) / 4;
    lastBranch << "s_branch 0x" << std::hex << trmpBranchImm;
    Instr lastBranchInstr{insts.back().addr, lastBranch.str(), 0};
    std::cout << "Last branch instruction: " << lastBranchInstr.instr << std::endl;
    std::string lastBranchAssembled = assemble({lastBranchInstr}, agent);
    std::memcpy(reinterpret_cast<void *>(insts.back().addr), lastBranchAssembled.data(), lastBranchAssembled.size());
    std::stringstream firstBranch;

    trmpPC = insts[nop_inst_idx + 1].addr;
    short origBranchImm = (trmpPC - instr.addr - 4) / 4;

    firstBranch << "s_branch 0x" << std::hex << origBranchImm;
    Instr firstBranchInstr{instr.addr, firstBranch.str(), 0};
    std::cout << "First branch instruction: " << firstBranchInstr.instr << std::endl;
    std::string firstBranchAssembled = assemble({firstBranchInstr}, agent);
    std::memcpy(reinterpret_cast<void *>(instr.addr), firstBranchAssembled.data(), firstBranchAssembled.size());
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
