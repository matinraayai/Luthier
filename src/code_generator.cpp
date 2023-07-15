#include "code_generator.hpp"
#include "amdgpu_elf.hpp"
#include "elfio/elfio.hpp"
#include "disassembler.hpp"
#include "context_manager.hpp"
#include "error_check.h"

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



std::string assemble(const std::vector<Instr>& instrVector, hsa_agent_t agent) {
    std::stringstream instStrStream;

    for (const auto& i: instrVector)
        instStrStream << i.instr << std::endl;
    std::string instString = instStrStream.str();

    std::cout << "Inst String's length" << instString.size() << std::endl;

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



//    std::string data;
//    data.resize(coElfIO.sections[".text"]->get_size());
//    coElfIO.sections[".text"]->set_data(data);
    for (unsigned int i = 0; i < sibir::elf::getSymbolNum(coElfIO); i++) {
        sibir::elf::SymbolInfo info;
        sibir::elf::getSymbolInfo(coElfIO, i, info);
        if (info.sec_name == ".text") {
            std::cout << "Symbol Name: " << sibir::ContextManager::getDemangledName(info.sym_name.c_str()) << std::endl;
            std::cout << "Address: " << reinterpret_cast<const void*>(info.address) << std::endl;
            std::cout << "Sec address: " << reinterpret_cast<const void*>(info.sec_addr) << std::endl;
            std::cout << "Size: " << info.size << std::endl;
            std::cout << "Sec Size: " << info.sec_size << std::endl;
            if (info.sym_name.find("instrumentation_kernel") != std::string::npos and info.sym_name.find("kd") == std::string::npos) {
                std::vector<Instr> writeInst{{0, "s_branch 0xffff"}};
                std::string instContent = assemble(writeInst, agent);
                memcpy((void *) info.address, instContent.data(), instContent.size() * sizeof(char));
                std::cout << "Assembled instruction size: " << instContent.size() << std::endl;
                auto insts = sibir::Disassembler::disassemble(agent, reinterpret_cast<sibir_address_t>(info.address), info.size);
                for (const auto& inst : insts)
                    std::cout << std::hex << inst.addr << std::dec << ": " << inst.instr << std::endl;
            }
            else if (info.sym_name.find("kd") != std::string::npos) {

                const auto kd = reinterpret_cast<const kernel_descriptor_t*>(info.address);
                std::cout << "KD: " << reinterpret_cast<const void *>(kd) << std::endl;
                std::cout << "Offset in KD: " << kd->kernel_code_entry_byte_offset << std::endl;
            }

        }
    }
}
