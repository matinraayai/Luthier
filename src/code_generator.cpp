#include "code_generator.hpp"
#include "amdgpu_elf.hpp"
#include "elfio/elfio.hpp"
#include "disassembler.hpp"

void sibir::CodeGenerator::instrument(hsa_agent_t agent, const Instr &instr, const std::string &elf, sibir_ipoint_t point) {
        ELFIO::elfio coElfIO;


        std::istringstream coStrStream{elf};

        coElfIO.load(coStrStream);

        std::cout << "Using ELFIO to iterate over the sections of the ELF" << std::endl;

        for (auto it = coElfIO.sections.begin(); it != coElfIO.sections.end(); it++) {
            auto sec = it->get();
            std::cout << "Name: " << sec->get_name() << std::endl;
            std::cout << "Address: " << sec->get_address() << std::endl;
        }

        std::cout << "Read the notes section: " << std::endl;

        ELFIO::section* noteSec = coElfIO.sections[".note"];

        ELFIO::note_section_accessor note_reader(coElfIO, noteSec);

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

        for (unsigned int i = 0; i < sibir::elf::getSymbolNum(coElfIO); i++) {
            sibir::elf::SymbolInfo info;
            sibir::elf::getSymbolInfo(coElfIO, i, info);
            std::cout << "Symbol Name: " << info.sym_name << std::endl;
            std::cout << "Section Name: " << info.sec_name << std::endl;
            std::cout << "Address: " << reinterpret_cast<const void*>(info.address) << std::endl;
            std::cout << "Sec address: " << reinterpret_cast<const void*>(info.sec_addr) << std::endl;
            std::cout << "Size: " << info.size << std::endl;
            std::cout << "Sec Size: " << info.sec_size << std::endl;
            if (info.sym_name.find("instrumentation_kernel") != std::string::npos and info.sym_name.find("kd") == std::string::npos) {
                auto insts = sibir::Disassembler::disassemble(agent, reinterpret_cast<sibir_address_t>(info.address), info.size);
                for (auto i : insts)
                    std::cout << std::hex << i.addr << std::dec << ": " << i.instr << std::endl;
            }
            else if (info.sym_name.find("kd") != std::string::npos) {

                const auto kd = reinterpret_cast<const kernel_descriptor_t*>(info.address);
                std::cout << "KD: " << reinterpret_cast<const void *>(kd) << std::endl;
                std::cout << "Offset in KD: " << kd->kernel_code_entry_byte_offset << std::endl;
            }
        }
}
