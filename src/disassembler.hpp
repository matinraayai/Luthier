#ifndef DISASSEMBLER_HPP
#define DISASSEMBLER_HPP
#include <amd_comgr/amd_comgr.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInst.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "code_view.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_instr.hpp"
#include "hsa_isa.hpp"
#include "luthier_types.h"

namespace luthier {

/**
 * \brief a singleton class in charge of disassembling device instructions and returning them as an
 * std::vector of \class Instr
 * Uses the AMD COMGR library internally
 */
class Disassembler {
 private:
    struct DisassemblyInfo {
        std::unique_ptr<const llvm::MCContext> context_;
        std::unique_ptr<const llvm::MCDisassembler> disAsm_;

        DisassemblyInfo() = delete;

        DisassemblyInfo(std::unique_ptr<const llvm::MCContext> context,
                        std::unique_ptr<const llvm::MCDisassembler> disAsm)
            : context_(std::move(context)),
              disAsm_(std::move(disAsm)) {
            assert(context_);
            assert(disAsm_);
        };
    };

    Disassembler() = default;

    ~Disassembler();

    const DisassemblyInfo &getDisassemblyInfo(const hsa::Isa &isa);

    std::unordered_map<hsa::Isa, DisassemblyInfo> disassemblyInfoMap_;

    std::unordered_set<hsa::Instr *> instrHandles_;

    hsa::Instr *createInstr(llvm::MCInst inst, hsa::ExecutableSymbol symbol);

 public:
    Disassembler(const Disassembler &) = delete;
    Disassembler &operator=(const Disassembler &) = delete;

    static inline Disassembler &instance() {
        static Disassembler instance;
        return instance;
    }

    void destroyInstr(hsa::Instr *instr);

    std::vector<hsa::Instr *> disassemble(const hsa::ExecutableSymbol &symbol,
                                          std::optional<size_t> size = std::nullopt);

    //TODO: ISA has to be detected from the ELF, not passed manually
    std::vector<llvm::MCInst> disassemble(const code::SymbolView &symbol, const hsa::Isa &isa,
                                          std::optional<size_t> size = std::nullopt);

    std::vector<llvm::MCInst> disassemble(const hsa::Isa &isa, byte_string_view code);
};

}// namespace luthier

#endif
