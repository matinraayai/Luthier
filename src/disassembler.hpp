#ifndef CODE_LIFTER_HPP
#define CODE_LIFTER_HPP
#include <llvm/MC/MCContext.h>
#include <llvm/Target/TargetOptions.h>
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

    // The vectors have to be allocated as a smart pointer to stop it from calling its destructor prematurely
    // The disassembler is in charge of destroying the disassembled symbols
    std::unordered_map<hsa::ExecutableSymbol, std::unique_ptr<std::vector<hsa::Instr>>> disassembledSymbols_;

 public:
    Disassembler(const Disassembler &) = delete;
    Disassembler &operator=(const Disassembler &) = delete;

    static inline Disassembler &instance() {
        static Disassembler instance;
        return instance;
    }

    const std::vector<hsa::Instr> *disassemble(const hsa::ExecutableSymbol &symbol,
                                               std::optional<hsa::Isa> isa = std::nullopt,
                                               std::optional<size_t> size = std::nullopt);

    //TODO: ISA has to be detected from the ELF, not passed manually
    std::vector<llvm::MCInst> disassemble(const code::SymbolView &symbol, const hsa::Isa &isa,
                                          std::optional<size_t> size = std::nullopt);

    std::vector<llvm::MCInst> disassemble(const hsa::Isa &isa, byte_string_view code);

    void liftKernelModule(const hsa::ExecutableSymbol &symbol);
};

}// namespace luthier

#endif
