

#ifndef LUTHIER_COMMON_LIFT_HPP
#define LUTHIER_COMMON_LIFT_HPP
#include <llvm/ADT/SmallVector.h>

namespace llvm {

namespace object {
class ELFSymbolRef;
}

class Error;

class MCAsmInfo;

class MCDisassembler;

class MCInst;

} // namespace llvm

namespace luthier {

llvm::Error disassemble(
    llvm::ArrayRef<uint8_t> Code, const llvm::MCDisassembler &DisAsm,
    const llvm::MCAsmInfo &MAI, llvm::SmallVectorImpl<llvm::MCInst> &Insts,
    llvm::SmallVectorImpl<uint64_t> *Offsets = nullptr);

} // namespace luthier

#endif