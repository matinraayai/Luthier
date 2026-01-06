

#ifndef LUTHIER_TOOLING_PSEUDO_OPCODE_AN_REG_MAPPER_H
#define LUTHIER_TOOLING_PSEUDO_OPCODE_AN_REG_MAPPER_H
#include <llvm/Support/Compiler.h>

namespace luthier {

LLVM_READONLY
unsigned short getPseudoOpcodeFromReal(unsigned short Opcode);

LLVM_READONLY
unsigned short RealToPseudoRegisterMapTable(unsigned short Reg);

} // namespace luthier

#endif