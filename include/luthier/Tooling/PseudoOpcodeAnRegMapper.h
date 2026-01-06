

#ifndef LUTHIER_TOOLING_PSEUDO_OPCODE_AN_REG_MAPPER_H
#define LUTHIER_TOOLING_PSEUDO_OPCODE_AN_REG_MAPPER_H

namespace luthier {

unsigned short getPseudoOpcodeFromReal(unsigned short Opcode);

unsigned short RealToPseudoRegisterMapTable(unsigned short Reg);

} // namespace luthier

#endif