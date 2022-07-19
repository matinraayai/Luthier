#include "disassembler.h"
#include <memory>
void Disassembler::initializeDecodeTable()
{
	addInstType(std::make_unique<InstType>("s_add_u32", 0, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0));
}