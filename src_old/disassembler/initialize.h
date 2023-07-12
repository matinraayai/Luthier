#ifndef INITIALIZE_H
#define INITIALIZE_H
#include "reg.h"
#include "inst.h"
#include <map>
#include <memory>
extern std::map<FormatType, Format> FormatTable;
void initFormatTable();
struct DecodeTable
{
	std::map<Opcode, InstType> insts;
};

#endif