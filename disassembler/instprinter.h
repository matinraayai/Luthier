#ifndef INSTPRINTER_H
#define INSTPRINTER_H
#include "../src/elf.h"
#include <string>
#include "inst.h"
struct InstPrinter
{
	elfio::File *file;
	std::string sop2String(Inst i);
	std::string print(Inst i);
};
#endif