#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include "format.h"
#include "inst.h"
#include <map>
#include "initialize.h"
#include <memory>
#include <vector>
#include <string>
#include "../src/elf.h"
typedef uint8_t byte;
class Disassembler
{
public:
	void Disassemble(elfio::File *file, std::string filename);

	Disassembler();

private:
	void initializeDecodeTable();
	void initFormatList();
	void addInstType(std::unique_ptr<InstType> info);
	Format *matchFormat(uint32_t firstFourBytes);
	Inst *decode(byte buf[]);
	bool isVOP3bOpcode(Opcode opcode);

	std::vector<Format *> formatList;
	std::map<FormatType, std::unique_ptr<DecodeTable>> decodeTables;
	int nextInstID;
	InstPrinter printer;
};
#endif