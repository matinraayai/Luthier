#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include "inst.h"
#include <map>
#include "initialize.h"
#include <memory>
#include <vector>
#include <string>
#include "elf.hpp"
#include "instprinter.h"

class Disassembler
{
public:
	void Disassemble(elfio::File *file, std::string filename, std::ostream &o);
	void Disassemble(elfio::File *file, std::string filename);
	std::vector<Inst> Disassemble(const unsigned char *buf);
	void Disassemble(std::vector<unsigned char> buf, std::ostream &o);
	void Disassemble(std::vector<unsigned char> buf, 
					 instnode *head, uint64_t off);

	Disassembler(elfio::File *file);
	Disassembler();
	int maxNumSReg();
	int maxNumVReg();

private:
	void initializeDecodeTable();
	void initFormatList();
	void addInstType(InstType info);
	Format matchFormat(uint32_t firstFourBytes);
    Inst decode(const unsigned char* buf);
	bool isVOP3bOpcode(Opcode opcode);
	InstType lookUp(const Format& format, Opcode opcode);
	void tryPrintSymbol(elfio::File *file, uint64_t offset, std::ostream &o);
	void decodeSOP2(Inst& inst, const unsigned char* buf);
	void decodeSOPK(Inst& inst, const unsigned char* buf);
	void decodeSOP1(Inst& inst, const unsigned char* buf);
	void decodeSOPC(Inst& inst, const unsigned char* buf);
	void decodeSOPP(Inst& inst, const unsigned char*buf);
	void decodeVOP1(Inst& inst, const unsigned char* buf);
	void decodeSMEM(Inst& inst, const unsigned char* buf);
	void decodeVOP2(Inst& inst, const unsigned char* buf);
	void decodeVOPC(Inst& inst, const unsigned char* buf);
	void decodeVOP3a(Inst& inst, const unsigned char* buf);
	void decodeVOP3b(Inst& inst, const unsigned char* buf);
	void decodeVOP3p(Inst& inst, const unsigned char* buf);
	void decodeFLAT(Inst& inst, const unsigned char* buf);
	void decodeDS(Inst& inst, const unsigned char* buf);
	void decodeSDWA(Inst& inst, const unsigned char* buf);
	void parseAbs(Inst& inst, int abs);
	void parseNeg(Inst& inst, int abs);
	void combineDSOffsets(Inst& inst);
	Operand setRegCountFromWidth(Operand o, int width);

	std::vector<Format> formatList;
	std::map<FormatType, std::unique_ptr<DecodeTable>> decodeTables;
	int nextInstID;
	InstPrinter printer;
};
#endif
