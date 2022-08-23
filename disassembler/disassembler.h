#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include "inst.h"
#include <map>
#include "initialize.h"
#include <memory>
#include <vector>
#include <string>
#include "elf.h"
#include "instprinter.h"

class Disassembler
{
public:
	void Disassemble(elfio::File *file, std::string filename, std::ostream &o);
	void Disassemble(elfio::File *file, std::string filename);
	Disassembler(elfio::File *file);
	int maxNumSReg();
	int maxNumVReg();

private:
	void initializeDecodeTable();
	void initFormatList();
	void addInstType(InstType info);
	Format matchFormat(uint32_t firstFourBytes);
	std::unique_ptr<Inst> decode(std::vector<unsigned char> buf);
	bool isVOP3bOpcode(Opcode opcode);
	InstType lookUp(Format format, Opcode opcode);
	void tryPrintSymbol(elfio::File *file, uint64_t offset, std::ostream &o);
	void decodeSOP2(Inst *inst, std::vector<unsigned char> buf);
	void decodeSOPK(Inst *inst, std::vector<unsigned char> buf);
	void decodeSOP1(Inst *inst, std::vector<unsigned char> buf);
	void decodeSOPC(Inst *inst, std::vector<unsigned char> buf);
	void decodeSOPP(Inst *inst, std::vector<unsigned char> buf);
	void decodeVOP1(Inst *inst, std::vector<unsigned char> buf);
	void decodeSMEM(Inst *inst, std::vector<unsigned char> buf);
	void decodeVOP2(Inst *inst, std::vector<unsigned char> buf);
	void decodeVOPC(Inst *inst, std::vector<unsigned char> buf);
	void decodeVOP3a(Inst *inst, std::vector<unsigned char> buf);
	void decodeVOP3b(Inst *inst, std::vector<unsigned char> buf);
	void decodeFLAT(Inst *inst, std::vector<unsigned char> buf);
	void decodeDS(Inst *inst, std::vector<unsigned char> buf);
	void decodeSDWA(Inst *inst, std::vector<unsigned char> buf);
	void parseAbs(Inst *inst, int abs);
	void parseNeg(Inst *inst, int abs);
	void combineDSOffsets(Inst *inst);
	Operand setRegCountFromWidth(Operand o, int width);

	std::vector<Format> formatList;
	std::map<FormatType, std::unique_ptr<DecodeTable>> decodeTables;
	int nextInstID;
	InstPrinter printer;
};
#endif