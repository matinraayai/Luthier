#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include "elf.hpp"
#include "initialize.h"
#include "inst.h"
#include "instmodifier.h"
#include "instprinter.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

struct DecodeTable {
  std::map<Opcode, InstType> insts;
};

class Disassembler {
public:
  void Disassemble(elfio::File *file, std::string filename, std::ostream &o);
  void Disassemble(elfio::File *file, std::string filename);
  void Disassemble(std::vector<unsigned char> buf, std::ostream &o);
  void Disassemble(std::vector<unsigned char> buf, instnode *head,
                   uint64_t off);

  Disassembler(elfio::File *file);
  Disassembler();
  int maxNumSReg();
  int maxNumVReg();
  void getMaxRegIdx(elfio::File *file, int *sRegMax, int *vRegMax);
  std::vector<std::unique_ptr<Inst>> GetInstruInsts(elfio::File *file);
  std::vector<std::unique_ptr<Inst>> GetOrigInsts(elfio::File *file);
  void SetModVal(int v_offset, int s_offset);

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
  void decodeVOP3p(Inst *inst, std::vector<unsigned char> buf);
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
  InstModifier modifier;
};
#endif
