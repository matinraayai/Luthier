#include "disassembler.h"
#include "operand.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

Disassembler::Disassembler(elfio::File *file) {
  nextInstID = 0;
  initFormatList();
  initRegs();
  initializeDecodeTable();
  auto printer = InstPrinter(file);
  this->printer = printer;
}
Disassembler::Disassembler() {
  nextInstID = 0;
  initFormatList();
  initRegs();
  initializeDecodeTable();
  auto printer = InstPrinter();
  this->printer = printer;
}

void Disassembler::addInstType(InstType info) {
  if (decodeTables.find(info.format.formatType) == decodeTables.end()) {
    decodeTables[info.format.formatType] =
        std::unique_ptr<DecodeTable>(new DecodeTable);
  }
  (decodeTables[info.format.formatType])->insts[info.opcode] = info;
  info.ID = nextInstID;
  nextInstID++;
}

void Disassembler::initFormatList() {
  initFormatTable();
  for (auto &item : FormatTable) {
    formatList.push_back(item.second);
  }

  std::sort(formatList.begin(), formatList.end(),
            [](Format const &f1, Format const &f2) -> bool {
              return f1.mask > f2.mask;
            });
}

void Disassembler::Disassemble(elfio::File *file, std::string filename,
                               std::ostream &o) {
  o << "\n" << filename << "\tfile format ELF64-amdgpu\n";
  o << "\n\nDisassembly of section .text:\n";
  auto text_section = file->GetSectionByName(".text");
  if (!text_section) {
    throw std::runtime_error("text section is not found");
  }
  std::vector<unsigned char> buf(text_section->Blob(),
                                 text_section->Blob() + text_section->size);
  auto pc = text_section->offset;
  while (!buf.empty()) {
    tryPrintSymbol(file, pc, o);
    std::unique_ptr<Inst> inst = decode(buf);
    inst->PC = pc;
    std::string instStr = printer.print(inst.get());
    o << "\t" << instStr;
    for (int i = instStr.size(); i < 59; i++) {
      o << " ";
    }
    o << std::setbase(10) << "//" << std::setw(12) << std::setfill('0') << pc
      << ": ";
    o << std::setw(8) << std::hex << convertLE(buf);
    if (inst->byteSize == 8) {
      std::vector<unsigned char> sfb(buf.begin() + 4, buf.begin() + 8);
      o << std::setw(8) << std::hex << convertLE(sfb) << std::endl;
    } else {
      o << std::endl;
    }
    buf.erase(buf.begin(), buf.begin() + inst->byteSize);
    pc += uint64_t(inst->byteSize);
  }
}
void Disassembler::Disassemble(elfio::File *file, std::string filename) {
  std::string line;
  std::fstream myfile(filename, std::ios::in);
  std::vector<unsigned char> buf, lo4, hi4;
  auto pc = file->GetSectionByName(".text")->offset;
  if (myfile.is_open()) {
    while (getline(myfile, line)) {
      int location1, location2;
      location1 = line.find_first_of(";");
      std::string inst_str = line.substr(0, location1);
      location2 = line.find_last_of(";");
      if (location1 == location2) {
        std::string str_bytes = line.substr(location1 + 1, 8);
        buf = stringToByteArray(str_bytes);
        try {
          std::unique_ptr<Inst> inst = decode(buf);
          inst->PC = pc;
          std::string instStr = printer.print(inst.get());

          if (instStr == inst_str) {
            std::cout << line + "[PASS]\n";
          } else {
            std::cout << line + "[MISMATCH]\n";
            std::cout << "output is " << instStr << std::endl;
          }
          pc += inst->byteSize;
        } catch (std::runtime_error &error) {
          std::cout << "Exception occured: " << error.what()
                    << " at inst: " << str_bytes << std::endl;
          std::cout << line + "[FAIL]\n";
        }
        buf.clear();
      } else {
        std::string str_bytes_lo = line.substr(location1 + 1, 8);
        lo4 = stringToByteArray(str_bytes_lo);
        std::string str_bytes_hi = line.substr(location2 + 1, 8);
        hi4 = stringToByteArray(str_bytes_hi);
        buf.reserve(lo4.size() + hi4.size());
        buf.insert(buf.end(), lo4.begin(), lo4.end());
        buf.insert(buf.end(), hi4.begin(), hi4.end());
        try {
          std::unique_ptr<Inst> inst = decode(buf);
          inst->PC = pc;
          std::string instStr = printer.print(inst.get());

          if (instStr == inst_str) {
            std::cout << line + "[PASS]\n";
          } else {
            std::cout << line + "[MISMATCH]\n";
            std::cout << "output is " << instStr << std::endl;
          }
          pc += inst->byteSize;
        } catch (std::runtime_error &error) {
          std::cout << "Exception occured: " << error.what()
                    << " at inst: " << str_bytes_lo << str_bytes_hi
                    << std::endl;
          std::cout << line + "[FAIL]\n";
        }
        buf.clear();
      }
    }
    myfile.close();
  } else {
    std::cout << "Error: " << strerror(errno);
    std::cout << "Unable to open file\n";
    return;
  }
}
void Disassembler::Disassemble(std::vector<unsigned char> buf) {
  uint64_t pc = 0;
  while (!buf.empty()) {
    std::unique_ptr<Inst> inst = decode(buf);
    inst->PC = pc;
    std::string instStr = printer.print(inst.get());
    std::cout << instStr << std::endl;

    buf.erase(buf.begin(), buf.begin() + inst->byteSize);

    pc += uint64_t(inst->byteSize);
  }
}
void Disassembler::tryPrintSymbol(elfio::File *file, uint64_t offset,
                                  std::ostream &o) {
  for (auto &symbol : file->GetSymbols()) {
    if (symbol->value == offset) {
      o << "\n" << std::setw(16) << std::setfill('0') << std::hex << offset;
      o << " " << symbol->name << ":\n";
    }
  }
}
Format Disassembler::matchFormat(uint32_t firstFourBytes) {
  for (auto &f : formatList) {
    if (f.formatType == VOP3b) {
      continue;
    }
    if (((firstFourBytes ^ f.encoding) & f.mask) == 0) {
      if (f.formatType == VOP3a) {
        auto opcode = f.retrieveOpcode(firstFourBytes);
        if (isVOP3bOpcode(opcode)) {
          return FormatTable[VOP3b];
        }

        uint32_t vop3pFormatCheck = extractBitsFromU32(firstFourBytes, 22, 25);
        if (vop3pFormatCheck == 14 || vop3pFormatCheck == 15) {
          return FormatTable[VOP3P];
        }
      }
      return f;
    }
  }
  std::stringstream stream;
  stream << "cannot find the instruction format, first four bytes are "
         << std::setw(8) << std::setfill('0') << std::hex << firstFourBytes;
  throw std::runtime_error(stream.str());
}
bool Disassembler::isVOP3bOpcode(Opcode opcode) {
  switch (opcode) {
  case 281:
    return true;
  case 282:
    return true;
  case 283:
    return true;
  case 284:
    return true;
  case 285:
    return true;
  case 286:
    return true;
  case 480:
    return true;
  case 481:
    return true;
  case 488:
    return true;
  case 489:
    return true;
  }
  return false;
}
InstType Disassembler::lookUp(Format format, Opcode opcode) {
  if (decodeTables.find(format.formatType) != decodeTables.end() &&
      decodeTables[format.formatType]->insts.find(opcode) !=
          decodeTables[format.formatType]->insts.end()) {
    return decodeTables[format.formatType]->insts[opcode];
  }
  std::stringstream stream;
  stream << "instruction format " << format.formatName << ", opcode " << opcode
         << " not found\n";
  throw std::runtime_error(stream.str());
}

std::unique_ptr<Inst> Disassembler::decode(std::vector<unsigned char> buf) {
  Format format = matchFormat(convertLE(buf));

  Opcode opcode = format.retrieveOpcode(convertLE(buf));

  InstType instType = lookUp(format, opcode);

  auto inst = std::make_unique<Inst>();
  inst->format = format;
  inst->instType = instType;
  inst->byteSize = format.byteSizeExLiteral;

  if (inst->byteSize > buf.size()) {
    throw std::runtime_error("no enough buffer");
  }

  switch (format.formatType) {
  case SOP2:
    decodeSOP2(inst.get(), buf);
    break;
  case SOPK:
    decodeSOPK(inst.get(), buf);
    break;
  case SOP1:
    decodeSOP1(inst.get(), buf);
    break;
  case SOPC:
    decodeSOPC(inst.get(), buf);
    break;
  case SOPP:
    decodeSOPP(inst.get(), buf);
    break;
  case SMEM:
    decodeSMEM(inst.get(), buf);
    break;
  case VOP1:
    decodeVOP1(inst.get(), buf);
    break;
  case VOP2:
    decodeVOP2(inst.get(), buf);
    break;
  case VOPC:
    decodeVOPC(inst.get(), buf);
    break;
  case VOP3a:
    decodeVOP3a(inst.get(), buf);
    break;
  case VOP3b:
    decodeVOP3b(inst.get(), buf);
    break;
  case VOP3P:
    decodeVOP3p(inst.get(), buf);
    break;
  case FLAT:
    decodeFLAT(inst.get(), buf);
    break;
  case DS:
    decodeDS(inst.get(), buf);
    break;
  }
  return std::move(inst);
}

void Disassembler::decodeSOP2(Inst *inst, std::vector<unsigned char> buf) {
  uint32_t bytes = convertLE(buf);
  uint32_t src0Value = extractBitsFromU32(bytes, 0, 7);
  inst->src0 = getOperandByCode(uint16_t(src0Value));
  if (inst->src0.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src0.literalConstant = convertLE(sub);
  }
  uint32_t src1Value = extractBitsFromU32(bytes, 8, 15);
  inst->src1 = getOperandByCode(uint16_t(src1Value));
  if (inst->src1.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src1.literalConstant = convertLE(sub);
  }
  uint32_t sdstValue = extractBitsFromU32(bytes, 16, 22);
  inst->dst = getOperandByCode(uint16_t(sdstValue));

  if (inst->instType.instName.find("64") != std::string::npos) {
    inst->src0.regCount = 2;
    inst->src1.regCount = 2;
    inst->dst.regCount = 2;
  }
}

void Disassembler::decodeSOPK(Inst *inst, std::vector<unsigned char> buf) {
  uint32_t bytes = convertLE(buf);

  uint32_t simm16val = extractBitsFromU32(bytes, 0, 15);
  inst->simm16 = getOperandByCode(uint16_t(simm16val));

  uint32_t sdstVal = extractBitsFromU32(bytes, 16, 22);
  inst->dst = getOperandByCode(uint16_t(sdstVal));
}

void Disassembler::decodeSOP1(Inst *inst, std::vector<unsigned char> buf) {
  uint32_t bytes = convertLE(buf);

  uint32_t src0Value = extractBitsFromU32(bytes, 0, 7);
  inst->src0 = getOperandByCode(uint16_t(src0Value));
  if (inst->instType.SRC0Width == 64)
    inst->src0.regCount = 2;

  uint32_t sdstValue = extractBitsFromU32(bytes, 16, 22);
  inst->dst = getOperandByCode(uint16_t(sdstValue));
  if (inst->instType.DSTWidth == 64)
    inst->dst.regCount = 2;

  if (inst->src0.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src0.literalConstant = convertLE(sub);
  }
}

void Disassembler::decodeSOPC(Inst *inst, std::vector<unsigned char> buf) {
  uint32_t bytes = convertLE(buf);

  uint32_t src0Value = extractBitsFromU32(bytes, 0, 7);
  inst->src0 = getOperandByCode(uint16_t(src0Value));
  if (inst->src0.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src0.literalConstant = convertLE(sub);
  }

  uint32_t src1Value = extractBitsFromU32(bytes, 8, 15);
  inst->src1 = getOperandByCode(uint16_t(src1Value));
  if (inst->src1.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src1.literalConstant = convertLE(sub);
  }
}

void Disassembler::decodeSOPP(Inst *inst, std::vector<unsigned char> buf) {
  uint32_t bytes = convertLE(buf);

  if (inst->instType.opcode == 12) // s_waitcnt
  {
    inst->VMCNT = extractBitsFromU32(uint32_t(bytes), 0, 3);
    inst->VMCNT += extractBitsFromU32(uint32_t(bytes), 14, 15) << 4;
    inst->LKGMCNT = extractBitsFromU32(uint32_t(bytes), 8, 11);
    inst->EXPCNT = extractBitsFromU32(uint32_t(bytes), 4, 6);
  } else {
    uint32_t simm16val = extractBitsFromU32(bytes, 0, 15);
    inst->simm16 = newIntOperand(0, long(simm16val));
  }
}

void Disassembler::decodeSMEM(Inst *inst, std::vector<unsigned char> buf) {
  auto bytesLo = convertLE(buf);
  std::vector<unsigned char> sfb(buf.begin() + 4, buf.end());
  auto bytesHi = convertLE(sfb);

  if (extractBitsFromU32(bytesLo, 16, 16) != 0) {
    inst->GlobalLevelCoherent = true;
  }

  if (extractBitsFromU32(bytesLo, 17, 17) != 0) {
    inst->Imm = true;
  }

  auto sbaseValue = extractBitsFromU32(bytesLo, 0, 5);
  auto bits = int(sbaseValue << 1);
  inst->base = newSRegOperand(bits, bits, 2);

  auto sdataValue = extractBitsFromU32(bytesLo, 6, 12);
  inst->data = getOperandByCode(uint16_t(sdataValue));
  if (inst->data.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> nfb(buf.begin() + 8, buf.begin() + 12);
    inst->data.literalConstant = convertLE(nfb);
  }

  if (inst->instType.opcode == 0) {
    inst->data.regCount = 1;
  } else if (inst->instType.opcode == 1 || inst->instType.opcode == 9 ||
             inst->instType.opcode == 17 || inst->instType.opcode == 25) {
    inst->data.regCount = 2;
  } else if (inst->instType.opcode == 2 || inst->instType.opcode == 10 ||
             inst->instType.opcode == 18 || inst->instType.opcode == 26) {
    inst->data.regCount = 4;
  } else if (inst->instType.opcode == 3 || inst->instType.opcode == 11 ||
             inst->instType.opcode == 19 || inst->instType.opcode == 27) {
    inst->data.regCount = 8;
  } else {
    inst->data.regCount = 16;
  }

  if (inst->Imm) {
    uint64_t bits64 = (uint64_t)extractBitsFromU32(bytesHi, 0, 19);
    inst->offset = newIntOperand(0, bits64);
  } else {
    int bits = (int)extractBitsFromU32(bytesHi, 0, 19);
    inst->offset = newSRegOperand(bits, bits, 1);
  }
}

void Disassembler::decodeVOP1(Inst *inst, std::vector<unsigned char> buf) {
  uint32_t bytes = convertLE(buf);
  if (extractBitsFromU32(bytes, 0, 8) == 249) {
    decodeSDWA(inst, buf);
  } else {
    uint32_t src0Value = extractBitsFromU32(bytes, 0, 8);
    inst->src0 = getOperandByCode(uint16_t(src0Value));
  }
  if (inst->src0.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src0.literalConstant = convertLE(sub);
  }

  if (inst->instType.SRC0Width == 64) {
    inst->src0.regCount = 2;
  }

  uint32_t sdstValue = extractBitsFromU32(bytes, 17, 24);
  switch (inst->instType.opcode) {
  case 2:
    inst->dst = getOperandByCode(uint16_t(sdstValue));
    break;
  default:
    inst->dst = getOperandByCode(uint16_t(sdstValue + 256));
  }

  if (inst->instType.DSTWidth == 64) {
    inst->dst.regCount = 2;
  }

  switch (inst->instType.opcode) {
  case 4: // v_cvt_f64_i32_e32
    inst->dst.regCount = 2;
    break;
  case 15: // v_cvt_f32_f64_e32
    inst->src0.regCount = 2;
    break;
  case 16: // v_cvt_f64_f32_e32
    inst->dst.regCount = 2;
    break;
  }
}

void Disassembler::decodeVOP2(Inst *inst, std::vector<unsigned char> buf) {
  uint32_t bytes = convertLE(buf);
  if (extractBitsFromU32(bytes, 0, 8) == 249) {
    decodeSDWA(inst, buf);
  } else {
    uint32_t src0Value = extractBitsFromU32(bytes, 0, 8);
    inst->src0 = getOperandByCode(uint16_t(src0Value));
  }

  if (inst->src0.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src0.literalConstant = convertLE(sub);
  }

  int bits = (int)extractBitsFromU32(bytes, 9, 16);
  inst->src1 = newVRegOperand(bits, bits, 0);

  bits = (int)extractBitsFromU32(bytes, 17, 24);
  inst->dst = newVRegOperand(bits, bits, 0);

  if (inst->instType.opcode == 24 || inst->instType.opcode == 37) { // madak
    inst->Imm = true;
    inst->byteSize += 4;
    Operand o;
    o.operandType = LiteralConstant;
    inst->src2 = o;
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src2.literalConstant = convertLE(sub);
  }
}

void Disassembler::decodeVOPC(Inst *inst, std::vector<unsigned char> buf) {
  uint32_t bytes = convertLE(buf);
  uint32_t src0Value = extractBitsFromU32(bytes, 0, 8);
  inst->src0 = getOperandByCode(uint16_t(src0Value));
  if (inst->src0.operandType == LiteralConstant) {
    inst->byteSize += 4;
    if (buf.size() < 8) {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src0.literalConstant = convertLE(sub);
  }

  int bits = (int)extractBitsFromU32(bytes, 9, 16);
  inst->src1 = newVRegOperand(bits, bits, 0);
}

void Disassembler::decodeVOP3a(Inst *inst, std::vector<unsigned char> buf) {
  auto bytesLo = convertLE(buf);
  std::vector<unsigned char> sfb(buf.begin() + 4, buf.end());
  auto bytesHi = convertLE(sfb);

  int bits = (int)extractBitsFromU32(bytesLo, 0, 7);
  if (inst->instType.opcode <= 255) {
    inst->dst = getOperandByCode(uint16_t(bits));
  } else {
    inst->dst = newVRegOperand(bits, bits, 0);
  }
  if (inst->instType.DSTWidth == 64) {
    inst->dst.regCount = 2;
  }

  inst->Abs = (int)extractBitsFromU32(bytesLo, 8, 10);
  parseAbs(inst, inst->Abs);

  if (extractBitsFromU32(bytesLo, 15, 15) != 0) {
    inst->Clamp = true;
  }

  inst->src0 = getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 0, 8)));
  if (inst->instType.SRC0Width == 64) {
    inst->src0.regCount = 2;
  }
  inst->src1 = getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 9, 17)));
  if (inst->instType.SRC1Width == 64) {
    inst->src1.regCount = 2;
  }

  if (inst->instType.SRC2Width != 0) {
    inst->src2 =
        getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 18, 26)));
    if (inst->instType.SRC2Width == 64) {
      inst->src2.regCount = 2;
    }
  }

  inst->Omod = (int)extractBitsFromU32(bytesHi, 27, 28);
  inst->Neg = (int)extractBitsFromU32(bytesHi, 29, 31);
  parseNeg(inst, inst->Neg);
}

void Disassembler::decodeSDWA(Inst *inst, std::vector<unsigned char> buf) {
  if (buf.size() < 8)
    throw std::runtime_error("no enough bytes");
  inst->byteSize = buf.size();
  inst->IsSdwa = true;

  std::vector<unsigned char> sdwabuf;
  for (int i = 4; i < 8; i++)
    sdwabuf.push_back(buf.at(i));
  uint32_t sdwaBytes = convertLE(sdwabuf);

  uint32_t src0Value = extractBitsFromU32(sdwaBytes, 0, 7);
  inst->src0 = newVRegOperand(int(src0Value), int(src0Value), 0);

  inst->DstSel = int(extractBitsFromU32(sdwaBytes, 8, 10));
  inst->DstUnused = int(extractBitsFromU32(sdwaBytes, 11, 12));

  inst->Src0Sel = int(extractBitsFromU32(sdwaBytes, 16, 18));
  extractBitsFromU32(sdwaBytes, 19, 19) == 1 ? inst->Src0Sext = true
                                             : inst->Src0Sext = false;
  extractBitsFromU32(sdwaBytes, 20, 20) == 1 ? inst->Src0Neg = true
                                             : inst->Src0Neg = false;
  extractBitsFromU32(sdwaBytes, 21, 21) == 1 ? inst->Src0Abs = true
                                             : inst->Src0Abs = false;

  inst->Src1Sel = int(extractBitsFromU32(sdwaBytes, 24, 26));
  extractBitsFromU32(sdwaBytes, 27, 27) == 1 ? inst->Src1Sext = true
                                             : inst->Src1Sext = false;
  extractBitsFromU32(sdwaBytes, 28, 28) == 1 ? inst->Src1Neg = true
                                             : inst->Src1Neg = false;
  extractBitsFromU32(sdwaBytes, 29, 29) == 1 ? inst->Src1Abs = true
                                             : inst->Src1Abs = false;
}

void Disassembler::parseAbs(Inst *inst, int abs) {
  if ((abs & 0b001) > 0) {
    inst->Src0Abs = true;
  }

  if ((abs & 0b010) > 0) {
    inst->Src1Abs = true;
  }
  if ((abs & 0b100) > 0) {
    inst->Src2Abs = true;
  }
}

void Disassembler::parseNeg(Inst *inst, int neg) {
  if ((neg & 0b001) > 0) {
    inst->Src0Neg = true;
  }

  if ((neg & 0b010) > 0) {
    inst->Src1Neg = true;
  }
  if ((neg & 0b100) > 0) {
    inst->Src2Neg = true;
  }
}

void Disassembler::decodeVOP3b(Inst *inst, std::vector<unsigned char> buf) {
  auto bytesLo = convertLE(buf);
  std::vector<unsigned char> sfb(buf.begin() + 4, buf.end());
  auto bytesHi = convertLE(sfb);

  if (inst->instType.opcode > 255) {
    int dstBits = (int)extractBitsFromU32(bytesLo, 0, 7);
    inst->dst = newVRegOperand(dstBits, dstBits, 1);
    if (inst->instType.DSTWidth == 64) {
      inst->dst.regCount = 2;
    }
  }
  inst->sdst = getOperandByCode(uint16_t(extractBitsFromU32(bytesLo, 8, 14)));
  if (inst->instType.SDSTWidth == 64) {
    inst->sdst.regCount = 2;
  }

  if (extractBitsFromU32(bytesLo, 15, 15) != 0) {
    inst->Clamp = true;
  }

  inst->src0 = getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 0, 8)));
  if (inst->instType.SRC0Width == 64) {
    inst->src0.regCount = 2;
  }

  inst->src1 = getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 9, 17)));
  if (inst->instType.SRC1Width == 64) {
    inst->src1.regCount = 2;
  }

  if (inst->instType.opcode > 255 && inst->instType.SRC2Width > 0) {
    inst->src2 =
        getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 18, 26)));
    if (inst->instType.SRC2Width == 64) {
      inst->src2.regCount = 2;
    }
  }

  inst->Omod = (int)extractBitsFromU32(bytesHi, 27, 28);
  inst->Neg = (int)extractBitsFromU32(bytesHi, 29, 31);
}

void Disassembler::decodeVOP3p(Inst *inst, std::vector<unsigned char> buf) {
  auto bytesLo = convertLE(buf);
  std::vector<unsigned char> sfb(buf.begin() + 4, buf.end());
  auto bytesHi = convertLE(sfb);

  int dstBits = (int)extractBitsFromU32(bytesLo, 0, 7);
  inst->dst = newVRegOperand(dstBits, dstBits, 1);
  if (inst->instType.DSTWidth == 64) {
    inst->dst.regCount = 2;
  }

  inst->src0 = getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 0, 8)));
  if (inst->instType.SRC0Width == 64) {
    inst->src0.regCount = 2;
  }

  inst->src1 = getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 9, 17)));
  if (inst->instType.SRC1Width == 64) {
    inst->src1.regCount = 2;
  }

  if ((inst->instType.opcode > 31 && inst->instType.opcode < 44) ||
      (inst->instType.opcode == 0) || (inst->instType.opcode == 9) ||
      (inst->instType.opcode == 14)) {
    inst->src2 =
        getOperandByCode(uint16_t(extractBitsFromU32(bytesHi, 18, 26)));
  }

  printf("Finished\n");
}

void Disassembler::decodeFLAT(Inst *inst, std::vector<unsigned char> buf) {
  auto bytesLo = convertLE(buf);
  std::vector<unsigned char> sfb(buf.begin() + 4, buf.end());
  auto bytesHi = convertLE(sfb);

  if (extractBitsFromU32(bytesLo, 17, 17) != 0) {
    inst->SystemLevelCoherent = true;
  }
  if (extractBitsFromU32(bytesLo, 16, 16) != 0) {
    inst->GlobalLevelCoherent = true;
  }
  if (extractBitsFromU32(bytesLo, 23, 23) != 0) {
    inst->TextureFailEnable = true;
  }

  int bits = (int)extractBitsFromU32(bytesLo, 14, 15);
  inst->Seg = bits;

  auto bitOffset = signExt(uint64_t(extractBitsFromU32(bytesLo, 0, 12)), 12);
  long bits64 = static_cast<long>(bitOffset);
  inst->offset = newIntOperand(int(bits64), bits64);

  int addrbits = (int)extractBitsFromU32(bytesHi, 0, 7);
  inst->addr = newVRegOperand(addrbits, addrbits, 2);
  bits = (int)extractBitsFromU32(bytesHi, 24, 31);
  inst->dst = newVRegOperand(bits, bits, 0);
  
  bits = (int)extractBitsFromU32(bytesHi, 8, 15);
  inst->data = newVRegOperand(bits, bits, 0);
  bits = (int)extractBitsFromU32(bytesHi, 16, 22);
  if (bits != 0x7f) {
    inst->sAddr = newSRegOperand(bits, bits, 2);
    inst->addr = newVRegOperand(addrbits, addrbits, 1);
  }

  switch (inst->instType.opcode) {
  case 21: //"_load_dwordx2"
    inst->dst.regCount = 2;
    break;
  case 23: //"_load_dwordx4"
    inst->dst.regCount = 4;
    break;
  case 31: //"_store_dwordx4"
    inst->data.regCount = 4;
    break;
  case 98:
    inst->data.regCount = 2;
    break;
  case 28:
    if (inst->sAddr.code != 0x7F) {
      inst->addr.regCount = 1;
    }
    inst->dst.regCount = 4;
    break;
  case 30:
    inst->sAddr.regCount = 4;
    inst->data.regCount = 4;
    inst->dst.regCount = 4;
    break;
  }
}

void Disassembler::decodeDS(Inst *inst, std::vector<unsigned char> buf) {
  auto bytesLo = convertLE(buf);
  std::vector<unsigned char> sfb(buf.begin() + 4, buf.end());
  auto bytesHi = convertLE(sfb);

  inst->Offset0 = extractBitsFromU32(bytesLo, 0, 7);
  inst->Offset1 = extractBitsFromU32(bytesLo, 8, 15);
  combineDSOffsets(inst);

  auto gdsBit = (int)extractBitsFromU32(bytesLo, 16, 16);
  if (gdsBit != 0) {
    inst->GDS = true;
  }

  auto addrBits = (int)extractBitsFromU32(bytesHi, 0, 7);
  inst->addr = newVRegOperand(addrBits, addrBits, 1);

  if (inst->instType.SRC0Width > 0) {
    auto data0Bits = (int)extractBitsFromU32(bytesHi, 8, 15);
    inst->data = newVRegOperand(data0Bits, data0Bits, 1);
    inst->data = setRegCountFromWidth(inst->data, inst->instType.SRC0Width);
  }

  if (inst->instType.SRC1Width > 0) {
    auto data1Bits = (int)extractBitsFromU32(bytesHi, 16, 23);
    inst->data1 = newVRegOperand(data1Bits, data1Bits, 1);
    inst->data1 = setRegCountFromWidth(inst->data1, inst->instType.SRC1Width);
  }

  if (inst->instType.DSTWidth > 0) {
    auto dstBits = (int)extractBitsFromU32(bytesHi, 24, 31);
    inst->dst = newVRegOperand(dstBits, dstBits, 1);      
    inst->dst = setRegCountFromWidth(inst->dst, inst->instType.DSTWidth);
  }
}

void Disassembler::combineDSOffsets(Inst *inst) {
  auto oc = inst->instType.opcode;
  switch (oc) {
  case 14:
    break;
  case 15:
    break;
  case 46:
    break;
  case 47:
    break;
  case 55:
    break;
  case 56:
    break;
  case 78:
    break;
  case 79:
    break;
  case 110:
    break;
  case 111:
    break;
  case 119:
    break;
  case 120:
    break;
  default:
    inst->Offset0 += inst->Offset1 << 8;
  }
}

Operand Disassembler::setRegCountFromWidth(Operand o, int width)
{
  // printf("WIDTH: val - %d\n", width);
  switch (width)
  {
  case 64:
    o.regCount = 2;
    return o;
  case 96:
    o.regCount = 3;
    return o;
  case 128:
    o.regCount = 4;
    return o;
  default:
    o.regCount = 1;
    return o;
  }
}

int Disassembler::maxNumSReg() {
  return *std::max_element(sRegNum.begin(), sRegNum.end());
}

int Disassembler::maxNumVReg() {
  return *std::max_element(vRegNum.begin(), vRegNum.end());
}
