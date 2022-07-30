#include "disassembler.h"
#include "operand.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>

Disassembler::Disassembler()
{
  nextInstID = 0;
  initFormatList();
  initializeDecodeTable();
}

void Disassembler::addInstType(InstType info)
{
  if (decodeTables.find(info.format.formatType) == decodeTables.end())
  {
    decodeTables[info.format.formatType] =
        std::unique_ptr<DecodeTable>(new DecodeTable);
  }
  (decodeTables[info.format.formatType])->insts[info.opcode] = info;
  info.ID = nextInstID;
  nextInstID++;
}

void Disassembler::initFormatList()
{
  for (auto &item : FormatTable)
  {
    formatList.push_back(item.second);
  }

  std::sort(formatList.begin(), formatList.end(),
            [](Format const &f1, Format const &f2) -> bool
            {
              return f1.mask > f2.mask;
            });
}
void Disassembler::Disassemble(elfio::File *file, std::string filename,
                               std::ostream &o)
{
  printer.file = file;
  o << "\n"
    << filename << "\tfile format ELF64-amdgpu\n";
  o << "\n\nDisassembly of section .text:\n";
  auto text_section = file->GetSectionByName(".text");
  if (!text_section)
  {
    throw std::runtime_error("text section is not found");
  }
  std::vector<unsigned char> buf(text_section->Blob(),
                                 text_section->Blob() + text_section->size);
  auto pc = text_section->offset;
  if (!buf.empty())
  {
    tryPrintSymbol(file, pc, o);
    std::unique_ptr<Inst> inst = decode(buf);
    inst->PC = pc;
    std::string instStr = printer.print(inst.get());
    o << "\t" << instStr;
    for (int i = instStr.size(); i < 59; i++)
    {
      o << " ";
    }
    o << std::setbase(10) << "//" << std::setw(12) << std::setfill('0') << pc
      << ": ";
    o << std::setw(8) << std::hex << convertLE(buf);
    if (inst->byteSize == 8)
    {
      std::vector<unsigned char> sfb(buf.begin() + 4, buf.begin() + 8);
      o << std::setw(8) << std::hex << convertLE(sfb) << std::endl;
    }
    buf.erase(buf.begin(), buf.begin() + inst->byteSize);
    pc += uint64_t(inst->byteSize);
  }
}
void Disassembler::Disassemble(std::string filename)
{
  std::string line;
  std::fstream myfile(filename, std::ios::in | std::ios::out);
  std::vector<unsigned char> buf, lo4, hi4;
  if (myfile.is_open())
  {
    while (getline(myfile, line))
    {
      int location1, location2;
      location1 = line.find_first_of(";");
      std::string inst_str = line.substr(0, location1);
      location2 = line.find_last_of(";");
      if (location1 == location2)
      {
        std::string str_bytes = line.substr(location1 + 1, 8);
        buf = stringToByteArray(str_bytes);
        try
        {
          std::unique_ptr<Inst> inst = decode(buf);
          std::string instStr = printer.print(inst.get());
          if (instStr == inst_str)
          {
            std::cout << line + "[PASS]\n";
          }
          else
          {
            std::cout << line + "[MISMATCH]\n";
            std::cout << "output is " << instStr << std::endl;
          }
        }
        catch (std::runtime_error &error)
        {
          std::cout << "Exception occured: " << error.what()
                    << " at inst: " << str_bytes << std::endl;
          std::cout << line + "[FAIL]\n";
        }
        buf.clear();
      }
      else
      {
        std::string str_bytes_lo = line.substr(location1 + 1, 8);
        lo4 = stringToByteArray(str_bytes_lo);
        std::string str_bytes_hi = line.substr(location2 + 1, 8);
        hi4 = stringToByteArray(str_bytes_hi);
        buf.reserve(lo4.size() + hi4.size());
        buf.insert(buf.end(), lo4.begin(), lo4.end());
        buf.insert(buf.end(), hi4.begin(), hi4.end());
        try
        {
          std::unique_ptr<Inst> inst = decode(buf);
          std::string instStr = printer.print(inst.get());
          if (instStr == inst_str)
          {
            std::cout << line + "[PASS]\n";
          }
          else
          {
            std::cout << line + "[MISMATCH]\n";
            std::cout << "output is " << instStr << std::endl;
          }
        }
        catch (std::runtime_error &error)
        {
          std::cout << "Exception occured: " << error.what()
                    << " at inst: " << str_bytes_lo << str_bytes_hi
                    << std::endl;
          std::cout << line + "[FAIL]\n";
        }
        buf.clear();
      }
    }
    myfile.close();
  }
  else
  {
    std::cout << "Unable to open file\n";
    return;
  }
}
void Disassembler::tryPrintSymbol(elfio::File *file, uint64_t offset,
                                  std::ostream &o)
{
  for (auto &symbol : file->GetSymbols())
  {
    if (symbol->value == offset)
    {
      o << "\n"
        << std::setw(16) << std::setfill('0') << std::hex << offset;
      o << " " << symbol->name << ":\n";
    }
  }
}
Format Disassembler::matchFormat(uint32_t firstFourBytes)
{
  for (auto &f : formatList)
  {
    if (f.formatType == VOP3b)
    {
      continue;
    }
    if (((firstFourBytes ^ f.encoding) & f.mask) == 0)
    {
      if (f.formatType == VOP3a)
      {
        auto opcode = f.retrieveOpcode(firstFourBytes);
        if (isVOP3bOpcode(opcode))
        {
          return FormatTable[VOP3b];
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
bool Disassembler::isVOP3bOpcode(Opcode opcode)
{
  switch (opcode)
  {
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
  }
  return false;
}
InstType Disassembler::lookUp(Format format, Opcode opcode)
{
  if (decodeTables.find(format.formatType) != decodeTables.end() &&
      decodeTables[format.formatType]->insts.find(opcode) !=
          decodeTables[format.formatType]->insts.end())
  {
    return decodeTables[format.formatType]->insts[opcode];
  }
  std::stringstream stream;
  stream << "instruction format " << format.formatName << ", opcode " << opcode
         << " not found\n";
  throw std::runtime_error(stream.str());
}

std::unique_ptr<Inst> Disassembler::decode(std::vector<unsigned char> buf)
{
  Format format = matchFormat(convertLE(buf));

  Opcode opcode = format.retrieveOpcode(convertLE(buf));

  InstType instType = lookUp(format, opcode);

  auto inst = std::make_unique<Inst>();
  inst->format = format;
  inst->instType = instType;
  inst->byteSize = format.byteSizeExLiteral;

  if (inst->byteSize > buf.size())
  {
    throw std::runtime_error("no enough buffer");
  }

  switch (format.formatType)
  {
  case SOP2:
    decodeSOP2(inst.get(), buf);
    break;
  case SMEM:
    decodeSMEM(inst.get(), buf);
    break;
  }
  return std::move(inst);
}
void Disassembler::decodeSOP2(Inst *inst, std::vector<unsigned char> buf)
{
  uint32_t bytes = convertLE(buf);
  uint32_t src0Value = extractBitsFromU32(bytes, 0, 7);
  inst->src0 = getOperandByCode(uint16_t(src0Value));
  if (inst->src0.operandType == LiteralConstant)
  {
    inst->byteSize += 4;
    if (buf.size() < 8)
    {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src0.literalConstant = convertLE(sub);
  }
  uint32_t src1Value = extractBitsFromU32(bytes, 8, 15);
  inst->src1 = getOperandByCode(uint16_t(src1Value));
  if (inst->src1.operandType == LiteralConstant)
  {
    inst->byteSize += 4;
    if (buf.size() < 8)
    {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> sub(&buf[4], &buf[8]);
    inst->src1.literalConstant = convertLE(sub);
  }
  uint32_t sdstValue = extractBitsFromU32(bytes, 16, 22);
  inst->dst = getOperandByCode(uint16_t(sdstValue));

  if (inst->instType.instName.find("64") != std::string::npos)
  {
    inst->src0.regCount = 2;
    inst->src1.regCount = 2;
    inst->dst.regCount = 2;
  }
}
void Disassembler::decodeSMEM(Inst *inst, std::vector<unsigned char> buf)
{
  auto bytesLo = convertLE(buf);
  std::vector<unsigned char> sfb(buf.begin() + 4, buf.end());
  auto bytesHi = convertLE(sfb);

  if (extractBitsFromU32(bytesLo, 16, 16) != 0)
  {
    inst->GlobalLevelCoherent = true;
  }

  if (extractBitsFromU32(bytesLo, 17, 17) != 0)
  {
    inst->Imm = true;
  }

  auto sbaseValue = extractBitsFromU32(bytesLo, 0, 5);
  auto bits = int(sbaseValue << 1);
  inst->base = newSRegOperand(bits, bits, 2);

  auto sdataValue = extractBitsFromU32(bytesLo, 6, 12);
  inst->data = getOperandByCode(uint16_t(sdataValue));
  if (inst->data.operandType == LiteralConstant)
  {
    inst->byteSize += 4;
    if (buf.size() < 8)
    {
      throw std::runtime_error("no enough bytes for literal");
    }
    std::vector<unsigned char> nfb(buf.begin() + 8, buf.begin() + 12);
    inst->data.literalConstant = convertLE(nfb);
  }

  if (inst->instType.opcode == 0)
  {
    inst->data.regCount = 1;
  }
  else if (inst->instType.opcode == 1 || inst->instType.opcode == 9 ||
           inst->instType.opcode == 17 || inst->instType.opcode == 25)
  {
    inst->data.regCount = 2;
  }
  else if (inst->instType.opcode == 2 || inst->instType.opcode == 10 ||
           inst->instType.opcode == 18 || inst->instType.opcode == 26)
  {
    inst->data.regCount = 4;
  }
  else if (inst->instType.opcode == 3 || inst->instType.opcode == 11 ||
           inst->instType.opcode == 19 || inst->instType.opcode == 27)
  {
    inst->data.regCount = 8;
  }
  else
  {
    inst->data.regCount = 16;
  }

  if (inst->Imm)
  {
    uint64_t bits64 = (uint64_t)extractBitsFromU32(bytesHi, 0, 19);
    inst->offset = newIntOperand(0, bits64);
  }
  else
  {
    int bits = (int)extractBitsFromU32(bytesHi, 0, 19);
    inst->offset = newSRegOperand(bits, bits, 1);
  }
}