#include "bitops.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

uint32_t extractBitsFromU32(uint32_t num, int loInclude, int hiInclude)
{
  uint32_t mask, extracted;
  mask = ((1 << (hiInclude - loInclude + 1)) - 1) << loInclude;
  extracted = (mask & num) >> loInclude;
  return extracted;
}

uint32_t convertLE(std::vector<unsigned char> b)
{
  auto r = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 |
           uint32_t(b[3]) << 24;
  return r;
}

std::vector<unsigned char> instcodeToByteArray(std::vector<uint32_t> inst)
{
  std::vector<unsigned char> bytes;
  uint8_t buf;
  uint32_t byteVal;

  for (int i = 0; i < inst.size(); i++)
  {
    for (int j = 0; j <= 24; j += 8)
    {
      byteVal = extractBitsFromU32(inst.at(i), j, j+7);
      buf = uint8_t(byteVal);
      bytes.push_back(buf);
    }
  }

  return bytes;
}

std::vector<unsigned char> stringToByteArray(std::string str)
{
  const char *ptr = str.c_str();
  std::vector<unsigned char> bytes;
  for (int i = 0; i < 8; i += 2)
  {
    int j = 8 - 1 - i;
    uint8_t h4, l4;
    if (ptr[j - 1] < 65)
    {
      h4 = (uint8_t)(ptr[j - 1] - '0');
    }
    else
    {
      h4 = (uint8_t)(ptr[j - 1] - 'A' + 10);
    }
    if (ptr[j] < 65)
    {
      l4 = (uint8_t)(ptr[j] - '0');
    }
    else
    {
      l4 = (uint8_t)(ptr[j] - 'A' + 10);
    }

    uint32_t byte = (h4 << 4) + l4;
    bytes.push_back((char)byte);
    // std::cout << std::hex << std::setw(2) << std::setfill('0') << byte;
  }
  std::cout << std::endl;
  return bytes;
}

std::vector<unsigned char> charToByteArray(char *blob, uint64_t size)
{
  std::vector<unsigned char> bytes;
  char *word = new char[4];
  
  for (uint64_t i = 0; i < size; i += 4)
  {
    std::memcpy(word, blob + i, 4);
    for (int j = 0; j < 4; j++)
    {
      bytes.push_back(uint8_t(word[j]));
    }
  }
  return bytes;
}

uint64_t signExt(uint64_t in, int signBit)
{
  uint64_t out = in;

  uint64_t mask;
  mask = ~((1 << (signBit + 1)) - 1);

  auto sign = (in >> signBit) & 1;

  if (sign > 0)
  {
    out = out | mask;
  }
  else
  {
    mask = ~mask;
    out = out & mask;
  }
  return out;
}
