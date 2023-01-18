#ifndef BITOPS_H
#define BITOPS_H
#include <stdint.h>
#include <string>
#include <vector>
uint32_t extractBitsFromU32(uint32_t num, int loInclude, int hiInclude);
uint32_t zerooutBitsFromU32(uint32_t num, int loInclude, int hiInclude);
uint32_t convertLE(std::vector<unsigned char> b);
uint32_t convertLEsec(std::vector<unsigned char> b);
std::vector<unsigned char> instcodeToByteArray(std::vector<uint32_t> inst);
std::vector<unsigned char> charToByteArray(char *blob, uint64_t size);
std::vector<unsigned char> stringToByteArray(std::string str);
uint64_t signExt(uint64_t in, int signBit);
#endif
