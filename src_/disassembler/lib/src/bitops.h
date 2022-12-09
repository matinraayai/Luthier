#ifndef BITOPS_H
#define BITOPS_H
#include <stdint.h>
#include <vector>
#include <string>
uint32_t extractBitsFromU32(uint32_t num, int loInclude, int hiInclude);
uint32_t convertLE(std::vector<unsigned char> b);
std::vector<unsigned char> stringToByteArray(std::string str);
std::vector<unsigned char> charToByteArray(char *blob, uint64_t size);
uint64_t signExt(uint64_t in, int signBit);
#endif
