#ifndef BITOPS_H
#define BITOPS_H
#include <stdint.h>
#include <vector>
#include <string>
uint32_t extractBitsFromU32(uint32_t num, int loInclude, int hiInclude);
uint32_t convertLE(std::vector<char> b);
std::vector<char> stringToByteArray(std::string str)
#endif