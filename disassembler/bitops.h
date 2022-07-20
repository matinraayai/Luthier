#ifndef BITOPS_H
#define BITOPS_H
#include <stdint.h>
uint32_t extractBitsFromU32(uint32_t num, int loInclude, int hiInclude);
uint32_t convertLE(char b[]);
#endif