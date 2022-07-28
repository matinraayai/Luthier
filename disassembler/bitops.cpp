#include "bitops.h"
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
uint32_t extractBitsFromU32(uint32_t num, int loInclude, int hiInclude)
{
	uint32_t mask, extracted;
	mask = ((1 << (hiInclude - loInclude + 1)) - 1) << loInclude;
	extracted = (mask & num) >> loInclude;
	return extracted;
}
uint32_t convertLE(std::vector<char> b)
{
	return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
}
std::vector<char> stringToByteArray(std::string str)
{

	const char *ptr = str.c_str();
	std::vector<char> bytes;
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
		std::cout << std::hex << std::setw(2) << std::setfill('0') << byte;
	}
	std::cout << std::endl;
	return bytes;
}
