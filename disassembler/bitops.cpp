#include <bitops.h>
uint32_t extractBitsFromU32(uint32_t num, int loInclude, int hiInclude)
{
	uint32_t mask, extracted;
	mask = ((1 << (hiInclude - loInclude + 1)) - 1) << loInclude;
	extracted = (mask & num) >> loInclude;
	return extracted;
}