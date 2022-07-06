#ifndef FORMAT_H
#define FORMAT_H
#include <string>
enum FormatType
{
	SOP2,
	SOPK,
	SOP1,
	SOPC,
	SOPP,
	SMEM,
	VOP2,
	VOP1,
	VOP3a,
	VOP3b,
	VOPC,
	VINTRP,
	DS,
	MUBUF,
	MTBUF,
	MIMG,
	EXP,
	FLAT
};
struct Format
{
	FormatType formatType;
	std::string formatName;
	uint32_t encoding;
	uint32_t mask;
	int byteSizeExLiteral;
	uint8_t opcodeLow;
	uint8_t opcodeHigh;
};

#endif