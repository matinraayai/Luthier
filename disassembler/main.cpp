#include <iostream>
#include <fstream>
#include <string>
#include "../src/elf.h"
#include "disassembler.h"
int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("a input elf file is required\n");
		return 1;
	}
	std::string filename = argv[1];
	std::streampos size;
	char *blob;
	std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open())
	{
		size = file.tellg();
		blob = new char[size];
		file.seekg(0, std::ios::beg);
		file.read(blob, size);
		file.close();
	}
	else
	{
		printf("unable to open file\n");
		return 1;
	}
	elfio::File elfFile;
	elfFile = elfFile.FromMem(blob);
	elfFile.PrintSymbolsForSection(".text");

	initRegs();
	initFormatTable();

	Disassembler d;
	d.Disassemble(&elfFile, filename, std::cout);

	return 0;
}
