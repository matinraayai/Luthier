#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "elf.h"
#include "disassembler.h"

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("a input elf file is required\n");
		return 1;
	}
	std::string filename = argv[1];
	std::string filename_elf;
	initRegs();
	initFormatTable();

	if (filename.find("csv") != std::string::npos)
	{
		filename_elf = filename.substr(0, filename.length() - 6);
	}
	else
	{
		filename_elf = filename;
	}
	std::streampos size;
	char *blob;
	std::ifstream file(filename_elf, std::ios::in | std::ios::binary | std::ios::ate);
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

	Disassembler d(&elfFile);
	if (filename.find("csv") != std::string::npos)
	{
		d.Disassemble(&elfFile, filename);
	}
	else
	{
		d.Disassemble(&elfFile, filename, std::cout);
	}
	std::cout << "The maximum number of sReg is " << d.maxNumSReg() << "\n";
	std::cout << "The maximum number of vReg is " << d.maxNumVReg() << "\n";

	return 0;
}
