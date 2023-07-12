#include "amdgpu_elf.hpp"
#include "disassembler.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("a input elf file is required\n");
		return 1;
	}
	std::string filename = argv[1];

	initRegs();
	initFormatTable();

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
	// elfFile.PrintSymbolsForSection(".text");

	Disassembler d(&elfFile);
	std::cout.setstate(std::ios_base::badbit);
	d.Disassemble(&elfFile, filename, std::cout);
	std::cout.clear();
	std::cout << "The maximum number of sReg is " << std::setbase(10) << d.maxNumSReg() << "\n";
	std::cout << "The maximum number of vReg is " << d.maxNumVReg() << "\n";

	return 0;
}
