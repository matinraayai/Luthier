#include <iostream>
#include <fstream>
#include "../src/internal.h"
void processBuddle(char *data)
{
	__ClangOffloadBundleHeader *header = reinterpret_cast<__ClangOffloadBundleHeader *>(data);

	std::string magic{data, 24};
	std::cout << magic << "\n";
	printf("The address of header is %p\n", (void *)header);

	uint64_t coSize;
	char *blob = reinterpret_cast<char *>(header);
	uint64_t offset =
		(uint64_t)blob + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
	std::cout << "num of bundles: " << header->numBundles << "\n";
	const __ClangOffloadBundleDesc *desc =
		reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
	uint64_t endOfHeader;
	for (int i = 0; i < header->numBundles; i++, desc = desc->next())
	{
		printf("desc struct is stored from address%p\n", (void *)desc);
		uint64_t trippleSize = desc->tripleSize;
		offset += 8 + 8 + 8 + trippleSize;
		printf("address after %d th desc is %p\n", i, (void *)offset);
		std::string triple{desc->triple, desc->tripleSize};
		std::cout << "triple " << triple << " 's offset is " << desc->offset
				  << "\n";

		coSize = desc->size;
		std::cout << "code object size is " << coSize << "\n";
		char *codeobj = reinterpret_cast<char *>(
			reinterpret_cast<uintptr_t>(header) + desc->offset);

		if (i == header->numBundles - 1)
		{
			endOfHeader = (uint64_t)codeobj + coSize;
			printf("address at the end of the last codeobject is %p\n",
				   (void *)endOfHeader);
		}
	}
}
int main(int argc, char **argv)
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
	processBuddle(blob);
}
