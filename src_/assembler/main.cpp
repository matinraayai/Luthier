#include <iostream>
#include <string>
#include "assembler.h"

int main(int argc, char *argv[])
{
    if(argc != 2)
        {
            std::cout << "Expected input instruction!\n";
            return 1;
        }
    std::string i(argv[1]);

    Assembler assembler;
    assembler.Assemble(i);

    return 0;
}

/*
int main(int argc, char *argv[])
{
	if (argc != 2) 
    {
		printf("An input assembly file is required\n");
		return 1;
	}

    Inst inst;
    std::string newInst = std::string(argv[1]);

    getInstData(&inst, newInst);
    return 0;
}
*/