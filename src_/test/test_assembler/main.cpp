#include <iostream>
#include <fstream>
#include <string>
#include "assembler.h"

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        printf("Expected 1 argument: input assembly file\n");
        return 1;
    }
        
    std::string fname(argv[1]);
    std::string line;
    std::string instr;

    std::ifstream file;
    file.open(fname);

    if(file.is_open())
    {
        Assembler assembler;
        // uint32_t *newasm = new uint32_t[2];

        while (file)
        {
            std::getline(file, line);
            std::size_t i;

            if(fname.find(".s") != std::string::npos)
            { 
                i = line.find("/");
            }
            else if(fname.find(".csv") != std::string::npos)
            {
                i = line.find(";");
            }
            else
            {
                printf("Invalid file type\n");
                file.close();
                return 2;
            }

            if(i != std::string::npos)
            {
                instr = line.substr(0, i);
                std::cout << "From input file:\t" << line << std::endl
			                    << "Output from asm:\t";
		            assembler.Assemble(instr, std::cout);
                std::cout << std::endl;
            }
        }
    }
    file.close();
    return 0;
}
