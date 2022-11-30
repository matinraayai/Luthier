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
        if(fname.find(".s") != std::string::npos)
        {
            while (file)
            {
                std::getline(file, line);
                auto i = line.find("/");
                if(i != std::string::npos)
                {
                    instr = line.substr(0, i);
                    printf("\n%s\n", instr.c_str());
                    assembler.Assemble(instr);
                }
            }
        }
        else if(fname.find(".csv") != std::string::npos)
        {
            while (file)
            {
                std::getline(file, line);
                auto i = line.find(";");
                if(i != std::string::npos)
                {
                    instr = line.substr(0, i);
                    printf("\n%s\n", instr.c_str());
                    assembler.Assemble(instr);
                }
            }
        }
        else
        {
            printf("Invalid file type\n");
        }
    }
    file.close();
    return 0;
}