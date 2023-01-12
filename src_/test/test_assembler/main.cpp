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
                std::cout << line << std::endl;

                // auto instbytes = assembler.Assemble(instr);
                
                // std::cout<<"Assembler Output:\n";
                // for (int i = instbytes.size()-1; i >= 0; i--)
                //     std::cout << std::hex << int(instbytes.at(i))<<" ";
                // std::cout << std::endl;

                std::cout << "Extract Reg Output:\n";
                auto instparams = assembler.getInstParams(instr);
                for (int i = 1; i < instparams.size(); i++) {
                    std::cout << std::hex << uint32_t(assembler.extractGPRbyte(instparams.at(i))) << " ";
                }
                std::cout << std::endl << std::endl;
                // printf("Expected: %s\n", line.substr(i+1, line.length()).c_str());
            }
        }
    }
    file.close();
    return 0;
}