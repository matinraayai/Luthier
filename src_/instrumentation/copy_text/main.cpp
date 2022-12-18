#include "assembler.h" // !!!! Cannot use Assembler bc of multiple definitions of 'FormatTable' !!!!
#include "bitops.h"    // !!!! We may need to move the shared libraries of the asm and disasm sooner rather than later
#include "disassembler.h"
#include "elf.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

elfio::File getELF(elfio::File* elf, std::string fname, size_t blobsize);
instnode* getInstFromPC(instnode *head, uint64_t pc);
void printInstList(instnode *head);
// void offsetInstruRegs(Assembler a, instnode *head, uint64_t offset, int smax, int vmax);
void offsetInstruRegs(instnode *head, uint64_t offset, int smax, int vmax);

std::string insertReg(std::string reg, int regval);

// These are functions copy-pasted from the Assembler:
std::vector<std::string> getInstParams(std::string inststr)
{
  std::vector<std::string> params;
  std::string delim = " ";
  
  size_t i;
  while(i != std::string::npos)
  {
    i = inststr.find(delim);
    params.push_back(inststr.substr(0, i));
    inststr.erase(0, i+1);
  }

  return params;
}
std::string extractreg(std::string reg)
{
  if(reg.find("v") != std::string::npos) 
  {
    reg.erase(reg.find("v"), 1);
  }
  else if(reg.find("s") != std::string::npos) 
  {
    reg.erase(reg.find("s"), 1);
  }

  if(reg.find("[") != std::string::npos) 
  {
    reg.erase(reg.find("["),1);
    reg.erase(reg.find(":"),reg.length());
    return reg;
  }
  else
  {
    reg.erase(1, reg.length());
    return reg;
  }
}
// Delete these after using the assembler with this works!


int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Expected 2 inputs: Program File, Instrumentation Function\n";
    return 1;
  }

  std::string prgmfile   = argv[1];
  std::string instrufile = argv[2];

  elfio::File *prgmelf   = new elfio::File;
  elfio::File *instruelf = new elfio::File;

  *prgmelf   = getELF(prgmelf, prgmfile, 50000);    // come up with a smart and/or clever way of 
  *instruelf = getELF(instruelf, instrufile, 50000);// getting blob size for elfio::File obj later

  auto prgmtexsec   = prgmelf->GetSectionByName(".text");
  auto instrutexsec = instruelf->GetSectionByName(".text");

  std::cout << "---------------------------------------" << std::endl;

  uint64_t prgmoff    = prgmtexsec->offset;
  uint64_t prgmsize   = prgmtexsec->size;
  uint64_t instruoff  = instrutexsec->offset;
  uint64_t instrusize = instrutexsec->size;

  std::cout << "Program Offset:\t" << prgmoff    << std::endl
            << "Program Size:\t"   << prgmsize   << std::endl;
  std::cout << "Instru Offset:\t"  << instruoff  << std::endl
            << "Instru Size:\t"    << instrusize << std::endl;


  char *newkernel = new char[prgmsize + instrusize];
  std::memcpy(newkernel, prgmtexsec->Blob(), prgmsize);
  std::memcpy(newkernel + prgmsize, instrutexsec->Blob(), instrusize);


  std::cout << "---------------------------------------" << std::endl;

  auto oldkernelbytes = charToByteArray(prgmtexsec->Blob(), prgmsize);
  auto newkernelbytes = charToByteArray(newkernel, prgmsize + instrusize);
  
  instnode *instrukernel = new instnode;
  
  Assembler a;
  Disassembler d(prgmelf);

  d.Disassemble(oldkernelbytes);

  int sregmax = d.maxNumSReg();
  int vregmax = d.maxNumVReg();
  
  d.Disassemble(newkernelbytes, instrukernel, prgmtexsec->offset);

  std::cout << "---------------------------------------" << std::endl;

  // printInstList(instrukernel);

  std::cout << "---------------------------------------" << std::endl;
  
  offsetInstruRegs(instrukernel, prgmsize + prgmoff, sregmax, vregmax);
  printInstList(instrukernel);

  return 0;
}

elfio::File getELF(elfio::File* elf, std::string fname, size_t blobsize) {
  std::fstream file;
  char *blob;

  file.open(fname, std::ios_base::in | std::ios_base::binary);
  if(file.is_open()){
    blob = new char[blobsize];
    file.read(blob, blobsize);
    file.close();
  }
  return elf->FromMem(blob);
}

instnode* getInstFromPC(instnode *head, uint64_t pc) {
  instnode *curr = head;
  while (curr->next != NULL) {
    if (curr->pc == pc) {
      return curr;
    }
    curr = curr->next;
  }
  return NULL;
}

void printInstList(instnode *head) {
  std::string instStr;
  instnode *curr = head;

  while(curr->next != NULL) {
    instStr = curr->instStr;
    std::cout << "\t" << instStr;
    for (int i = instStr.size(); i < 59; i++) {
      std::cout << " ";
    }

    std::cout << std::hex << "//" << std::setw(12) << std::setfill('0') 
              << curr->pc << ": ";
    std::cout << std::setw(8) << std::hex << convertLE(curr->bytes);
    if (curr->byteSize == 8) {
      std::vector<unsigned char> sfb(curr->bytes.begin() + 4, 
                                 curr->bytes.begin() + 8);
      std::cout << std::setw(8) << std::hex << convertLE(sfb) << std::endl;
    } else {
      std::cout << std::endl;
    }
    curr = curr->next;
  }
}

std::string insertReg(std::string reg, int regval) {
  std::string newval;
  std::stringstream stream;
  bool comma;

  int charlen = 1;
  if(regval > 9) {
    charlen++;
  }
  if(reg.at(reg.length()-1) == ',') {
    comma = true;
  } else {
    comma = false;
  }

  if(reg.find("[") != std::string::npos) {
    int i = reg.find("[");
    int j = reg.find(":");
    std::string reglow  = reg.substr(i+1, j-i-1);
    std::string reghigh = reg.substr(j+1, reg.find("]")-j-1);

    int diff = stoi(reghigh) - stoi(reglow);

    stream << regval;
    stream >> newval;
    reg.replace(i+1, charlen, newval);
    reg.replace(i+1+charlen, 1, ":");
    j = reg.find(":");

    stream.clear();
    regval += diff;
    if(regval > 9) {
      charlen++;
    }

    stream << regval;
    stream >> newval;
    reg.replace(j+1, charlen, newval);
    if(reg.find("]") == std::string::npos) {
      if(reg.at(reg.length()-1) == ',') {
        reg.replace(reg.length()-1, 1, "]");
        reg.append(",");
      } else {
        reg.append("]");
      }
    }
    return reg;
  }

  stream << regval;
  stream >> newval;
  reg.replace(1, charlen, newval);

  if(comma && reg.at(reg.length()-1) != ',') {
    reg.append(",");
  }

  return reg;
}

// void offsetInstruRegs(Assembler a, instnode *head, uint64_t offset, int smax, int vmax) {
void offsetInstruRegs(instnode *head, uint64_t offset, int smax, int vmax) {
  printf("Max S reg:\t%d\nMax V reg:\t%d\n", smax, vmax);
  instnode *curr = head;

  std::vector<std::string> params;
  std::string inststr;
  std::string regstr;
  std::stringstream regstream;
  uint32_t regval;
  
  while (curr->next != NULL) {
    if(curr->pc > offset) {
      inststr = curr->instStr;
      params  = getInstParams(inststr);
      curr->instStr = params.at(0);
      curr->instStr.append(" ");

      for (int i = 1; i < params.size(); i++) {
        if (params.at(i).find("v") != std::string::npos) {
          regstr = extractreg(params.at(i));
          try {
            regval = stoi(regstr);
          } catch(const std::exception& e) {
            continue;
          }
          if (regval < vmax) {
            regval += vmax;
          } 
          params.at(i) = insertReg(params.at(i), regval);
        } else if (params.at(i).find("s") != std::string::npos) {
          regstr = extractreg(params.at(i));
          try {
            regval = stoi(regstr);
          } catch(const std::exception& e) {
            continue;
          }
          if (regval < smax) {
            regval += smax;
          } 
          params.at(i) = insertReg(params.at(i), regval);
        }
        curr->instStr.append(params.at(i));
        curr->instStr.append(" ");
      }
    }
    curr = curr->next;
  }
}

