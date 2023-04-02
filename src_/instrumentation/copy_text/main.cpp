#include "trampoline.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Expected 2 inputs: Program File, Instrumentation Function\n";
    return 1;
  }
  std::string pFilename = argv[1];
  std::string iFilename = argv[2];

  elfio::File elfFileP;
  char *blob = getELF(pFilename);
  elfFileP = elfFileP.FromMem(blob);

  elfio::File elfFileI;
  char *blob1 = getELF(iFilename);
  elfFileI = elfFileI.FromMem(blob1);

  Assembler a;
  Disassembler d(&elfFileP);

  auto ptexsec = elfFileP.GetSectionByName(".text");
  uint64_t poff = ptexsec->offset;
  uint64_t psize = ptexsec->size;

  auto itexsec = elfFileI.GetSectionByName(".text");
  uint64_t ioff = itexsec->offset;
  uint64_t isize = itexsec->size;

  std::cout << "Program Offset:\t" << poff << std::endl
            << "Program Size:\t" << psize << std::endl;
  std::cout << "Instru Offset:\t" << ioff << std::endl
            << "Instru Size:\t" << isize << std::endl
            << std::endl;

  int sRegMax, vRegMax;
  d.getMaxRegIdx(&elfFileP, &sRegMax, &vRegMax);
  std::cout << "Max S reg:\t" << sRegMax << std::endl
            << "Max V reg:\t" << vRegMax << std::endl
            << std::endl;
  std::cout << "---------------------------------------" << std::endl;

  auto newkernel = newKernel(ptexsec, itexsec);
  std::vector<std::shared_ptr<Inst>> instList = d.GetInsts(newkernel, poff);

  offsetInstruRegs(instList, a, sRegMax, vRegMax);

  makeTrampoline(instList, a, 0);
  d.Disassemble(a.ilstbuf(instList), std::cout);

  return 0;
}
