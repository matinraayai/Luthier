#include <iostream>

#include "elf.h"
#include "parser.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Expected 1 arg: Executable File\n";
    return -1;
  }

  std::string file = std::string(argv[1]);
  // processBundle(file);
  // registerFatBinary(file, "gfx906");
  auto elf = extractFromBundle(file);

  return 0;
}
