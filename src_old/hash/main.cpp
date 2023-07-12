#include <iostream>
#include <string>

using namespace std;
unsigned long elf_Hash(const char *name) {
  unsigned long h = 0, g;

  while (*name) {
    h = (h << 4) + *name++;
    if (g = h & 0xf0000000)
      h ^= g >> 24;
    h &= ~g;
  }
  return h;
}

int main() {
  std::string str_array[5] = {"", "counter", "kernel", "kernel.kd",
                              "counter.managed"};
  std::string str_array1[3] = {"", "_Z15vectoradd_floatPfPKfS1_ii",
                               "_Z15vectoradd_floatPfPKfS1_ii.kd"};
  std::string str_array2[5] = {"", "_Z15vectoradd_floatPfPKfS1_ii",
                               "_Z15vectoradd_floatPfPKfS1_ii.kd", "counter",
                               "counter.managed"};
  for (int i = 0; i < 5; i++) {
    unsigned long idx = elf_Hash(str_array2[i].c_str());
    std::cout << str_array2[i] << ":" << idx % 5 << "\n";
  }

  return 0;
}