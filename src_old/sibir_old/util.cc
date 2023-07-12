#include "src/util.h"

void panic(const std::string &msg) {
  printf("Panic: %s\n", msg.c_str());
  exit(1);
}
