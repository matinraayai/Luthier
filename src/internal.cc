#include "internal.h"

#include <cstdio>
#include <vector>

extern "C" std::vector<hipModule_t>* __hipRegisterFatBinary(const void* data) {
  printf("Here\n");
  return nullptr;
}