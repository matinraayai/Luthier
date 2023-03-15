#include "elf.hpp"
#include "internal.h"
#include <fstream>
#include <iostream>

void processBundle(std::string filename) {
  std::streampos size;
  char *data;
  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    data = new char[size];
    file.seekg(0, std::ios::beg);
    file.read(data, size);
    file.close();
  } else {
    printf("unable to open file\n");
    return;
  }
  __ClangOffloadBundleHeader *header =
      reinterpret_cast<__ClangOffloadBundleHeader *>(data);

  std::string magic{data, 24};
  std::cout << magic << "\n";
  printf("The address of header is %p\n", (void *)header);

  uint64_t coSize;
  char *blob = reinterpret_cast<char *>(header);
  uint64_t offset =
      (uint64_t)blob + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  std::cout << "num of bundles: " << header->numBundles << "\n";
  const __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
  uint64_t endOfHeader;
  for (int i = 0; i < header->numBundles; i++, desc = desc->next()) {
    printf("desc struct is stored from address%p\n", (void *)desc);
    uint64_t trippleSize = desc->tripleSize;
    offset += 8 + 8 + 8 + trippleSize;
    printf("address after %d th desc is %p\n", i, (void *)offset);
    std::string triple{desc->triple, desc->tripleSize};
    std::cout << "triple " << triple << " 's offset is " << desc->offset
              << "\n";

    coSize = desc->size;
    std::cout << "code object size is " << coSize << "\n";
  }
}

uint64_t processBundle(char *data) {
  __ClangOffloadBundleHeader *header =
      reinterpret_cast<__ClangOffloadBundleHeader *>(data);

  std::string magic{data, 24};
  printf("Here in %s\n", __FUNCTION__);
  std::cout << magic << "\n\n";
  printf("The address of header is %p\n", (void *)header);

  uint64_t coSize;
  char *blob = reinterpret_cast<char *>(header);
  uint64_t offset =
      (uint64_t)blob + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  std::cout << "num of bundles: " << header->numBundles << "\n";
  const __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
  uint64_t endOfHeader;
  for (int i = 0; i < header->numBundles; i++, desc = desc->next()) {
    printf("desc struct is stored from address%p\n", (void *)desc);
    uint64_t trippleSize = desc->tripleSize;
    offset += 8 + 8 + 8 + trippleSize;
    printf("address after %d th desc is %p\n", i, (void *)offset);
    std::string triple{desc->triple, desc->tripleSize};
    std::cout << "triple " << triple << " 's offset is " << desc->offset
              << "\n";

    coSize = desc->size;
    std::cout << "code object size is " << coSize << "\n";
    char *codeobj = reinterpret_cast<char *>(
        reinterpret_cast<uintptr_t>(header) + desc->offset);

    if (i == header->numBundles - 1) {
      endOfHeader = (uint64_t)codeobj + coSize;
      printf("address at the end of the last codeobject is %p\n",
             (void *)endOfHeader);
    }
  }
  printf("\n");
  return endOfHeader;
}

void registerFatBinary(std::string filename, std::string target) {
  std::streampos size;
  char *data;
  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    data = new char[size];
    file.seekg(0, std::ios::beg);
    file.read(data, size);
    file.close();
  } else {
    printf("unable to open file\n");
    return;
  }

  char *data_copy = new char[sizeof(__CudaFatBinaryWrapper)];
  std::memcpy(data_copy, data, sizeof(__CudaFatBinaryWrapper));

  __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<__CudaFatBinaryWrapper *>(data);

  __ClangOffloadBundleHeader *header = fbwrapper->binary;
  uint64_t endOfHeader;
  endOfHeader = processBundle((char *)fbwrapper->binary);
}

elfio::File extractFromBundle(std::string filename) {
  std::streampos size;
  char *blob;
  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    blob = new char[size];
    file.seekg(0, std::ios::beg);
    file.read(blob, size);
    file.close();
  } else {
    printf("unable to open file\n");
    return elfio::File();
  }
  __ClangOffloadBundleHeader *header =
      reinterpret_cast<__ClangOffloadBundleHeader *>(blob);

  std::string magic{blob, 24};
  std::cout << magic << "\n";
  uint64_t offset =
      (uint64_t)header + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  const __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
  std::string curr_target{"hipv4-amdgcn-amd-amdhsa--gfx906"};
  elfio::File elfFile;
  for (int i = 0; i < header->numBundles; i++, desc = desc->next()) {
    uint64_t trippleSize = desc->tripleSize;
    std::string triple{desc->triple, desc->tripleSize};
    if (triple.compare(curr_target)) {
      continue;
    }
    std::cout << "matching triple name is " << triple << "\n";
    std::cout << "code object size is " << desc->size << "\n";
    char *codeobj = reinterpret_cast<char *>(
        reinterpret_cast<uintptr_t>(header) + desc->offset);
    elfFile = elfFile.FromMem(codeobj); // load elf file from code object
    break;
  }
  return elfFile;
}
