#include "src/internal.h"
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <fstream>
#include <elf.h>

#include <cstdio>
#include <vector>

#include "nlohmann/json.hpp"
#include "src/util.h"
#include "src/elf.h"

std::vector<hipModule_t> *call_original_hip_register_fat_binary(const void *data);
nlohmann::json getKernelArgumentMetaData(elfio::File* elf);

extern "C" std::vector<hipModule_t> *__hipRegisterFatBinary(char *data)
{
  printf("Here in %s\n", __FUNCTION__);

  __CudaFatBinaryWrapper *fbwrapper = reinterpret_cast<__CudaFatBinaryWrapper *>(data);
  __CudaFatBinaryWrapper newwrapper(*fbwrapper);

  const __ClangOffloadBundleHeader *header = fbwrapper->binary;
  // char header_buffer[50000];

  std::string magic(reinterpret_cast<const char *>(header), sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);

  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC))
  {
    return nullptr;
  }

  int device_count = 0;
  int err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess) {
    panic("cannot get device count.");
  }

  const __ClangOffloadBundleDesc *desc = &header->desc[0];

  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC))
  {
    return nullptr;
  }

  //We want this one, not the "host-x86_64-unknown-linux" bc that one does not have vgpr count
  std::string curr_target{"hipv4-amdgcn-amd-amdhsa--gfx908", sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};
  elfio::File elfFile; //create file object that contains our ELF binary info

  for (uint64_t i = 0; i < header->numBundles; ++i, desc = desc->next())
  {

    std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};

    printf("triple: %s ", &desc->triple[0]);
    if(triple.compare(curr_target)) continue;

    std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)], desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};
                       
    printf("Found hip-clang bundle for %s\n", target.c_str());

    auto *codeobj = reinterpret_cast<char *>(reinterpret_cast<uintptr_t>(header) + desc->offset);
    elfFile = elfFile.FromMem(codeobj);
  }

  printf("\n------------ELF Header------------ \n");
  for (int i = 0; i < 64; i++)
  {
    printf("%02X ", (unsigned int)(unsigned char)elfFile.Blob()[i]);
    if ( (i+1) % 16 == 0) printf("\n");  
  } printf("---------------------------------- \n\n");
  
  /*
  auto textsec = elfFile.GetSectionByName(".text");
  if(!textsec) panic("text section not found");
  char* text = new char[textsec->size];
  std::memcpy(text, textsec->Blob(), textsec->size);

  printf("\n------------Text Section------------ \n");
  for (int i = 0; i < textsec->size; i++)
  {
    printf("%02X ", (unsigned int)(unsigned char)text[i]);
    if ( (i+1) % 4 == 0) printf("\n");  
  } printf("---------------------------------- \n\n");
  */

  char *headbuff = new char[elfFile.size];
  std::memcpy(headbuff, elfFile.Blob(), elfFile.size);

  newwrapper.binary = reinterpret_cast<__ClangOffloadBundleHeader *>(headbuff);

  auto modules = call_original_hip_register_fat_binary(&newwrapper);
  return modules;
}

std::vector<hipModule_t> *call_original_hip_register_fat_binary(
    const void *data)
{
  std::vector<hipModule_t> *(*func)(const void *);
  func = (decltype(func))dlsym(RTLD_NEXT, "__hipRegisterFatBinary");
  std::vector<hipModule_t> *ret = func(data);

  return ret;
}

// Returns the binary data of the .note section of an ELF file in JSON format 
nlohmann::json getKernelArgumentMetaData(elfio::File* elf) {
  printf("Here in %s\n", __FUNCTION__);

  auto note_section = elf->GetSectionByType("SHT_NOTE");
  if (!note_section) {
    panic("note section is not found");
  }

  char* blog = note_section->Blob();
  int offset = 0;
  while (offset < note_section->size) {
    auto note = std::make_unique<elfio::Note>(elf, blog + offset);
    offset += note->TotalSize();

    if (note->name.rfind("AMDGPU") == 0) {
      auto json = nlohmann::json::from_msgpack(note->desc);
      return json;
    }
  }

  panic("note not found");
  return nlohmann::json();
}