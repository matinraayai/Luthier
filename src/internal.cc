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

extern "C" std::vector<hipModule_t> *__hipRegisterFatBinary(const void *data)
{
  printf("Here in %s\n", __FUNCTION__);

  const __CudaFatBinaryWrapper *fbwrapper = reinterpret_cast<const __CudaFatBinaryWrapper *>(data);

  const __ClangOffloadBundleHeader *header = fbwrapper->binary;


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

    auto *codeobj = reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(header) + desc->offset);

    char *p = const_cast<char*>(codeobj);

    elfFile = elfFile.FromMem(p);  

    // nlohmann::json kernelArgMetaData = getKernelArgumentMetaData(&elfFile);

    // if(kernelArgMetaData.find("amdhsa.kernels") == kernelArgMetaData.end())
    //   panic("Cannot find kernel data");
    // nlohmann::json kernels = kernelArgMetaData["amdhsa.kernels"];
    
    // for (int i = 0; i < kernels.size(); i++)
    // {
    //   if(kernels[i].find(".name") == kernels[i].end() |kernels[i].find(".vgpr_count") == kernels[i].end())
    //     panic("Cannot find kernel name or vgpr_count");
      
    //   printf("For kernel %s", kernels[i].value(".name", "please work").c_str());
    //   printf("vgpr_count = %d\n", kernels[i].value(".vgpr_count", 0));
    // }
  }

  auto modules = call_original_hip_register_fat_binary(fbwrapper);
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

// Shamelessly copied from RHIPO:
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