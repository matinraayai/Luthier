#include "src/internal.h"
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <fstream>
#include <elf.h>

#include <cstdio>
#include <vector>

#include "src/util.h"

std::vector<hipModule_t> *call_original_hip_register_fat_binary(
    const void *data);

extern "C" std::vector<hipModule_t> *__hipRegisterFatBinary(const void *data)
{
  // hipError_t err;

  printf("Here in %s\n", __FUNCTION__);

  const __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<const __CudaFatBinaryWrapper *>(data);

  __builtin_dump_struct(fbwrapper, &printf);

  // if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
  //   return call_original_hip_register_fat_binary(data);
  // }

  const __ClangOffloadBundleHeader *header = fbwrapper->binary;

  printf("Printing header\n");
  __builtin_dump_struct(header, &printf);

  std::string magic(reinterpret_cast<const char *>(header),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);

  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC))
  {
    return nullptr;
  }

  int device_count = 0;
  int err = hipGetDeviceCount(&device_count);
  // if (err != hipSuccess) {
  //   panic("cannot get device count.");
  // }

  const __ClangOffloadBundleDesc *desc = &header->desc[0];
  printf("Printing desc\n");
  __builtin_dump_struct(desc, &printf);

  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC))
  {
    return nullptr;
  }

  for (uint64_t i = 0; i < header->numBundles; ++i, desc = desc->next())
  {

    std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};

    printf("%s", &desc->triple[0]);
    // if (triple.compare(AMDGCN_AMDHSA_TRIPLE))
    //   continue;

    std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)],
                       desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};
                       
    printf("Found hip-clang bundle for %s\n", target.c_str());

    // codeobject
    auto *codeobj = reinterpret_cast<const char *>(
        reinterpret_cast<uintptr_t>(header) + desc->offset);

    // inspect elf file
    const char *p = codeobj;

    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)p;
    Elf64_Shdr *shdr = (Elf64_Shdr *)(p + ehdr->e_shoff);

    int shnum = ehdr->e_shnum;

    Elf64_Shdr *sh_strtab = &shdr[ehdr->e_shstrndx];

    const char *const sh_strtab_p = p + sh_strtab->sh_offset;
    // print sections in elf file (code is in .text)
    for (int i = 0; i < shnum; ++i)
    {
      printf("%2d: %4d '%s'\n", i, shdr[i].sh_name, sh_strtab_p + shdr[i].sh_name);
    }
  }

  // print instructions in elf .text section
  // To get the contents of the section, dump .sh_size bytes located at (char *)p + shdr->sh_offset.

  // auto modules = new std::vector<hipModule_t>(device_count);

  for (uint64_t i = 0; i < header->numBundles; i++, desc = desc->next())
  {
    printf("%lu\n", i);
    printf("%s\n", desc->triple);
  }
  //   std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};
  //   if (triple.compare(AMDGCN_AMDHSA_TRIPLE)) {
  //     continue;
  //   }

  //   std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)],
  //                      desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};
  //   printf("Found bundle for %s\n", target.c_str());

  //   for (int device_id = 1; device_id <= device_count; device_id++) {
  //     struct hipDeviceProp_t device_prop;
  //     err = hipGetDeviceProperties(&device_prop, device_id);
  //     if (err != hipSuccess) {
  //       panic("cannot get device properties.");
  //     }

  //     ihipModule_t* module = new ihipModule_t;

  //     auto image_ptr = reinterpret_cast<uintptr_t>(header) + desc->offset;

  //     std::string image{reinterpret_cast<const char*>(image_ptr),
  //     desc->size};
  //     // __hipDumpCodeObject(image);

  //     module->executable = hsa_executable_t();
  //     module->executable.handle = reinterpret_cast<uint64_t>(image_ptr);

  //     modules->at(device_id - 1) = module;
  //   }
  // }

  auto modules = call_original_hip_register_fat_binary(data);

  // printf("Number of modules: %zu\n", modules->size());

  // __builtin_dump_struct(modules,&printf);

  // for (auto module : *modules) {
  //       count +=1;

  //       if (count > 2) {
  //          printf(module->fileName.c_str());
  ///         __builtin_dump_struct(module,&printf);
  //      };
  // printf("%d\n", count);

  //}

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