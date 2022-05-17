#include "src/internal.h"
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <fstream>
#include <elf.h>
#include <string.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include "src/util.h"
#include <string.h>



std::vector<hipModule_t> *call_original_hip_register_fat_binary(
    const void *data);

extern "C" std::vector<hipModule_t> *__hipRegisterFatBinary(char *data)
{
 
  printf("Here in %s\n", __FUNCTION__);
  // copy data into completely new location

  char data_copy[50000];
  // need to figure out correct size of buffer
  std::memcpy(data_copy, data, 10000);

  __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<__CudaFatBinaryWrapper *>(data_copy);

  __CudaFatBinaryWrapper fbwrapper2(*fbwrapper);

  __CudaFatBinaryWrapper *pointer = &fbwrapper2;

  //__ClangOffloadBundleHeader *newbinary = new __ClangOffloadBundleHeader;
  __ClangOffloadBundleHeader newbinary(*fbwrapper->binary);

  __ClangOffloadBundleHeader *header = fbwrapper->binary;

  printf("data copy address %p\n", data_copy);
  printf("header address %p\n", header);

  std::string magic(reinterpret_cast<char *>(header),
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

  for (uint64_t i = 0; i < header->numBundles; ++i, desc = desc->next())
  {

    printf("Printing desc\n");
    __builtin_dump_struct(&header->desc[i], &printf);

    std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};

    printf("Desc triptle: %s \n", &desc->triple[i]);

    std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)],
                       desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};

    auto *codeobj = reinterpret_cast<const char *>(
        reinterpret_cast<uintptr_t>(header) + desc->offset);

    
    // inspect elf file
    const char *p = (const char*) codeobj;
   
    //
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)p;
    Elf64_Shdr *shdr = (Elf64_Shdr *)(p + ehdr->e_shoff);
    Elf64_Phdr *phdr = (Elf64_Phdr *)(p + ehdr->e_phnum);

    int shnum = ehdr->e_shnum;

    Elf64_Shdr *sh_strtab = &shdr[ehdr->e_shstrndx];
    
    const char *const sh_strtab_p = p + sh_strtab->sh_offset;
    // print sections in elf file (code is in .text)
    for (int i = 0; i < shnum; ++i)
    {
      
      const char *sec_name = sh_strtab_p + shdr[i].sh_name;

      char str[15];
      int ret;

      strcpy(str, ".note");
      //strcpy(str, ".rodata");

      ret = strcmp(sec_name, str);

      // print ret name

      if (ret == 0) {
        printf("%d\n", ret);
        //printf("%2d: %4d '%s'\n", i, shdr[i].sh_name, sh_strtab_p + shdr[i].sh_name);
        //Get section size of note section
        printf("Size %lu\n", shdr[i].sh_size);
 

      }
    }
  }
   //make a copy of the binary
   char header_buffer[50000];

   //copy binary to new memory location
   std::memcpy(header_buffer, fbwrapper->binary, 15000);

   // small modification to the binary (probably break the program)
   
   //header_buffer[8096+128] = 'h';

   //TODO: Copy the modified note section to header_buffer. If it's exactly the same size, might be OK to ignore ELF offsets. Otherwise, adjust offsets.

   // set the pointer to the copy of the header buffer
   fbwrapper2.binary = reinterpret_cast<__ClangOffloadBundleHeader *>(header_buffer);

  // print instructions in elf .text section
  // To get the contents of the section, dump .sh_size bytes located at (char *)p + shdr->sh_offset.

  // auto modules = new std::vector<hipModule_t>(device_count);
  

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

  auto modules = call_original_hip_register_fat_binary(pointer);

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