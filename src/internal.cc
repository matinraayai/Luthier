#include "src/internal.h"
#include "src/util.h"
#include "src/elf.h"
#include "nlohmann/json.hpp"

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <elf.h>
#include <string.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>


std::vector<hipModule_t> *call_original_hip_register_fat_binary(const void *data);
nlohmann::json getKernelArgumentMetaData(elfio::File* elf);

void getNoteSectionData(nlohmann::json noteData, std::string kernelName, std::string valueName);

extern "C" std::vector<hipModule_t> *__hipRegisterFatBinary(char *data) { 
  printf("Here in %s\n", __FUNCTION__);

  // copy char* data into new location of size equal to 
  // length of data times size of character pointer:
  char *data_copy = new char[strlen(data)*sizeof(char*)];
  std::memcpy(data_copy, data, strlen(data)*sizeof(char*));

  // __builtin_dump_struct(fbwrapper, &printf);

  // create two wrappers for the copied data:
  __CudaFatBinaryWrapper *fbwrapper = reinterpret_cast<__CudaFatBinaryWrapper *>(data_copy);
  __CudaFatBinaryWrapper *newWrapper = new __CudaFatBinaryWrapper(*fbwrapper);

  // create the binary header
  __ClangOffloadBundleHeader *header = fbwrapper->binary;

  //make a copy of the binary header:
  char *header_buffer = new char[sizeof(*header)*sizeof(__ClangOffloadBundleHeader*)];
  std::memcpy(header_buffer, header, sizeof(*header)*sizeof(__ClangOffloadBundleHeader*));

  printf("input data address: %p | data copy address: %p\n", data, data_copy);
  printf("header address %p | header buffer address: %p \n", header, header_buffer);

  std::string magic(reinterpret_cast<char *>(header), sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC))
  {
    printf("Invalid magic: %s\n Expected: %s\n", magic.c_str(), CLANG_OFFLOAD_BUNDLER_MAGIC);
    return nullptr;
  }

  int device_count = 0;
  int err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess) {
    panic("cannot get device count.");
  }

  const __ClangOffloadBundleDesc *desc = &header->desc[0];
  // printf("Printing desc\n");
  // __builtin_dump_struct(desc, &printf);

  //We want this one, not the "host-x86_64-unknown-linux" bc that one does not have vgpr count
  std::string curr_target{"hipv4-amdgcn-amd-amdhsa--gfx908", sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};

  for (uint64_t i = 0; i < header->numBundles; ++i, desc = desc->next())
  {

    // printf("Printing desc\n");
    // __builtin_dump_struct(&header->desc[i], &printf);

    std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};

    printf("Desc triple: %s \n", &desc->triple[i]);

    if(triple.compare(curr_target)) continue;

    std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)],
                        desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};


    auto *codeobj = reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(header) + desc->offset);
    
    elfio::File elfFile; // file object that contains our ELF binary info
    elfFile = elfFile.FromMem(const_cast<char*>(codeobj));

    nlohmann::json noteSectionData = getKernelArgumentMetaData(&elfFile);

    // getNoteSectionData(kernelArgMetaData, "amdhsa.kernels", ".vgpr_count");
    // output kernelArgMetaData as JSON file:
    // std::ofstream note_section_json("note_section.json");
    // note_section_json << kernelArgMetaData;

    // Elf64_Ehdr *ehdr = (Elf64_Ehdr *)elfFile.Blob();  //Blob() returns the ELF header as char ptr
    // Elf64_Shdr *shdr = (Elf64_Shdr *)(elfFile.Blob() + ehdr->e_shoff);
    // Elf64_Phdr *phdr = (Elf64_Phdr *)(elfFile.Blob() + ehdr->e_phnum);

    // int shnum = ehdr->e_shnum;
    // Elf64_Shdr *sh_strtab = &shdr[ehdr->e_shstrndx];
    
    // const char *const sh_strtab_p = elfFile.Blob() + sh_strtab->sh_offset;
    // // print sections in elf file (code is in .text)
    // for (int i = 0; i < shnum; ++i)
    // {
    //   const char *sec_name = sh_strtab_p + shdr[i].sh_name;
    //   char str[15];
    //   int ret;
    //   strcpy(str, ".note");
    //   //strcpy(str, ".rodata");

    //   ret = strcmp(sec_name, str);

    //   // print ret name
    //   if (ret == 0) {
    //     printf("This section is the %s section! \n", sec_name);
    //     printf("ret %d\n", ret);
    //     //printf("%2d: %4d '%s'\n", i, shdr[i].sh_name, sh_strtab_p + shdr[i].sh_name);
    //     //Get section size of note section
    //     printf("Size %lu\n", shdr[i].sh_size);
    //   }
    // }
  }

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

  
  
  // small modification to the binary (probably break the program)
  header_buffer[8096+128] = 'h';

  //TODO: Copy the modified note section to header_buffer. If it's exactly the same size, might be OK to ignore ELF offsets. Otherwise, adjust offsets.
  elfio::File newelfFile;
  newelfFile = newelfFile.FromMem(header_buffer);
  // nlohmann::json newnoteSectionData = getKernelArgumentMetaData(&newelfFile);

  // // output kernelArgMetaData as JSON file:
  // std::ofstream new_note_section_json("modified_note_section.json");
  // new_note_section_json << newnoteSectionData;

  
  // set the pointer to the copy of the header buffer
  newWrapper->binary = reinterpret_cast<__ClangOffloadBundleHeader *>(header_buffer);

  //pass new wrapper into original register fat binary func:
  auto modules = call_original_hip_register_fat_binary(newWrapper); 

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

std::vector<hipModule_t> *call_original_hip_register_fat_binary(const void *data) {
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

void getNoteSectionData(nlohmann::json noteData, std::string kernelName, std::string valueName) {
    if(noteData.find(kernelName) == noteData.end())
      panic("Cannot find kernel data");
    
    for (int i = 0; i < noteData[kernelName].size(); i++)
    {
      if(noteData[kernelName][i].find(".name") == noteData[kernelName][i].end() 
        | noteData[kernelName][i].find(valueName) == noteData[kernelName][i].end())
        panic("Cannot find kernel name or vgpr_count");
      
      // for now, just hope that valueName refers to an integer value
      printf("For kernel %s: Parameter %s = %d\n",
              noteData[kernelName][i].value(".name", "kernel name").c_str(),
              valueName.c_str(),
              noteData[kernelName][i].value(valueName, 0));
    }
}