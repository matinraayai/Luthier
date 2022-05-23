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
elfio::Note getNoteSection(elfio::File* elf);
void editNoteSectionData(elfio::Note &note);

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

  __ClangOffloadBundleHeader *header = newWrapper->binary;
  __ClangOffloadBundleHeader *modifiedHeader;
  //make a copy of the wrapper header:
  char *header_buffer = new char[sizeof(*header)*sizeof(__ClangOffloadBundleHeader*)];
  std::memcpy(header_buffer, header, sizeof(*header)*sizeof(__ClangOffloadBundleHeader*));

  // printf("input data address: %p | data copy address: %p\n", data, data_copy);
  // printf("header address %p | header buffer address: %p \n", header, header_buffer);

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
  elfio::File elfFile; // file object that will contain our ELF binary info

  for (uint64_t i = 0; i < header->numBundles; ++i, desc = desc->next())
  {

    // printf("Printing desc\n");
    // __builtin_dump_struct(&header->desc[i], &printf);

    std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};

    if(triple.compare(curr_target)) continue;
    printf("Desc triple: %s \n", &desc->triple[i]);

    // std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)],
    //                     desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};


    //create code object:
    char *codeobj = reinterpret_cast<char *>(reinterpret_cast<uintptr_t>(header) + desc->offset);

    elfFile = elfFile.FromMem(codeobj); // load elf file from code object
  }

  elfio::Note noteSec = getNoteSection(&elfFile);
  editNoteSectionData(noteSec); 
  
  //OK, now I have to take this note section, put it back into the elf file, and then get a new header from it...
  auto origNoteSec = elfFile.GetSectionByType("SHT_NOTE");

  //memcopy the modified note desc into the memory space of the original note desc
  // Elf64_Shdr *new_note_hdr = reinterpret_cast<Elf64_Shdr *>(noteSec.Blob());
  // Elf64_Shdr *old_note_hdr = reinterpret_cast<Elf64_Shdr *>(origNoteSec->Blob());

  // printf("%u | %u | %lu\n", new_note_hdr->sh_name, new_note_hdr->sh_type, new_note_hdr->sh_addr);
  // printf("%u | %u | %lu\n", old_note_hdr->sh_name, old_note_hdr->sh_type, old_note_hdr->sh_addr);

  // std::memcpy(old_note_hdr, new_note_hdr, noteSec.TotalSize()); //seg faults

  //turns out, new_note_hdr and old_note_hdr have the same address. That's probs why this seg faults

  //Right now, this doesn't do anything.
  modifiedHeader = reinterpret_cast<__ClangOffloadBundleHeader *>(reinterpret_cast<uintptr_t>(elfFile.Blob()) - desc->offset);


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
  // header_buffer[8096+128] = 'h';

  //TODO: Copy the modified note section to header_buffer. If it's exactly the same size, might be OK to ignore ELF offsets. Otherwise, adjust offsets.


  // set the pointer to the copy of the header buffer
  // newWrapper->binary = reinterpret_cast<__ClangOffloadBundleHeader *>(header_buffer);

  newWrapper->binary = modifiedHeader;

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

// This function returns the note section of an elf file as an elfio::Note obj
// Uses the same algorithm that getKernelArgumentMetaData in Rhipo uses.
// By passing elfio::Note.desc into nlohmann::json::from_msgpack(), you can get
// the note section as a JSON file. elfio::Note.desc is just a big string.
elfio::Note getNoteSection(elfio::File* elf) {
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
      return elfio::Note(elf, note->Blob());
    }
  }
}

// This function changes things in the note section by taking an elfio::Note obj
// and passes the desc param it into a nlohmann::json obj. Then this edits the
// desc param, and passes that back into the elfio::Note obj, which is why we
// pass the note obj by reference.
void editNoteSectionData(elfio::Note &note) {
  printf("Here in %s\n", __FUNCTION__);
  auto json = nlohmann::json::from_msgpack(note.desc);

  // I'm gonna make a change here for now. If/when this function is implemented,
  // changes to the note section might be done elsewhere.
  json["amdhsa.target"] = "gibberish";  
  json["amdhsa.kernels"][0][".vgpr_count"] = 5000000;

  //to_msgpack() returns std::vector<std::uint8_t> which is "great"...
  auto blog = nlohmann::json::to_msgpack(json);
  std::string newDesc(blog.begin(), blog.end());
  note.desc = newDesc;       
}