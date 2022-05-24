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
char * getNoteSection2(elfio::File* elf);
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
  __CudaFatBinaryWrapper newWrapper(*fbwrapper);

  __CudaFatBinaryWrapper *fbwrapper_copy = &newWrapper;

  __ClangOffloadBundleHeader *header = fbwrapper->binary;
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

  char *noteSec2 = reinterpret_cast<char*>(noteSec.Blob());
  
   //make a copy of the binary
   
   // not sure if size is correct
   std::memcpy(header_buffer, fbwrapper->binary, 15000);

   // small modification to the binary (probably break the program)
  printf("header_bbuffer\n");
  // this is the offset where the actual ELF starts in the fbwrapper->binary
  int codeobjstart = 4096;
  // copy edited note section back to object
  for (int i = 512 + codeobjstart; i < 1610 + codeobjstart; i ++) {

     header_buffer[i] = noteSec2[i-codeobjstart-512];
   }

   //TODO: Copy the modified note section to header_buffer. If it's exactly the same size, might be OK to ignore ELF offsets. Otherwise, adjust offsets.

   // set the pointer to the copy of the header buffer
   newWrapper.binary = reinterpret_cast<__ClangOffloadBundleHeader *>(header_buffer);
 
  auto origNoteSec = elfFile.GetSectionByType("SHT_NOTE");


  //pass new wrapper into original register fat binary func:
  auto modules = call_original_hip_register_fat_binary(fbwrapper_copy); 

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
      printf("Offset %d\n", offset);
      printf("Total Size %d\n", note->TotalSize());
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
  json["amdhsa.kernels"][0][".vgpr_count"] = 10;

  //to_msgpack() returns std::vector<std::uint8_t> which is "great"...
  auto blog = nlohmann::json::to_msgpack(json);
  std::string newDesc(blog.begin(), blog.end());
  note.desc = newDesc;       
}