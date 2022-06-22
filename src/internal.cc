#include "src/internal.h"
#include "src/util.h"
#include "src/elf.h"
#include "nlohmann/json.hpp"

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <elf.h>
#include <string.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>


std::vector<hipModule_t> *call_original_hip_register_fat_binary(const void *data);
elfio::Note getNoteSection(elfio::File* elf);

elfio::Section *getTextSection(elfio::File* elf);

char * getNoteSection2(elfio::File* elf);
void editNoteSectionData(elfio::Note &note);
void verifyNoteSectionData(std::string note);
std::__cxx11::string editReturnNoteSectionData(elfio::Note &note);


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
  elfio::File elfFile2; // file object that will contain our ELF binary info
  char *codeobj;
  
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
    codeobj = reinterpret_cast<char *>(reinterpret_cast<uintptr_t>(header) + desc->offset);
    printf("Address of codeobj is %p\n", codeobj);
    elfFile = elfFile.FromMem(codeobj); // load elf file from code object

    std::cout << "Printing code obj" << std::endl;

    elfio::Note noteSec5 = getNoteSection(&elfFile);
  

  }


  //get the notesection
 
   elfio::Note noteSec = getNoteSection(&elfFile);

  //modify the notesection
  std::string newNote;
  newNote = editReturnNoteSectionData(noteSec);

  //verify the ntoesection is changed
  verifyNoteSectionData(newNote); 

  elfio::Note noteSec4 = getNoteSection(&elfFile);
 
  char *noteSec2 = reinterpret_cast<char*>(noteSec.Blob());
  //make a copy of the binary
  
  // not sure if size is correct
  //make a copy of the wrapper header:
  char header_buffer[500000];
  
  std::memcpy(header_buffer, fbwrapper->binary, 20000);

  char temp[sizeof(newNote)*sizeof(std::string)];

  std::strcpy(temp, newNote.c_str());

   // small modification to the binary:
  printf("header_buffer\n");
  // this is the offset where the actual ELF starts in the fbwrapper->binary
  int codeobjstart = 4096;
  int extraoffset = 532;
  //+20 bytes to section offset

  printf("size of nnew note %d\n", sizeof(newNote)*sizeof(char));
  // copy edited note section back to object
  for (int i = extraoffset + codeobjstart; i < extraoffset + noteSec.desc_size + codeobjstart; i ++) {
     
     header_buffer[i] = newNote[i-codeobjstart-extraoffset];

   }

  // edit .text instruction - different offset
  extraoffset = 4096-5;
  // copy edited note section back to object
  for (int i = extraoffset + codeobjstart; i < extraoffset + 188 + codeobjstart; i ++) {
     //header_buffer[i] = newNote[i-codeobjstart-extraoffset];
     //0 here is only printed once, but actually 00
     printf("%0X", header_buffer[i]);

   }
   std::cout << "\nPrinting individual location:" << std::endl;

   //change just a single instrucion (I believe C0 is a load)

   header_buffer[extraoffset + codeobjstart +21] = 0xFFFFFFF6;
   //printf("%0X\n", header_buffer[extraoffset + codeobjstart +12]);
   //printf("%0X\n", header_buffer[extraoffset + codeobjstart +14]);

   //insert a whole instruction instruction

   //1. find insert point
   //2. cut first half and append
   //3. append second half?
   //4. update elf offsets (challenging but try without it first)


  elfio::Section* sec = getTextSection(&elfFile);
  char * textsec;
  textsec = sec->Blob();
  for (int k = 0; k< 188; k++) {
    printf("%02X", textsec[k]);
 
  }
  printf("\n");
   // set the pointer to the copy of the header buffer
  newWrapper.binary = reinterpret_cast<__ClangOffloadBundleHeader *>(header_buffer);
 
  //auto origNoteSec = elfFile.GetSectionByType("SHT_NOTE");

  //pass new wrapper into original register fat binary func:
  auto modules = call_original_hip_register_fat_binary(&newWrapper); 
  //auto modules = call_original_hip_register_fat_binary(&newWrapper); 

  /*
  __ClangOffloadBundleHeader *header2 = reinterpret_cast<__ClangOffloadBundleHeader *>(data);
  
  const __ClangOffloadBundleDesc *desc2 = &header2->desc[0];
  // printf("Printing desc\n");
  // __builtin_dump_struct(desc, &printf);


  for (uint64_t i = 0; i < header2->numBundles; ++i, desc2 = desc2->next())
  {

    // printf("Printing desc\n");
    // __builtin_dump_struct(&header->desc[i], &printf);

    std::string triple{&desc2->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};

    if(triple.compare(curr_target)) continue;
    printf("Desc triple: %s \n", &desc2->triple[i]);

    char *codeobj2 = reinterpret_cast<char *>(reinterpret_cast<uintptr_t>(header2) + desc2->offset);

    elfFile2 = elfFile.FromMem(codeobj2); // load elf file from code object
  }

  noteSec = getNoteSection(&elfFile2);

  verifyNoteSectionData(noteSec.desc); 

  */
  
  printf("Number of modules: %ul\n", modules->size());

  // __builtin_dump_struct(modules,&printf);

 // for (auto module : *modules) {
         //count +=1;

        //printf("here");
        //printf(module->fileName.c_str());
        //__builtin_dump_struct(module,&printf);
      
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
  
  char* blob = note_section->Blob();

  printf("Address of note_section is %p\n", blob);
  int offset = 0;
  while (offset < note_section->size) {
    auto note = std::make_unique<elfio::Note>(elf, blob + offset);
    offset += note->TotalSize();
    if (note->name.rfind("AMDGPU") == 0) {
      printf("Offset %d\n", offset);
      printf("Total Size %d\n", note->TotalSize());
      return elfio::Note(elf, note->Blob());
    }
  }
}

// This function returns the .text section of an ELF file as an 
// elfio::Section object.
elfio::Section* getTextSection(elfio::File* elf) {
  auto text = elf->GetSectionByName(".text");
  if(!text) panic("can't find text section");
  // where is .hip_fatbin section?
  
  //section name, type, offset, and size are all public members:
  //printf("Hi! I'm the %s section!!!\n", text->name.c_str());
  //printf("I'm a %s type section!", text->type.c_str());

  //we can get the section header as a char* with this:
  //auto textHdr = text->Blob();

  return text;
}


// This function changes the note section by taking an elfio::Note obj
// and passes the desc param it into a nlohmann::json obj. Then this edits the
// desc param, and passes that back into the elfio::Note obj, which is why we
// pass the note obj by reference.
void editNoteSectionData(elfio::Note &note) {
  printf("Here in %s\n", __FUNCTION__);
  auto json = nlohmann::json::from_msgpack(note.desc);
  
  std::string dump = json.dump();
  printf("%s\n", dump);
  // I'm gonna make a change here for now. If/when this function is implemented,
  // changes to the note section might be done elsewhere.
  //json["amdhsa.target"] = "gibberish";  
  printf("before %d\n", (int)json["amdhsa.kernels"][0][".vgpr_count"]);
  json["amdhsa.kernels"][0][".vgpr_count"] = 8;
  printf("after %d\n", (int)json["amdhsa.kernels"][0][".vgpr_count"]);
  //to_msgpack() returns std::vector<std::uint8_t> 
  auto blog = nlohmann::json::to_msgpack(json);
  std::string newDesc(blog.begin(), blog.end());
  note.desc = newDesc;       
}

std::__cxx11::string editReturnNoteSectionData(elfio::Note &note) {
  printf("Here in %s\n", __FUNCTION__);

  auto json = nlohmann::json::from_msgpack(note.desc);
  printf("size of old desc %d", note.desc_size);
  std::string dump = json.dump();
  printf("%s\n", dump);
  // I'm gonna make a change here for now. If/when this function is implemented,
  // changes to the note section might be done elsewhere.
  //json["amdhsa.target"] = "gibberish";  
  printf("before %d\n", (int)json["amdhsa.kernels"][0][".vgpr_count"]);
  json["amdhsa.kernels"][0][".vgpr_count"] = 5;
  printf("after %d\n", (int)json["amdhsa.kernels"][0][".vgpr_count"]);
  //to_msgpack() returns std::vector<std::uint8_t> 
  auto blog = nlohmann::json::to_msgpack(json);
  std::string newDesc(blog.begin(), blog.end());

  note.desc = newDesc; 

  return newDesc;       
}

void verifyNoteSectionData(std::string note) {
  auto json = nlohmann::json::from_msgpack(note);
  printf("verify %d\n", (int)json["amdhsa.kernels"][0][".vgpr_count"]);


}
