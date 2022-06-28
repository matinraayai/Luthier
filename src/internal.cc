#include "nlohmann/json.hpp"
#include "src/elf.h"
#include "src/internal.h"
#include "src/util.h"

#include <dlfcn.h>
#include <elf.h>
#include <hip/hip_runtime.h>
#include <string.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

std::vector<hipModule_t> *
call_original_hip_register_fat_binary(const void *data);
elfio::Note getNoteSection(elfio::File *elf);
char *getNoteSection2(elfio::File *elf);
void editNoteSectionData(elfio::Note &note);
void editTextSectionData(elfio::File *elf);

uint64_t getHeaderSize(__ClangOffloadBundleHeader *header) {
  char *blob = reinterpret_cast<char *>(header);
  auto offset = blob + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  std::cout << "num of bundles: " << header->numBundles << "\n";
  const __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
  uint64_t totalCoSize = 0;
  for (int i = 0; i < header->numBundles; i++, desc = desc->next()) {
    totalCoSize += desc->size;
    uint64_t trippleSize = desc->tripleSize;
    offset += 8 + 8 + 8 + trippleSize;
    std::string triple{desc->triple, desc->tripleSize};
    std::cout << "triple name is " << triple << "\n";
    std::cout << "desc size is " << desc->size << "\n";
    printf("desc struct is stored from address%p\n", (void *)desc);
    printf("bundle triple name is stored from address %p\n", desc->triple);
    printf("bundle %d offset is %p\n", i, (void *)desc->offset);
    printf("address after %d th desc is %p\n", i, (void *)offset);
    // desc = reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
  }
  return 0;
}

extern "C" std::vector<hipModule_t> *__hipRegisterFatBinary(char *data) {
  printf("Here in %s\n", __FUNCTION__);

  char *data_copy = new char[sizeof(__CudaFatBinaryWrapper)];
  std::memcpy(data_copy, data, sizeof(__CudaFatBinaryWrapper));

  __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<__CudaFatBinaryWrapper *>(data);

  __ClangOffloadBundleHeader *header = fbwrapper->binary;
  printf("The address of header is %p\n", (void *)header);

  uint64_t coSize;
  std::string curr_target{"hipv4-amdgcn-amd-amdhsa--gfx906"};
  char *blob = reinterpret_cast<char *>(header);
  uint64_t offset =
      (uint64_t)blob + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  std::cout << "num of bundles: " << header->numBundles << "\n";
  const __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
  elfio::File elfFile;
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
  char *header_copy = new char[endOfHeader - (uint64_t)header];
  std::memcpy(header_copy, header, endOfHeader - (uint64_t)header);
  offset = (uint64_t)header_copy + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  desc = reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
  for (int i = 0; i < header->numBundles; i++, desc = desc->next()) {
    uint64_t trippleSize = desc->tripleSize;
    std::string triple{desc->triple, desc->tripleSize};
    if (triple.compare(curr_target)) {
      continue;
    }
    std::cout << "matching triple name is " << triple << "\n";
    std::cout << "code object size is " << desc->size << "\n";
    char *codeobj = reinterpret_cast<char *>(
        reinterpret_cast<uintptr_t>(header_copy) + desc->offset);
    elfFile = elfFile.FromMem(codeobj); // load elf file from code object
  }

  // elfio::Note noteSec = getNoteSection(&elfFile);

  // editNoteSectionData(noteSec);
  editTextSectionData(&elfFile);
  reinterpret_cast<__CudaFatBinaryWrapper *>(data_copy)->binary =
      reinterpret_cast<__ClangOffloadBundleHeader *>(header_copy);
  // pass new wrapper into original register fat binary func:
  auto modules = call_original_hip_register_fat_binary(data_copy);

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
  // return NULL;
}

std::vector<hipModule_t> *
call_original_hip_register_fat_binary(const void *data) {
  std::vector<hipModule_t> *(*func)(const void *);
  func = (decltype(func))dlsym(RTLD_NEXT, "__hipRegisterFatBinary");

  std::vector<hipModule_t> *ret = func(data);

  return ret;
}

// This function returns the note section of an elf file as an elfio::Note obj
// Uses the same algorithm that getKernelArgumentMetaData in Rhipo uses.
// By passing elfio::Note.desc into nlohmann::json::from_msgpack(), you can
// get the note section as a JSON file. elfio::Note.desc is just a big string.
elfio::Note getNoteSection(elfio::File *elf) {
  printf("Here in %s\n", __FUNCTION__);

  auto note_section = elf->GetSectionByType("SHT_NOTE");
  if (!note_section) {
    panic("note section is not found");
  }

  char *blog = note_section->Blob();
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
void editTextSectionData(elfio::File *elf) {
  auto text_section = elf->GetSectionByName(".text");
  if (!text_section) {
    panic("text section is not found");
  }
  auto text_header = reinterpret_cast<Elf64_Shdr *>(text_section->Blob());
  std::cout << "text section size is " << text_section->size << "\n";
}

// This function changes things in the note section by taking an elfio::Note
// obj and passes the desc param it into a nlohmann::json obj. Then this edits
// the desc param, and passes that back into the elfio::Note obj, which is why
// we pass the note obj by reference.
void editNoteSectionData(elfio::Note &note) {
  printf("Here in %s\n", __FUNCTION__);
  auto json = nlohmann::json::from_msgpack(note.desc);
  std::cout << std::setw(4) << json << '\n';
  // I'm gonna make a change here for now. If/when this function is
  // implemented, changes to the note section might be done elsewhere.
  json["amdhsa.target"] = "gibberish";
  json["amdhsa.kernels"][0][".vgpr_count"] = 256;

  // to_msgpack() returns std::vector<std::uint8_t> which is "great"...
  auto blog = nlohmann::json::to_msgpack(json);
  std::string newDesc(blog.begin(), blog.end());
  note.desc = newDesc;
}