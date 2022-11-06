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
#define ALIGN_UP(offset, align)                                                \
  (offset % align == 0 ? offset : (offset + align - offset % align))

std::vector<hipModule_t> *
call_original_hip_register_fat_binary(const void *data);
void editNoteSectionData(elfio::File *elf);
void editTextSectionData();
void editShr(elfio::File *elf);

uint64_t processBuddle(char *data) {
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
    char *codeobj = reinterpret_cast<char *>(
        reinterpret_cast<uintptr_t>(header) + desc->offset);

    if (i == header->numBundles - 1) {
      endOfHeader = (uint64_t)codeobj + coSize;
      printf("address at the end of the last codeobject is %p\n",
             (void *)endOfHeader);
    }
  }
  return endOfHeader;
}

extern "C" std::vector<hipModule_t> *__hipRegisterFatBinary(char *data) {
  printf("Here in %s\n", __FUNCTION__);

  char *data_copy = new char[sizeof(__CudaFatBinaryWrapper)];
  std::memcpy(data_copy, data, sizeof(__CudaFatBinaryWrapper));

  __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<__CudaFatBinaryWrapper *>(data);

  __ClangOffloadBundleHeader *header = fbwrapper->binary;
  uint64_t endOfHeader;
  endOfHeader = processBuddle((char *)fbwrapper->binary);

  std::string curr_target{"hipv4-amdgcn-amd-amdhsa--gfx906"};

  elfio::File elfFile;

  char *header_copy = new char[endOfHeader - (uint64_t)header];
  std::memcpy(header_copy, header, endOfHeader - (uint64_t)header);
  uint64_t offset =
      (uint64_t)header_copy + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  const __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
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

  // elfio::Note noteSec = getNoteSection();

  editNoteSectionData(&elfFile);
  // editTextSectionData(&elfFile);
  // editShr(&elfFile);
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

void editTextSectionData(elfio::File *elf) {
  auto text_section = elf->GetSectionByName(".text");
  if (!text_section) {
    panic("text section is not found");
  }
  std::cout << "text section size is " << text_section->size << "\n";
  std::cout << "text section offset is " << text_section->offset << "\n";
  char newInst[4];
  newInst[0] = (unsigned char)0xf2;
  newInst[1] = (unsigned char)0x02;
  newInst[2] = (unsigned char)0x00;
  newInst[3] = (unsigned char)0x7e;

  char *oldInst = text_section->Blob() + text_section->size - 16;
  printf("start address of text is %02X\n",
         (unsigned int)(unsigned char)*oldInst);
  std::memcpy(oldInst, newInst, 4);
  printf("start address of text is %02X\n",
         (unsigned int)(unsigned char)*(oldInst + 3));
}

// This function changes things in the note section by taking an elfio::Note
// obj and passes the desc param it into a nlohmann::json obj. Then this edits
// the desc param, and passes that back into the elfio::Note obj, which is why
// we pass the note obj by reference.
void editNoteSectionData(elfio::File *elf) {
  printf("Here in %s\n", __FUNCTION__);
  auto note_section = elf->GetSectionByType("SHT_NOTE");
  if (!note_section) {
    panic("note section is not found");
  }

  char *blog = note_section->Blob();
  int offset = 0, size;
  while (offset < note_section->size) {
    auto note = std::make_unique<elfio::Note>(elf, blog + offset);

    if (note->name.rfind("AMDGPU") == 0) {
      printf("Offset %d\n", offset);
      elfio::Note AMDGPU_note = elfio::Note(elf, note->Blob());
      auto json = nlohmann::json::from_msgpack(AMDGPU_note.desc);
      std::cout << std::setw(4) << json << '\n';
      json["amdhsa.target"] = "gibberish";
      json["amdhsa.kernels"][0][".vgpr_count"] = 256;
      // to_msgpack() returns std::vector<std::uint8_t> which is "great"...
      auto newStr = nlohmann::json::to_msgpack(json);
      auto newStr_size = newStr.size();
      char *newDesc = reinterpret_cast<char *>(newStr.data());
      std::memcpy(blog + offset + 4, &newStr_size, 4);
      std::memcpy(blog + offset + sizeof(Elf64_Nhdr) +
                      ALIGN_UP(AMDGPU_note.name_size, 4),
                  newDesc, newStr.size());
      break;
    }
    offset += sizeof(Elf64_Nhdr) + ALIGN_UP(note->name_size, 4) +
              ALIGN_UP(note->desc_size, 4);
  }
}

// void editShr(elfio::File *elf) {
//   Elf64_Ehdr *header = elf->GetHeader();
//   auto *shdr = reinterpret_cast<Elf64_Shdr *>(elf->Blob() + header->e_shoff);
//   char *shrEInstru = extractShrE();
//   std::memcpy((char *)(shdr + 64 * header->e_shnum), shrEInstru, 64);
// }