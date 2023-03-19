#include "nlohmann/json.hpp"
#include "src/elf.h"
#include "src/internal.h"
#include "src/util.h"

#include "assembler.h"
#include "bitops.h"
#include "disassembler.h"
#include "sectiongenerator.h"
#include "trampoline.h"

#include <dlfcn.h>
#include <elf.h>
#include <hip/hip_runtime.h>
#include <string.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#define ALIGN_UP(offset, align)                                                \
  (offset % align == 0 ? offset : (offset + align - offset % align))

std::vector<hipModule_t> *
call_original_hip_register_fat_binary(const void *data);
void editNoteSectionData(elfio::File *elf);
void editTextSectionData(elfio::File *elf);
void editShr(elfio::File *elf);
void printSymbolTable(elfio::File *elf);
std::vector<unsigned char> trampoline(char *codeobj, char *ipath);

uint64_t processBundle(char *data) {
  printf("Here in %s\n", __FUNCTION__);

  __ClangOffloadBundleHeader *header =
      reinterpret_cast<__ClangOffloadBundleHeader *>(data);

  std::string magic{data, 24};
  std::cout << magic << "\n\n";
  printf("The address of header is %p\n", (void *)header);

  uint64_t coSize;
  char *blob = reinterpret_cast<char *>(header);
  uint64_t offset =
      (uint64_t)blob + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  std::cout << "num of bundles: " << header->numBundles << "\n\n";
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
    std::cout << "code object size is " << coSize << "\n\n";
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
  printf("Here in %s\n\n", __FUNCTION__);

  char *data_copy = new char[sizeof(__CudaFatBinaryWrapper)];
  std::memcpy(data_copy, data, sizeof(__CudaFatBinaryWrapper));

  __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<__CudaFatBinaryWrapper *>(data);

  __ClangOffloadBundleHeader *header = fbwrapper->binary;
  uint64_t endOfHeader;
  endOfHeader = processBundle((char *)header);

  std::string curr_target{"hipv4-amdgcn-amd-amdhsa--gfx906"};

  char *header_copy = new char[endOfHeader - (uint64_t)header];
  std::memcpy(header_copy, header, endOfHeader - (uint64_t)header);
  uint64_t offset =
      (uint64_t)header_copy + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  const __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);

  char *elfBinary;

  for (int i = 0; i < header->numBundles; i++, desc = desc->next()) {
    uint64_t trippleSize = desc->tripleSize;
    std::string triple{desc->triple, desc->tripleSize};
    if (triple.compare(curr_target)) {
      continue;
    }
    std::cout << "matching triple name is " << triple << "\n";
    std::cout << "code object size is " << desc->size << "\n\n";

    elfBinary = reinterpret_cast<char *>(
        reinterpret_cast<uintptr_t>(header_copy) + desc->offset);
  }
  elfio::File elfFilep, elfFilei;
  elfFilep = elfFilep.FromMem(elfBinary); // load elf file from code object
  Disassembler d(&elfFilep);
  // elfio::Note noteSec = getNoteSection();

  // editNoteSectionData(&elfFile);
  int newSize = 0x25f6;
  char *newELFBinary = new char[newSize];

  // load instrumentation code
  char *ipath = std::getenv("INSTRU_FUNC");
  char *iBinary = getELF(std::string(ipath));
  elfFilei = elfFilei.FromMem(iBinary);

  // copy NULL and .note section
  std::memcpy(newELFBinary, elfBinary,
              elfFilep.GetSectionByName(".dynsym")->offset);
  // generate new .dynsym section
  int newSecSize = elfFilep.GetSectionByName(".dynsym")->size +
                   elfFilep.GetSectionByName(".dynsym")->entsize * 2;
  char *newSecBinary = new char[newSecSize];
  getDynsymSecBinary(newSecBinary, elfFilep.GetSectionByName(".dynsym"),
                     elfFilei.GetSectionByName(".dynsym"));

  if (ipath != NULL) {
    auto buf = trampoline(codeobj, ipath);

    d.Disassemble(buf, std::cout);

    reinterpret_cast<__ClangOffloadBundleDesc *>(desc_copy)->size += buf.size();
    newcodeobj = new char[buf.size()];

    std::memcpy(newcodeobj, byteArrayToChar(buf), buf.size());

    reinterpret_cast<__ClangOffloadBundleDesc *>(desc_copy)->offset =
        (uint64_t)newcodeobj - (uint64_t)header_copy;

    std::cout << "New code object offset: "
              << reinterpret_cast<__ClangOffloadBundleDesc *>(desc_copy)->offset
              << std::endl;
    std::cout << "New code object size: "
              << reinterpret_cast<__ClangOffloadBundleDesc *>(desc_copy)->size
              << std::endl;
    std::cout << "matching triple name is "
              << reinterpret_cast<__ClangOffloadBundleDesc *>(desc_copy)->triple
              << "\n";
  }
  // editTextSectionData(&elfFile);

  // editShr(&elfFile);
  // printSymbolTable(&elfFile);

  reinterpret_cast<__ClangOffloadBundleHeader *>(header_copy)->desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(desc_copy);
  reinterpret_cast<__CudaFatBinaryWrapper *>(data_copy)->binary =
      reinterpret_cast<__ClangOffloadBundleHeader *>(header_copy);

  // pass new wrapper into original register fat binary func:
  auto modules = call_original_hip_register_fat_binary(data_copy);

  printf("Number of modules: %zu\n", modules->size());

  // __builtin_dump_struct(modules,&printf);

  // for (auto module : *modules) {
  //       count +=1;

  //       if (count > 2) {
  //          printf(module->fileName.c_str());
  // /         __builtin_dump_struct(module,&printf);
  //      };
  // printf("%d\n", count);

  // }

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
  auto tex = elf->GetSectionByName(".text");
  if (!tex) {
    panic("text section is not found");
  }
  Assembler a;
  Disassembler d(elf);

  d.Disassemble(elf, "vectoradd_hip.exe -- before edit", std::cout);
  std::vector<unsigned char> prgmByteArray =
      charToByteArray(tex->Blob(), tex->size);
  std::vector<std::shared_ptr<Inst>> instList =
      d.GetInsts(prgmByteArray, tex->offset);

  char *newInst = a.Assemble("s_nop");
  for (uint64_t i = 0; i < instList.size(); i++) {
    // for(uint64_t i = 0; i < tex->size; i += 4) {

    if (instList.at(i)->instType.instName == "s_endpgm")
      break;

    char *oldInst = tex->Blob() + (instList.at(i)->PC - tex->offset);
    // char *newInst = byteArrayToChar(instList.at(i)->bytes);

    std::memcpy(oldInst, newInst, 4);
  }

  // char *oldInst = tex->Blob();
  // char *newInst = a.Assemble("s_nop");
  // std::memcpy(oldInst, newInst, 4);
  // std::memcpy(oldInst + 4, newInst, 4);

  d.Disassemble(elf, "vectoradd_hip.exe -- after edit", std::cout);
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

void printSymbolTable(elfio::File *elf) {
  elf->PrintSymbolsForSection(".text");
  elf->PrintSymbolsForSection(".rodata");
}

std::vector<unsigned char> trampoline(char *codeobj, char *ipath) {
  char *iblob = getELF(std::string(ipath));
  elfio::File elfp, elfi;
  elfp = elfp.FromMem(codeobj);
  elfi = elfi.FromMem(iblob);

  auto ptex = elfp.GetSectionByName(".text");
  auto itex = elfi.GetSectionByName(".text");

  if (!ptex) {
    panic("text section is not found for program");
  }
  if (!ptex) {
    panic("text section is not found for instrumentation function");
  }

  Assembler a;
  Disassembler d(&elfp);

  uint64_t poff = ptex->offset;
  uint64_t psize = ptex->size;
  uint64_t ioff = itex->offset;
  uint64_t isize = itex->size;

  std::cout << "---------------------------------------" << std::endl;
  std::cout << "Program Offset:\t" << poff << std::endl
            << "Program Size:\t" << psize << std::endl
            << "Instru Offset:\t" << ioff << std::endl
            << "Instru Size:\t" << isize << std::endl
            << std::endl;

  int sRegMax, vRegMax;
  d.getMaxRegIdx(&elfp, &sRegMax, &vRegMax);
  std::cout << "Max S reg:\t" << sRegMax << std::endl
            << "Max V reg:\t" << vRegMax << std::endl;
  std::cout << "---------------------------------------" << std::endl;

  auto newkernel = newKernel(ptex, itex);
  std::vector<std::shared_ptr<Inst>> instList = d.GetInsts(newkernel, poff);

  offsetInstruRegs(instList, a, sRegMax, vRegMax);

  makeTrampoline(instList, a, 0);

  return a.ilstbuf(instList);
}
