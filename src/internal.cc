
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
#define ALIGN_UP(offset, align) \
  (offset % align == 0 ? offset : (offset + align - offset % align))

std::vector<hipModule_t> *
call_original_hip_register_fat_binary(const void *data);
void editNoteSectionData(elfio::File *elf);
void editTextSectionData(elfio::File *elf);
void editShr(elfio::File *elf);
void printSymbolTable(elfio::File *elf);
std::vector<unsigned char> trampoline(char *codeobj, char *ipath);
elfio::Note getNote(elfio::File *elf);

uint64_t processBundle(char *data)
{
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
  __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);
  uint64_t endOfHeader;
  for (int i = 0; i < header->numBundles; i++, desc = desc->next())
  {
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

    if (i == header->numBundles - 1)
    {
      endOfHeader = (uint64_t)codeobj + coSize;
      printf("address at the end of the last codeobject is %p\n",
             (void *)endOfHeader);
    }
  }
  return endOfHeader;
}

extern "C" std::vector<hipModule_t> *__hipRegisterFatBinary(char *data)
{
  printf("Here in %s\n\n", __FUNCTION__);

  char *data_copy = new char[sizeof(__CudaFatBinaryWrapper)];
  std::memcpy(data_copy, data, sizeof(__CudaFatBinaryWrapper));

  __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<__CudaFatBinaryWrapper *>(data);

  __ClangOffloadBundleHeader *header = fbwrapper->binary;
  uint64_t endOfHeader;
  endOfHeader = processBundle((char *)header);
  endOfHeader = 0x209680;

  std::string curr_target{"hipv4-amdgcn-amd-amdhsa--gfx908"};

  char *header_copy = new char[endOfHeader - (uint64_t)header];
  std::memcpy(header_copy, header, endOfHeader - (uint64_t)header);
  uint64_t offset =
      (uint64_t)header_copy + sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1 + 8;
  __ClangOffloadBundleDesc *desc =
      reinterpret_cast<__ClangOffloadBundleDesc *>(offset);

  char *elfBinary;

  for (int i = 0; i < header->numBundles; i++, desc = desc->next())
  {
    uint64_t trippleSize = desc->tripleSize;
    std::string triple{desc->triple, desc->tripleSize};
    if (triple.compare(curr_target))
    {
      continue;
    }
    std::cout << "matching triple name is " << triple << "\n";
    std::cout << "code object size is " << desc->size << "\n\n";

    elfBinary = reinterpret_cast<char *>(
        reinterpret_cast<uintptr_t>(header_copy) + desc->offset);
    break;
  }
  elfio::File elfFilep, elfFilei;
  elfFilep = elfFilep.FromMem(elfBinary); // load elf file from code object
  // Disassembler d(&elfFilep);
  elfio::Note note = getNote(&elfFilep);
  printf("%s", note.desc);

  // editNoteSectionData(&elfFile);

  int newSize = 0x3680;
  char *newELFBinary = new char[newSize];

  // load instrumentation code
  char *ipath = std::getenv("INSTRU_FUNC");
  char *iBinary = getELF(std::string(ipath));
  elfFilei = elfFilei.FromMem(iBinary);

  std::vector<int> offsets, sizes;

  offset = 0x200; // .note's offset

  // copy .note section
  int copySize = elfFilep.GetSectionByName(".note")->size;
  std::memcpy(newELFBinary + offset, elfFilep.GetSectionByName(".note")->Blob(),
              copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  // generate new .dynsym section
  int newSecSize =
      elfFilep.GetSectionByName(".dynsym")->size +
      elfFilep.GetSectionByName(".dynsym")->entsize * 2; // hex num * dec num
  char *newSecBinary = new char[newSecSize];
  getDynsymSecBinary(newSecBinary, elfFilep.GetSectionByName(".dynsym"),
                     elfFilei.GetSectionByName(".dynsym"));
  // copy new .dynsym section
  int align_req = elfFilep.GetSectionByName(".dynsym")->align;
  if (offset % align_req != 0)
  {
    offset += align_req - offset % align_req;
  }
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // copy .gnu.hash sections
  copySize = elfFilep.GetSectionByName(".gnu.hash")->size;
  align_req = elfFilep.GetSectionByName(".gnu.hash")->align;
  if (offset % align_req != 0)
  {
    offset += align_req - offset % align_req;
  }
  std::memcpy(newSecBinary + offset,
              elfFilep.GetSectionByName(".gnu.hash")->Blob(), copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  // find size for new .hash section
  int numEntry = elfFilep.GetSectionByName(".dynsym")->size /
                 elfFilep.GetSectionByName(".dynsym")->entsize;
  int newHashSize =
      elfFilep.GetSectionByName(".hash")->entsize * (1 + 1 + 2 * numEntry);

  // generate new .dynstr section
  newSecSize = elfFilep.GetSectionByName(".dynstr")->size + strlen("counter") +
               strlen("counter.managed") + 2; //\0 null character problem
  newSecBinary = new char[newSecSize];
  char *newHashBinary = new char[newHashSize];
  getDynstrSecBinary(newSecBinary, elfFilep.GetSectionByName(".dynstr"),
                     elfFilei.GetSectionByName(".dynstr"));
  // generate new .hash section
  getHashSecBinary(newHashBinary, newSecBinary, numEntry);
  // copy new .hash section
  align_req = elfFilep.GetSectionByName(".hash")->align;
  if (offset % align_req != 0)
  {
    offset += align_req - offset % align_req;
  }
  std::memcpy(newELFBinary + offset, newHashBinary, newHashSize);
  offsets.push_back(offset);
  sizes.push_back(newHashSize);
  free(newHashBinary);
  offset += newHashSize; // find the begining of .dynstr section

  // copy new .dynstr section
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // copy .rodata
  align_req = elfFilep.GetSectionByName(".rodata")->align;
  if (offset % align_req != 0)
  {
    offset += align_req - offset % align_req;
  }
  copySize = elfFilep.GetSectionByName(".rodata")->size;
  std::memcpy(newELFBinary + offset,
              elfFilep.GetSectionByName(".rodata")->Blob(), copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  // generate new .text section
  int newTextSize = elfFilep.GetSectionByName(".text")->size +
                    elfFilei.GetSectionByName(".text")->size +
                    32; // trampoline size :32 bytes;
  char *newTextBinary = new char[newTextSize];
  getNewTextBinary(newTextBinary, elfBinary, ipath);

  // copy new .text section
  offset = 0x1000;
  std::memcpy(newELFBinary + offset, newTextBinary, newTextSize);
  offsets.push_back(offset);
  sizes.push_back(newTextSize);
  free(newTextBinary);

  // copy .dynamic and .comment sections
  copySize = elfFilep.GetSectionByName(".dynamic")->size;

  offset = 0x2000;
  std::memcpy(newELFBinary + offset,
              elfFilep.GetSectionByName(".dynamic")->Blob(), copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  copySize = elfFilep.GetSectionByName(".comment")->size;
  std::memcpy(newELFBinary + offset,
              elfFilep.GetSectionByName(".comment")->Blob(), copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  // generate new .symtab section
  newSecSize = elfFilep.GetSectionByName(".symtab")->size +
               elfFilep.GetSectionByName(".symtab")->entsize * 4;
  newSecBinary = new char[newSecSize];
  getSymtabSecBinary(newSecBinary, elfFilep.GetSectionByName(".symtab"),
                     elfFilei.GetSectionByName(".symtab"));

  // copy new .symtab
  align_req = elfFilep.GetSectionByName(".symtab")->align;
  if (offset % align_req != 0)
  {
    offset += align_req - offset % align_req;
  }
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // generate new .shstrtab section
  newSecSize = elfFilep.GetSectionByName(".shstrtab")->size + strlen(".bss") +
               1; //\0 null character problem
  newSecBinary = new char[newSecSize];
  getShstrtabSecBinary(newSecBinary, elfFilep.GetSectionByName(".shstrtab"));

  // copy new .shstrtab
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // generate new .strtab section
  newSecSize = elfFilep.GetSectionByName(".strtab")->size +
               strlen("incr_counter") + strlen("counter") +
               strlen("counter.managed") + strlen("trampoline") +
               4; //\0 null character problem
  newSecBinary = new char[newSecSize];
  getStrtabSecBinary(newSecBinary, elfFilep.GetSectionByName(".strtab"),
                     elfFilei.GetSectionByName(".strtab"));

  // copy new .strtab
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // generate new section header table
  Elf64_Ehdr *Eheader = elfFilep.GetHeader();
  // add .bss section header into section header table
  int newShdrSize = (Eheader->e_shnum + 1) * Eheader->e_shentsize;
  char *newShdrBinary = new char[newShdrSize];
  Elf64_Shdr *shr =
      reinterpret_cast<Elf64_Shdr *>(elfFilep.Blob() + Eheader->e_shoff);
  // modify current section header table: offset, size and addr
  for (int i = 1; i < Eheader->e_shnum; i++)
  {
    shr[i].sh_offset = offsets[i - 1];
    shr[i].sh_size = sizes[i - 1];
  }
  for (int i = 1; i < 9; i++)
  {
    shr[i].sh_addr = shr[i].sh_offset;
  }
  std::memcpy(newShdrBinary, shr, Eheader->e_shnum * Eheader->e_shentsize);
  int endOld = Eheader->e_shnum * Eheader->e_shentsize;
  int bssidx = 9;
  Elf64_Shdr *bssshr = elfFilei.ExtractShr(bssidx);
  std::memcpy(newShdrBinary + endOld, bssshr, Eheader->e_shentsize);

  offset = 0x3000;
  std::memcpy(newELFBinary + offset, newShdrBinary, newShdrSize);
  free(newShdrBinary);

  std::memcpy((void *)(desc->offset), newELFBinary, 0x3680);
  desc->size = 0x3680;
  // reinterpret_cast<__ClangOffloadBundleHeader *>(header_copy)->numBundles = 2;

  // editTextSectionData(&elfFile);

  // editShr(&elfFile);
  // printSymbolTable(&elfFile);

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
call_original_hip_register_fat_binary(const void *data)
{
  std::vector<hipModule_t> *(*func)(const void *);
  func = (decltype(func))dlsym(RTLD_NEXT, "__hipRegisterFatBinary");

  std::vector<hipModule_t> *ret = func(data);

  return ret;
}

void editTextSectionData(elfio::File *elf)
{
  auto tex = elf->GetSectionByName(".text");
  if (!tex)
  {
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
  for (uint64_t i = 0; i < instList.size(); i++)
  {
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

// This function returns the note section of an elf file as an elfio::Note obj
// Uses the same algorithm that getKernelArgumentMetaData in Rhipo uses.
// By passing elfio::Note.desc into nlohmann::json::from_msgpack(), you can
// get the note section as a JSON file. elfio::Note.desc is just a big string.
elfio::Note getNote(elfio::File *elf)
{
  printf("Here in %s\n", __FUNCTION__);

  auto note_section = elf->GetSectionByType("SHT_NOTE");
  if (!note_section)
  {
    panic("note section is not found");
  }

  char *blog = note_section->Blob();
  int offset = 0;
  while (offset < note_section->size)
  {
    auto note = std::make_unique<elfio::Note>(elf, blog + offset);
    offset += note->TotalSize();
    if (note->name.rfind("AMDGPU") == 0)
    {
      printf("Offset %d\n", offset);
      printf("Total Size %d\n", note->TotalSize());
      return elfio::Note(elf, note->Blob());
    }
  }
}

// This function changes things in the note section by taking an elfio::Note
// obj and passes the desc param it into a nlohmann::json obj. Then this edits
// the desc param, and passes that back into the elfio::Note obj, which is why
// we pass the note obj by reference.
void editNoteSectionData(elfio::File *elf)
{
  printf("Here in %s\n", __FUNCTION__);
  auto note_section = elf->GetSectionByType("SHT_NOTE");
  if (!note_section)
  {
    panic("note section is not found");
  }

  char *blog = note_section->Blob();
  int offset = 0, size;
  while (offset < note_section->size)
  {
    auto note = std::make_unique<elfio::Note>(elf, blog + offset);

    if (note->name.rfind("AMDGPU") == 0)
    {
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

void printSymbolTable(elfio::File *elf)
{
  elf->PrintSymbolsForSection(".text");
  elf->PrintSymbolsForSection(".rodata");
}
