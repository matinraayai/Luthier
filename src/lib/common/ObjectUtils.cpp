//===-- ObjectUtils.cpp - Luthier's Object File Utility  ------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements all operations related to dealing with parsing and
/// processing AMDGPU code objects using LLVM object file and DWARF utilities.
//===----------------------------------------------------------------------===//
#include "common/ObjectUtils.hpp"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include "luthier/llvm/LLVMError.h"
#include <llvm/ADT/StringMap.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-object-utils"

using namespace llvm;

using namespace llvm::object;

namespace luthier {

llvm::Expected<std::unique_ptr<ELFObjectFileBase>>
parseELFObjectFile(llvm::StringRef ObjectFile) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(ObjectFile, "", false);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      Buffer != nullptr,
      "Failed to create a llvm::MemoryBuffer to encapsulate the object file."));
  auto ObjFileOrErr = ObjectFile::createELFObjectFile(*Buffer);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      ObjFileOrErr.get() != nullptr,
      "Failed to create an llvm::object::ELFObjectFileBase."));
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(ObjFileOrErr.takeError()));
  return unique_dyn_cast<ELFObjectFileBase>(std::move(*ObjFileOrErr));
}

llvm::Expected<std::unique_ptr<ELFObjectFileBase>>
parseELFObjectFile(llvm::ArrayRef<uint8_t> ObjectFile) {
  return parseELFObjectFile(toStringRef(ObjectFile));
}

llvm::Expected<
    std::tuple<llvm::Triple, llvm::StringRef, llvm::SubtargetFeatures>>
getELFObjectFileISA(const ELFObjectFileBase &Obj) {
  llvm::Triple TT = Obj.makeTriple();
  std::string CPU = Obj.tryGetCPUName().value_or("").str();
  llvm::Expected<llvm::SubtargetFeatures> FeaturesOrErr = Obj.getFeatures();
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(FeaturesOrErr.takeError()));
  return std::make_tuple(TT, CPU, *FeaturesOrErr);
}

llvm::Expected<bool> isAMDGPUKernelDescriptor(const ELFSymbolRef &Symbol) {
  const ELFObjectFileBase *Obj = Symbol.getObject();
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Obj != nullptr, "Object file of symbol is nullptr."));
  if (!Obj->makeTriple().isAMDGPU())
    return false;
  uint8_t Type = Symbol.getELFType();
  uint64_t Size = Symbol.getSize();
  llvm::Expected<llvm::StringRef> SymNameOrErr = Symbol.getName();
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(SymNameOrErr.takeError()));
  if (Type == llvm::ELF::STT_OBJECT && SymNameOrErr->ends_with(".kd") &&
      Size == 64) {
    return true;
  } else
    return Type == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64;
}

llvm::Expected<ELFSymbolRef> getKernelFunctionForAMDGPUKernelDescriptor(
    const ELFSymbolRef &KernelDescriptorSymbol) {
  const ELFObjectFileBase *Obj = KernelDescriptorSymbol.getObject();
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Obj != nullptr, "Object file of symbol is nullptr."));
  llvm::Expected<bool> IsKdOrErr =
      isAMDGPUKernelDescriptor(KernelDescriptorSymbol);
  LUTHIER_RETURN_ON_ERROR(IsKdOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(*IsKdOrErr, "Symbol is not a kernel descriptor"));
  // Look for the kernel function symbol using the KD's name without the ".kd"
  // suffix
  llvm::Expected<llvm::StringRef> SymbolNameOrErr =
      KernelDescriptorSymbol.getName();
  LUTHIER_RETURN_ON_ERROR(SymbolNameOrErr.takeError());
  auto KDSymbolOrErr = lookupSymbolByName(
      *KernelDescriptorSymbol.getObject(),
      SymbolNameOrErr->substr(0, SymbolNameOrErr->rfind(".kd")));
  LUTHIER_RETURN_ON_ERROR(KDSymbolOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(KDSymbolOrErr->has_value(),
                          "Failed to find the kernel function associated with "
                          "kernel descriptor {0}.",
                          *SymbolNameOrErr));
  return **KDSymbolOrErr;
}

llvm::Expected<bool> isAMDGPUKernelFunction(const ELFSymbolRef &Symbol) {
  const ELFObjectFileBase *Obj = Symbol.getObject();
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Obj != nullptr, "Object file of symbol is nullptr."));
  if (!Obj->makeTriple().isAMDGPU())
    return false;
  uint8_t Type = Symbol.getELFType();
  llvm::Expected<llvm::StringRef> SymNameOrErr = Symbol.getName();
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(SymNameOrErr.takeError()));

  if (Type == llvm::ELF::STT_FUNC) {
    auto KDSymbolOrErr = lookupSymbolByName(*Symbol.getObject(),
                                            std::string(*SymNameOrErr) + ".kd");
    LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(KDSymbolOrErr.takeError()));
    return KDSymbolOrErr->has_value();
  }
  return false;
}

llvm::Expected<ELFSymbolRef> getKernelDescriptorForAMDGPUKernelFunction(
    const ELFSymbolRef &KernelFunctionSymbol) {
  const ELFObjectFileBase *Obj = KernelFunctionSymbol.getObject();
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Obj != nullptr, "Object file of symbol is nullptr."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      KernelFunctionSymbol.getObject()->makeTriple().isAMDGPU(),
      "Passed symbol's object file is not for AMD GPUs."));

  llvm::Expected<llvm::StringRef> SymNameOrErr = KernelFunctionSymbol.getName();
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(SymNameOrErr.takeError()));

  uint8_t Type = KernelFunctionSymbol.getELFType();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(Type == llvm::ELF::STT_FUNC,
                                              "Symbol {0} is not a function.",
                                              *SymNameOrErr));

  auto KDSymbolOrErr = lookupSymbolByName(*KernelFunctionSymbol.getObject(),
                                          std::string(*SymNameOrErr) + ".kd");
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(KDSymbolOrErr.takeError()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      KDSymbolOrErr->has_value(),
      "Failed to find a kernel descriptor associated with function {0}.",
      *SymNameOrErr));
  return **KDSymbolOrErr;
}

llvm::Expected<bool> isAMDGPUDeviceFunction(const ELFSymbolRef &Symbol) {
  const ELFObjectFileBase *ObjectFile = Symbol.getObject();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      ObjectFile != nullptr, "Object file of the symbol is nullptr."));
  if (!ObjectFile->makeTriple().isAMDGPU())
    return false;
  llvm::Expected<bool> IsKernelOrErr = isAMDGPUKernelFunction(Symbol);
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(IsKernelOrErr.takeError()));
  return Symbol.getELFType() == llvm::ELF::STT_FUNC && !*IsKernelOrErr;
}

llvm::Error categorizeAMDGPUSymbols(
    const ELFObjectFileBase &AMDGPUObjectFile,
    llvm::SmallVectorImpl<ELFSymbolRef> *KernelDescriptorSymbols,
    llvm::SmallVectorImpl<ELFSymbolRef> *KernelFunctionSymbols,
    llvm::SmallVectorImpl<ELFSymbolRef> *DeviceFunctionSymbols,
    llvm::SmallVectorImpl<ELFSymbolRef> *VariableSymbols,
    llvm::SmallVectorImpl<ELFSymbolRef> *ExternSymbols,
    llvm::SmallVectorImpl<ELFSymbolRef> *MiscSymbols) {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(AMDGPUObjectFile.makeTriple().isAMDGPU(),
                          "Passed object file is not for AMD GPUs."));
  for (const object::ELFSymbolRef &Symbol : AMDGPUObjectFile.symbols()) {
    llvm::Expected<bool> IsKdOrErr = isAMDGPUKernelDescriptor(Symbol);
    LUTHIER_RETURN_ON_ERROR(IsKdOrErr.takeError());
    if (*IsKdOrErr && KernelDescriptorSymbols) {
      KernelDescriptorSymbols->emplace_back(Symbol);
      continue;
    }
    llvm::Expected<bool> IsKfOrErr = isAMDGPUKernelFunction(Symbol);
    LUTHIER_RETURN_ON_ERROR(IsKfOrErr.takeError());
    if (*IsKfOrErr && KernelFunctionSymbols) {
      KernelFunctionSymbols->emplace_back(Symbol);
      continue;
    }
    llvm::Expected<bool> IsDfOrErr = isAMDGPUDeviceFunction(Symbol);
    LUTHIER_RETURN_ON_ERROR(IsDfOrErr.takeError());
    if (*IsDfOrErr && DeviceFunctionSymbols) {
      DeviceFunctionSymbols->emplace_back(Symbol);
      continue;
    }
    if (isVariable(Symbol) && VariableSymbols) {
      VariableSymbols->emplace_back(Symbol);
    } else if (isExtern(Symbol) && ExternSymbols) {
      ExternSymbols->emplace_back(Symbol);
    } else if (MiscSymbols) {
      MiscSymbols->emplace_back(Symbol);
    }
  }
  return llvm::Error::success();
}

template <class ELFT>
static llvm::Expected<std::optional<ELFSymbolRef>>
lookupSymbolByNameFromGnuHashTable(const ELFObjectFile<ELFT> &Elf,
                                   llvm::StringRef Name,
                                   const typename ELFT::Shdr *Sec,
                                   const typename ELFT::GnuHash &HashTab,
                                   llvm::ArrayRef<typename ELFT::Sym> SymTab,
                                   llvm::StringRef StrTab) {
  const uint32_t NameHash = llvm::object::hashGnu(Name);
  const typename ELFT::Word NBucket = HashTab.nbuckets;
  const typename ELFT::Word SymOffset = HashTab.symndx;
  llvm::ArrayRef<typename ELFT::Off> Filter = HashTab.filter();
  llvm::ArrayRef<typename ELFT::Word> Bucket = HashTab.buckets();
  llvm::ArrayRef<typename ELFT::Word> Chain = HashTab.values(SymTab.size());

  // Check the bloom filter and exit early if the symbol is not present.
  uint64_t ElfClassBits = ELFT::Is64Bits ? 64 : 32;
  typename ELFT::Off Word =
      Filter[(NameHash / ElfClassBits) % HashTab.maskwords];
  uint64_t Mask = (0x1ull << (NameHash % ElfClassBits)) |
                  (0x1ull << ((NameHash >> HashTab.shift2) % ElfClassBits));
  if ((Word & Mask) != Mask)
    return std::nullopt;

  // The symbol may or may not be present, check the hash values.
  for (typename ELFT::Word I = Bucket[NameHash % NBucket];
       I >= SymOffset && I < SymTab.size(); I = I + 1) {
    const uint32_t ChainHash = Chain[I - SymOffset];

    if ((NameHash | 0x1) != (ChainHash | 0x1))
      continue;

    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        SymTab[I].st_name < StrTab.size(), "Invalid symbol name size"));

    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return Elf.toSymbolRef(Sec, I);

    if (ChainHash & 0x1)
      return std::nullopt;
  }
  return std::nullopt;
}

template <class ELFT>
static llvm::Expected<std::optional<ELFSymbolRef>>
lookupSymbolByNameFromSysVHashTable(const ELFObjectFile<ELFT> &Elf,
                                    llvm::StringRef Name,
                                    const typename ELFT::Shdr *Sec,
                                    const typename ELFT::Hash &HashTab,
                                    llvm::ArrayRef<typename ELFT::Sym> SymTab,
                                    llvm::StringRef StrTab) {
  const uint32_t Hash = llvm::object::hashSysV(Name);
  const typename ELFT::Word NBucket = HashTab.nbucket;
  llvm::ArrayRef<typename ELFT::Word> Bucket = HashTab.buckets();
  llvm::ArrayRef<typename ELFT::Word> Chain = HashTab.chains();
  for (typename ELFT::Word I = Bucket[Hash % NBucket];
       I != llvm::ELF::STN_UNDEF; I = Chain[I]) {
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(I < SymTab.size(), "Invalid symbol table index"));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        SymTab[I].st_name < StrTab.size(), "Invalid symbol name size"));

    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return Elf.toSymbolRef(Sec, I);
  }
  return std::nullopt;
}

template <typename ELFT>
static llvm::Expected<std::optional<ELFSymbolRef>>
lookupSymbolByNameViaHashLookup(const ELFObjectFile<ELFT> &Elf,
                                const typename ELFT::Shdr *Sec,
                                llvm::StringRef SymbolName) {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Sec->sh_type == llvm::ELF::SHT_HASH ||
                              Sec->sh_type == llvm::ELF::SHT_GNU_HASH,
                          "The passed section is not a symbol hash type."));

  auto &ElfFile = Elf.getELFFile();

  llvm::Expected<typename ELFT::ShdrRange> SectionsOrError = ElfFile.sections();
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(SectionsOrError.takeError()));

  auto SymTabOrErr = getSection<ELFT>(*SectionsOrError, Sec->sh_link);
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(SymTabOrErr.takeError()));

  auto StrTabOrErr =
      ElfFile.getStringTableForSymtab(**SymTabOrErr, *SectionsOrError);
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(StrTabOrErr.takeError()));

  llvm::StringRef StrTab = *StrTabOrErr;
  auto SymsOrErr = ElfFile.symbols(*SymTabOrErr);
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(SymsOrErr.takeError()));

  llvm::ArrayRef<typename ELFT::Sym> SymTab = *SymsOrErr;

  // If this is a GNU hash table we verify its size and search the symbol
  // table using the GNU hash table format.
  if (Sec->sh_type == llvm::ELF::SHT_GNU_HASH) {
    const auto *HashTab = reinterpret_cast<const typename ELFT::GnuHash *>(
        ElfFile.base() + Sec->sh_offset);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        Sec->sh_offset + Sec->sh_size < ElfFile.getBufSize(),
        "Invalid GNU Hash section size"));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        Sec->sh_size >= sizeof(typename ELFT::GnuHash) &&
            Sec->sh_size >=
                sizeof(typename ELFT::GnuHash) +
                    sizeof(typename ELFT::Word) * HashTab->maskwords +
                    sizeof(typename ELFT::Word) * HashTab->nbuckets +
                    sizeof(typename ELFT::Word) *
                        (SymTab.size() - HashTab->symndx),
        "Invalid GNU Hash section size"));
    return lookupSymbolByNameFromGnuHashTable<ELFT>(
        Elf, SymbolName, *SymTabOrErr, *HashTab, SymTab, StrTab);
  }

  // If this is a Sys-V hash table we verify its size and search the symbol
  // table using the Sys-V hash table format.
  if (Sec->sh_type == llvm::ELF::SHT_HASH) {
    const auto *HashTab = reinterpret_cast<const typename ELFT::Hash *>(
        ElfFile.base() + Sec->sh_offset);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(Sec->sh_offset + Sec->sh_size <
                                                    ElfFile.getBufSize(),
                                                "Invalid hash section size"));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        Sec->sh_size >= sizeof(typename ELFT::Hash) &&
            Sec->sh_size >= sizeof(typename ELFT::Hash) +
                                sizeof(typename ELFT::Word) * HashTab->nbucket +
                                sizeof(typename ELFT::Word) * HashTab->nchain,
        "Invalid Hash section size"));
    return lookupSymbolByNameFromSysVHashTable<ELFT>(
        Elf, SymbolName, *SymTabOrErr, *HashTab, SymTab, StrTab);
  }

  return std::nullopt;
}

template <typename ELT>
static llvm::Expected<std::optional<ELFSymbolRef>>
lookupSymbolByName(const ELFObjectFile<ELT> &ObjectFile,
                   llvm::StringRef SymbolName) {
  for (auto Section = elf_section_iterator(ObjectFile.section_begin());
       Section != elf_section_iterator(ObjectFile.section_end()); ++Section) {
    auto SectionAsSHdr = ObjectFile.getSection(Section->getRawDataRefImpl());
    if ((SectionAsSHdr->sh_type == llvm::ELF::SHT_HASH) ||
        (SectionAsSHdr->sh_type == llvm::ELF::SHT_GNU_HASH)) {
      return lookupSymbolByNameViaHashLookup(ObjectFile, SectionAsSHdr,
                                             SymbolName);
    }
  }
  return LUTHIER_CREATE_ERROR(
      "ELF object does not have a hash table for its symbols.");
}

llvm::Expected<std::optional<ELFSymbolRef>>
lookupSymbolByName(const ELFObjectFileBase &ObjectFile,
                   llvm::StringRef SymbolName) {
  if (auto *ObjectFileLE32 = dyn_cast<ELF32LEObjectFile>(&ObjectFile)) {
    return lookupSymbolByName(*ObjectFileLE32, SymbolName);
  } else if (auto *ObjectFileLE64 = dyn_cast<ELF64LEObjectFile>(&ObjectFile)) {
    return lookupSymbolByName(*ObjectFileLE64, SymbolName);
  } else if (auto *ObjectFileBE32 = dyn_cast<ELF32BEObjectFile>(&ObjectFile)) {
    return lookupSymbolByName(*ObjectFileBE32, SymbolName);
  } else {
    return lookupSymbolByName(*dyn_cast<ELF64BEObjectFile>(&ObjectFile),
                              SymbolName);
  }
}

template <class ELFT>
static llvm::Expected<uint64_t>
getLoadedMemoryOffset(const ELFFile<ELFT> &Obj, const ELFSectionRef &Sec) {
  auto PhdrRangeOrErr = Obj.program_headers();
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(PhdrRangeOrErr.takeError()));

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRangeOrErr)
    if ((Phdr.p_type == ELF::PT_LOAD) &&
        (isSectionInSegment<ELFT>(
            Phdr, *cast<const ELFObjectFile<ELFT>>(Sec.getObject())
                       ->getSection(Sec.getRawDataRefImpl()))))
      return Sec.getAddress() - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return Sec.getAddress();
}

llvm::Expected<uint64_t> getLoadedMemoryOffset(const ELFSectionRef &Section) {
  if (const auto *ELF32LEObj = dyn_cast<ELF32LEObjectFile>(Section.getObject()))
    return getLoadedMemoryOffset(ELF32LEObj->getELFFile(), Section);
  else if (const auto *ELF32BEObj =
               dyn_cast<ELF32BEObjectFile>(Section.getObject()))
    return getLoadedMemoryOffset(ELF32BEObj->getELFFile(), Section);
  else if (const auto *ELF64LEObj =
               dyn_cast<ELF64LEObjectFile>(Section.getObject()))
    return getLoadedMemoryOffset(ELF64LEObj->getELFFile(), Section);
  const auto *ELF64BEObj = cast<ELF64BEObjectFile>(Section.getObject());
  return getLoadedMemoryOffset(ELF64BEObj->getELFFile(), Section);
}

template <class ELFT>
static llvm::Expected<uint64_t>
getLoadedMemoryOffset(const ELFFile<ELFT> &Obj, const ELFSymbolRef &Symbol) {
  auto SymbolSectionOrErr = Symbol.getSection();
  LUTHIER_RETURN_ON_ERROR(SymbolSectionOrErr.takeError());

  auto SymbolAddressOrErr = Symbol.getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolAddressOrErr.takeError());

  auto PhdrRangeOrErr = Obj.program_headers();
  LUTHIER_RETURN_ON_ERROR(LLVM_ERROR_CHECK(PhdrRangeOrErr.takeError()));

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the symbol's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRangeOrErr)
    if ((Phdr.p_type == ELF::PT_LOAD) &&
        (isSectionInSegment<ELFT>(
            Phdr,
            *cast<const ELFObjectFile<ELFT>>(Symbol.getObject())
                 ->getSection(SymbolSectionOrErr.get()->getRawDataRefImpl()))))
      return *SymbolAddressOrErr - Phdr.p_vaddr + Phdr.p_paddr;

  // Return symbol's VMA if it isn't in a PT_LOAD segment.
  return *SymbolAddressOrErr;
}

llvm::Expected<uint64_t> getLoadedMemoryOffset(const ELFSymbolRef &Symbol) {
  if (const auto *ELF32LEObj = dyn_cast<ELF32LEObjectFile>(Symbol.getObject()))
    return getLoadedMemoryOffset(ELF32LEObj->getELFFile(), Symbol);
  else if (const auto *ELF32BEObj =
               dyn_cast<ELF32BEObjectFile>(Symbol.getObject()))
    return getLoadedMemoryOffset(ELF32BEObj->getELFFile(), Symbol);
  else if (const auto *ELF64LEObj =
               dyn_cast<ELF64LEObjectFile>(Symbol.getObject()))
    return getLoadedMemoryOffset(ELF64LEObj->getELFFile(), Symbol);
  const auto *ELF64BEObj = cast<ELF64BEObjectFile>(Symbol.getObject());
  return getLoadedMemoryOffset(ELF64BEObj->getELFFile(), Symbol);
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
getELFSymbolRefContents(const ELFSymbolRef &Sym) {
  // Get the symbol's section
  Expected<section_iterator> SectionOrErr = Sym.getSection();
  LUTHIER_RETURN_ON_ERROR(SectionOrErr.takeError());
  // Get the section's address
  Expected<uint64_t> SectionAddrOrErr = SectionOrErr.get()->getAddress();
  LUTHIER_RETURN_ON_ERROR(SectionAddrOrErr.takeError());
  // Get the section's contents
  Expected<StringRef> SectionContentsOrErr = SectionOrErr.get()->getContents();
  LUTHIER_RETURN_ON_ERROR(SectionContentsOrErr.takeError());
  // Get the symbol's address
  Expected<uint64_t> SymbolAddrOrErr = Sym.getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolAddrOrErr.takeError());
  // Calculate the beginning address of the symbol
  uint64_t SymbolContentStart =
      reinterpret_cast<uint64_t>(SectionContentsOrErr->data()) +
      (*SymbolAddrOrErr - *SectionAddrOrErr);
  // Return the symbol's contents
  return llvm::ArrayRef{reinterpret_cast<uint8_t *>(SymbolContentStart),
                        Sym.getSize()};
}

} // namespace luthier
