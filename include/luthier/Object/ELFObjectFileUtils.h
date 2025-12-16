//===-- ELFObjectUtils.h - ELF Object File Utilities ------------*- C++ -*-===//
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
/// \file This file defines a set of ELF object file utilities.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_ELF_OBJECT_FILE_H
#define LUTHIER_COMMON_ELF_OBJECT_FILE_H
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/LLVMError.h"
#include <llvm/Object/ELFObjectFile.h>

namespace luthier::object {

/// Returns the <tt>Sec</tt>'s load memory offset from the <tt>ELF</tt>'s
/// load base; If the section is not in the program headers (i.e. is not
/// to be loaded) or if the ELF is a relocatable, returns \c std::nullopt
/// \note Function was adapted from LLVM's object dump library
/// \tparam ELFT type of ELF used
/// \param ELF the ELF file being queried
/// \param Sec the ELF's section being queried
/// \return on success, the loaded offset of \p Sec with respect to the
/// ELF's load base if \p Sec is in the program headers; \c std::nullopt if the
/// section is not loadable; an \c llvm::Error on failure
template <typename ELFT>
llvm::Expected<std::optional<uint64_t>>
getLoadOffset(const llvm::object::ELFFile<ELFT> &ELF,
              const typename llvm::object::ELFFile<ELFT>::Elf_Shdr &Sec) {
  /// If the ELF is a relocatable, return as it does not have any program
  /// headers yet
  if (ELF.getHeader().e_type == llvm::ELF::ET_REL)
    return std::nullopt;

  llvm::Expected<typename llvm::object::ELFFile<ELFT>::Elf_Phdr_Range>
      PhdrRangeOrErr = ELF.program_headers();
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_LLVM_ERROR_CHECK(PhdrRangeOrErr.takeError(),
                               "Failed to get the program headers of the ELF"));

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRangeOrErr)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<ELFT>(Phdr, Sec)))
      return Sec.sh_addr - Phdr.p_vaddr + Phdr.p_paddr;

  // Return nullopt if the section was not found in the program headers
  return std::nullopt;
}

/// Returns the <tt>Sec</tt>'s load memory offset from its object file's
/// load base; If the section is not in the program headers (i.e. is not
/// to be loaded), returns \c std::nullopt
/// \note Function was adapted from LLVM's object dump library
/// \param Sec the  section being queried
/// \return on success, the loaded offset of \p Sec with respect to the
/// object file's load base; an \c llvm::Error on failure
llvm::Expected<std::optional<uint64_t>> inline getLoadOffset(
    const llvm::object::ELFSectionRef &Sec) {
  const llvm::object::ELFObjectFileBase *ObjFile = Sec.getObject();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      ObjFile != nullptr, "Object file of section is nullptr."));

  if (const auto *ELF64LE =
          llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(ObjFile))
    return getLoadOffset(ELF64LE->getELFFile(),
                         *ELF64LE->getSection(Sec.getRawDataRefImpl()));
  else if (const auto *ELF64BE =
               llvm::dyn_cast<llvm::object::ELF64BEObjectFile>(ObjFile))
    return getLoadOffset(ELF64BE->getELFFile(),
                         *ELF64BE->getSection(Sec.getRawDataRefImpl()));
  else if (const auto *ELF32LE =
               llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(ObjFile))
    return getLoadOffset(ELF32LE->getELFFile(),
                         *ELF32LE->getSection(Sec.getRawDataRefImpl()));
  else {
    auto *ELF32BE = llvm::cast<llvm::object::ELF32BEObjectFile>(ObjFile);
    return getLoadOffset(ELF32BE->getELFFile(),
                         *ELF32BE->getSection(Sec.getRawDataRefImpl()));
  }
}

/// Returns the <tt>Sym</tt>'s load offset from the <tt>ELF</tt>'s
/// load base; If \p Sym is not loadable, returns \c std::nullopt
/// \note Function was adapted from LLVM's object library
/// \tparam ELFT type of ELF used
/// \param ELF the Object file being queried
/// \param Sym the ELF's symbol being queried
/// \return on success, the load offset of the \c Sym with respect to the
/// ELF's load base if the symbol is loaded, \c std::nullopt if not;
/// an \c llvm::Error on failure
template <typename ELFT>
llvm::Expected<std::optional<uint64_t>>
getLoadOffset(const llvm::object::ELFFile<ELFT> &ELF,
              const typename llvm::object::ELFFile<ELFT>::Elf_Sym &Sym) {
  /// If the ELF is a relocatable, return as it does not have any program
  /// headers yet
  if (ELF.getHeader().e_type == llvm::ELF::ET_REL)
    return std::nullopt;

  llvm::Expected<typename llvm::object::ELFFile<ELFT>::Elf_Phdr_Range>
      PhdrRangeOrErr = ELF.program_headers();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
      PhdrRangeOrErr.takeError(),
      "Failed to obtain the program headers of the ELF"));

  /// Get the symbol's section
  llvm::Expected<const typename ELFT::Shdr *> SymSectionOrErr =
      ELF.getSection(Sym.st_shndx);

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_LLVM_ERROR_CHECK(SymSectionOrErr.takeError(),
                               "Failed to get the ELF section for the symbol"));

  // Search for a PT_LOAD segment containing the symbol's section. Use this
  // segment's p_addr to calculate the symbol's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRangeOrErr)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<ELFT>(Phdr, **SymSectionOrErr)))
      return Sym.st_value - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return std::nullopt;
}

/// Returns the <tt>Sym</tt>'s load offset from its object file's
/// load base; If \p Sym is not loadable, returns \c std::nullopt
/// \note Function was adapted from LLVM's object library
/// \param Sym the ELF's symbol being queried
/// \return on success, the load offset of the \c Sym with respect to the
/// object file's load base if the symbol is loaded, \c std::nullopt if not;
/// an \c llvm::Error on failure
llvm::Expected<std::optional<uint64_t>> inline getLoadOffset(
    const llvm::object::ELFSymbolRef &SymbolRef) {
  const llvm::object::ELFObjectFileBase *ObjFile = SymbolRef.getObject();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      ObjFile != nullptr, "Object file of symbol is nullptr."));

  if (const auto *ELF64LE =
          llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(ObjFile)) {
    auto ElfSymOrErr = ELF64LE->getSymbol(SymbolRef.getRawDataRefImpl());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
        ElfSymOrErr.takeError(),
        "Failed to get the ELF symbol from its Dataref"));
    return getLoadOffset(ELF64LE->getELFFile(), **ElfSymOrErr);
  } else if (const auto *ELF64BE =
                 llvm::dyn_cast<llvm::object::ELF64BEObjectFile>(ObjFile)) {
    auto ElfSymOrErr = ELF64BE->getSymbol(SymbolRef.getRawDataRefImpl());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
        ElfSymOrErr.takeError(),
        "Failed to get the ELF symbol from its Dataref"));
    return getLoadOffset(ELF64BE->getELFFile(), **ElfSymOrErr);
  } else if (const auto *ELF32LE =
                 llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(ObjFile)) {
    auto ElfSymOrErr = ELF32LE->getSymbol(SymbolRef.getRawDataRefImpl());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
        ElfSymOrErr.takeError(),
        "Failed to get the ELF symbol from its Dataref"));
    return getLoadOffset(ELF32LE->getELFFile(), **ElfSymOrErr);
  } else {
    auto *ELF32BE = llvm::cast<llvm::object::ELF32BEObjectFile>(ObjFile);
    auto ElfSymOrErr = ELF32BE->getSymbol(SymbolRef.getRawDataRefImpl());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
        ElfSymOrErr.takeError(),
        "Failed to get the ELF symbol from its Dataref"));
    return getLoadOffset(ELF32BE->getELFFile(), **ElfSymOrErr);
  }
}

/// Returns an \c llvm::ArrayRef to the contents of the symbol inside its
/// object file.
/// \param SymbolRef the symbol being queried; If its object file is
/// relocatable, it should not be have a \c llvm::object::SymbolRef::SF_Common
/// flag
/// \return on success, an \c llvm::ArrayRef encapsulating the contents
/// of the symbol inside its object file; an \c llvm::Error on failure
llvm::Expected<llvm::ArrayRef<uint8_t>> inline getContents(
    const llvm::object::ELFSymbolRef &SymbolRef) {
  const llvm::object::ELFObjectFileBase *ObjFile = SymbolRef.getObject();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      ObjFile != nullptr, "Object file of symbol is nullptr."));

  /// Get everything needed to calculate where the contents of the symbol
  /// is
  bool IsRelocatable = ObjFile->isRelocatableObject();

  llvm::Expected<uint64_t> SymValueOrErr = SymbolRef.getValue();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
      SymValueOrErr.takeError(), "Failed to get the symbol's value"));
  size_t SymbolSize = SymbolRef.getSize();

  llvm::Expected<llvm::object::section_iterator> SectionOrErr =
      SymbolRef.getSection();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
      SectionOrErr.takeError(), "Failed to get the section of the symbol"));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_GENERIC_ERROR_CHECK(*SectionOrErr != ObjFile->section_end(),
                                  "Failed to find the symbol's section"));

  llvm::Expected<llvm::StringRef> SectionContentsOrErr =
      (*SectionOrErr)->getContents();
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_LLVM_ERROR_CHECK(SectionContentsOrErr.takeError(),
                               "Failed to get the contents of the section"));

  uint64_t SymbolOffset;

  if (IsRelocatable) {
    llvm::Expected<uint32_t> SymbolFlagsOrErr = SymbolRef.getFlags();
    LUTHIER_RETURN_ON_ERROR(SymbolFlagsOrErr.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        !(*SymbolFlagsOrErr & llvm::object::SymbolRef::SF_Common),
        "Symbol has common linkage inside the relocatable; Cannot infer its "
        "contents using the st_value field."));

    // in relocatable objects, st_value is the offset from the beginning of the
    // section
    SymbolOffset = *SymValueOrErr;
  } else {
    // In shared objects and executable, symbol value is the load address;
    // We use it with Section's address to calculate the symbol's offset inside
    // the section's contents
    SymbolOffset = *SymValueOrErr - (**SectionOrErr).getAddress();
  }

  auto SymbolStart = reinterpret_cast<const uint8_t *>(
      &((SectionContentsOrErr->data())[SymbolOffset]));

  return llvm::ArrayRef{SymbolStart, SymbolSize};
}

template <class ELFT>
static llvm::Expected<const typename ELFT::Sym *> getSymbolFromGnuHashTable(
    llvm::StringRef Name, const typename ELFT::GnuHash &HashTab,
    llvm::ArrayRef<typename ELFT::Sym> SymTab, llvm::StringRef StrTab) {
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
    return nullptr;

  // The symbol may or may not be present, check the hash values.
  for (typename ELFT::Word I = Bucket[NameHash % NBucket];
       I >= SymOffset && I < SymTab.size(); I = I + 1) {
    const uint32_t ChainHash = Chain[I - SymOffset];

    if ((NameHash | 0x1) != (ChainHash | 0x1))
      continue;

    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        SymTab[I].st_name < StrTab.size(),
        llvm::formatv("symbol [index {0}] has invalid st_name: {1}", I,
                      SymTab[I].st_name)));
    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return &SymTab[I];

    if (ChainHash & 0x1)
      return nullptr;
  }
  return nullptr;
}

template <class ELFT>
static llvm::Expected<const typename ELFT::Sym *> getSymbolFromSysVHashTable(
    llvm::StringRef Name, const typename ELFT::Hash &HashTab,
    llvm::ArrayRef<typename ELFT::Sym> SymTab, llvm::StringRef StrTab) {
  const uint32_t Hash = llvm::object::hashSysV(Name);
  const typename ELFT::Word NBucket = HashTab.nbucket;
  llvm::ArrayRef<typename ELFT::Word> Bucket = HashTab.buckets();
  llvm::ArrayRef<typename ELFT::Word> Chain = HashTab.chains();
  for (typename ELFT::Word I = Bucket[Hash % NBucket];
       I != llvm::ELF::STN_UNDEF; I = Chain[I]) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        I < SymTab.size(),
        llvm::formatv(
            "symbol [index {0}] is greater than the number of symbols: {1}", I,
            SymTab.size())));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        SymTab[I].st_name < StrTab.size(),
        llvm::formatv("symbol [index {0}] has invalid st_name: {1}", I,
                      SymTab[I].st_name)));

    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return &SymTab[I];
  }
  return nullptr;
}

template <class ELFT>
static llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
getHashTableSymbol(const llvm::object::ELFObjectFile<ELFT> &ELFObj,
                   const llvm::object::ELFSectionRef &HashSecRef,
                   llvm::StringRef Name) {
  auto HashSecOrErr = ELFObj.getELFFile().getSection(HashSecRef.getIndex());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
      HashSecOrErr.takeError(),
      "Failed to get the hash section using its index"));

  const typename ELFT::Shdr &HashSec = **HashSecOrErr;

  const llvm::object::ELFFile<ELFT> &Elf = ELFObj.getELFFile();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      HashSec.sh_type == llvm::ELF::SHT_HASH ||
          HashSec.sh_type == llvm::ELF::SHT_GNU_HASH,
      "invalid sh_type for hash table, expected SHT_HASH or SHT_GNU_HASH"));

  llvm::Expected<typename ELFT::ShdrRange> SectionsOrError = Elf.sections();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
      SectionsOrError.takeError(),
      "Failed to get the sections range of the ELF file"));

  auto SymTabOrErr = getSection<ELFT>(*SectionsOrError, HashSec.sh_link);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
      SymTabOrErr.takeError(),
      "Failed to get the symtab section of the ELF file"));

  auto StrTabOrErr =
      Elf.getStringTableForSymtab(**SymTabOrErr, *SectionsOrError);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
      StrTabOrErr.takeError(), "Failed to get the strtab of the ELF file"));

  llvm::StringRef StrTab = *StrTabOrErr;

  auto SymsOrErr = Elf.symbols(*SymTabOrErr);
  if (!SymsOrErr)
    return SymsOrErr.takeError();
  llvm::ArrayRef<typename ELFT::Sym> SymTab = *SymsOrErr;

  // If this is a GNU hash table we verify its size and search the symbol
  // table using the GNU hash table format.
  if (HashSec.sh_type == llvm::ELF::SHT_GNU_HASH) {
    const auto *HashTab = reinterpret_cast<const typename ELFT::GnuHash *>(
        Elf.base() + HashSec.sh_offset);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        HashSec.sh_offset + HashSec.sh_size < Elf.getBufSize(),
        llvm::formatv("section has invalid sh_offset: {0}",
                      HashSec.sh_offset)));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        HashSec.sh_size >= sizeof(typename ELFT::GnuHash) &&
            HashSec.sh_size >=
                sizeof(typename ELFT::GnuHash) +
                    sizeof(typename ELFT::Word) * HashTab->maskwords +
                    sizeof(typename ELFT::Word) * HashTab->nbuckets +
                    sizeof(typename ELFT::Word) *
                        (SymTab.size() - HashTab->symndx),
        llvm::formatv("section has invalid sh_size: {0}", HashSec.sh_size)));

    auto Sym = getSymbolFromGnuHashTable<ELFT>(Name, *HashTab, SymTab, StrTab);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
        Sym.takeError(), "Failed to get symbol using GNU hash table"));
    if (!*Sym)
      return std::nullopt;
    return ELFObj.toSymbolRef(*SymTabOrErr, *Sym - &SymTab[0]);
  }

  // If this is a Sys-V hash table we verify its size and search the symbol
  // table using the Sys-V hash table format.
  if (HashSec.sh_type == llvm::ELF::SHT_HASH) {
    const auto *HashTab = reinterpret_cast<const typename ELFT::Hash *>(
        Elf.base() + HashSec.sh_offset);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        HashSec.sh_offset + HashSec.sh_size < Elf.getBufSize(),
        llvm::formatv("section has invalid sh_offset: {0}",
                      HashSec.sh_offset)));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        HashSec.sh_size >= sizeof(typename ELFT::Hash) &&
            HashSec.sh_size >=
                sizeof(typename ELFT::Hash) +
                    sizeof(typename ELFT::Word) * HashTab->nbucket +
                    sizeof(typename ELFT::Word) * HashTab->nchain,
        llvm::formatv("section has invalid sh_size: {0}", HashSec.sh_size)));

    auto Sym = getSymbolFromSysVHashTable<ELFT>(Name, *HashTab, SymTab, StrTab);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_LLVM_ERROR_CHECK(
        Sym.takeError(), "Failed to get symbol using Sys hash table"));
    if (!*Sym)
      return std::nullopt;
    return ELFObj.toSymbolRef(*SymTabOrErr, *Sym - &SymTab[0]);
  }

  return std::nullopt;
}

/// Looks up a symbol by its name in the given \p ELFObj
/// This function first tries to look up the symbol inside the dynamic symbol
/// table section using the hash section of \p ELFObj if present.
/// If \p ELFObj doesn't have a hash section, or if
/// the hash look up fails to find the symbol in the dynamic symbol table,
/// \p FallbackToIteration can be set so that the function fall backs to using
/// simple iteration over the symbol table after hash lookup
/// \note If this function is being used to look for a symbol inside a
/// relocatable ELF, \p FallbackToSymTabIteration must be set to \c true
/// as relocatables don't have a hash table
/// \note Function was adapted from LLVM's offload library
/// \param ELFObj the ELF object being queried
/// \param Name Name of the symbol being looked up
/// \param FallbackToSymTabIteration If \c true the function will fall back to
/// iteration over the symbol table if hash lookup fails
/// \return an \c llvm::object::ELFSymbolRef if the Symbol was found,
/// an \c std::nullopt if the symbol was not found, and \c llvm::Error if
/// any issue was encountered during the process
template <class ELFT>
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
lookupSymbolByName(const llvm::object::ELFObjectFile<ELFT> &ELFObj,
                   llvm::StringRef Name,
                   bool FallbackToSymTabIteration = true) {
  // First try to look up the symbol via the hash table
  for (llvm::object::ELFSectionRef Sec : ELFObj.sections()) {
    if (Sec.getType() != llvm::ELF::SHT_HASH &&
        Sec.getType() != llvm::ELF::SHT_GNU_HASH)
      continue;
    auto Out = getHashTableSymbol<ELFT>(ELFObj, Sec, Name);
    LUTHIER_RETURN_ON_ERROR(Out.takeError());
    // If we have found the symbol here return it
    if (Out->has_value())
      return Out;
    // If the lookup fails then there's no point looking for any other hash
    // tables; break and fall back to symbol table iteration if necessary
    break;
  }

  if (FallbackToSymTabIteration) {
    for (llvm::object::ELFSymbolRef CurSym : ELFObj.symbols()) {
      llvm::Expected<llvm::StringRef> CurSymNameOrErr = CurSym.getName();
      LUTHIER_RETURN_ON_ERROR(CurSymNameOrErr.takeError());
      if (*CurSymNameOrErr == Name)
        return CurSym;
    }
  }
  return std::nullopt;
}

/// Checks if the data ref of two ELF symbols \p SymA and \p SymB
/// point to the same entry inside \p ELFObjFile
/// \tparam ELFT type of the ELF object file
/// \param ELFObjFile object file of the symbols
/// \param SymA data ref of the first symbol
/// \param SymB data ref of the second symbol
/// \return \c true if the symbols are equal, \c false if not, an \c llvm::Error
/// if an issue was encountered
template <typename ELFT>
llvm::Expected<bool>
areSymbolsEqual(const llvm::object::ELFObjectFile<ELFT> &ELFObjFile,
                llvm::object::DataRefImpl SymA,
                llvm::object::DataRefImpl SymB) {
  const typename ELFT::Sym *ELFSymA;
  const typename ELFT::Sym *ELFSymB;
  LUTHIER_RETURN_ON_ERROR(ELFObjFile.getSymbol(SymA).moveInto(ELFSymA));
  LUTHIER_RETURN_ON_ERROR(ELFObjFile.getSymbol(SymB).moveInto(ELFSymB));
  return ELFSymA->st_name == ELFSymB->st_name &&
         ELFSymA->st_value == ELFSymB->st_value &&
         ELFSymA->st_size == ELFSymB->st_size &&
         ELFSymA->st_info == ELFSymB->st_info &&
         ELFSymA->st_other == ELFSymB->st_other &&
         ELFSymA->st_shndx == ELFSymB->st_shndx;
}

} // namespace luthier::object

#endif