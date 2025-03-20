//===-- ELFObjectFile.h - Luthier's ELF Object File Wrapper  --------------===//
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
/// \file This file defines a wrapper around LLVM's
/// \c llvm::object::ELFObjectFile that can be used via casting.
//===----------------------------------------------------------------------===//
#include <llvm/Object/ELFObjectFile.h>
#include <luthier/common/ErrorCheck.h>

namespace luthier::object {

template <class ELFT> class ELFObjectFile;

class ELFSectionRef : public llvm::object::ELFSectionRef {
public:
  ELFSectionRef(const llvm::object::ELFSectionRef &B)
      : llvm::object::ELFSectionRef(B) {}

  /// Returns the section's loaded memory offset from its object's
  /// loaded base
  /// \note Function was adapted from LLVM's object utilities
  /// \return on success, the loaded offset of this section with respect to its
  /// object file's load base; an \c llvm::Error on failure
  llvm::Expected<uint64_t> getLoadedOffset() const;

private:
  template <typename ELFT>
  llvm::Expected<uint64_t>
  getLoadedOffset(const luthier::object::ELFObjectFile<ELFT> &ObjFile) const;
};

class elf_section_iterator : public llvm::object::elf_section_iterator {
public:
  elf_section_iterator(const llvm::object::elf_section_iterator &B)
      : llvm::object::elf_section_iterator(B) {}

  const ELFSectionRef *operator->() const {
    return static_cast<const ELFSectionRef *>(
        llvm::object::elf_section_iterator::operator->());
  }

  const ELFSectionRef &operator*() const {
    return static_cast<const ELFSectionRef &>(
        llvm::object::elf_section_iterator::operator*());
  }
};

class ELFSymbolRef : public llvm::object::ELFSymbolRef {
public:
  ELFSymbolRef(const SymbolRef &B) : llvm::object::ELFSymbolRef(B) {};

  /// Returns the symbol's loaded memory offset from its object's
  /// loaded base
  /// \note Function was adapted from LLVM's object utilities
  /// \return on success, the loaded offset of this symbol with respect to its
  /// object file's load base; an \c llvm::Error on failure
  llvm::Expected<uint64_t> getLoadedOffset() const;

private:
  template <typename ELFT>
  llvm::Expected<uint64_t>
  getLoadedOffset(const luthier::object::ELFObjectFile<ELFT> &ObjFile) const;
};

class elf_symbol_iterator : public llvm::object::elf_symbol_iterator {
public:
  elf_symbol_iterator(const llvm::object::elf_symbol_iterator &B)
      : llvm::object::elf_symbol_iterator(B) {}

  const ELFSymbolRef *operator->() const {
    return static_cast<const ELFSymbolRef *>(symbol_iterator::operator->());
  }

  const ELFSymbolRef &operator*() const {
    return static_cast<const ELFSymbolRef &>(symbol_iterator::operator*());
  }
};

/// Luthier's wrapper around \c llvm::object::ELFObjectFile functionality;
/// Use casting on a parsed \c llvm::object::ELFObjectFile to use it:
/// \code
/// auto* LuthierObject =
/// llvm::Cast<luthier::object::ELFObjectFile>(MyLLVMObjFile)
/// \endcode
template <class ELFT>
class ELFObjectFile : public llvm::object::ELFObjectFile<ELFT> {
public:
  static bool classof(llvm::object::Binary *V) {
    return llvm::object::ELFObjectFile<ELFT>::classof(V);
  }

  /// Looks up a symbol by its name in the given \p Elf from its symbol hash
  /// table
  /// \note Function was adapted from LLVM's OpenMP library
  /// \param SymbolName Name of the symbol being looked up
  /// \return an \c llvm::object::ELFSymbolRef if the Symbol was found,
  /// an \c std::nullopt if the symbol was not found, and \c llvm::Error if
  /// any issue was encountered during the process
  llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
  lookupSymbol(llvm::StringRef SymbolName);

private:
  llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
  hashLookup(const typename ELFT::Shdr *Sec, llvm::StringRef SymbolName);

  llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
  getSymbolFromGnuHashTable(llvm::StringRef Name,
                            const typename ELFT::Shdr *Sec,
                            const typename ELFT::GnuHash &HashTab,
                            llvm::ArrayRef<typename ELFT::Sym> SymTab,
                            llvm::StringRef StrTab);

  llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
  getSymbolFromSysVHashTable(llvm::StringRef Name,
                             const typename ELFT::Shdr *Sec,
                             const typename ELFT::Hash &HashTab,
                             llvm::ArrayRef<typename ELFT::Sym> SymTab,
                             llvm::StringRef StrTab);
};

using ELF32LEObjectFile = luthier::object::ELFObjectFile<llvm::object::ELF32LE>;
using ELF64LEObjectFile = luthier::object::ELFObjectFile<llvm::object::ELF64LE>;
using ELF32BEObjectFile = luthier::object::ELFObjectFile<llvm::object::ELF32BE>;
using ELF64BEObjectFile = luthier::object::ELFObjectFile<llvm::object::ELF64BE>;

//===----------------------------------------------------------------------===//
// Implementation Details
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t> ELFSectionRef::getLoadedOffset() const {
  if (auto *ELF64LE =
          llvm::dyn_cast<luthier::object::ELF64LEObjectFile>(getObject()))
    return getLoadedOffset(*ELF64LE);
  else if (auto *ELF64BE =
               llvm::dyn_cast<luthier::object::ELF64BEObjectFile>(getObject()))
    return getLoadedOffset(*ELF64BE);
  else if (auto *ELF32LE =
               llvm::dyn_cast<luthier::object::ELF32LEObjectFile>(getObject()))
    return getLoadedOffset(*ELF32LE);
  else
    return getLoadedOffset(
        *llvm::cast<luthier::object::ELF32BEObjectFile>(getObject()));
}

template <typename ELFT>
llvm::Expected<uint64_t> ELFSectionRef::getLoadedOffset(
    const luthier::object::ELFObjectFile<ELFT> &Obj) const {
  auto PhdrRange = Obj.getELFFile().program_headers();
  LUTHIER_RETURN_ON_ERROR(PhdrRange.takeError());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRange)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<ELFT>(
            Phdr, *Obj.getSection(getRawDataRefImpl()))))
      return getAddress() - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return getAddress();
}

llvm::Expected<uint64_t> ELFSymbolRef::getLoadedOffset() const {
  if (auto *ELF64LE =
          llvm::dyn_cast<luthier::object::ELF64LEObjectFile>(getObject()))
    return getLoadedOffset(*ELF64LE);
  else if (auto *ELF64BE =
               llvm::dyn_cast<luthier::object::ELF64BEObjectFile>(getObject()))
    return getLoadedOffset(*ELF64BE);
  else if (auto *ELF32LE =
               llvm::dyn_cast<luthier::object::ELF32LEObjectFile>(getObject()))
    return getLoadedOffset(*ELF32LE);
  else
    return getLoadedOffset(
        *llvm::cast<luthier::object::ELF32BEObjectFile>(getObject()));
}

template <typename ELFT>
llvm::Expected<uint64_t> ELFSymbolRef::getLoadedOffset(
    const luthier::object::ELFObjectFile<ELFT> &Obj) const {
  auto PhdrRangeOrErr = Obj.getELFFile().program_headers();
  LUTHIER_RETURN_ON_ERROR(PhdrRangeOrErr.takeError());

  llvm::Expected<llvm::object::section_iterator> SymbolSectionOrErr =
      getSection();
  LUTHIER_RETURN_ON_ERROR(SymbolSectionOrErr.takeError());

  auto SymbolAddressOrErr = getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolAddressOrErr.takeError());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRangeOrErr)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<ELFT>(
            Phdr,
            *Obj.getSection(SymbolSectionOrErr.get()->getRawDataRefImpl()))))
      return *SymbolAddressOrErr - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return *SymbolAddressOrErr;
}

template <class ELFT>
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
ELFObjectFile<ELFT>::lookupSymbol(llvm::StringRef SymbolName) {
  for (auto Section = llvm::object::elf_section_iterator(this->section_begin());
       Section != llvm::object::elf_section_iterator(this->section_end());
       ++Section) {
    auto SectionAsSHdr = this->getSection(Section->getRawDataRefImpl());
    if ((SectionAsSHdr->sh_type == llvm::ELF::SHT_HASH) ||
        (SectionAsSHdr->sh_type == llvm::ELF::SHT_GNU_HASH)) {
      return hashLookup(SectionAsSHdr, SymbolName);
    }
  }
}

template <typename ELFT>
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
ELFObjectFile<ELFT>::hashLookup(const typename ELFT::Shdr *Sec,
                                llvm::StringRef SymbolName) {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Sec->sh_type == llvm::ELF::SHT_HASH ||
                              Sec->sh_type == llvm::ELF::SHT_GNU_HASH,
                          "The passed section is not a symbol hash type."));

  auto &ElfFile = this->getELFFile();

  llvm::Expected<typename ELFT::ShdrRange> SectionsOrError = ElfFile.sections();
  LUTHIER_RETURN_ON_ERROR(SectionsOrError.takeError());

  auto SymTabOrErr = getSection<ELFT>(*SectionsOrError, Sec->sh_link);
  LUTHIER_RETURN_ON_ERROR(SymTabOrErr.takeError());

  auto StrTabOrErr =
      ElfFile.getStringTableForSymtab(**SymTabOrErr, *SectionsOrError);
  LUTHIER_RETURN_ON_ERROR(StrTabOrErr.takeError());

  llvm::StringRef StrTab = *StrTabOrErr;
  auto SymsOrErr = ElfFile.symbols(*SymTabOrErr);
  LUTHIER_RETURN_ON_ERROR(SymsOrErr.takeError());

  llvm::ArrayRef<typename ELFT::Sym> SymTab = *SymsOrErr;

  // If this is a GNU hash table we verify its size and search the symbol
  // table using the GNU hash table format.
  if (Sec->sh_type == llvm::ELF::SHT_GNU_HASH) {
    const auto *HashTab = reinterpret_cast<const typename ELFT::GnuHash *>(
        ElfFile.base() + Sec->sh_offset);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        Sec->sh_offset + Sec->sh_size < ElfFile.getBufSize(), ""));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        Sec->sh_size >= sizeof(typename ELFT::GnuHash) &&
            Sec->sh_size >=
                sizeof(typename ELFT::GnuHash) +
                    sizeof(typename ELFT::Word) * HashTab->maskwords +
                    sizeof(typename ELFT::Word) * HashTab->nbuckets +
                    sizeof(typename ELFT::Word) *
                        (SymTab.size() - HashTab->symndx),
        ""));
    return getSymbolFromGnuHashTable<ELFT>(SymbolName, *SymTabOrErr, *HashTab,
                                           SymTab, StrTab);
  }

  // If this is a Sys-V hash table we verify its size and search the symbol
  // table using the Sys-V hash table format.
  if (Sec->sh_type == llvm::ELF::SHT_HASH) {
    const auto *HashTab = reinterpret_cast<const typename ELFT::Hash *>(
        ElfFile.base() + Sec->sh_offset);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        Sec->sh_offset + Sec->sh_size < ElfFile.getBufSize(), ""));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        Sec->sh_size >= sizeof(typename ELFT::Hash) &&
            Sec->sh_size >= sizeof(typename ELFT::Hash) +
                                sizeof(typename ELFT::Word) * HashTab->nbucket +
                                sizeof(typename ELFT::Word) * HashTab->nchain,
        ""));
    return getSymbolFromSysVHashTable<ELFT>(SymbolName, *SymTabOrErr, *HashTab,
                                            SymTab, StrTab);
  }

  return std::nullopt;
}

template <class ELFT>
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
ELFObjectFile<ELFT>::getSymbolFromGnuHashTable(
    llvm::StringRef Name, const typename ELFT::Shdr *Sec,
    const typename ELFT::GnuHash &HashTab,
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
    return std::nullopt;

  // The symbol may or may not be present, check the hash values.
  for (typename ELFT::Word I = Bucket[NameHash % NBucket];
       I >= SymOffset && I < SymTab.size(); I = I + 1) {
    const uint32_t ChainHash = Chain[I - SymOffset];

    if ((NameHash | 0x1) != (ChainHash | 0x1))
      continue;

    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(SymTab[I].st_name < StrTab.size(), ""));

    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return toSymbolRef(Sec, I);

    if (ChainHash & 0x1)
      return std::nullopt;
  }
  return std::nullopt;
}

template <class ELFT>
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
ELFObjectFile<ELFT>::getSymbolFromSysVHashTable(
    llvm::StringRef Name, const typename ELFT::Shdr *Sec,
    const typename ELFT::Hash &HashTab,
    llvm::ArrayRef<typename ELFT::Sym> SymTab, llvm::StringRef StrTab) {
  const uint32_t Hash = llvm::object::hashSysV(Name);
  const typename ELFT::Word NBucket = HashTab.nbucket;
  llvm::ArrayRef<typename ELFT::Word> Bucket = HashTab.buckets();
  llvm::ArrayRef<typename ELFT::Word> Chain = HashTab.chains();
  for (typename ELFT::Word I = Bucket[Hash % NBucket];
       I != llvm::ELF::STN_UNDEF; I = Chain[I]) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(I < SymTab.size(), ""));
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(SymTab[I].st_name < StrTab.size(), ""));

    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return toSymbolRef(Sec, I);
  }
  return std::nullopt;
}

} // namespace luthier::object