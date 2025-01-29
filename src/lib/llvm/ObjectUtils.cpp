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
#include "llvm/ObjectUtils.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/AMDGPUMetadata.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/YAMLTraits.h>
#include <ranges>

#include <string>

#include "common/Error.hpp"

#include "luthier/common/ErrorCheck.h"

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-object-utils"

using namespace llvm;

using namespace llvm::object;

using namespace llvm::msgpack;

static llvm::Error
parseVersionMDOptional(MapDocNode &Map, llvm::StringRef Key,
                       std::optional<luthier::hsa::md::Version> &Out) {
  auto VersionMD = Map.find(Key);
  if (VersionMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        VersionMD->second.isArray(),
        "Version metadata was found but it is not an array metadata."));
    auto VersionMDAsArray = VersionMD->second.getArray();
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(VersionMDAsArray.size() == 2, ""));
    auto MajorVersionMD = VersionMDAsArray[0];
    auto MinorVersionMD = VersionMDAsArray[1];
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(MajorVersionMD.isScalar(),
                            "The major number of the metadata is not scalar."));
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(MinorVersionMD.isScalar(),
                            "The minor number of the metadata is not scalar."));
    Out = {MajorVersionMD.getUInt(), MinorVersionMD.getUInt()};
  }
  return llvm::Error::success();
}

llvm::Error parseVersionMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                   luthier::hsa::md::Version &Out) {
  auto VersionMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(VersionMD->second.isArray(), ""));
  auto VersionMDAsArray = VersionMD->second.getArray();
  Out = {VersionMDAsArray[0].getUInt(), VersionMDAsArray[1].getUInt()};
  return llvm::Error::success();
}

llvm::Error parseStringMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                  std::optional<std::string> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD->second.isString(), ""));
    Out = NodeMD->second.getString();
  }
  return llvm::Error::success();
}

llvm::Error parseStringMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                  std::string &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD != Map.end(), ""));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD->second.isString(), ""));
  Out = NodeMD->second.getString();
  return llvm::Error::success();
}

static const llvm::StringMap<llvm::AMDGPU::HSAMD::AccessQualifier>
    AccessQualifierEnumMap = {
        {"read_only", llvm::AMDGPU::HSAMD::AccessQualifier::ReadOnly},
        {"write_only", llvm::AMDGPU::HSAMD::AccessQualifier::WriteOnly},
        {"read_write", llvm::AMDGPU::HSAMD::AccessQualifier::ReadWrite}};

static const llvm::StringMap<unsigned> AMDGPUAddressSpaceEnumMap = {
    {"private", llvm::AMDGPUAS::PRIVATE_ADDRESS},
    {"global", llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {"constant", llvm::AMDGPUAS::CONSTANT_ADDRESS},
    {"local", llvm::AMDGPUAS::LOCAL_ADDRESS},
    {"generic", llvm::AMDGPUAS::FLAT_ADDRESS},
    {"region", llvm::AMDGPUAS::REGION_ADDRESS}};

static const llvm::StringMap<luthier::hsa::md::ValueKind> ValueKindEnumMap = {
    {"by_value", luthier::hsa::md::ValueKind::ByValue},
    {"global_buffer", luthier::hsa::md::ValueKind::GlobalBuffer},
    {"dynamic_shared_pointer",
     luthier::hsa::md::ValueKind::DynamicSharedPointer},
    {"sampler", luthier::hsa::md::ValueKind::Sampler},
    {"image", luthier::hsa::md::ValueKind::Image},
    {"pipe", luthier::hsa::md::ValueKind::Pipe},
    {"queue", luthier::hsa::md::ValueKind::Queue},
    {"hidden_global_offset_x",
     luthier::hsa::md::ValueKind::HiddenGlobalOffsetX},
    {"hidden_global_offset_y",
     luthier::hsa::md::ValueKind::HiddenGlobalOffsetY},
    {"hidden_global_offset_z",
     luthier::hsa::md::ValueKind::HiddenGlobalOffsetZ},
    {"hidden_none", luthier::hsa::md::ValueKind::HiddenNone},
    {"hidden_printf_buffer", luthier::hsa::md::ValueKind::HiddenPrintfBuffer},
    {"hidden_hostcall_buffer",
     luthier::hsa::md::ValueKind::HiddenHostcallBuffer},
    {"hidden_default_queue", luthier::hsa::md::ValueKind::HiddenDefaultQueue},
    {"hidden_completion_action",
     luthier::hsa::md::ValueKind::HiddenCompletionAction},
    {"hidden_multigrid_sync_arg",
     luthier::hsa::md::ValueKind::HiddenMultiGridSyncArg},
    {"hidden_block_count_x", luthier::hsa::md::ValueKind::HiddenBlockCountX},
    {"hidden_block_count_y", luthier::hsa::md::ValueKind::HiddenBlockCountY},
    {"hidden_block_count_z", luthier::hsa::md::ValueKind::HiddenBlockCountZ},
    {"hidden_group_size_x", luthier::hsa::md::ValueKind::HiddenGroupSizeX},
    {"hidden_group_size_y", luthier::hsa::md::ValueKind::HiddenGroupSizeY},
    {"hidden_group_size_z", luthier::hsa::md::ValueKind::HiddenGroupSizeZ},
    {"hidden_remainder_x", luthier::hsa::md::ValueKind::HiddenRemainderX},
    {"hidden_remainder_y", luthier::hsa::md::ValueKind::HiddenRemainderY},
    {"hidden_remainder_z", luthier::hsa::md::ValueKind::HiddenRemainderZ},
    {"hidden_grid_dims", luthier::hsa::md::ValueKind::HiddenGridDims},
    {"hidden_heap_v1", luthier::hsa::md::ValueKind::HiddenHeapV1},
    {"hidden_dynamic_lds_size",
     luthier::hsa::md::ValueKind::HiddenDynamicLDSSize},
    {"hidden_private_base", luthier::hsa::md::ValueKind::HiddenPrivateBase},
    {"hidden_shared_base", luthier::hsa::md::ValueKind::HiddenSharedBase},
    {"hidden_queue_ptr", luthier::hsa::md::ValueKind::HiddenQueuePtr}};

static const llvm::StringMap<luthier::hsa::md::KernelKind> KernelKindEnumMap = {
    {"normal", luthier::hsa::md::KernelKind::Normal},
    {"init", luthier::hsa::md::KernelKind::Init},
    {"fini", luthier::hsa::md::KernelKind::Fini}};

template <typename ET>
llvm::Error parseEnumMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                const llvm::StringMap<ET> &EnumMap,
                                std::optional<ET> &Out) {
  std::optional<std::string> EnumString;
  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(Map, Key, EnumString));
  if (EnumString.has_value()) {
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(EnumMap.contains(*EnumString), ""));
    Out = EnumMap.at(*EnumString);
  }
  return llvm::Error::success();
}

template <typename ET>
llvm::Error parseEnumMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                const llvm::StringMap<ET> &EnumMap, ET &Out) {
  std::string EnumString;
  LUTHIER_RETURN_ON_ERROR(parseStringMDRequired(Map, Key, EnumString));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      EnumMap.contains(EnumString), "Key {0} is not present in Enum Map.", Key));
  Out = EnumMap.at(EnumString);
  return llvm::Error::success();
}

llvm::Error parseDim3MDOptional(MapDocNode &Map, llvm::StringRef Key,
                                std::optional<hsa_dim3_t> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD->second.isArray(), ""));
    auto NodeMDAsArray = NodeMD->second.getArray();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMDAsArray.size() == 3, ""));

    auto XMD = NodeMDAsArray[0];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(XMD.isScalar(), ""));

    auto YMD = NodeMDAsArray[1];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(YMD.isScalar(), ""));

    auto ZMD = NodeMDAsArray[2];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(ZMD.isScalar(), ""));

    Out = {static_cast<uint32_t>(XMD.getUInt()),
           static_cast<uint32_t>(YMD.getUInt()),
           static_cast<uint32_t>(ZMD.getUInt())};
  }
  return llvm::Error::success();
}

llvm::Error parseDim3MDRequired(MapDocNode &Map, llvm::StringRef Key,
                                hsa_dim3_t &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD != Map.end(), ""));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD->second.isArray(), ""));
  auto NodeMDAsArray = NodeMD->second.getArray();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMDAsArray.size() == 3, ""));

  auto XMD = NodeMDAsArray[0];
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(XMD.isScalar(), ""));

  auto YMD = NodeMDAsArray[1];
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(YMD.isScalar(), ""));

  auto ZMD = NodeMDAsArray[2];
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(ZMD.isScalar(), ""));

  Out = {static_cast<uint32_t>(XMD.getUInt()),
         static_cast<uint32_t>(YMD.getUInt()),
         static_cast<uint32_t>(ZMD.getUInt())};
  return llvm::Error::success();
}

template <typename T>
llvm::Error parseUIntMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                std::optional<T> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD->second.isScalar(), ""));
    Out = static_cast<T>(NodeMD->second.getUInt());
  }
  return llvm::Error::success();
}

template <typename T>
llvm::Error parseUIntMDRequired(MapDocNode &Map, llvm::StringRef Key, T &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD != Map.end(), ""));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD->second.isScalar(), ""));
  Out = static_cast<T>(NodeMD->second.getUInt());
  return llvm::Error::success();
}

llvm::Error parseBoolMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                bool &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD->second.isScalar(), ""));
    Out = NodeMD->second.getBool();
  }
  return llvm::Error::success();
}

llvm::Error parseBoolMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                bool &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD != Map.end(), ""));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(NodeMD->second.isScalar(), ""));
  Out = NodeMD->second.getBool();
  return llvm::Error::success();
}

template <class ELFT>
static llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
getSymbolFromGnuHashTable(const llvm::object::ELFObjectFile<ELFT> &Elf,
                          llvm::StringRef Name, const typename ELFT::Shdr *Sec,
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

    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(SymTab[I].st_name < StrTab.size(), ""));

    if (StrTab.drop_front(SymTab[I].st_name).data() == Name)
      return Elf.toSymbolRef(Sec, I);

    if (ChainHash & 0x1)
      return std::nullopt;
  }
  return std::nullopt;
}

template <class ELFT>
static llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
getSymbolFromSysVHashTable(const llvm::object::ELFObjectFile<ELFT> &Elf,
                           llvm::StringRef Name, const typename ELFT::Shdr *Sec,
                           const typename ELFT::Hash &HashTab,
                           llvm::ArrayRef<typename ELFT::Sym> SymTab,
                           llvm::StringRef StrTab) {
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
      return Elf.toSymbolRef(Sec, I);
  }
  return std::nullopt;
}

template <typename ELFT>
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
hashLookup(const llvm::object::ELFObjectFile<ELFT> &Elf,
           const typename ELFT::Shdr *Sec, llvm::StringRef SymbolName) {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Sec->sh_type == llvm::ELF::SHT_HASH ||
                              Sec->sh_type == llvm::ELF::SHT_GNU_HASH,
                          ""));

  auto &ElfFile = Elf.getELFFile();

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
    return getSymbolFromGnuHashTable<ELFT>(Elf, SymbolName, *SymTabOrErr,
                                           *HashTab, SymTab, StrTab);
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
    return getSymbolFromSysVHashTable<ELFT>(Elf, SymbolName, *SymTabOrErr,
                                            *HashTab, SymTab, StrTab);
  }

  return std::nullopt;
}

namespace luthier {

Expected<std::unique_ptr<ELF64LEObjectFile>>
parseAMDGCNObjectFile(llvm::StringRef ELF) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(ELF, "", false);
  Expected<std::unique_ptr<ObjectFile>> ObjectFile =
      ObjectFile::createELFObjectFile(*Buffer);
  LUTHIER_RETURN_ON_ERROR(ObjectFile.takeError());
  return unique_dyn_cast<ELF64LEObjectFile>(std::move(*ObjectFile));
}

Expected<std::unique_ptr<ELF64LEObjectFile>>
parseAMDGCNObjectFile(llvm::ArrayRef<uint8_t> ELF) {
  return parseAMDGCNObjectFile(toStringRef(ELF));
}

llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
lookupSymbolByName(const luthier::AMDGCNObjectFile &ObjectFile,
                   llvm::StringRef SymbolName) {
  for (auto Section =
           llvm::object::elf_section_iterator(ObjectFile.section_begin());
       Section != llvm::object::elf_section_iterator(ObjectFile.section_end());
       ++Section) {
    auto SectionAsSHdr = ObjectFile.getSection(Section->getRawDataRefImpl());
    if ((SectionAsSHdr->sh_type == llvm::ELF::SHT_HASH) ||
        (SectionAsSHdr->sh_type == llvm::ELF::SHT_GNU_HASH)) {
      return hashLookup(ObjectFile, SectionAsSHdr, SymbolName);
    }
  }
  return LUTHIER_CREATE_ERROR(
      "ELF object does not have a hash table for its symbols.");
}

/// Returns the <tt>Sec</tt>'s loaded memory offset from the <tt>ELF</tt>'s
/// loaded base
/// \note Function was adapted from LLVM's object library
/// \tparam ELFT type of ELF used
/// \param ELF the Object file being queried
/// \param Sec the ELF's section being queried
/// \return on success, the loaded offset of \p Sec with respect to the
/// ELF's load base; an \c llvm::Error on failure
llvm::Expected<uint64_t>
getLoadedMemoryOffset(const luthier::AMDGCNELFFile &ELF,
                      const llvm::object::ELFSectionRef &Sec) {
  auto PhdrRange = ELF.program_headers();
  LUTHIER_RETURN_ON_ERROR(PhdrRange.takeError());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const auto &Phdr : *PhdrRange)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<llvm::object::ELF64LE>(
            Phdr,
            *llvm::cast<
                 const llvm::object::ELFObjectFile<llvm::object::ELF64LE>>(
                 Sec.getObject())
                 ->getSection(Sec.getRawDataRefImpl()))))
      return Sec.getAddress() - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return Sec.getAddress();
}

llvm::Error parseArgMD(llvm::msgpack::MapDocNode &KernelMetaNode,
                       hsa::md::Kernel::Arg::Metadata &Out) {
  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::Name, Out.Name));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::TypeName,
      Out.TypeName));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::Size, Out.Size));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::Offset, Out.Offset));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDRequired(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::ValueKind,
      ValueKindEnumMap, Out.ValueKind));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::PointeeAlign,
      Out.PointeeAlign));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::AddressSpace,
      AMDGPUAddressSpaceEnumMap, Out.AddressSpace));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::AccQual,
      AccessQualifierEnumMap, Out.AccQual));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::ActualAccQual,
      AccessQualifierEnumMap, Out.ActualAccQual));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::IsConst,
      Out.IsConst));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::IsRestrict,
      Out.IsRestrict));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::IsVolatile,
      Out.IsVolatile));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Arg::Key::IsPipe, Out.IsPipe));

  return llvm::Error::success();
};

llvm::Error parseKernelMD(llvm::msgpack::MapDocNode &KernelMetaNode,
                          hsa::md::Kernel::Metadata &Out) {

  LUTHIER_RETURN_ON_ERROR(parseStringMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::Name, Out.Name));

  LUTHIER_RETURN_ON_ERROR(parseStringMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::Symbol, Out.Symbol));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, hsa::md::Kernel::Key::Language, Out.Language));

  LUTHIER_RETURN_ON_ERROR(parseVersionMDOptional(
      KernelMetaNode, hsa::md::Kernel::Key::LanguageVersion,
      Out.LanguageVersion));

  auto ArgsMD = KernelMetaNode.find(hsa::md::Kernel::Key::Args);
  if (ArgsMD != KernelMetaNode.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(ArgsMD->second.isArray(), ""));
    Out.Args.emplace();
    for (auto &ArgMD : ArgsMD->second.getArray()) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(ArgMD.isMap(), ""));
      Out.Args->emplace_back();
      LUTHIER_RETURN_ON_ERROR(parseArgMD(ArgMD.getMap(), Out.Args->back()));
    }
  }

  LUTHIER_RETURN_ON_ERROR(parseDim3MDOptional(
      KernelMetaNode, hsa::md::Kernel::Key::ReqdWorkGroupSize,
      Out.ReqdWorkGroupSize));

  LUTHIER_RETURN_ON_ERROR(parseDim3MDOptional(
      KernelMetaNode, hsa::md::Kernel::Key::WorkGroupSizeHint,
      Out.WorkGroupSizeHint));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, hsa::md::Kernel::Key::VecTypeHint, Out.VecTypeHint));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, hsa::md::Kernel::Key::DeviceEnqueueSymbol,
      Out.DeviceEnqueueSymbol));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::KernArgSegmentSize,
      Out.KernArgSegmentSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::GroupSegmentFixedSize,
      Out.GroupSegmentFixedSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::PrivateSegmentFixedSize,
      Out.PrivateSegmentFixedSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::KernArgSegmentAlign,
      Out.KernArgSegmentAlign));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::WaveFrontSize, Out.WaveFrontSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::SGPRCount, Out.SGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::VGPRCount, Out.VGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::AGPRCount, Out.AGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, hsa::md::Kernel::Key::MaxFlatWorkgroupSize,
      Out.MaxFlatWorkgroupSize));

  LUTHIER_RETURN_ON_ERROR(
      parseUIntMDOptional(KernelMetaNode, hsa::md::Kernel::Key::SGPRSpillCount,
                          Out.SGPRSpillCount));

  LUTHIER_RETURN_ON_ERROR(
      parseUIntMDOptional(KernelMetaNode, hsa::md::Kernel::Key::VGPRSpillCount,
                          Out.VGPRSpillCount));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Key::KernelKind,
      KernelKindEnumMap, Out.KernelKind));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Key::UsesDynamicStack,
      Out.UsesDynamicStack));

  std::optional<unsigned> WorkgroupProcessorMode{0};

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Key::WorkgroupProcessorMode,
      WorkgroupProcessorMode));

  Out.WorkgroupProcessorMode = *WorkgroupProcessorMode == 1;

  std::optional<unsigned> UniformWorkgroupSize{0};

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Key::UniformWorkgroupSize,
      UniformWorkgroupSize));

  Out.UniformWorkgroupSize = *UniformWorkgroupSize == 1;

  return llvm::Error::success();
}

llvm::Expected<
    std::tuple<llvm::Triple, llvm::StringRef, llvm::SubtargetFeatures>>
getELFObjectFileISA(const luthier::AMDGCNObjectFile &Obj) {
  llvm::Triple TT = Obj.makeTriple();
  std::optional<llvm::StringRef> CPU = Obj.tryGetCPUName();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      CPU.has_value(), "Failed to get the CPU name of the object file."));
  llvm::SubtargetFeatures Features;
  LUTHIER_RETURN_ON_ERROR(Obj.getFeatures().moveInto(Features));
  return std::make_tuple(TT, *CPU, Features);
}

/// Returns the <tt>Sym</tt>'s loaded memory offset from the <tt>ELF</tt>'s
/// loaded base
/// \note Function was adapted from LLVM's object library
/// \tparam ELFT type of ELF used
/// \param ELF the Object file being queried
/// \param Sym the ELF's symbol being queried
/// \return on success, the loaded offset of the \c Sym with respect to the
/// ELF's load base; an \c llvm::Error on failure
llvm::Expected<uint64_t>
getLoadedMemoryOffset(const luthier::AMDGCNELFFile &ELF,
                      const llvm::object::ELFSymbolRef &Sym) {
  auto PhdrRange = ELF.program_headers();
  LUTHIER_RETURN_ON_ERROR(PhdrRange.takeError());

  auto SymbolSection = Sym.getSection();
  LUTHIER_RETURN_ON_ERROR(SymbolSection.takeError());

  auto SymbolAddress = Sym.getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolAddress.takeError());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename llvm::object::ELF64LE::Phdr &Phdr : *PhdrRange)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<llvm::object::ELF64LE>(
            Phdr,
            *llvm::cast<
                 const llvm::object::ELFObjectFile<llvm::object::ELF64LE>>(
                 Sym.getObject())
                 ->getSection(SymbolSection.get()->getRawDataRefImpl()))))
      return *SymbolAddress - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return *SymbolAddress;
}

/// Parses the msgpack document inside the \p Note section or program header
/// into the given \p Doc
/// \note this only supports code objects V2+
/// \tparam ELFT type of the ELF
/// \param Note
/// \param NoteDescString
/// \param Doc
/// \return

bool parseNoteSection(const typename llvm::object::ELF64LE::Note &Note,
                      llvm::msgpack::Document &Doc) {
  if (((Note.getName() == "AMD" || Note.getName() == "AMDGPU") &&
       Note.getType() == llvm::ELF::NT_AMD_PAL_METADATA) ||
      (Note.getName() == "AMDGPU" &&
       Note.getType() == llvm::ELF::NT_AMDGPU_METADATA)) {
    return Doc.readFromBlob(Note.getDescAsStringRef(4), false);
  } else
    return false;
}

/// Reads the \c llvm::msgpack::Document obtained by parsing the note section
/// of \p Obj into a \c hsa::md::Metadata structure for easier access
/// to the document's metadata fields
/// \param Obj the \c luthier::AMDGCNObjectFile to be inspected
/// \return on success, the \c hsa::md::Metadata of the document, or an
/// \c llvm::Error describing the issue encountered during the process
static llvm::Expected<luthier::hsa::md::Metadata>
parseMetaDoc(llvm::msgpack::Document &KernelMetaDataDoc) {
  luthier::hsa::md::Metadata Out;
  auto MetaDataRoot = KernelMetaDataDoc.getRoot();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(MetaDataRoot.isMap(), ""));
  auto RootMap = MetaDataRoot.getMap();

  LUTHIER_RETURN_ON_ERROR(
      parseVersionMDRequired(RootMap, hsa::md::Key::Version, Out.Version));

  bool IsV2 = Out.Version.Minor == 0;
  // We don't support Code Object V2
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      !IsV2, "Parsing of Code Object V2 metadata is not supported."));

  auto PrintfMD = RootMap.find(hsa::md::Key::Printf);
  if (PrintfMD != RootMap.end()) {
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(PrintfMD->second.isArray(), ""));
    Out.Printf.emplace();
    for (const auto &P : PrintfMD->second.getArray()) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(P.isString(), ""));
      Out.Printf->emplace_back(P.getString());
    }
  }

  auto KernelsMD = RootMap.find(hsa::md::Key::Kernels);
  if (KernelsMD != RootMap.end()) {
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(KernelsMD->second.isArray(), ""));
    auto KernelsMDAsArray = KernelsMD->second.getArray();
    Out.Kernels.reserve(KernelsMDAsArray.size());
    for (auto &KernelMD : KernelsMDAsArray) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(KernelMD.isMap(), ""));

      auto KernelMDAsMap = KernelMD.getMap();
      auto SymbolMD = KernelMDAsMap.find(hsa::md::Kernel::Key::Symbol);
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ERROR_CHECK(SymbolMD != KernelMDAsMap.end(), ""));
      Out.Kernels.emplace_back();
      LUTHIER_RETURN_ON_ERROR(parseKernelMD(KernelMDAsMap, Out.Kernels.back()));
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Number of kernels parsed from the metadata: "
                          << Out.Kernels.size() << "\n";);
  return Out;
}

llvm::Expected<hsa::md::Metadata>
parseNoteMetaData(const luthier::AMDGCNObjectFile &Obj) {
  // First try to find the note program header and parse it
  llvm::msgpack::Document Doc;
  const auto &ELFFile = Obj.getELFFile();
  auto ProgramHeaders = ELFFile.program_headers();
  LUTHIER_RETURN_ON_ERROR(ProgramHeaders.takeError());
  for (const auto &Phdr : *ProgramHeaders) {
    if (Phdr.p_type == llvm::ELF::PT_NOTE) {
      llvm::Error Err = llvm::Error::success();
      for (const auto &Note : ELFFile.notes(Phdr, Err)) {
        if (parseNoteSection(Note, Doc)) {
          LLVM_DEBUG(llvm::dbgs() << "Parsed metadata in YAML:\n";
                     Doc.toYAML(llvm::dbgs()); llvm::dbgs() << "\n";);
          return parseMetaDoc(Doc);
        }
      }
      LUTHIER_RETURN_ON_ERROR(Err);
    }
  }
  // Try to find the note section and parse it
  auto Sections = ELFFile.sections();
  LUTHIER_RETURN_ON_ERROR(Sections.takeError());

  for (const auto &Shdr : *Sections) {
    if (Shdr.sh_type == llvm::ELF::SHT_NOTE) {
      llvm::Error Err = llvm::Error::success();
      for (const auto &Note : ELFFile.notes(Shdr, Err)) {
        if (parseNoteSection(Note, Doc)) {
          LLVM_DEBUG(llvm::dbgs() << "Parsed metadata in YAML:\n";
                     Doc.toYAML(llvm::dbgs()); llvm::dbgs() << "\n";);
          return parseMetaDoc(Doc);
        }
      }
      LUTHIER_RETURN_ON_ERROR(Err);
    }
  }

  return LUTHIER_CREATE_ERROR(
      "Failed to find the note section to parse its metadata.");
}

} // namespace luthier
