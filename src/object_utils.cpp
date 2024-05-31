#include "object_utils.hpp"

#include "llvm/IR/DIBuilder.h"
#include <llvm/ADT/StringExtras.h>
#include <llvm/DebugInfo/DWARF/DWARFContext.h>
#include <llvm/DebugInfo/DWARF/DWARFDie.h>
#include <llvm/Support/AMDGPUMetadata.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/YAMLTraits.h>
#include <ranges>
#include <llvm/IR/Module.h>
#include <string>

#include "error.hpp"

using namespace llvm;

using namespace llvm::object;

using namespace llvm::msgpack;

llvm::Error
parseVersionMDOptional(MapDocNode &Map, llvm::StringRef Key,
                       std::optional<luthier::hsa::md::Version> &Out) {
  auto VersionMD = Map.find(Key);
  if (VersionMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(VersionMD->second.isArray()));
    auto VersionMDAsArray = VersionMD->second.getArray();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(VersionMDAsArray.size() == 2));
    auto MajorVersionMD = VersionMDAsArray[0];
    auto MinorVersionMD = VersionMDAsArray[1];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MajorVersionMD.isScalar()));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MinorVersionMD.isScalar()));
    Out = {MajorVersionMD.getUInt(), MinorVersionMD.getUInt()};
  }
  return llvm::Error::success();
}

llvm::Error parseVersionMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                   luthier::hsa::md::Version &Out) {
  auto VersionMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(VersionMD->second.isArray()));
  auto VersionMDAsArray = VersionMD->second.getArray();
  Out = {VersionMDAsArray[0].getUInt(), VersionMDAsArray[1].getUInt()};
  return llvm::Error::success();
}

llvm::Error parseStringMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                  std::optional<std::string> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isString()));
    Out = NodeMD->second.getString();
  }
  return llvm::Error::success();
}

llvm::Error parseStringMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                  std::string &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD != Map.end()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isString()));
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
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(EnumMap.contains(*EnumString)));
    Out = EnumMap.at(*EnumString);
  }
  return llvm::Error::success();
}

template <typename ET>
llvm::Error parseEnumMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                const llvm::StringMap<ET> &EnumMap,
                                ET &Out) {
  std::string EnumString;
  LUTHIER_RETURN_ON_ERROR(parseStringMDRequired(Map, Key, EnumString));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(EnumMap.contains(EnumString)));
  Out = EnumMap.at(EnumString);
  //  llvm::outs() << EnumString << "\n";
  return llvm::Error::success();
}

llvm::Error parseDim3MDOptional(MapDocNode &Map, llvm::StringRef Key,
                                std::optional<dim3> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isArray()));
    auto NodeMDAsArray = NodeMD->second.getArray();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMDAsArray.size() == 3));

    auto XMD = NodeMDAsArray[0];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(XMD.isScalar()));

    auto YMD = NodeMDAsArray[1];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(YMD.isScalar()));

    auto ZMD = NodeMDAsArray[2];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ZMD.isScalar()));

    Out = {static_cast<uint32_t>(XMD.getUInt()),
           static_cast<uint32_t>(YMD.getUInt()),
           static_cast<uint32_t>(ZMD.getUInt())};
    //    llvm::outs() << Out->x << "," << Out->y << "," << Out->z << ","
    //                 << "\n";
  }
  return llvm::Error::success();
}

llvm::Error parseDim3MDRequired(MapDocNode &Map, llvm::StringRef Key,
                                dim3 &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD != Map.end()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isArray()));
  auto NodeMDAsArray = NodeMD->second.getArray();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMDAsArray.size() == 3));

  auto XMD = NodeMDAsArray[0];
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(XMD.isScalar()));

  auto YMD = NodeMDAsArray[1];
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(YMD.isScalar()));

  auto ZMD = NodeMDAsArray[2];
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ZMD.isScalar()));

  Out = {static_cast<uint32_t>(XMD.getUInt()),
         static_cast<uint32_t>(YMD.getUInt()),
         static_cast<uint32_t>(ZMD.getUInt())};
  //  llvm::outs() << Out.x << "," << Out.y << "," << Out.z << ","
  //               << "\n";
  return llvm::Error::success();
}

template <typename T>
llvm::Error parseUIntMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                std::optional<T> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isScalar()));
    Out = static_cast<T>(NodeMD->second.getUInt());
    //    llvm::outs() << Out << "\n";
  }
  return llvm::Error::success();
}

template <typename T>
llvm::Error parseUIntMDRequired(MapDocNode &Map, llvm::StringRef Key, T &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD != Map.end()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isScalar()));
  Out = static_cast<T>(NodeMD->second.getUInt());
  return llvm::Error::success();
}

llvm::Error parseBoolMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                bool &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isScalar()));
    Out = NodeMD->second.getBool();
  }
  return llvm::Error::success();
}

llvm::Error parseBoolMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                bool &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD != Map.end()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isScalar()));
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
        LUTHIER_ASSERTION(SymTab[I].st_name < StrTab.size()));

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
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(I < SymTab.size()));
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(SymTab[I].st_name < StrTab.size()));

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
      LUTHIER_ARGUMENT_ERROR_CHECK(Sec->sh_type == llvm::ELF::SHT_HASH ||
                                   Sec->sh_type == llvm::ELF::SHT_GNU_HASH));

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
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Sec->sh_offset + Sec->sh_size <
                                              ElfFile.getBufSize()));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        Sec->sh_size >= sizeof(typename ELFT::GnuHash) &&
        Sec->sh_size >= sizeof(typename ELFT::GnuHash) +
                            sizeof(typename ELFT::Word) * HashTab->maskwords +
                            sizeof(typename ELFT::Word) * HashTab->nbuckets +
                            sizeof(typename ELFT::Word) *
                                (SymTab.size() - HashTab->symndx)));
    return getSymbolFromGnuHashTable<ELFT>(Elf, SymbolName, *SymTabOrErr,
                                           *HashTab, SymTab, StrTab);
  }

  // If this is a Sys-V hash table we verify its size and search the symbol
  // table using the Sys-V hash table format.
  if (Sec->sh_type == llvm::ELF::SHT_HASH) {
    const auto *HashTab = reinterpret_cast<const typename ELFT::Hash *>(
        ElfFile.base() + Sec->sh_offset);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Sec->sh_offset + Sec->sh_size <
                                              ElfFile.getBufSize()));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        Sec->sh_size >= sizeof(typename ELFT::Hash) &&
        Sec->sh_size >= sizeof(typename ELFT::Hash) +
                            sizeof(typename ELFT::Word) * HashTab->nbucket +
                            sizeof(typename ELFT::Word) * HashTab->nchain));
    return getSymbolFromSysVHashTable<ELFT>(Elf, SymbolName, *SymTabOrErr,
                                            *HashTab, SymTab, StrTab);
  }

  return std::nullopt;
}

namespace luthier {

llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
lookupSymbolByName(const luthier::AMDGCNObjectFile &Elf,
                llvm::StringRef SymbolName) {
  for (auto Section = llvm::object::elf_section_iterator(Elf.section_begin());
       Section != llvm::object::elf_section_iterator(Elf.section_end());
       ++Section) {
    auto SectionAsSHdr = Elf.getSection(Section->getRawDataRefImpl());
    if ((SectionAsSHdr->sh_type == llvm::ELF::SHT_HASH) ||
        (SectionAsSHdr->sh_type == llvm::ELF::SHT_GNU_HASH)) {
      return hashLookup(Elf, SectionAsSHdr, SymbolName);
    }
  }
  llvm_unreachable("Symbol hash table was not found");
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
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ArgsMD->second.isArray()));
    Out.Args.emplace();
    for (auto &ArgMD : ArgsMD->second.getArray()) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ArgMD.isMap()));
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

  std::optional<unsigned> UniformWorkgroupSize{0};

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, luthier::hsa::md::Kernel::Key::UniformWorkgroupSize,
      UniformWorkgroupSize));

  Out.UniformWorkgroupSize = *UniformWorkgroupSize == 1;

  return llvm::Error::success();
}

llvm::Expected<luthier::hsa::md::Metadata>
parseMetaDoc(llvm::msgpack::Document &KernelMetaNode) {
  luthier::hsa::md::Metadata Out;
  auto MetaDataRoot = KernelMetaNode.getRoot();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(MetaDataRoot.isMap()));
  auto RootMap = MetaDataRoot.getMap();

  LUTHIER_RETURN_ON_ERROR(
      parseVersionMDRequired(RootMap, hsa::md::Key::Version, Out.Version));

  bool IsV2 = Out.Version.Minor == 0;
  // TODO: Write a V2 Parser if needed
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(!IsV2));

  auto PrintfMD = RootMap.find(hsa::md::Key::Printf);
  if (PrintfMD != RootMap.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(PrintfMD->second.isArray()));
    Out.Printf.emplace();
    for (const auto &P : PrintfMD->second.getArray()) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(P.isString()));
      Out.Printf->emplace_back(P.getString());
    }
  }

  auto KernelsMD = RootMap.find(hsa::md::Key::Kernels);
  //  llvm::outs() << "Is kernel MD found? " << (KernelsMD != RootMap.end())
  //               << "\n";
  for (auto &[k, v] : RootMap) {
    //    llvm::outs() << "Key is string: " << k.isString() << "\n";
    //    llvm::outs() << k.toString() << "\n";
  }
  //  llvm::outs() << RootMap.toString() << "\n";
  if (KernelsMD != RootMap.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(KernelsMD->second.isArray()));
    auto KernelsMDAsArray = KernelsMD->second.getArray();
    Out.Kernels.reserve(KernelsMDAsArray.size());
    for (auto &KernelMD : KernelsMDAsArray) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(KernelMD.isMap()));

      auto KernelMDAsMap = KernelMD.getMap();
      auto SymbolMD = KernelMDAsMap.find(hsa::md::Kernel::Key::Symbol);
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ASSERTION(SymbolMD != KernelMDAsMap.end()));
      Out.Kernels.emplace_back();
      LUTHIER_RETURN_ON_ERROR(parseKernelMD(KernelMDAsMap, Out.Kernels.back()));
    }
  }
  return Out;
}

Expected<std::unique_ptr<ELF64LEObjectFile>>
getAMDGCNObjectFile(StringRef Elf) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(Elf, "", false);
  Expected<std::unique_ptr<ObjectFile>> ObjectFile =
      ObjectFile::createELFObjectFile(*Buffer);
  LUTHIER_RETURN_ON_ERROR(ObjectFile.takeError());
  return unique_dyn_cast<ELF64LEObjectFile>(std::move(*ObjectFile));
}

Expected<std::unique_ptr<ELF64LEObjectFile>>
getAMDGCNObjectFile(ArrayRef<uint8_t> Elf) {
  return getAMDGCNObjectFile(toStringRef(Elf));
}

template <class ELFT>
static std::optional<Expected<typename ELFT::Phdr>>
getNotesFromProgramHeader(const ELFObjectFile<ELFT> *obj) {
  const ELFFile<ELFT> &ELFFile = obj->getELFFile();

  auto programHeadersOrError = ELFFile.program_headers();
  LUTHIER_RETURN_ON_ERROR(programHeadersOrError.takeError());

  for (const auto &pHdr : *programHeadersOrError) {
    if (pHdr.p_type == llvm::ELF::PT_NOTE) {
      return pHdr;
    }
  }
  return std::nullopt;
}

template <class ELFT>
static std::optional<Expected<typename ELFT::Shdr>>
getNotesFromSectionHeader(const ELFObjectFile<ELFT> *Obj) {
  const ELFFile<ELFT> &ELFFile = Obj->getELFFile();

  auto sectionsOrError = ELFFile.sections();
  if (errorToBool(sectionsOrError.takeError())) {
    return sectionsOrError.takeError();
  }

  for (const auto &sHdr : *sectionsOrError) {
    if (sHdr.sh_type == llvm::ELF::SHT_NOTE)
      return sHdr;
  }

  return std::nullopt;
}

// Try to merge "amdhsa.kernels" from DocNode @p From to @p To.
// The merge is allowed only if
// 1. "amdhsa.printf" record is not existing in either of the nodes.
// 2. "amdhsa.version" exists and is same.
// 3. "amdhsa.kernels" exists in both nodes.
//
// If merge is possible the function merges Kernel records
// to @p To and returns @c true.
static bool mergeNoteRecords(llvm::msgpack::DocNode &From,
                             llvm::msgpack::DocNode &To,
                             const StringRef VersionStrKey,
                             const StringRef PrintfStrKey,
                             const StringRef KernelStrKey) {
  if (!From.isMap()) {
    return false;
  }

  if (To.isEmpty()) {
    To = From;
    return true;
  }

  assert(To.isMap());

  if (From.getMap().find(PrintfStrKey) != From.getMap().end()) {
    /* Check if both have Printf records */
    if (To.getMap().find(PrintfStrKey) != To.getMap().end()) {
      return false;
    }

    /* Add Printf record for 'To' */
    To.getMap()[PrintfStrKey] = From.getMap()[PrintfStrKey];
  }

  auto &FromMapNode = From.getMap();
  auto &ToMapNode = To.getMap();

  auto FromVersionArrayNode = FromMapNode.find(VersionStrKey);
  auto ToVersionArrayNode = ToMapNode.find(VersionStrKey);

  if ((FromVersionArrayNode == FromMapNode.end() ||
       !FromVersionArrayNode->second.isArray()) ||
      (ToVersionArrayNode == ToMapNode.end() ||
       !ToVersionArrayNode->second.isArray())) {
    return false;
  }

  auto FromVersionArray = FromMapNode[VersionStrKey].getArray();
  auto ToVersionArray = ToMapNode[VersionStrKey].getArray();

  if (FromVersionArray.size() != ToVersionArray.size()) {
    return false;
  }

  for (size_t I = 0, E = FromVersionArray.size(); I != E; ++I) {
    if (FromVersionArray[I] != ToVersionArray[I]) {
      return false;
    }
  }

  auto FromKernelArray = FromMapNode.find(KernelStrKey);
  auto ToKernelArray = ToMapNode.find(KernelStrKey);

  if ((FromKernelArray == FromMapNode.end() ||
       !FromKernelArray->second.isArray()) ||
      (ToKernelArray == ToMapNode.end() || !ToKernelArray->second.isArray())) {
    return false;
  }

  auto &ToKernelRecords = ToKernelArray->second.getArray();
  for (auto Kernel : FromKernelArray->second.getArray()) {
    ToKernelRecords.push_back(Kernel);
  }

  return true;
}



// helper method (need to test this, small chance it might loop forever)
DWARFDie findSymbolDie(const llvm::DWARFDie Die,
                                       std::string &symbolName) {
  auto tag = Die.getTag();
  // Check current DIE for symbol name
  if (tag == dwarf::DW_TAG_subprogram ||
      tag == dwarf::DW_TAG_variable &&
          (*Die.find(dwarf::DW_AT_name)->getAsCString() == symbolName)) {
    return Die;
  }
  // check children of the current DIE
  for (const auto &Child : Die.children()) {
    DWARFDie Result = findSymbolDie(Child, symbolName);
    if (Result.isValid()) {
      return Result;
    }
  }
  // default die with invalid state
  return DWARFDie();
}

// const llvm::object::ELFObjectFile<ELFT> &Elf
//
llvm::Expected<llvm::DWARFDie> getDWARFDie(llvm::DWARFContext &ctx,
                                     std::string symbolName) {

  for (const auto &CU : ctx.compile_units()) {
    if (llvm::DWARFDie DIE = CU->getUnitDIE(false)) {
      auto symbolDie = findSymbolDie(DIE, symbolName);
      if (symbolDie.isValid()) {
        return symbolDie;
      }
    }
  }
  // Return an invalid DWARFDie if not found
  // For error throwing:
  // LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
  //       DisAsm->getInstruction(Inst, InstSize, ReadBytes, CurrentAddress,
  //                              llvm::nulls()) ==
  //       llvm::MCDisassembler::Success));
  return DWARFDie();
}

llvm::Expected<DebugLoc> getDebugLoc(const llvm::DWARFDie &die,
                                    llvm::Module module) {
  auto line = die.find(dwarf::DW_AT_decl_line)->getAsUnsignedConstant().value();
  auto col =  die.find(dwarf::DW_AT_decl_column)->getAsUnsignedConstant().value();
  // llvm::DWARFCompileUnit * compileUnit = (llvm::DWARFCompileUnit *)die.getDwarfUnit(); // cast the
  // I need to pass in a DICompileUnit (how do I get that?) -> needs to be resolved for more accurate debug info
  llvm::DIBuilder DIB(module, true, nullptr); // allowUnresolved = true (i don't really know what that means)
  llvm::Metadata *Scope = nullptr;
  // // Determine the scope
  if (auto ScopeAttr = die.find(dwarf::DW_AT_start_scope)) {
    const uint64_t ScopeOffset = ScopeAttr->getAsReference().value();
    if (auto ScopeDie = die.getDwarfUnit()->getDIEForOffset(ScopeOffset)) {
      if (ScopeDie.isValid()) {
        // if its a subprogram (func) or lexical block /scope ({})
        // need to handle case where: ScopeDie.getTag() == dwarf::DW_TAG_lexical_block
        if (ScopeDie.getTag() == dwarf::DW_TAG_subprogram) {
          Scope = DIB.createFunction(nullptr, ScopeDie.getShortName(), StringRef(), nullptr, ScopeDie.getDeclLine(), nullptr, 0, DINode::FlagZero, llvm::DISubprogram::SPFlagZero);
        } else if (ScopeDie.getTag() ==
                   dwarf::DW_TAG_file_type) { // else, if it's a file
          Scope =
              DIB.createFile(ScopeDie.getShortName(), ScopeDie.getFilename().str());
        }
      }
    }
  }
  if (!Scope) {
    // Fallback to using the CU's file as the scope
    auto CU = die.getDwarfUnit()->getUnitDIE();
    if (auto FileAttr = CU.find(dwarf::DW_AT_name)) {
      std::string FileName = std::string(CU.getShortName());
      if (!FileName) {
        // return a default DebugLoc
        // Instead, need to throw an Error (can do this by returning an Expected<DebugLoc>)
        return DebugLoc();
      }
      Scope = DIB.createFile(FileName, CU.getDeclFile().str()); // declFile
    }
  }
  // get might not be returning a pointer! We need a pointer to pass into the
  // DebugLoc constructor
  // DIB.createBasicType()
  // return DebugLoc(DILocation::get(line, col, scope));
  // DIBasicType::get(unsigned Tag, StringRef Name) -> Tag ->
  // DebugLoc(DILocation::get(line, col, scope));

  // Updates:
  // disassemble(hsa::ExecutableSymbol Symbol, bool includeDebugInfo) -> creates the DWARFDie, and adds it to the Instr.


  // scope is a Metadata* and represents the "scope" of this symbol
  // a scope is one of: subroutine (function) / lexical scope ({}) / File
  // Issue: I need the "scope" to create the DebugLoc
  // - I need the Module in order to get construct a DIBuilder (I got confused and thought we could use the LLVMContext)
  // - I need the DIBuilder to create the scope and pass it into the DILocation::get()
  // Solutions:
  // - the liftAndAddToModule function has access to the Module, use it to get the DIBuilder -> DebugLoc
  // - create a dummy Metadata* scope object (check: DIBasicType, also having trouble doing that)
  // - Matin, any ideas on how the scope could be accessed without all this hassle?
  //    - I feel like scope of a symbol should be easily accessible
  //    - I checked: Module, MachineModuleInfo, Function, MachineBasicBlock, MachineFunction....
  return llvm::DebugLoc();
}
} // namespace luthier

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const luthier::hsa::md::Kernel::Metadata &MD) {
  OS << luthier::hsa::md::Kernel::Key::Name << ": " << MD.Name << "\n";
  OS << luthier::hsa::md::Kernel::Key::Symbol << ": " << MD.Symbol << "\n";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const luthier::hsa::md::Metadata &MD) {
  OS << luthier::hsa::md::Key::Version << ": " << MD.Version.Major << ", "
     << MD.Version.Minor << "\n";

  if (MD.Printf.has_value()) {
    OS << luthier::hsa::md::Key::Printf << ": \n";
    for (const auto &P : *MD.Printf) {
      OS.indent(2);
      OS << P << "\n";
    }
  }

  if (!MD.Kernels.empty()) {
    OS << luthier::hsa::md::Key::Kernels << ": \n";
    for (const auto &Kernel : MD.Kernels) {
      OS.indent(2);
      OS << Kernel << "\n";
    }
  }

  return OS;
}
