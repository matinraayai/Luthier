#include "object_utils.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/AMDGPUMetadata.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/YAMLTraits.h>
#include <ranges>

#include <string>

#include "error.hpp"

using namespace llvm;

using namespace llvm::object;

using namespace llvm::msgpack;

llvm::Error
parseVersionMDOptional(MapDocNode &Map, llvm::StringRef Key,
                       std::optional<luthier::HSAMD::Version> &Out) {
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
                                   luthier::HSAMD::Version &Out) {
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
//    llvm::outs() << Out << "\n";
  }
  return llvm::Error::success();
}

llvm::Error parseStringMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                  std::string &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD != Map.end()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isString()));
  Out = NodeMD->second.getString();
//  llvm::outs() << Out << "\n";
  return llvm::Error::success();
}

static llvm::DenseMap<llvm::StringRef, llvm::AMDGPU::HSAMD::AccessQualifier>
    AccessQualifierEnumMap = {
        {"read_only", llvm::AMDGPU::HSAMD::AccessQualifier::ReadOnly},
        {"write_only", llvm::AMDGPU::HSAMD::AccessQualifier::WriteOnly},
        {"read_write", llvm::AMDGPU::HSAMD::AccessQualifier::ReadWrite}};

static llvm::DenseMap<llvm::StringRef,
                      decltype(llvm::AMDGPUAS::MAX_AMDGPU_ADDRESS)>
    AMDGPUAddressSpaceEnumMap = {{"private", llvm::AMDGPUAS::PRIVATE_ADDRESS},
                                 {"global", llvm::AMDGPUAS::GLOBAL_ADDRESS},
                                 {"constant", llvm::AMDGPUAS::CONSTANT_ADDRESS},
                                 {"local", llvm::AMDGPUAS::LOCAL_ADDRESS},
                                 {"generic", llvm::AMDGPUAS::FLAT_ADDRESS},
                                 {"region", llvm::AMDGPUAS::REGION_ADDRESS}};

static llvm::DenseMap<llvm::StringRef, luthier::HSAMD::ValueKind>
    ValueKindEnumMap = {
        {"by_value", luthier::HSAMD::ValueKind::ByValue},
        {"global_buffer", luthier::HSAMD::ValueKind::GlobalBuffer},
        {"dynamic_shared_pointer",
         luthier::HSAMD::ValueKind::DynamicSharedPointer},
        {"sampler", luthier::HSAMD::ValueKind::Sampler},
        {"image", luthier::HSAMD::ValueKind::Image},
        {"pipe", luthier::HSAMD::ValueKind::Pipe},
        {"queue", luthier::HSAMD::ValueKind::Queue},
        {"hidden_global_offset_x",
         luthier::HSAMD::ValueKind::HiddenGlobalOffsetX},
        {"hidden_global_offset_y",
         luthier::HSAMD::ValueKind::HiddenGlobalOffsetY},
        {"hidden_global_offset_z",
         luthier::HSAMD::ValueKind::HiddenGlobalOffsetZ},
        {"hidden_none", luthier::HSAMD::ValueKind::HiddenNone},
        {"hidden_printf_buffer", luthier::HSAMD::ValueKind::HiddenPrintfBuffer},
        {"hidden_hostcall_buffer",
         luthier::HSAMD::ValueKind::HiddenHostcallBuffer},
        {"hidden_default_queue", luthier::HSAMD::ValueKind::HiddenDefaultQueue},
        {"hidden_completion_action",
         luthier::HSAMD::ValueKind::HiddenCompletionAction},
        {"hidden_multigrid_sync_arg",
         luthier::HSAMD::ValueKind::HiddenMultiGridSyncArg},
        {"hidden_block_count_x", luthier::HSAMD::ValueKind::HiddenBlockCountX},
        {"hidden_block_count_y", luthier::HSAMD::ValueKind::HiddenBlockCountY},
        {"hidden_block_count_z", luthier::HSAMD::ValueKind::HiddenBlockCountZ},
        {"hidden_group_size_x", luthier::HSAMD::ValueKind::HiddenGroupSizeX},
        {"hidden_group_size_y", luthier::HSAMD::ValueKind::HiddenGroupSizeY},
        {"hidden_group_size_z", luthier::HSAMD::ValueKind::HiddenGroupSizeZ},
        {"hidden_remainder_x", luthier::HSAMD::ValueKind::HiddenRemainderX},
        {"hidden_remainder_y", luthier::HSAMD::ValueKind::HiddenRemainderY},
        {"hidden_remainder_z", luthier::HSAMD::ValueKind::HiddenRemainderZ},
        {"hidden_grid_dims", luthier::HSAMD::ValueKind::HiddenGridDims},
        {"hidden_heap_v1", luthier::HSAMD::ValueKind::HiddenHeapV1},
        {"hidden_dynamic_lds_size",
         luthier::HSAMD::ValueKind::HiddenDynamicLDSSize},
        {"hidden_private_base", luthier::HSAMD::ValueKind::HiddenPrivateBase},
        {"hidden_shared_base", luthier::HSAMD::ValueKind::HiddenSharedBase},
        {"hidden_queue_ptr", luthier::HSAMD::ValueKind::HiddenQueuePtr}};

static llvm::DenseMap<llvm::StringRef, luthier::HSAMD::KernelKind>
    KernelKindEnumMap = {{"normal", luthier::HSAMD::KernelKind::Normal},
                         {"init", luthier::HSAMD::KernelKind::Init},
                         {"fini", luthier::HSAMD::KernelKind::Fini}};

template <typename ET>
llvm::Error parseEnumMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                llvm::DenseMap<llvm::StringRef, ET> &EnumMap,
                                std::optional<ET> &Out) {
  std::optional<std::string> EnumString;
  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(Map, Key, EnumString));
  if (EnumString.has_value()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(EnumMap.contains(*EnumString)));
    Out = EnumMap[*EnumString];
//    llvm::outs() << EnumString << "\n";
  }
  return llvm::Error::success();
}

template <typename ET>
llvm::Error parseEnumMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                llvm::DenseMap<llvm::StringRef, ET> &EnumMap,
                                ET &Out) {
  std::string EnumString;
  LUTHIER_RETURN_ON_ERROR(parseStringMDRequired(Map, Key, EnumString));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(EnumMap.contains(EnumString)));
  Out = EnumMap[EnumString];
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
//  llvm::outs() << Out << "\n";
  return llvm::Error::success();
}

llvm::Error parseBoolMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                std::optional<bool> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(NodeMD->second.isScalar()));
    Out = NodeMD->second.getBool();
  }
//  llvm::outs() << Out << "\n";
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

namespace luthier {

llvm::Error parseArgMD(llvm::msgpack::MapDocNode &KernelMetaNode,
                       HSAMD::Kernel::Arg::Metadata &Out) {
  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::Name, Out.Name));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::TypeName,
      Out.TypeName));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::Size, Out.Size));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::Offset, Out.Offset));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDRequired(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::ValueKind,
      ValueKindEnumMap, Out.ValueKind));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::PointeeAlign,
      Out.PointeeAlign));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::AddressSpace,
      AMDGPUAddressSpaceEnumMap, Out.AddressSpace));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::AccQual,
      AccessQualifierEnumMap, Out.AccQual));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::ActualAccQual,
      AccessQualifierEnumMap, Out.ActualAccQual));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::IsConst, Out.IsConst));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::IsRestrict,
      Out.IsRestrict));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::IsVolatile,
      Out.IsVolatile));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Arg::Key::IsPipe, Out.IsPipe));

  return llvm::Error::success();
};

llvm::Error parseKernelMD(llvm::msgpack::MapDocNode &KernelMetaNode,
                          HSAMD::Kernel::Metadata &Out) {

  LUTHIER_RETURN_ON_ERROR(parseStringMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::Name, Out.Name));

  LUTHIER_RETURN_ON_ERROR(parseStringMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::Symbol, Out.Symbol));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, HSAMD::Kernel::Key::Language, Out.Language));

  LUTHIER_RETURN_ON_ERROR(parseVersionMDOptional(
      KernelMetaNode, HSAMD::Kernel::Key::LanguageVersion,
      Out.LanguageVersion));

  auto ArgsMD = KernelMetaNode.find(HSAMD::Kernel::Key::Args);
  if (ArgsMD != KernelMetaNode.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ArgsMD->second.isArray()));
    Out.Args.emplace();
    for (auto &ArgMD : ArgsMD->second.getArray()) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ArgMD.isMap()));
      Out.Args->emplace_back();
      LUTHIER_RETURN_ON_ERROR(parseArgMD(ArgMD.getMap(), Out.Args->back()));
    }
  }

  LUTHIER_RETURN_ON_ERROR(
      parseDim3MDOptional(KernelMetaNode, HSAMD::Kernel::Key::ReqdWorkGroupSize,
                          Out.ReqdWorkGroupSize));

  LUTHIER_RETURN_ON_ERROR(
      parseDim3MDOptional(KernelMetaNode, HSAMD::Kernel::Key::WorkGroupSizeHint,
                          Out.WorkGroupSizeHint));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, HSAMD::Kernel::Key::VecTypeHint, Out.VecTypeHint));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, HSAMD::Kernel::Key::DeviceEnqueueSymbol,
      Out.DeviceEnqueueSymbol));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::KernArgSegmentSize,
      Out.KernArgSegmentSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::GroupSegmentFixedSize,
      Out.GroupSegmentFixedSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::PrivateSegmentFixedSize,
      Out.PrivateSegmentFixedSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::KernArgSegmentAlign,
      Out.KernArgSegmentAlign));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::WaveFrontSize, Out.WaveFrontSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::SGPRCount, Out.SGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::VGPRCount, Out.VGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::AGPRCount, Out.AGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, HSAMD::Kernel::Key::MaxFlatWorkgroupSize,
      Out.MaxFlatWorkgroupSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, HSAMD::Kernel::Key::SGPRSpillCount, Out.SGPRSpillCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, HSAMD::Kernel::Key::VGPRSpillCount, Out.VGPRSpillCount));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Key::KernelKind,
      KernelKindEnumMap, Out.KernelKind));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, luthier::HSAMD::Kernel::Key::UniformWorkgroupSize,
      Out.UniformWorkgroupSize));

  return llvm::Error::success();
}

llvm::Expected<luthier::HSAMD::Metadata>
parseMetaDoc(llvm::msgpack::Document &KernelMetaNode) {
  luthier::HSAMD::Metadata Out;
  auto MetaDataRoot = KernelMetaNode.getRoot();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(MetaDataRoot.isMap()));
  auto RootMap = MetaDataRoot.getMap();

  LUTHIER_RETURN_ON_ERROR(
      parseVersionMDRequired(RootMap, HSAMD::Key::Version, Out.Version));

  bool IsV2 = Out.Version.Minor == 0;
  // TODO: Write a V2 Parser if needed
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(!IsV2));

  auto PrintfMD = RootMap.find(HSAMD::Key::Printf);
  if (PrintfMD != RootMap.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(PrintfMD->second.isArray()));
    Out.Printf.emplace();
    for (const auto &P : PrintfMD->second.getArray()) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(P.isString()));
      Out.Printf->emplace_back(P.getString());
    }
  }

  auto KernelsMD = RootMap.find(HSAMD::Key::Kernels);
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
      auto SymbolMD = KernelMDAsMap.find(HSAMD::Kernel::Key::Symbol);
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

} // namespace luthier

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const luthier::HSAMD::Kernel::Metadata &MD) {
  OS << luthier::HSAMD::Kernel::Key::Name << ": " << MD.Name << "\n";
  OS << luthier::HSAMD::Kernel::Key::Symbol << ": " << MD.Symbol << "\n";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const luthier::HSAMD::Metadata &MD) {
  OS << luthier::HSAMD::Key::Version << ": " << MD.Version.Major << ", "
     << MD.Version.Minor << "\n";

  if (MD.Printf.has_value()) {
    OS << luthier::HSAMD::Key::Printf << ": \n";
    for (const auto &P : *MD.Printf) {
      OS.indent(2);
      OS << P << "\n";
    }
  }

  if (!MD.Kernels.empty()) {
    OS << luthier::HSAMD::Key::Kernels << ": \n";
    for (const auto &Kernel : MD.Kernels) {
      OS.indent(2);
      OS << Kernel << "\n";
    }
  }

  return OS;
}
