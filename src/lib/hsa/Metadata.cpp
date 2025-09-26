//===-- Metadata.cpp - Metadata Struct for HSA COV3+ ----------------------===//
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
/// Implements parsing of the HSA metadata, as well as the pass used
/// to parse it inside the lifting procedure.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/Metadata.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/GenericLuthierError.h"
#include <llvm/IR/Module.h>

using namespace llvm::msgpack;

namespace luthier {

namespace amdgpu::hsamd {

static llvm::Error parseVersionMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                          std::optional<Version> &Out) {
  auto VersionMD = Map.find(Key);
  if (VersionMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        VersionMD->second.isArray(),
        "Version metadata was found but it is not an array metadata"));
    auto VersionMDAsArray = VersionMD->second.getArray();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        VersionMDAsArray.size() == 2,
        llvm::formatv(
            "Expected the version metadata to have 2 entries, got {0} instead",
            VersionMDAsArray.size())));
    auto MajorVersionMD = VersionMDAsArray[0];
    auto MinorVersionMD = VersionMDAsArray[1];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        MajorVersionMD.isScalar(),
        "The major number of the metadata is not scalar."));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        MinorVersionMD.isScalar(),
        "The minor number of the metadata is not scalar."));
    Out = {MajorVersionMD.getUInt(), MinorVersionMD.getUInt()};
  }
  return llvm::Error::success();
}

static llvm::Error parseVersionMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                          Version &Out) {
  std::optional<Version> OptionalVersion;
  LUTHIER_RETURN_ON_ERROR(parseVersionMDOptional(Map, Key, OptionalVersion));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      OptionalVersion.has_value(), "Failed to find the version metadata"));
  Out = *OptionalVersion;
  return llvm::Error::success();
}

static llvm::Error parseStringMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                         std::optional<std::string> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NodeMD->second.isString(),
        llvm::formatv(
            "Found metadata entry with key name {0} but it is not a string",
            Key)));
    Out = NodeMD->second.getString();
  }
  return llvm::Error::success();
}

static llvm::Error parseStringMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                         std::string &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      NodeMD != Map.end(),
      llvm::formatv("Failed to find the key {0} inside the metadata map",
                    Key)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      NodeMD->second.isString(),
      llvm::formatv(
          "Found key {0} inside the metadata map but it is not a string",
          Key)));
  Out = NodeMD->second.getString();
  return llvm::Error::success();
}

template <typename ET>
llvm::Error parseEnumMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                const llvm::StringMap<ET> &EnumMap,
                                std::optional<ET> &Out) {
  std::optional<std::string> EnumString;
  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(Map, Key, EnumString));
  if (EnumString.has_value()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        EnumMap.contains(*EnumString),
        "Failed to find the string in the enum map"));
    Out = EnumMap.at(*EnumString);
  }
  return llvm::Error::success();
}

template <typename ET>
llvm::Error parseEnumMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                const llvm::StringMap<ET> &EnumMap, ET &Out) {
  std::string EnumString;
  LUTHIER_RETURN_ON_ERROR(parseStringMDRequired(Map, Key, EnumString));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      EnumMap.contains(EnumString),
      llvm::formatv("Key {0} is not present in Enum Map", Key)));
  Out = EnumMap.at(EnumString);
  return llvm::Error::success();
}

llvm::Error parseDim3MDOptional(MapDocNode &Map, llvm::StringRef Key,
                                std::optional<dim3> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NodeMD->second.isArray(), "Dim3 metadata is not array type"));
    auto NodeMDAsArray = NodeMD->second.getArray();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NodeMDAsArray.size() == 3,
        "Dim3 metadata entry does not have 3 entries"));

    auto XMD = NodeMDAsArray[0];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        XMD.isScalar(), "Dim3 x metadata entry is not scalar"));

    auto YMD = NodeMDAsArray[1];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        YMD.isScalar(), "Dim3 y metadata entry is not scalar"));

    auto ZMD = NodeMDAsArray[2];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        ZMD.isScalar(), "Dim3 z metadata entry is not scalar"));

    Out = {static_cast<uint32_t>(XMD.getUInt()),
           static_cast<uint32_t>(YMD.getUInt()),
           static_cast<uint32_t>(ZMD.getUInt())};
  }
  return llvm::Error::success();
}

llvm::Error parseDim3MDRequired(MapDocNode &Map, llvm::StringRef Key,
                                dim3 &Out) {
  std::optional<dim3> OptionalDim3;
  LUTHIER_RETURN_ON_ERROR(parseDim3MDOptional(Map, Key, OptionalDim3));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      OptionalDim3.has_value(),
      llvm::formatv(
          "Failed to find the dim3 metadata node associated with key {0}",
          Key)));
  return llvm::Error::success();
}

template <typename T>
llvm::Error parseUIntMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                std::optional<T> &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NodeMD->second.isScalar(),
        llvm::formatv("Found key {0} in the map but it is not scalar", Key)));
    Out = static_cast<T>(NodeMD->second.getUInt());
  }
  return llvm::Error::success();
}

template <typename T>
llvm::Error parseUIntMDRequired(MapDocNode &Map, llvm::StringRef Key, T &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      NodeMD != Map.end(),
      llvm::formatv("Failed to find entry {0} inside the map", Key)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(NodeMD->second.isScalar(),
                                                      "Entry is not scalar"));
  Out = static_cast<T>(NodeMD->second.getUInt());
  return llvm::Error::success();
}

static llvm::Error parseBoolMDOptional(MapDocNode &Map, llvm::StringRef Key,
                                       bool &Out) {
  auto NodeMD = Map.find(Key);
  if (NodeMD != Map.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        NodeMD->second.isScalar(),
        llvm::formatv("Found key {0} in the metadata map but it is not scalar",
                      Key)));
    Out = NodeMD->second.getBool();
  }
  return llvm::Error::success();
}

static llvm::Error parseBoolMDRequired(MapDocNode &Map, llvm::StringRef Key,
                                       bool &Out) {
  auto NodeMD = Map.find(Key);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      NodeMD != Map.end(),
      llvm::formatv("Failed to find the key {0} in the metadata", Key)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      NodeMD->second.isScalar(),
      llvm::formatv("Found key {0} in the metadata note but it is not scalar",
                    Key)));
  Out = NodeMD->second.getBool();
  return llvm::Error::success();
}

static llvm::Error
parseArgMD(llvm::msgpack::MapDocNode &KernelMetaNode,
           const llvm::StringMap<AccessQualifier> &AccessQualifierEnumMap,
           const llvm::StringMap<unsigned> &AMDGPUAddressSpaceEnumMap,
           const llvm::StringMap<ValueKind> &ValueKindEnumMap,
           Kernel::Arg::Metadata &Out) {
  LUTHIER_RETURN_ON_ERROR(
      parseStringMDOptional(KernelMetaNode, Kernel::Arg::Key::Name, Out.Name));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, Kernel::Arg::Key::TypeName, Out.TypeName));

  LUTHIER_RETURN_ON_ERROR(
      parseUIntMDRequired(KernelMetaNode, Kernel::Arg::Key::Size, Out.Size));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, Kernel::Arg::Key::Offset, Out.Offset));

  LUTHIER_RETURN_ON_ERROR(parseEnumMDRequired(KernelMetaNode,
                                              Kernel::Arg::Key::ValueKind,
                                              ValueKindEnumMap, Out.ValKind));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, Kernel::Arg::Key::PointeeAlign, Out.PointeeAlign));

  LUTHIER_RETURN_ON_ERROR(
      parseEnumMDOptional(KernelMetaNode, Kernel::Arg::Key::AddressSpace,
                          AMDGPUAddressSpaceEnumMap, Out.AddressSpace));

  LUTHIER_RETURN_ON_ERROR(
      parseEnumMDOptional(KernelMetaNode, Kernel::Arg::Key::AccQual,
                          AccessQualifierEnumMap, Out.AccQual));

  LUTHIER_RETURN_ON_ERROR(
      parseEnumMDOptional(KernelMetaNode, Kernel::Arg::Key::ActualAccQual,
                          AccessQualifierEnumMap, Out.ActualAccQual));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, Kernel::Arg::Key::IsConst, Out.IsConst));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, Kernel::Arg::Key::IsRestrict, Out.IsRestrict));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, Kernel::Arg::Key::IsVolatile, Out.IsVolatile));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, Kernel::Arg::Key::IsPipe, Out.IsPipe));

  return llvm::Error::success();
};

static llvm::Error
parseKernelMD(llvm::msgpack::MapDocNode &KernelMetaNode,
              const llvm::StringMap<AccessQualifier> &AccessQualifierEnumMap,
              const llvm::StringMap<unsigned> &AMDGPUAddressSpaceEnumMap,
              const llvm::StringMap<ValueKind> &ValueKindEnumMap,
              const llvm::StringMap<KernelKind> &KernelKindEnumMap,
              Kernel::Metadata &Out) {

  LUTHIER_RETURN_ON_ERROR(
      parseStringMDRequired(KernelMetaNode, Kernel::Key::Name, Out.Name));

  LUTHIER_RETURN_ON_ERROR(
      parseStringMDRequired(KernelMetaNode, Kernel::Key::Symbol, Out.Symbol));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, Kernel::Key::Language, Out.Language));

  LUTHIER_RETURN_ON_ERROR(parseVersionMDOptional(
      KernelMetaNode, Kernel::Key::LanguageVersion, Out.LanguageVersion));

  auto ArgsMD = KernelMetaNode.find(Kernel::Key::Args);
  if (ArgsMD != KernelMetaNode.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        ArgsMD->second.isArray(), "Argument node is not an array"));
    for (auto &ArgMD : ArgsMD->second.getArray()) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          ArgMD.isMap(), "Argument metadata is not a map"));
      Out.Args->emplace_back();
      LUTHIER_RETURN_ON_ERROR(parseArgMD(ArgMD.getMap(), AccessQualifierEnumMap,
                                         AMDGPUAddressSpaceEnumMap,
                                         ValueKindEnumMap, Out.Args->back()));
    }
  }

  LUTHIER_RETURN_ON_ERROR(parseDim3MDOptional(
      KernelMetaNode, Kernel::Key::ReqdWorkGroupSize, Out.ReqdWorkGroupSize));

  LUTHIER_RETURN_ON_ERROR(parseDim3MDOptional(
      KernelMetaNode, Kernel::Key::WorkGroupSizeHint, Out.WorkGroupSizeHint));

  LUTHIER_RETURN_ON_ERROR(parseStringMDOptional(
      KernelMetaNode, Kernel::Key::VecTypeHint, Out.VecTypeHint));

  LUTHIER_RETURN_ON_ERROR(
      parseStringMDOptional(KernelMetaNode, Kernel::Key::DeviceEnqueueSymbol,
                            Out.DeviceEnqueueSymbol));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, Kernel::Key::KernArgSegmentSize, Out.KernArgSegmentSize));

  LUTHIER_RETURN_ON_ERROR(
      parseUIntMDRequired(KernelMetaNode, Kernel::Key::GroupSegmentFixedSize,
                          Out.GroupSegmentFixedSize));

  LUTHIER_RETURN_ON_ERROR(
      parseUIntMDRequired(KernelMetaNode, Kernel::Key::PrivateSegmentFixedSize,
                          Out.PrivateSegmentFixedSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(KernelMetaNode,
                                              Kernel::Key::KernArgSegmentAlign,
                                              Out.KernArgSegmentAlign));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, Kernel::Key::WaveFrontSize, Out.WaveFrontSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, Kernel::Key::SGPRCount, Out.SGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(
      KernelMetaNode, Kernel::Key::VGPRCount, Out.VGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, Kernel::Key::AGPRCount, Out.AGPRCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDRequired(KernelMetaNode,
                                              Kernel::Key::MaxFlatWorkgroupSize,
                                              Out.MaxFlatWorkgroupSize));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, Kernel::Key::SGPRSpillCount, Out.SGPRSpillCount));

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, Kernel::Key::VGPRSpillCount, Out.VGPRSpillCount));

  LUTHIER_RETURN_ON_ERROR(
      parseEnumMDOptional(KernelMetaNode, Kernel::Key::KernelKind,
                          KernelKindEnumMap, Out.KernelKind));

  LUTHIER_RETURN_ON_ERROR(parseBoolMDOptional(
      KernelMetaNode, Kernel::Key::UsesDynamicStack, Out.UsesDynamicStack));

  std::optional<unsigned> WorkgroupProcessorMode{0};

  LUTHIER_RETURN_ON_ERROR(
      parseUIntMDOptional(KernelMetaNode, Kernel::Key::WorkgroupProcessorMode,
                          WorkgroupProcessorMode));

  Out.WorkgroupProcessorMode = *WorkgroupProcessorMode == 1;

  std::optional<unsigned> UniformWorkgroupSize{0};

  LUTHIER_RETURN_ON_ERROR(parseUIntMDOptional(
      KernelMetaNode, Kernel::Key::UniformWorkgroupSize, UniformWorkgroupSize));

  Out.UniformWorkgroupSize = *UniformWorkgroupSize == 1;

  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<Metadata>>
MetadataParser::parseNoteMetaData(llvm::msgpack::Document &Doc) const {
  auto Out = std::make_unique<Metadata>();
  DocNode &MetaDataRoot = Doc.getRoot();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      MetaDataRoot.isMap(), "The metadata doc is not a map"));
  auto RootMap = MetaDataRoot.getMap();

  LUTHIER_RETURN_ON_ERROR(
      parseVersionMDRequired(RootMap, Key::Version, Out->Version));

  auto PrintfMD = RootMap.find(Key::Printf);
  if (PrintfMD != RootMap.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        PrintfMD->second.isArray(), "The printf metadata is not an array"));
    for (const auto &P : PrintfMD->second.getArray()) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          P.isString(), "Printf entry is not a string"));
      Out->Printf->emplace_back(P.getString());
    }
  }
  return Out;
}

llvm::Expected<llvm::StringMap<std::unique_ptr<Kernel::Metadata>>>
MetadataParser::parseAllKernelsMetadata(llvm::msgpack::Document &Doc) const {
  llvm::StringMap<std::unique_ptr<Kernel::Metadata>> Out;
  DocNode &MetaDataRoot = Doc.getRoot();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      MetaDataRoot.isMap(), "The metadata doc is not a map"));
  auto &RootMap = MetaDataRoot.getMap();
  auto KernelsMD = RootMap.find(Key::Kernels);
  if (KernelsMD != RootMap.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        KernelsMD->second.isArray(),
        "The kernels entry inside the metadata is not an array"));
    auto &KernelsMDAsArray = KernelsMD->second.getArray();
    for (auto &KernelMD : KernelsMDAsArray) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          KernelMD.isMap(), "The kernel metadata entry is not a map"));

      auto &KernelMDAsMap = KernelMD.getMap();
      auto SymbolMD = KernelMDAsMap.find(Kernel::Key::Symbol);
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          SymbolMD != KernelMDAsMap.end(),
          "Failed to find the symbol name of the kernel inside the metadata"));
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          SymbolMD->second.isString(),
          "Metadata name of the kernel is not string"));
      auto &KMD = *Out.insert({SymbolMD->second.getString(),
                               std::make_unique<Kernel::Metadata>()})
                       .first->second;
      LUTHIER_RETURN_ON_ERROR(parseKernelMD(
          KernelMDAsMap, AccessQualifierEnumMap, AMDGPUAddressSpaceEnumMap,
          ValueKindEnumMap, KernelKindEnumMap, KMD));
    }
  }
  return Out;
}

llvm::Expected<std::unique_ptr<Kernel::Metadata>>
MetadataParser::parseKernelMetadata(llvm::msgpack::Document &Doc,
                                    llvm::StringRef KernelName) const {
   auto Out = std::make_unique<Kernel::Metadata>();
  DocNode &MetaDataRoot = Doc.getRoot();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      MetaDataRoot.isMap(), "The metadata doc is not a map"));
  auto &RootMap = MetaDataRoot.getMap();
  auto KernelsMD = RootMap.find(Key::Kernels);
  if (KernelsMD != RootMap.end()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        KernelsMD->second.isArray(),
        "The kernels entry inside the metadata is not an array"));
    auto &KernelsMDAsArray = KernelsMD->second.getArray();
    for (auto &KernelMD : KernelsMDAsArray) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          KernelMD.isMap(), "The kernel metadata entry is not a map"));

      auto &KernelMDAsMap = KernelMD.getMap();
      auto SymbolMD = KernelMDAsMap.find(Kernel::Key::Symbol);
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          SymbolMD != KernelMDAsMap.end(),
          "Failed to find the symbol name of the kernel inside the metadata"));
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          SymbolMD->second.isString(),
          "Metadata name of the kernel is not string"));
      if (KernelName == SymbolMD->second.getString()) {
        LUTHIER_RETURN_ON_ERROR(parseKernelMD(
            KernelMDAsMap, AccessQualifierEnumMap, AMDGPUAddressSpaceEnumMap,
            ValueKindEnumMap, KernelKindEnumMap, *Out));
        return Out;
      }
    }
  }
  return Out;
}
} // namespace amdgpu::hsamd

} // namespace luthier