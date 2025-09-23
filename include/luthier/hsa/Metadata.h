//===-- Metadata.h - Metadata Struct for HSA COV3+ --------------*- C++ -*-===//
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
/// Defines the Metadata structs and enums used by Luthier to
/// parse the code object metadata into an easy-access form, as well as an
/// analysis pass for parsing the metadata during lifting.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_METADATA_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_METADATA_H
#include <llvm/ADT/StringMap.h>
#include <llvm/BinaryFormat/MsgPackDocument.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <optional>
#include <string>
#include <vector>

namespace luthier {
namespace amdgpu::hsamd {

struct Version {
  uint64_t Major{0};
  uint64_t Minor{0};
};

struct dim3 {
  uint32_t X;
  uint32_t Y;
  uint32_t Z;
};

enum class AccessQualifier : uint8_t {
  Default = 0,
  ReadOnly = 1,
  WriteOnly = 2,
  ReadWrite = 3,
  Unknown = 0xff
};

/// Value kinds.
enum class ValueKind : uint8_t {
  ByValue = 0,
  GlobalBuffer = 1,
  DynamicSharedPointer = 2,
  Sampler = 3,
  Image = 4,
  Pipe = 5,
  Queue = 6,
  HiddenGlobalOffsetX = 7,
  HiddenArgKindBegin = HiddenGlobalOffsetX,
  HiddenGlobalOffsetY = 8,
  HiddenGlobalOffsetZ = 9,
  HiddenNone = 10,
  HiddenPrintfBuffer = 11,
  HiddenHostcallBuffer = 12,
  HiddenDefaultQueue = 13,
  HiddenCompletionAction = 14,
  HiddenMultiGridSyncArg = 15,
  HiddenBlockCountX = 16,
  HiddenBlockCountY = 17,
  HiddenBlockCountZ = 18,
  HiddenGroupSizeX = 19,
  HiddenGroupSizeY = 20,
  HiddenGroupSizeZ = 21,
  HiddenRemainderX = 22,
  HiddenRemainderY = 23,
  HiddenRemainderZ = 24,
  HiddenGridDims = 25,
  HiddenHeapV1 = 26,
  HiddenDynamicLDSSize = 27,
  HiddenPrivateBase = 28,
  HiddenSharedBase = 29,
  HiddenQueuePtr = 30,
  HiddenArgKindEnd = HiddenQueuePtr,
  LuthierHiddenStackPtr = 31,
  /// The Luthier hidden args are only specific to Luthier and have nothing
  /// to do with the HSA Code Object Metadata standards
  LuthierHiddenArgBegin = LuthierHiddenStackPtr,
  LuthierHiddenUserArgOffset = 32,
  LuthierHiddenArgEnd = LuthierHiddenUserArgOffset,
  Unknown = 0xff
};

enum class KernelKind : uint8_t { Normal, Init, Fini };

//===----------------------------------------------------------------------===//
// Kernel Metadata.
//===----------------------------------------------------------------------===//
namespace Kernel {

//===----------------------------------------------------------------------===//
// Kernel Argument Metadata.
//===----------------------------------------------------------------------===//
namespace Arg {

namespace Key {
/// Key for Kernel::Arg::Metadata::mName.
constexpr char Name[] = ".name";
/// Key for Kernel::Arg::Metadata::mTypeName.
constexpr char TypeName[] = ".type_name";
/// Key for Kernel::Arg::Metadata::mSize.
constexpr char Size[] = ".size";
/// Key for Kernel::Arg::Metadata::mOffset.
constexpr char Offset[] = ".offset";
/// Key for Kernel::Arg::Metadata::mValueKind.
constexpr char ValueKind[] = ".value_kind";
/// Key for Kernel::Arg::Metadata::mPointeeAlign.
constexpr char PointeeAlign[] = ".pointee_align";
/// Key for Kernel::Arg::Metadata::mAddrSpaceQual.
constexpr char AddressSpace[] = ".address_space";
/// Key for Kernel::Arg::Metadata::mAccQual.
constexpr char AccQual[] = ".access";
/// Key for Kernel::Arg::Metadata::mActualAccQual.
constexpr char ActualAccQual[] = ".actual_access";
/// Key for Kernel::Arg::Metadata::mIsConst.
constexpr char IsConst[] = ".is_const";
/// Key for Kernel::Arg::Metadata::mIsRestrict.
constexpr char IsRestrict[] = "is_restrict";
/// Key for Kernel::Arg::Metadata::mIsVolatile.
constexpr char IsVolatile[] = "is_volatile";
/// Key for Kernel::Arg::Metadata::mIsPipe.
constexpr char IsPipe[] = "is_pipe";
} // end namespace Key

/// In-memory representation of kernel argument metadata.
struct Metadata final {
  /// Name. Optional.
  std::optional<std::string> Name{std::nullopt};
  /// Type name. Optional.
  std::optional<std::string> TypeName{std::nullopt};
  /// Size in bytes. Required.
  uint32_t Size{0};
  /// Offset in bytes. Required for code object v3, unused for code object v2.
  uint32_t Offset{0};
  /// Value kind. Required.
  ValueKind ValueKind{ValueKind::Unknown};
  /// Pointee alignment in bytes. Optional.
  std::optional<uint32_t> PointeeAlign{0};
  /// Address space qualifier. Optional.
  std::optional<unsigned> AddressSpace{std::nullopt};
  /// Access qualifier. Optional.
  std::optional<AccessQualifier> AccQual{std::nullopt};
  /// Actual access qualifier. Optional.
  std::optional<AccessQualifier> ActualAccQual{std::nullopt};
  /// True if 'const' qualifier is specified. Optional.
  bool IsConst{false};
  /// True if 'restrict' qualifier is specified. Optional.
  bool IsRestrict{false};
  /// True if 'volatile' qualifier is specified. Optional.
  bool IsVolatile{false};
  /// True if 'pipe' qualifier is specified. Optional.
  bool IsPipe{false};

  /// Default constructor.
  Metadata() = default;
};

} // end namespace Arg

namespace Key {
/// Key for Kernel::Metadata::Name.
constexpr char Name[] = ".name";
/// Key for Kernel::Metadata::SymbolName.
constexpr char Symbol[] = ".symbol";
/// Key for Kernel::Metadata::Language.
constexpr char Language[] = ".language";
/// Key for Kernel::Metadata::LanguageVersion.
constexpr char LanguageVersion[] = ".language_version";
/// Key for Kernel::Metadata::Args.
constexpr char Args[] = ".args";

constexpr char ReqdWorkGroupSize[] = ".reqd_workgroup_size";

constexpr char WorkGroupSizeHint[] = ".workgroup_size_hint";

constexpr char VecTypeHint[] = ".vec_type_hint";

constexpr char DeviceEnqueueSymbol[] = ".device_enqueue_symbol";

constexpr char KernArgSegmentSize[] = ".kernarg_segment_size";

constexpr char GroupSegmentFixedSize[] = ".group_segment_fixed_size";

constexpr char PrivateSegmentFixedSize[] = ".private_segment_fixed_size";

constexpr char KernArgSegmentAlign[] = ".kernarg_segment_align";

constexpr char WaveFrontSize[] = ".wavefront_size";

constexpr char SGPRCount[] = ".sgpr_count";

constexpr char VGPRCount[] = ".vgpr_count";

constexpr char AGPRCount[] = ".agpr_count";

constexpr char MaxFlatWorkgroupSize[] = ".max_flat_workgroup_size";

constexpr char SGPRSpillCount[] = ".sgpr_spill_count";

constexpr char VGPRSpillCount[] = ".vgpr_spill_count";

constexpr char KernelKind[] = ".kind";

constexpr char UsesDynamicStack[] = ".uses_dynamic_stack";

constexpr char WorkgroupProcessorMode[] = ".workgroup_processor_mode";

constexpr char UniformWorkgroupSize[] = ".uniform_work_group_size";

} // end namespace Key

/// In-memory representation of kernel metadata.
struct Metadata final {
  /// Kernel source name. Required.
  std::string Name{};
  /// Kernel descriptor name. Required.
  std::string Symbol{};
  /// Language. Optional.
  std::optional<std::string> Language{std::nullopt};
  /// Language version. Optional.
  std::optional<Version> LanguageVersion{std::nullopt};
  /// Arguments metadata. Optional.
  std::optional<std::vector<Arg::Metadata>> Args{std::nullopt};
  /// 'reqd_work_group_size' attribute. Optional.
  std::optional<dim3> ReqdWorkGroupSize{std::nullopt};
  /// 'work_group_size_hint' attribute. Optional.
  std::optional<dim3> WorkGroupSizeHint{std::nullopt};
  /// 'vec_type_hint' attribute. Optional.
  std::optional<std::string> VecTypeHint{std::nullopt};
  /// External symbol created by runtime to store the kernel address
  /// for enqueued blocks.
  std::optional<std::string> DeviceEnqueueSymbol{std::nullopt};
  /// Size in bytes of the kernarg segment memory. Kernarg segment memory
  /// holds the values of the arguments to the kernel. Required.
  uint32_t KernArgSegmentSize{};
  /// Size in bytes of the group segment memory required by a workgroup.
  /// This value does not include any dynamically allocated group segment memory
  /// that may be added when the kernel is dispatched. Required.
  uint32_t GroupSegmentFixedSize{};
  /// Size in bytes of the private segment memory required by a workitem.
  /// Private segment memory includes arg, spill and private segments. Required.
  uint32_t PrivateSegmentFixedSize{};
  /// Maximum byte alignment of variables used by the kernel in the
  /// kernarg memory segment. Required.
  uint32_t KernArgSegmentAlign{};
  /// Wavefront size. Required.
  uint32_t WaveFrontSize{};
  /// Total number of SGPRs used by a wavefront. Optional.
  uint32_t SGPRCount{};
  /// Total number of VGPRs used by a workitem. Optional.
  uint32_t VGPRCount{};
  /// Total number of AGPRs required by each workitem for GFX90A, GFX908.
  std::optional<uint32_t> AGPRCount{};
  /// Maximum flat work-group size supported by the kernel. Optional.
  uint32_t MaxFlatWorkgroupSize{};
  /// Number of SGPRs spilled by a wavefront. Optional.
  std::optional<uint32_t> SGPRSpillCount{0};
  /// Number of VGPRs spilled by a workitem. Optional.
  std::optional<uint32_t> VGPRSpillCount{0};
  /// The kind of the kernel
  std::optional<hsamd::KernelKind> KernelKind{KernelKind::Normal};
  /// Indicates if the generated kernel machine code is using a
  /// dynamically sized stack.
  bool UsesDynamicStack{false};
  /// (GFX10+) Controls ENABLE_WGP_MODE in Code Object V3 Kernel Descriptor.
  /// Defaults to true if cumode is disabled
  bool WorkgroupProcessorMode{true};
  /// Indicates if the kernel requires that each dimension of global size
  /// is a multiple of corresponding dimension of work-group size.
  /// Only emitted when value is 1.
  bool UniformWorkgroupSize{false};

  /// Default constructor.
  Metadata() = default;
};

} // end namespace Kernel

namespace Key {
/// Key for hsa::Metadata::Version.
constexpr char Version[] = "amdhsa.version";
/// Key for hsa::Metadata::Printf.
constexpr char Printf[] = "amdhsa.printf";
/// Key for hsa::Metadata::Kernels.
constexpr char Kernels[] = "amdhsa.kernels";
} // end namespace Key

/// In-memory representation of HSA metadata.
struct Metadata final {
  /// HSA metadata version. Required.
  amdgpu::hsamd::Version Version;
  /// Printf metadata. Optional.
  std::optional<std::vector<std::string>> Printf{std::nullopt};
  /// Default constructor.
  Metadata() = default;
};

/// \brief Class in charge of parsing the HSA code object metadata
/// \details We opt to make this a class due to having to define a mapping
/// between the string name and the enums of the access qualifiers, address
/// spaces, argument value kinds, and kernel kinds. Instead of defining these
/// mappings inside a set of maps with static lifetime, we instead
/// follow the general design paradigm of Luthier and put them inside an
/// object here so that we can more control over their lifetime.
class MetadataParser {
private:
  const llvm::StringMap<AccessQualifier> AccessQualifierEnumMap = {
      {"read_only", AccessQualifier::ReadOnly},
      {"write_only", AccessQualifier::WriteOnly},
      {"read_write", AccessQualifier::ReadWrite}};

  const llvm::StringMap<unsigned> AMDGPUAddressSpaceEnumMap = {
      {"private", llvm::AMDGPUAS::PRIVATE_ADDRESS},
      {"global", llvm::AMDGPUAS::GLOBAL_ADDRESS},
      {"constant", llvm::AMDGPUAS::CONSTANT_ADDRESS},
      {"local", llvm::AMDGPUAS::LOCAL_ADDRESS},
      {"generic", llvm::AMDGPUAS::FLAT_ADDRESS},
      {"region", llvm::AMDGPUAS::REGION_ADDRESS}};

  const llvm::StringMap<ValueKind> ValueKindEnumMap = {
      {"by_value", ValueKind::ByValue},
      {"global_buffer", ValueKind::GlobalBuffer},
      {"dynamic_shared_pointer", ValueKind::DynamicSharedPointer},
      {"sampler", ValueKind::Sampler},
      {"image", ValueKind::Image},
      {"pipe", ValueKind::Pipe},
      {"queue", ValueKind::Queue},
      {"hidden_global_offset_x", ValueKind::HiddenGlobalOffsetX},
      {"hidden_global_offset_y", ValueKind::HiddenGlobalOffsetY},
      {"hidden_global_offset_z", ValueKind::HiddenGlobalOffsetZ},
      {"hidden_none", ValueKind::HiddenNone},
      {"hidden_printf_buffer", ValueKind::HiddenPrintfBuffer},
      {"hidden_hostcall_buffer", ValueKind::HiddenHostcallBuffer},
      {"hidden_default_queue", ValueKind::HiddenDefaultQueue},
      {"hidden_completion_action", ValueKind::HiddenCompletionAction},
      {"hidden_multigrid_sync_arg", ValueKind::HiddenMultiGridSyncArg},
      {"hidden_block_count_x", ValueKind::HiddenBlockCountX},
      {"hidden_block_count_y", ValueKind::HiddenBlockCountY},
      {"hidden_block_count_z", ValueKind::HiddenBlockCountZ},
      {"hidden_group_size_x", ValueKind::HiddenGroupSizeX},
      {"hidden_group_size_y", ValueKind::HiddenGroupSizeY},
      {"hidden_group_size_z", ValueKind::HiddenGroupSizeZ},
      {"hidden_remainder_x", ValueKind::HiddenRemainderX},
      {"hidden_remainder_y", ValueKind::HiddenRemainderY},
      {"hidden_remainder_z", ValueKind::HiddenRemainderZ},
      {"hidden_grid_dims", ValueKind::HiddenGridDims},
      {"hidden_heap_v1", ValueKind::HiddenHeapV1},
      {"hidden_dynamic_lds_size", ValueKind::HiddenDynamicLDSSize},
      {"hidden_private_base", ValueKind::HiddenPrivateBase},
      {"hidden_shared_base", ValueKind::HiddenSharedBase},
      {"hidden_queue_ptr", ValueKind::HiddenQueuePtr}};

  const llvm::StringMap<KernelKind> KernelKindEnumMap = {
      {"normal", KernelKind::Normal},
      {"init", KernelKind::Init},
      {"fini", KernelKind::Fini}};

public:
  MetadataParser() = default;

  /// Parses the high-level metadata included in the \p Doc
  /// Does not parse the kernels portion of the metadata
  llvm::Expected<std::unique_ptr<Metadata>>
  parseNoteMetaData(llvm::msgpack::Document &Doc) const;

  /// Parses the all the kernels metadata included in the \p Doc
  llvm::Expected<llvm::StringMap<std::unique_ptr<Kernel::Metadata>>>
  parseAllKernelsMetadata(llvm::msgpack::Document &Doc) const;

  /// Parses the kernel metadata of \p KernelName in \p Doc
  llvm::Expected<std::unique_ptr<Kernel::Metadata>>
  parseKernelMetadata(llvm::msgpack::Document &Doc,
                      llvm::StringRef KernelName) const;
};

}; // namespace amdgpu::hsamd

} // namespace luthier
#endif