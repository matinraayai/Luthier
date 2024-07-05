/**
 * \file object_utils.hpp
 * \brief contains all operations related to dealing with parsing and processing
 * AMDGPU code objects using LLVM object file and DWARF utilities
 *
 * Luthier uses LLVM's object library (under llvm/Object folder in LLVM)
 * to parse and inspect AMDGPU code objects, and if present, uses LLVM's
 * DebugInfo library (under llvm/DebugInfo) to parse and process DWARF
 * information from them.
 *
 * object_utils.hpp is meant to \b only include functionality that:
 * - concerns ELF object files and ELF file section parsing and processing,
 *   including DWARF debug information.
 * - is specific to AMDGPU GCN code objects. Some examples include
 *   parsing an AMDGCN object file and converting them to
 *   \p llvm::object::ELF64LEObjectFile, or parsing the note section
 *   of an AMDGPU code object into a \c luthier::hsa::md::Metadata.
 * - does not exist in LLVM's object library, and/or is implemented in other
 *   LLVM-based tools or project. Some examples include retrieving symbols by
 *   name, or getting the loaded address of a symbol.
 *
 * Although not strictly restricted for this specific purpose, object_utils.hpp
 * is only used to supplement ROCr functionality, by parsing the Storage
 * memory ELF of an \c luthier::hsa::LoadedCodeObject, which is exposed in
 * hsa wrapper primitives in the \c luthier::hsa namespace.
 *
 */
#ifndef OBJECT_UTILS_HPP
#define OBJECT_UTILS_HPP
#include <hip/hip_runtime_api.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/BinaryFormat/MsgPackDocument.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Object/ELFTypes.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/AMDGPUAddrSpace.h>

#include <map>
#include <optional>
#include <utility>
#include <common/error.hpp>
#include <luthier/types.h>

namespace llvm::AMDGPU::HSAMD {

enum class AccessQualifier : uint8_t;
}

namespace luthier {

/**
 * As per <a href="https://llvm.org/docs/AMDGPUUsage.html#elf-code-object">
 * AMDGPU backend documentation</a>, AMDGCN object files are 64-bit LSB.
 * Luthier does not support the R600 target, hence it is safe to assume for now
 * all ELF object files encountered by Luthier are of this type.
 */
typedef llvm::object::ELF64LEObjectFile AMDGCNObjectFile;

/**
 * Parses the ELF file pointed to by \p Elf into a \c AMDGCNObjectFile.
 * \param Elf \c llvm::StringRef encompassing the ELF file in memory
 * \return a \c std::unique_ptr<llvm::object::ELF64LEObjectFile> on successful
 * parsing, an \c llvm::Error on failure
 */
llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
getAMDGCNObjectFile(llvm::StringRef Elf);

/**
 * Parses the ELF file pointed to by \b Elf into a \b AMDGCNObjectFile.
 * \param Elf \p llvm::ArrayRef<uint8_t> encompassing the ELF file in memory
 * \return a \c std::unique_ptr<llvm::object::ELF64LEObjectFile> on successful
 * parsing, an \c llvm::Error on failure
 */
llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
getAMDGCNObjectFile(llvm::ArrayRef<uint8_t> Elf);

/**
 * Looks up a symbol by its name in the given ELF from its symbol hash table.
 * \param Elf
 * \param SymbolName
 * \return
 */
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
lookupSymbolByName(const luthier::AMDGCNObjectFile &Elf,
                   llvm::StringRef SymbolName);

template <class ELFT>
llvm::Expected<uint64_t> getSectionLMA(const llvm::object::ELFFile<ELFT> &Obj,
                                       const llvm::object::ELFSectionRef &Sec) {
  auto PhdrRange = Obj.program_headers();
  LUTHIER_RETURN_ON_ERROR(PhdrRange.takeError());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRange)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<ELFT>(
            Phdr, *llvm::cast<const llvm::object::ELFObjectFile<ELFT>>(
                       Sec.getObject())
                       ->getSection(Sec.getRawDataRefImpl()))))
      return Sec.getAddress() - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return Sec.getAddress();
}

template <class ELFT>
llvm::Expected<uint64_t> getSymbolLMA(const llvm::object::ELFFile<ELFT> &Obj,
                                      const llvm::object::ELFSymbolRef &Sym) {
  auto PhdrRange = Obj.program_headers();
  LUTHIER_RETURN_ON_ERROR(PhdrRange.takeError());

  auto SymbolSection = Sym.getSection();
  LUTHIER_RETURN_ON_ERROR(SymbolSection.takeError());

  auto SymbolAddress = Sym.getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolAddress.takeError());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRange)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<ELFT>(
            Phdr, *llvm::cast<const llvm::object::ELFObjectFile<ELFT>>(
                       Sym.getObject())
                       ->getSection(SymbolSection.get()->getRawDataRefImpl()))))
      return *SymbolAddress - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return *SymbolAddress;
}

template <typename ELFT>
llvm::Expected<std::tuple<llvm::Triple, llvm::StringRef, llvm::SubtargetFeatures>>
getELFObjectFileISA(const llvm::object::ELFObjectFile<ELFT> &Obj) {
  llvm::Triple TT = Obj.makeTriple();
  std::optional<llvm::StringRef> CPU = Obj.tryGetCPUName();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CPU.has_value()));
  llvm::SubtargetFeatures Features;
  LUTHIER_RETURN_ON_ERROR(Obj.getFeatures().moveInto(Features));
  return std::make_tuple(TT, *CPU, Features);
}

namespace hsa::md {

struct Version {
  uint64_t Major{0};
  uint64_t Minor{0};
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
  hsa::md::ValueKind ValueKind{ValueKind::Unknown};
  /// Pointee alignment in bytes. Optional.
  std::optional<uint32_t> PointeeAlign{0};
  /// Address space qualifier. Optional.
  std::optional<unsigned> AddressSpace{std::nullopt};
  /// Access qualifier. Optional.
  std::optional<llvm::AMDGPU::HSAMD::AccessQualifier> AccQual{std::nullopt};
  /// Actual access qualifier. Optional.
  std::optional<llvm::AMDGPU::HSAMD::AccessQualifier> ActualAccQual{
      std::nullopt};
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
  uint32_t AGPRCount{};
  /// Maximum flat work-group size supported by the kernel. Optional.
  uint32_t MaxFlatWorkgroupSize{};
  /// Number of SGPRs spilled by a wavefront. Optional.
  std::optional<uint32_t> SGPRSpillCount{0};
  /// Number of VGPRs spilled by a workitem. Optional.
  std::optional<uint32_t> VGPRSpillCount{0};
  /// The kind of the kernel
  std::optional<hsa::md::KernelKind> KernelKind{KernelKind::Normal};
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
  hsa::md::Version Version;
  /// Printf metadata. Optional.
  std::optional<std::vector<std::string>> Printf{std::nullopt};
  /// Kernels metadata. Required.
  std::vector<Kernel::Metadata> Kernels{};
  /// Default constructor.
  Metadata() = default;
};

}; // namespace hsa::md

llvm::Expected<hsa::md::Metadata>
parseMetaDoc(llvm::msgpack::Document &KernelMetaNode);

template <class ELFT>
static bool
processNote(const typename ELFT::Note &Note, const std::string &NoteDescString,
            llvm::msgpack::Document &Doc, llvm::msgpack::DocNode &Root) {

  if (Note.getName() == "AMD" &&
      Note.getType() == llvm::ELF::NT_AMD_HSA_METADATA) {
    if (!Root.isEmpty()) {
      return false;
    }
    if (!Doc.fromYAML(NoteDescString)) {
      return false;
    }
    return true;
  }
  if (((Note.getName() == "AMD" || Note.getName() == "AMDGPU") &&
       Note.getType() == llvm::ELF::NT_AMD_PAL_METADATA) ||
      (Note.getName() == "AMDGPU" &&
       Note.getType() == llvm::ELF::NT_AMDGPU_METADATA)) {
    if (!Doc.readFromBlob(NoteDescString, false)) {
      return false;
    }
    return true;
  }
  return false;
}

template <typename ELFT>
llvm::Expected<hsa::md::Metadata>
parseNoteMetaData(const llvm::object::ELFObjectFile<ELFT> &Obj) {
  bool Found = false;
  llvm::msgpack::Document Doc;
  auto &Root = Doc.getRoot();
  const llvm::object::ELFFile<ELFT> &ELFFile = Obj.getELFFile();
  auto ProgramHeaders = ELFFile.program_headers();
  std::string DescString;
  LUTHIER_RETURN_ON_ERROR(ProgramHeaders.takeError());
  for (const auto &Phdr : *ProgramHeaders) {
    if (Phdr.p_type == llvm::ELF::PT_NOTE) {
      llvm::Error Err = llvm::Error::success();
      for (const auto &Note : ELFFile.notes(Phdr, Err)) {
        DescString = Note.getDescAsStringRef(4);
        if (processNote<ELFT>(Note, DescString, Doc, Root)) {
          //          Doc.getRoot() = Root;
          //          llvm::outs() << "Is Map? " << Doc.getRoot().isMap() <<
          //          "\n";
          Found = true;
        }
      }
      LUTHIER_RETURN_ON_ERROR(Err);
    }
  }
  if (Found) {
    //    llvm::outs() << "Is Map? " << Doc.getRoot().isMap() << "\n";
    return parseMetaDoc(Doc);
  }

  auto Sections = ELFFile.sections();
  LUTHIER_RETURN_ON_ERROR(Sections.takeError());

  for (const auto &Shdr : *Sections) {
    if (Shdr.sh_type != llvm::ELF::SHT_NOTE) {
      continue;
    }
    llvm::Error Err = llvm::Error::success();
    for (const auto &Note : ELFFile.notes(Shdr, Err)) {
      DescString = Note.getDescAsStringRef(4);
      if (processNote<ELFT>(Note, DescString, Doc, Root)) {
        Found = true;
      }
    }
    LUTHIER_RETURN_ON_ERROR(Err);
  }

  if (Found)
    return parseMetaDoc(Doc);
  else
    return LUTHIER_ASSERTION(Found);
}

} // namespace luthier

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const luthier::hsa::md::Kernel::Metadata &MD);

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const luthier::hsa::md::Metadata &MD);

#endif