#ifndef AMDGPU_ELF_HPP
#define AMDGPU_ELF_HPP
#include <llvm/ADT/ArrayRef.h>
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/MemoryBuffer.h>

#include <map>
#include <optional>
#include <utility>

#include "luthier_types.h"

namespace luthier {

/**
 * Returns the demangled name of the input symbol name
 * @param mangledName mangled name string
 * @return demangled name as std::string
 */
std::string getDemangledName(const std::string &mangledName);

typedef enum {
    LLVMIR = 0,
    SOURCE,
    ILTEXT,
    ASTEXT,
    CAL,
    DLL,
    STRTAB,
    SYMTAB,
    RODATA,
    SHSTRTAB,
    NOTES,
    COMMENT,
    ILDEBUG,
    DEBUG_INFO,
    DEBUG_ABBREV,
    DEBUG_LINE,
    DEBUG_PUBNAMES,
    DEBUG_PUBTYPES,
    DEBUG_LOC,
    DEBUG_ARANGES,
    DEBUG_RANGES,
    DEBUG_MACINFO,
    DEBUG_STR,
    DEBUG_FRAME,
    JITBINARY,
    CODEGEN,
    TEXT,
    INTERNAL,
    SPIR,
    SPIRV,
    RUNTIME_METADATA,
    ELF_SECTIONS_LAST = RUNTIME_METADATA
} ElfSections;

typedef struct {
    ElfSections id;
    const char *name;
    uint64_t d_align;   // section alignment in bytes
    llvm::ELF::Elf32_Word sh_type; // section type
    llvm::ELF::Elf32_Word sh_flags;// section flags
    const char *desc;
} ElfSectionsDesc;

namespace {
// Objects that are visible only within this module
constexpr ElfSectionsDesc ElfSecDesc[] = {
    {LLVMIR, ".llvmir", 1, llvm::ELF::SHT_PROGBITS, 0, "ASIC-independent LLVM IR"},
    {SOURCE, ".source", 1, llvm::ELF::SHT_PROGBITS, 0, "OpenCL source"},
    {ILTEXT, ".amdil", 1, llvm::ELF::SHT_PROGBITS, 0, "AMD IL text"},
    {ASTEXT, ".astext", 1, llvm::ELF::SHT_PROGBITS, 0, "X86 assembly text"},
    {CAL, ".text", 1, llvm::ELF::SHT_PROGBITS, llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR, "AMD CalImage"},
    {DLL, ".text", 1, llvm::ELF::SHT_PROGBITS, llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR, "x86 dll"},
    {STRTAB, ".strtab", 1, llvm::ELF::SHT_STRTAB, llvm::ELF::SHF_STRINGS, "String table"},
    {SYMTAB, ".symtab", sizeof(llvm::ELF::Elf64_Xword), llvm::ELF::SHT_SYMTAB, 0, "Symbol table"},
    {RODATA, ".rodata", 1, llvm::ELF::SHT_PROGBITS, llvm::ELF::SHF_ALLOC, "Read-only data"},
    {SHSTRTAB, ".shstrtab", 1, llvm::ELF::SHT_STRTAB, llvm::ELF::SHF_STRINGS, "Section names"},
    {NOTES, ".note", 1, llvm::ELF::SHT_NOTE, 0, "used by loader for notes"},
    {COMMENT, ".comment", 1, llvm::ELF::SHT_PROGBITS, 0, "Version string"},
    {ILDEBUG, ".debugil", 1, llvm::ELF::SHT_PROGBITS, 0, "AMD Debug IL"},
    {DEBUG_INFO, ".debug_info", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug info"},
    {DEBUG_ABBREV, ".debug_abbrev", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug abbrev"},
    {DEBUG_LINE, ".debug_line", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug line"},
    {DEBUG_PUBNAMES, ".debug_pubnames", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug pubnames"},
    {DEBUG_PUBTYPES, ".debug_pubtypes", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug pubtypes"},
    {DEBUG_LOC, ".debug_loc", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug loc"},
    {DEBUG_ARANGES, ".debug_aranges", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug aranges"},
    {DEBUG_RANGES, ".debug_ranges", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug ranges"},
    {DEBUG_MACINFO, ".debug_macinfo", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug macinfo"},
    {DEBUG_STR, ".debug_str", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug str"},
    {DEBUG_FRAME, ".debug_frame", 1, llvm::ELF::SHT_PROGBITS, 0, "Dwarf debug frame"},
    {JITBINARY, ".text", 1, llvm::ELF::SHT_PROGBITS, llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR, "x86 JIT Binary"},
    {CODEGEN, ".cg", 1, llvm::ELF::SHT_PROGBITS, 0, "Target dependent IL"},
    {TEXT, ".text", 1, llvm::ELF::SHT_PROGBITS, llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR, "Device specific ISA"},
    {INTERNAL, ".internal", 1, llvm::ELF::SHT_PROGBITS, 0, "Internal usage"},
    {SPIR, ".spir", 1, llvm::ELF::SHT_PROGBITS, 0, "Vendor/Device-independent LLVM IR"},
    {SPIRV, ".spirv", 1, llvm::ELF::SHT_PROGBITS, 0, "SPIR-V Binary"},
    {RUNTIME_METADATA, ".AMDGPU.runtime_metadata", 1, llvm::ELF::SHT_PROGBITS, 0, "AMDGPU runtime metadata"},
};
}// namespace

typedef enum clk_value_type_t {
    T_VOID,
    T_CHAR,
    T_SHORT,
    T_INT,
    T_LONG,
    T_FLOAT,
    T_DOUBLE,
    T_POINTER,
    T_CHAR2,
    T_CHAR3,
    T_CHAR4,
    T_CHAR8,
    T_CHAR16,
    T_SHORT2,
    T_SHORT3,
    T_SHORT4,
    T_SHORT8,
    T_SHORT16,
    T_INT2,
    T_INT3,
    T_INT4,
    T_INT8,
    T_INT16,
    T_LONG2,
    T_LONG3,
    T_LONG4,
    T_LONG8,
    T_LONG16,
    T_FLOAT2,
    T_FLOAT3,
    T_FLOAT4,
    T_FLOAT8,
    T_FLOAT16,
    T_DOUBLE2,
    T_DOUBLE3,
    T_DOUBLE4,
    T_DOUBLE8,
    T_DOUBLE16,
    T_SAMPLER,
    T_SEMA,
    T_STRUCT,
    T_QUEUE,
    T_PAD
} clk_value_type_t;

struct KernelParameterDescriptor {
    enum {
        Value = 0,
        MemoryObject = 1,
        ReferenceObject = 2,
        ValueObject = 3,
        ImageObject = 4,
        SamplerObject = 5,
        QueueObject = 6,
        HiddenNone = 7,
        HiddenGlobalOffsetX = 8,
        HiddenGlobalOffsetY = 9,
        HiddenGlobalOffsetZ = 10,
        HiddenPrintfBuffer = 11,
        HiddenDefaultQueue = 12,
        HiddenCompletionAction = 13,
        HiddenMultiGridSync = 14,
        HiddenHeap = 15,
        HiddenHostcallBuffer = 16,
        HiddenBlockCountX = 17,
        HiddenBlockCountY = 18,
        HiddenBlockCountZ = 19,
        HiddenGroupSizeX = 20,
        HiddenGroupSizeY = 21,
        HiddenGroupSizeZ = 22,
        HiddenRemainderX = 23,
        HiddenRemainderY = 24,
        HiddenRemainderZ = 25,
        HiddenGridDims = 26,
        HiddenPrivateBase = 27,
        HiddenSharedBase = 28,
        HiddenQueuePtr = 29,
        HiddenLast = 30
    };
    clk_value_type_t type_;//!< The parameter's type
    size_t offset_;        //!< Its offset in the parameter's stack
    size_t size_;          //!< Its size in bytes
    union InfoData {
        struct {
            uint32_t oclObject_ : 6;  //!< OCL object type
            uint32_t readOnly_ : 1;   //!< OCL object is read only, applied to memory only
            uint32_t rawPointer_ : 1; //!< Arguments have a raw GPU VA
            uint32_t defined_ : 1;    //!< The argument was defined by the app
            uint32_t hidden_ : 1;     //!< It's a hidden argument
            uint32_t shared_ : 1;     //!< Dynamic shared memory
            uint32_t reserved_ : 1;   //!< Reserved
            uint32_t arrayIndex_ : 20;//!< Index in the objects array or LDS alignment
        };
        uint32_t allValues_;
        InfoData() : allValues_(0) {}
    } info_;

    //    cl_kernel_arg_address_qualifier addressQualifier_ = CL_KERNEL_ARG_ADDRESS_PRIVATE;//!< Argument's address qualifier
    //    cl_kernel_arg_access_qualifier accessQualifier_ = CL_KERNEL_ARG_ACCESS_NONE;      //!< Argument's access qualifier
    //    cl_kernel_arg_type_qualifier typeQualifier_;                                      //!< Argument's type qualifier

    std::string name_;    //!< The parameter's name in the source
    std::string typeName_;//!< Argument's type name
    uint32_t alignment_;  //!< Argument's alignment
};

struct WorkGroupInfo {
    std::string symbolName_;
    std::vector<KernelParameterDescriptor> parameters_;
    std::vector<KernelParameterDescriptor> hiddenParameters_;
    uint32_t kernargSegmentByteSize_ = 0;//!< Size of kernel argument buffer
    size_t size_;                        //!< kernel workgroup size
    size_t compileSize_[3];              //!< kernel compiled workgroup size
    uint64_t localMemSize_;              //!< amount of used local memory
    size_t preferredSizeMultiple_;       //!< preferred multiple for launch
    uint64_t privateMemSize_;            //!< amount of used private memory
    size_t scratchRegs_;                 //!< amount of used scratch registers
    size_t wavefrontPerSIMD_;            //!< number of wavefronts per SIMD
    size_t wavefrontSize_;               //!< number of threads per wavefront
    size_t availableGPRs_;               //!< GPRs available to the program
    size_t usedGPRs_;                    //!< GPRs used by the program
    size_t availableSGPRs_;              //!< SGPRs available to the program
    size_t usedSGPRs_;                   //!< SGPRs used by the program
    size_t availableVGPRs_;              //!< VGPRs available to the program
    size_t usedVGPRs_;                   //!< VGPRs used by the program
    size_t availableLDSSize_;            //!< available LDS size
    size_t usedLDSSize_;                 //!< used LDS size
    size_t availableStackSize_;          //!< available stack size
    size_t usedStackSize_;               //!< used stack size
    size_t compileSizeHint_[3];          //!< kernel compiled workgroup size hint
    std::string compileVecTypeHint_;     //!< kernel compiled vector type hint
    bool uniformWorkGroupSize_;          //!< uniform work group size option
    size_t wavesPerSimdHint_;            //!< waves per simd hit
    int maxOccupancyPerCu_;              //!< Max occupancy per compute unit in threads
    size_t constMemSize_;                //!< size of user-allocated constant memory
    bool isWGPMode_;                     //!< kernel compiled in WGP/cumode
    uint32_t workgroupGroupSegmentByteSize_ = 0;
    uint32_t workitemPrivateSegmentByteSize_ = 0;
    uint32_t kernargSegmentAlignment_ = 0;
    std::string runtimeHandle_;//!< Runtime handle for context loader
    bool isDynamicCallStack_;
    bool isXNACKEnabled_;
    size_t numSpilledSGPRs_;
    size_t numSpilledVGPRs_;

    enum KernelKind { Normal = 0, Init = 1, Fini = 2 };

    KernelKind kind_{Normal};//!< Kernel kind, is normal unless specified otherwise
    union Flags {
        struct {
            uint imageEna_ : 1;          //!< Kernel uses images
            uint imageWriteEna_ : 1;     //!< Kernel uses image writes
            uint dynamicParallelism_ : 1;//!< Dynamic parallelism enabled
            uint internalKernel_ : 1;    //!< True: internal kernel
            uint hsa_ : 1;               //!< HSA kernel
        };
        uint value_;
        Flags() : value_(0) {}
    } flags_;

    void SetKernelKind(const std::string &kind) {
        kind_ = (kind == "init") ? Init : ((kind == "fini") ? Fini : Normal);
    }
};

llvm::Expected<std::unique_ptr<llvm::object::ELFObjectFileBase>> getELFObjectFileBase(llvm::ArrayRef<uint8_t> elf);
//class SymbolView;

///**
// * \briefs a non-owning read-only view of an AMDGPU ELF Code object located on the host
// */
//class ElfView : public std::enable_shared_from_this<ElfView> {
// public:
//    ElfView() = delete;
//
//    ~ElfView() {
//        //        if (kernelMetadataMap_.has_value()) {
//        //            for (auto &kMap: *kernelMetadataMap_) amd_comgr_destroy_metadata(kMap.second);
//        //        }
//        //        if (kernelsMetadata_.has_value()) amd_comgr_destroy_metadata(*kernelsMetadata_);
//    }
//
//    static std::shared_ptr<ElfView> makeView(byte_string_view elf) {
//        return std::shared_ptr<ElfView>(new ElfView(elf));
//    }
//
//    static std::shared_ptr<ElfView> makeView(const byte_string_t &elf) {
//        return std::shared_ptr<ElfView>(new ElfView(byte_string_view(elf)));
//    }
//    const ELFIO::elfio &getElfIo() const {
//        if (io_ == std::nullopt) {
//            io_.emplace();
//            // All elfio objects are loaded with lazy=true in ElfViewImpl to prevent additional memory copy
//            if (not io_->load(*dataStringStream_, true)) { llvm::report_fatal_error("Failed to load the ELF file."); }
//        }
//        return io_.value();
//    }
//
//    byte_string_view getView() const { return data_; }
//
//    unsigned int getNumSymbols();
//
//    std::optional<SymbolView> getSymbol(unsigned int index);
//
//    std::optional<SymbolView> getSymbol(const std::string &name);
//
//    uint32_t getCodeObjectVersion() const {
//        //        if (not codeObjectVer_.has_value()) initializeComgrMetaData();
//        return codeObjectVer_.value();
//    }
//
//    //    amd_comgr_metadata_node_t getComgrMetaData() const {
//    //        if (not metadata_.has_value()) initializeComgrMetaData();
//    //        return metadata_.value();
//    //    }
//
//    //    amd_comgr_metadata_node_t getKernelMetaDataMap(const std::string &kernelSymbolName) const {
//    //        if (not kernelMetadataMap_.has_value()) initializeComgrMetaData();
//    //        auto kdInSymbolName = kernelSymbolName.find(".kd");
//    //        auto key = kdInSymbolName != std::string::npos ? kernelSymbolName.substr(0, kdInSymbolName) : kernelSymbolName;
//    //        return kernelMetadataMap_.value().at(key);
//    //    }
//
//    //    WorkGroupInfo getAttrCodePropMetadata(amd_comgr_metadata_node_t kernelMetaNode);
//
// private:
//    explicit ElfView(byte_string_view elf);
//
//    //    amd_comgr_status_t initializeComgrMetaData() const;
//
//    mutable std::optional<ELFIO::elfio> io_{std::nullopt};
//    //    mutable std::optional<amd_comgr_data_t> comgrData_{std::nullopt};        //!< COMgr Data for the Executable ELF
//    //    mutable std::optional<amd_comgr_metadata_node_t> metadata_{std::nullopt};//!< COMgr metadata
//    mutable std::optional<uint32_t> codeObjectVer_{std::nullopt};//!< version of code object
//    //    mutable std::optional<amd_comgr_metadata_node_t> kernelsMetadata_{std::nullopt};
//    //    mutable std::optional<std::unordered_map<std::string, amd_comgr_metadata_node_t>> kernelMetadataMap_{std::nullopt};
//    const byte_string_view data_;
//    const std::unique_ptr<byte_char_stream_t> dataStringStream_;//! Used to construct the elfio object;
//                                                                //! Without keeping a reference to this stream,
//                                                                //! we cannot use the elfio in lazy mode
//};

//class SymbolView {
// private:
//    friend class ElfView;
//    const std::shared_ptr<ElfView> elf_;//!   section's parent elfio class
//    const ELFIO::section *section_;     //!   symbol's section
//    std::string name_;                  //!   symbol name
//    luthier_address_t address_;
//    size_t size_;
//    ELFIO::Elf64_Addr value_;//!   value of the symbol
//    unsigned char type_;     //!   type of the symbol
//
//    SymbolView(const std::shared_ptr<ElfView> &elf, const ELFIO::section *section, std::string name,
//               luthier_address_t address, size_t size, size_t value, unsigned char type)
//        : elf_(elf),
//          section_(section),
//          name_(std::move(name)),
//          address_(address),
//          size_(size),
//          value_(value),
//          type_(type){};
//
// public:
//    SymbolView() = delete;
//
//    [[nodiscard]] std::shared_ptr<ElfView> getElfView() const { return elf_; };
//
//    [[nodiscard]] const ELFIO::section *getSection() const { return section_; };
//
//    [[nodiscard]] const std::string &getName() const { return name_; };
//
//    [[nodiscard]] luthier_address_t getAddress() const { return address_; }
//
//    [[nodiscard]] byte_string_view getView() const {
//        return byte_string_view(reinterpret_cast<std::byte *>(address_), size_);
//    }
//
//    [[nodiscard]] size_t getSize() const { return size_; }
//
//    [[nodiscard]] ELFIO::Elf64_Addr getValue() const { return value_; };
//
//    [[nodiscard]] unsigned char getType() const { return type_; };
//
//    [[nodiscard]] const std::byte *getData() const {
//        return reinterpret_cast<const std::byte *>(section_->get_data() + (size_t) value_
//                                                   - (size_t) section_->get_offset());
//    }
//
//    //    [[nodiscard]] WorkGroupInfo getMetaData() const {
//    //        auto metadata = elf_->getKernelMetaDataMap(name_);
//    //        return elf_->getAttrCodePropMetadata(metadata);
//    //    }
//};

}// namespace luthier

#endif