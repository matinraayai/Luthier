/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef AMDGPU_CODE_OBJECT_MANIPULATION_HPP
#define AMDGPU_CODE_OBJECT_MANIPULATION_HPP

#include "luthier_types.hpp"
#include <amd_comgr/amd_comgr.h>
#include <any>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <elfio/elfio.hpp>
#include <llvm/BinaryFormat/MsgPackDocument.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/MemoryBuffer.h>
#include <CL/cl.h>
#include <map>
#include <optional>
#include <utility>

namespace boost_ios = boost::iostreams;

namespace luthier::co_manip {

/**
 * \brief a non-owning view of memory portion that contains AMDGPU code object bytes
 * Can be passed by value or reference and can be returned. Is trivially copyable.
 */
typedef std::basic_string_view<std::byte> code_view_t;

/**
 * \brief owns the memory portion that contains device code
 * Can be passed by reference only. Cannot be returned. Is not trivially copyable.
 */
typedef std::basic_string<std::byte> code_t;


inline std::string convertToString(const code_t& code) {
    return {reinterpret_cast<const char*>(code.data()), code.size()};
}

inline std::string_view convertToStringView(code_view_t code) {
    return {reinterpret_cast<const char*>(code.data()), code.size()};
}


/**
 * \briefs a non-owning read-only view of an AMDGPU ELF Code object located on the host
 */
class ElfViewImpl : public std::enable_shared_from_this<ElfViewImpl> {
 public:
    ElfViewImpl() = delete;

    ~ElfViewImpl() {
        if (kernelMetadataMap_.has_value()) {
            for (auto& kMap: *kernelMetadataMap_)
                amd_comgr_destroy_metadata(kMap.second);
        }
        if (kernelsMetadata_.has_value())
            amd_comgr_destroy_metadata(*kernelsMetadata_);
    }

    static std::shared_ptr<ElfViewImpl> makeView(code_view_t elf) {
        return std::shared_ptr<ElfViewImpl>(new ElfViewImpl(elf));
    }

    static std::shared_ptr<ElfViewImpl> makeView(const code_t &elf) {
        return std::shared_ptr<ElfViewImpl>(new ElfViewImpl(code_view_t(elf)));
    }
    const ELFIO::elfio &getElfIo() const {
        if (io_ == std::nullopt) {
            io_.emplace();
            // All elfio objects are loaded with lazy=true in ElfViewImpl to prevent additional memory copy
            if (not io_->load(*dataStringStream_, true)) {
                throw std::runtime_error("Failed to load the ELF file.");
            }
        }
        return io_.value();
    }

    code_view_t getView() const {
        return data_;
    }

    uint32_t getCodeObjectVersion() const {
        if (not codeObjectVer_.has_value())
            initializeComgrMetaData();
        return codeObjectVer_.value();
    }

    amd_comgr_metadata_node_t getComgrMetaData() const {
        if (not metadata_.has_value())
            initializeComgrMetaData();
        return metadata_.value();
    }

    amd_comgr_metadata_node_t getKernelMetaDataMap(const std::string& kernelSymbolName) const {
        if (not kernelMetadataMap_.has_value())
            initializeComgrMetaData();
        auto kdInSymbolName = kernelSymbolName.find(".kd");
        auto key = kdInSymbolName != std::string::npos ? kernelSymbolName.substr(0, kdInSymbolName) : kernelSymbolName;
        return kernelMetadataMap_.value().at(key);
    }


 private:
    explicit ElfViewImpl(code_view_t elf);

    amd_comgr_status_t initializeComgrMetaData() const;

    mutable std::optional<ELFIO::elfio> io_{std::nullopt};
    mutable std::optional<amd_comgr_data_t> comgrData_{std::nullopt};              //!< COMgr Data for the Executable ELF
    mutable std::optional<amd_comgr_metadata_node_t> metadata_{std::nullopt};               //!< COMgr metadata
    mutable std::optional<uint32_t> codeObjectVer_{std::nullopt};                  //!< version of code object
    mutable std::optional<amd_comgr_metadata_node_t> kernelsMetadata_{std::nullopt};
    mutable std::optional<std::unordered_map<std::string, amd_comgr_metadata_node_t>> kernelMetadataMap_{std::nullopt};
    const code_view_t data_;
    const std::unique_ptr<boost_ios::stream<boost_ios::basic_array_source<char>>> dataStringStream_;//! Used to construct the elfio object;
                                                                                                    //! Without keeping a reference to this stream,
                                                                                                    //! we cannot use the elfio in lazy mode
};

typedef std::shared_ptr<ElfViewImpl> ElfView;

/**
 * Factory method to construct an ElfView
 * To ensure correct passing of arguments of ElfView between different functions and scope management,
 * only shared pointers of ElfViewImpls are allowed to be constructed
 * \return an ElfView object, which is a std::shared_ptr of an ElfViewImpl
 */
//ElfView makeElfView(code_view_t elf) {
//    return ElfViewImpl::make_view(elf);
//}
//
//ElfView makeElfView(const code_t &elf) {
//    return ElfViewImpl::make_view(code_view_t(elf));
//}

class SymbolView {
 private:
    const ElfView elf_;            //!   section's parent elfio class
    const ELFIO::section *section_;//!   symbol's section
    std::string name_;             //!   symbol name
    code_view_t data_;             //!   symbol's raw data
    size_t value_;                 //!   value of the symbol
    unsigned char type_;           //!   type of the symbol
 public:
    SymbolView() = delete;

    SymbolView(const ElfView &elf, unsigned int symIndex);

    [[nodiscard]] ElfView getElfview() const {
        return elf_;
    };

    [[nodiscard]] const ELFIO::section *getSection() const {
        return section_;
    };

    [[nodiscard]] const std::string &getName() const {
        return name_;
    };
    [[nodiscard]] code_view_t getView() const {
        return data_;
    }

    [[nodiscard]] size_t getValue() const {
        return value_;
    };

    [[nodiscard]] unsigned char getType() const {
        return type_;
    };

    [[nodiscard]] const std::byte *getData() const {
        return reinterpret_cast<const std::byte *>(section_->get_data() + (size_t) value_ - (size_t) section_->get_offset());
    }
};


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

    cl_kernel_arg_address_qualifier addressQualifier_ =
        CL_KERNEL_ARG_ADDRESS_PRIVATE;//!< Argument's address qualifier
    cl_kernel_arg_access_qualifier accessQualifier_ =
        CL_KERNEL_ARG_ACCESS_NONE;              //!< Argument's access qualifier
    cl_kernel_arg_type_qualifier typeQualifier_;//!< Argument's type qualifier

    std::string name_;    //!< The parameter's name in the source
    std::string typeName_;//!< Argument's type name
    uint32_t alignment_;  //!< Argument's alignment
};

struct WorkGroupInfo {
    std::string symbolName_;
    std::vector<KernelParameterDescriptor> parameters_;
    std::vector<KernelParameterDescriptor> hiddenParameters_;
    uint32_t kernargSegmentByteSize_ = 0;   //!< Size of kernel argument buffer
    size_t size_;                     //!< kernel workgroup size
    size_t compileSize_[3];           //!< kernel compiled workgroup size
    uint64_t localMemSize_;           //!< amount of used local memory
    size_t preferredSizeMultiple_;    //!< preferred multiple for launch
    uint64_t privateMemSize_;         //!< amount of used private memory
    size_t scratchRegs_;              //!< amount of used scratch registers
    size_t wavefrontPerSIMD_;         //!< number of wavefronts per SIMD
    size_t wavefrontSize_;            //!< number of threads per wavefront
    size_t availableGPRs_;            //!< GPRs available to the program
    size_t usedGPRs_;                 //!< GPRs used by the program
    size_t availableSGPRs_;           //!< SGPRs available to the program
    size_t usedSGPRs_;                //!< SGPRs used by the program
    size_t availableVGPRs_;           //!< VGPRs available to the program
    size_t usedVGPRs_;                //!< VGPRs used by the program
    size_t availableLDSSize_;         //!< available LDS size
    size_t usedLDSSize_;              //!< used LDS size
    size_t availableStackSize_;       //!< available stack size
    size_t usedStackSize_;            //!< used stack size
    size_t compileSizeHint_[3];       //!< kernel compiled workgroup size hint
    std::string compileVecTypeHint_;  //!< kernel compiled vector type hint
    bool uniformWorkGroupSize_;       //!< uniform work group size option
    size_t wavesPerSimdHint_;         //!< waves per simd hit
    int maxOccupancyPerCu_;           //!< Max occupancy per compute unit in threads
    size_t constMemSize_;           //!< size of user-allocated constant memory
    bool isWGPMode_;                  //!< kernel compiled in WGP/cumode
    uint32_t workgroupGroupSegmentByteSize_ = 0;
    uint32_t workitemPrivateSegmentByteSize_ = 0;
    uint32_t kernargSegmentAlignment_ = 0;
    std::string runtimeHandle_;       //!< Runtime handle for context loader
    bool isDynamicCallStack_;
    bool isXNACKEnabled_;
    size_t numSpilledSGPRs_;
    size_t numSpilledVGPRs_;

    enum KernelKind{
        Normal = 0,
        Init   = 1,
        Fini   = 2
    };

    KernelKind kind_{Normal};  //!< Kernel kind, is normal unless specified otherwise
    union Flags {
        struct {
            uint imageEna_ : 1;           //!< Kernel uses images
            uint imageWriteEna_ : 1;      //!< Kernel uses image writes
            uint dynamicParallelism_ : 1; //!< Dynamic parallelism enabled
            uint internalKernel_ : 1;     //!< True: internal kernel
            uint hsa_ : 1;                //!< HSA kernel
        };
        uint value_;
        Flags() : value_(0) {}
    } flags_;




    void SetKernelKind(const std::string& kind) {
        kind_ = (kind == "init") ? Init : ((kind == "fini") ? Fini : Normal);
    }


};


WorkGroupInfo GetAttrCodePropMetadata(const ElfView& elfView, amd_comgr_metadata_node_t kernelMetaNode);




//struct Kernel {
// public:
//    typedef std::vector<KernelParameterDescriptor> parameters_t;
//
//    //! \struct The device kernel workgroup info structure
//
//
//    //! Default constructor
//    Kernel(const std::string& name);
//
//    //! Default destructor
//    virtual ~Kernel();
//
//    //! Returns the kernel info structure
//    const WorkGroupInfo* workGroupInfo() const { return &workGroupInfo_; }
//    //! Returns the kernel info structure for filling in
//    WorkGroupInfo* workGroupInfo() { return &workGroupInfo_; }
//
//    //! Returns the kernel name
//    const std::string& name() const { return name_; }
//
//
//    void setUniformWorkGroupSize(bool u) { workGroupInfo_.uniformWorkGroupSize_ = u; }
//
//    bool getUniformWorkGroupSize() const { return workGroupInfo_.uniformWorkGroupSize_; }
//
//    void setReqdWorkGroupSize(size_t x, size_t y, size_t z) {
//        workGroupInfo_.compileSize_[0] = x;
//        workGroupInfo_.compileSize_[1] = y;
//        workGroupInfo_.compileSize_[2] = z;
//    }
//
//    size_t getReqdWorkGroupSize(int dim) { return workGroupInfo_.compileSize_[dim]; }
//
//    void setWorkGroupSizeHint(size_t x, size_t y, size_t z) {
//        workGroupInfo_.compileSizeHint_[0] = x;
//        workGroupInfo_.compileSizeHint_[1] = y;
//        workGroupInfo_.compileSizeHint_[2] = z;
//    }
//
//    size_t getWorkGroupSizeHint(int dim) const { return workGroupInfo_.compileSizeHint_[dim]; }
//
//    void setVecTypeHint(const std::string& hint) { workGroupInfo_.compileVecTypeHint_ = hint; }
//
//    void setLocalMemSize(size_t size) { workGroupInfo_.localMemSize_ = size; }
//
//    void setPreferredSizeMultiple(size_t size) { workGroupInfo_.preferredSizeMultiple_ = size; }
//
//    const std::string& RuntimeHandle() const { return runtimeHandle_; }
//    void setRuntimeHandle(const std::string& handle) { runtimeHandle_ = handle; }
//
//    //! Return the build log
//    const std::string& buildLog() const { return buildLog_; }
//
//    const std::unordered_map<size_t, size_t>& patch() const { return patchReferences_; }
//
//    //! Returns TRUE if kernel uses dynamic parallelism
//    bool dynamicParallelism() const { return (flags_.dynamicParallelism_) ? true : false; }
//
//    //! set dynamic parallelism flag
//    void setDynamicParallelFlag(bool flag) { flags_.dynamicParallelism_ = flag; }
//
//    //! Returns TRUE if kernel is internal kernel
//    bool isInternalKernel() const { return (flags_.internalKernel_) ? true : false; }
//
//    //! set internal kernel flag
//    void setInternalKernelFlag(bool flag) { flags_.internalKernel_ = flag; }
//
//    //! Return TRUE if kernel uses images
//    bool imageEnable() const { return (flags_.imageEna_) ? true : false; }
//
//    //! Return TRUE if kernel wirtes images
//    bool imageWrite() const { return (flags_.imageWriteEna_) ? true : false; }
//
//    //! Returns TRUE if it's a HSA kernel
//    bool hsa() const { return (flags_.hsa_) ? true : false; }
//
//    //! Finds local workgroup size
//    void FindLocalWorkSize(
//        size_t workDim,                   //!< Work dimension
//        const amd::NDRange& gblWorkSize,  //!< Global work size
//        amd::NDRange& lclWorkSize         //!< Calculated local work size
//    ) const;
//
//    const uint64_t KernelCodeHandle() const { return kernelCodeHandle_; }
//
//    const uint32_t WorkgroupGroupSegmentByteSize() const { return workgroupGroupSegmentByteSize_; }
//    void SetWorkgroupGroupSegmentByteSize(uint32_t size) { workgroupGroupSegmentByteSize_ = size; }
//
//    const uint32_t WorkitemPrivateSegmentByteSize() const { return workitemPrivateSegmentByteSize_; }
//    void SetWorkitemPrivateSegmentByteSize(uint32_t size) { workitemPrivateSegmentByteSize_ = size; }
//    const bool KernalHasDynamicCallStack() const { return kernelHasDynamicCallStack_; }
//
//    const uint32_t KernargSegmentByteSize() const { return kernargSegmentByteSize_; }
//    void SetKernargSegmentByteSize(uint32_t size) { kernargSegmentByteSize_ = size; }
//
//    const uint8_t KernargSegmentAlignment() const { return kernargSegmentAlignment_; }
//    void SetKernargSegmentAlignment(uint32_t align) { kernargSegmentAlignment_ = align; }
//
//    void SetSymbolName(const std::string& name) { symbolName_ = name; }
//
//    void SetKernelKind(const std::string& kind) {
//        kind_ = (kind == "init") ? Init : ((kind == "fini") ? Fini : Normal);
//    }
//
//    void SetWGPMode(bool wgpMode) {
//        workGroupInfo_.isWGPMode_ = wgpMode;
//    }
//
//    bool isInitKernel() const { return kind_ == Init; }
//
//    bool isFiniKernel() const { return kind_ == Fini; }
//
//
//    std::string name_;                //!< kernel name
//    std::string symbolName_;          //!< kernel symbol name
//    WorkGroupInfo workGroupInfo_;     //!< device kernel info structure
//    std::string buildLog_;            //!< build log
//
//
//    uint64_t kernelCodeHandle_ = 0;   //!< Kernel code handle (aka amd_kernel_code_t)
//    uint32_t workgroupGroupSegmentByteSize_ = 0;
//    uint32_t workitemPrivateSegmentByteSize_ = 0;
//    uint32_t kernargSegmentByteSize_ = 0;   //!< Size of kernel argument buffer
//    uint32_t kernargSegmentAlignment_ = 0;
//    bool kernelHasDynamicCallStack_ = 0;
//
//    union Flags {
//        struct {
//            uint imageEna_ : 1;           //!< Kernel uses images
//            uint imageWriteEna_ : 1;      //!< Kernel uses image writes
//            uint dynamicParallelism_ : 1; //!< Dynamic parallelism enabled
//            uint internalKernel_ : 1;     //!< True: internal kernel
//            uint hsa_ : 1;                //!< HSA kernel
//        };
//        uint value_;
//        Flags() : value_(0) {}
//    } flags_;
//
//
// private:
//    //! Disable default copy constructor
//    Kernel(const Kernel&);
//
//    //! Disable operator=
//    Kernel& operator=(const Kernel&);
//
//    std::unordered_map<size_t, size_t> patchReferences_;  //!< Patch table for references
//
//    enum KernelKind{
//        Normal = 0,
//        Init   = 1,
//        Fini   = 2
//    };
//
//    KernelKind kind_{Normal};  //!< Kernel kind, is normal unless specified otherwise
//};



/**
 * Returns the number of symbols in the ElfView
 * \param elfView a view of the AMDGPU ELF in host memory
 * \return number of symbols in the ELF
 */
unsigned int getSymbolNum(const ElfView &elfView);

/**
 * Returns the demangled name of the input symbol name
 * @param mangledName mangled name string
 * @return demangled name as std::string
 */
std::string getDemangledName(const std::string &mangledName);

amd_comgr_status_t getCodeObjectElfsFromFatBinary(const void *data, std::vector<ELFIO::elfio> &fatBinaryElfs);

code_view_t getFunctionFromSymbol(ELFIO::elfio &elfio, const std::string &functionName);

std::vector<code_view_t> getDeviceLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent);

std::vector<code_view_t> getHostLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent);

//struct kd_rsrc_1 {
//    uint8_t granulated_vgpr_count;
//    uint8_t granulated_sgpr_count;
//    uint8_t priority;
//    uint8_t float_round_mode_32;
//    uint8_t float_denorm_mode_32;
//    uint8_t float_denorm_mode_16_64;
//    bool priv;
//    bool enable_dx10_clamp;
//    bool debug_mode;
//    bool enable_ieee_mode;
//    bool bulky;
//    bool cdbg_user;
//    uint8_t reserved1;
//};


#define AMD_HSA_BITS_GET_RSRC1(kd, prop) AMD_HSA_BITS_GET(kd->rsrc1, prop));

#define AMD_HSA_BITS_GET_RSRC2(kd, prop) AMD_HSA_BITS_GET(kd->rsrc2, prop));

void printRSR1(const kernel_descriptor_t *kd);

void printRSR2(const kernel_descriptor_t *kd);

void printCodeProperties(const kernel_descriptor_t *kd);

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

// ELFIO::elfio createAMDGPUElf(const ELFIO::elfio &elfIoIn, hsa_agent_t agent);
ElfView createAMDGPUElf(const ELFIO::elfio &elfIoIn, hsa_agent_t agent);

ELFIO::section *newSection(
    ELFIO::elfio &elfIo,
    ElfSections id,
    co_manip::code_view_t data);

bool addSection(
    ELFIO::elfio &elfIo,
    ElfSections id,
    co_manip::code_view_t data);

bool addSectionData(
    ELFIO::elfio &elfIo,
    ELFIO::Elf_Xword &outOffset,
    ElfSections id,
    co_manip::code_view_t data);

bool addSymbol(
    ELFIO::elfio &elfIo,
    ElfSections id,
    const char *symbolName,
    code_view_t data);

}// namespace luthier::co_manip

#endif
