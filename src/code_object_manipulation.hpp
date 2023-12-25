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
#ifndef CODE_OBJECT_MANIPULATION_HPP
#define CODE_OBJECT_MANIPULATION_HPP

#include "luthier_types.h"
#include "code_view.hpp"
#include "hsa_agent.hpp"
#include <elfio/elfio.hpp>


namespace luthier::co_manip {


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

ELFIO::elfio createAMDGPUElf(const ELFIO::elfio &elfIoIn, const luthier::hsa::GpuAgent &agent);

ELFIO::section *newSection(
    ELFIO::elfio &elfIo,
    luthier::code::ElfSections id,
    luthier::byte_string_view data);

bool addSection(
    ELFIO::elfio &elfIo,
    luthier::code::ElfSections id,
    luthier::byte_string_view data);

bool addSectionData(
    ELFIO::elfio &elfIo,
    ELFIO::Elf_Xword &outOffset,
    luthier::code::ElfSections id,
    luthier::byte_string_view data);

bool addSymbol(
    ELFIO::elfio &elfIo,
    luthier::code::ElfSections id,
    const char *symbolName,
    luthier::byte_string_view data);

}// namespace luthier::co_manip

#endif
