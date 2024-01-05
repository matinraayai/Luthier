#include "hsa_kernel_descriptor.hpp"

#include <hsa/amd_hsa_common.h>
#include <hsa/amd_hsa_kernel_code.h>

#define REG_BIT_GETTER(registerName, registerVar, propName, prop) \
    uint32_t luthier::hsa::KernelDescriptor::get##registerName##propName() const {                \
        return AMD_HSA_BITS_GET(registerVar, prop);               \
    };

#define REG_BIT_SETTER(registerName, registerVar, propFuncName, prop) \
    void luthier::hsa::KernelDescriptor::set##registerName##propFuncName(uint32_t value) {            \
        AMD_HSA_BITS_SET(registerVar, prop, value);                   \
    }

REG_BIT_GETTER(Rsrc1,
               computePgmRsrc1,
               GranulatedWorkItemVGPRCount,
               AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT)
REG_BIT_GETTER(Rsrc1,
               computePgmRsrc1,
               GranulatedWaveFrontSGPRCount,
               AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, Priority, AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, FloatRoundMode32, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, FloatRoundMode16_64, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, FloatDenormMode32, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, FloatDenormMode16_64, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, Priv, AMD_COMPUTE_PGM_RSRC_ONE_PRIV)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, EnableDx10Clamp, AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, DebugMode, AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, EnableIeeeMode, AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, EnableBulky, AMD_COMPUTE_PGM_RSRC_ONE_BULKY)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, CdbgUser, AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER)
REG_BIT_GETTER(Rsrc1, computePgmRsrc1, Reserved1, AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1)

REG_BIT_SETTER(Rsrc1,
               computePgmRsrc1,
               GranulatedWorkItemVGPRCount,
               AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT)
REG_BIT_SETTER(Rsrc1,
               computePgmRsrc1,
               GranulatedWaveFrontSGPRCount,
               AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, Priority, AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, FloatRoundMode32, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, FloatRoundMode16_64, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, FloatDenormMode32, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, FloatDenormMode16_64, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, Priv, AMD_COMPUTE_PGM_RSRC_ONE_PRIV)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, EnableDx10Clamp, AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, DebugMode, AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, EnableIEEEMode, AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, EnableBulky, AMD_COMPUTE_PGM_RSRC_ONE_BULKY)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, CdbgUser, AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER)
REG_BIT_SETTER(Rsrc1, computePgmRsrc1, Reserved1, AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1)

REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableSgprPrivateSegmentWaveByteOffset,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, UserSgprCount, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, EnableTrapHandler, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, EnableSgprWorkgroupIdX, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, EnableSgprWorkgroupIdY, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, EnableSgprWorkgroupIdZ, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, EnableSgprInfo, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, EnableVgprWorkitemId, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, ExceptionAddressWatch, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH)
REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionMemoryViolation,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, GranulatedLdsSize, AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE)
REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPInvalidOperation,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION)
REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionFPDenormalSource,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE)
REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPDivisionByZero,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO)
REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPOverflow,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW)
REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPUnderflow,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW)
REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPInexact,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT)
REG_BIT_GETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIntDivisionByZero,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO)
REG_BIT_GETTER(Rsrc2, computePgmRsrc2, Reserved1, AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1)

REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableSgprPrivateSegmentWaveByteOffset,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, UserSgprCount, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, EnableTrapHandler, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, EnableSgprWorkgroupIdX, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, EnableSgprWorkgroupIdY, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, EnableSgprWorkgroupIdZ, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, EnableSgprInfo, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, EnableVgprWorkitemId, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, ExceptionAddressWatch, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH)
REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionMemoryViolation,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, GranulatedLdsSize, AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE)
REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPInvalidOperation,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION)
REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionFPDenormalSource,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE)
REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPDivisionByZero,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO)
REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPOverflow,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW)
REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPUnderflow,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW)
REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIEEE754FPInexact,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT)
REG_BIT_SETTER(Rsrc2,
               computePgmRsrc2,
               EnableExceptionIntDivisionByZero,
               AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO)
REG_BIT_SETTER(Rsrc2, computePgmRsrc2, Reserved1, AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1)

REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprPrivateSegmentBuffer,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprDispatchPtr,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprQueuePtr,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprKernArgSegmentPtr,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprDispatchId,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprFlatScratchInit,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprPrivateSegmentSize,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprGridWorkgroupCountX,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprGridWorkgroupCountY,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprGridWorkgroupCountZ,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z)
REG_BIT_GETTER(KernelCodeProperties, kernelCodeProperties, Reserved1, AMD_KERNEL_CODE_PROPERTIES_RESERVED1)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableOrderedAppendGds,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               PrivateElementSize,
               AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE)
REG_BIT_GETTER(KernelCodeProperties, kernelCodeProperties, IsPtr64, AMD_KERNEL_CODE_PROPERTIES_IS_PTR64)
REG_BIT_GETTER(KernelCodeProperties,
               kernelCodeProperties,
               IsDynamicCallStack,
               AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK)
REG_BIT_GETTER(KernelCodeProperties, kernelCodeProperties, IsDebugEnabled, AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED)
REG_BIT_GETTER(KernelCodeProperties, kernelCodeProperties, IsXnackEnabled, AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED)
REG_BIT_GETTER(KernelCodeProperties, kernelCodeProperties, Reserved2, AMD_KERNEL_CODE_PROPERTIES_RESERVED2)

REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprPrivateSegmentBuffer,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprDispatchPtr,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprQueuePtr,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprKernArgSegmentPtr,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprDispatchId,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprFlatScratchInit,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprPrivateSegmentSize,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprGridWorkgroupCountX,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprGridWorkgroupCountY,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableSgprGridWorkgroupCountZ,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z)
REG_BIT_SETTER(KernelCodeProperties, kernelCodeProperties, Reserved1, AMD_KERNEL_CODE_PROPERTIES_RESERVED1)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               EnableOrderedAppendGds,
               AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               PrivateElementSize,
               AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE)
REG_BIT_SETTER(KernelCodeProperties, kernelCodeProperties, IsPtr64, AMD_KERNEL_CODE_PROPERTIES_IS_PTR64)
REG_BIT_SETTER(KernelCodeProperties,
               kernelCodeProperties,
               IsDynamicCallStack,
               AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK)
REG_BIT_SETTER(KernelCodeProperties, kernelCodeProperties, IsDebugEnabled, AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED)
REG_BIT_SETTER(KernelCodeProperties, kernelCodeProperties, IsXnackEnabled, AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED)
REG_BIT_SETTER(KernelCodeProperties, kernelCodeProperties, Reserved2, AMD_KERNEL_CODE_PROPERTIES_RESERVED2)

#undef REG_BIT_SETTER
#undef REG_BIT_GETTER

luthier_address_t luthier::hsa::KernelDescriptor::getEntryPoint() const {
    return reinterpret_cast<luthier_address_t>(this) + this->kernelCodeEntryByteOffset;
    //        const auto& amdTable = luthier::HsaInterceptor::instance().getHsaVenAmdLoaderTable();
    //        LUTHIER_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(kernelObject),
    //                                                                         reinterpret_cast<const void **>(&kernelDescriptor)));
}
