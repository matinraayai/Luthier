#ifndef KERNEL_DESCRIPTOR_HPP
#define KERNEL_DESCRIPTOR_HPP
#include <cstdint>
#include "luthier_types.h"

namespace luthier::hsa {

//TODO: Make this work with KDs in executables not created with the FULL profile

struct KernelDescriptor {
    uint32_t groupSegmentFixedSize;
    uint32_t privateSegmentFixedSize;
    uint32_t kernArgSize;
    uint8_t reserved0[4];
    int64_t kernelCodeEntryByteOffset;
    uint8_t reserved1[20];
    uint32_t computePgmRsrc3;// GFX10+ and GFX90A+
    uint32_t computePgmRsrc1;
    uint32_t computePgmRsrc2;
    uint16_t kernelCodeProperties;
    uint16_t kernArgPreload;
    uint8_t reserved2[4];

#define REG_BIT_GETTER(registerName, propName) \
    uint32_t get##registerName##propName() const;

#define REG_BIT_SETTER(registerName, propFuncName) \
    void set##registerName##propFuncName(uint32_t value);

    REG_BIT_GETTER(Rsrc1, GranulatedWorkItemVGPRCount)
    REG_BIT_GETTER(Rsrc1, GranulatedWaveFrontSGPRCount)
    REG_BIT_GETTER(Rsrc1, Priority)
    REG_BIT_GETTER(Rsrc1, FloatRoundMode32)
    REG_BIT_GETTER(Rsrc1, FloatRoundMode16_64)
    REG_BIT_GETTER(Rsrc1, FloatDenormMode32)
    REG_BIT_GETTER(Rsrc1, FloatDenormMode16_64)
    REG_BIT_GETTER(Rsrc1, Priv)
    REG_BIT_GETTER(Rsrc1, EnableDx10Clamp)
    REG_BIT_GETTER(Rsrc1, DebugMode)
    REG_BIT_GETTER(Rsrc1, EnableIeeeMode)
    REG_BIT_GETTER(Rsrc1, EnableBulky)
    REG_BIT_GETTER(Rsrc1, CdbgUser)
    REG_BIT_GETTER(Rsrc1, Reserved1)
    REG_BIT_SETTER(Rsrc1, GranulatedWorkItemVGPRCount)
    REG_BIT_SETTER(Rsrc1, GranulatedWaveFrontSGPRCount)
    REG_BIT_SETTER(Rsrc1, Priority)
    REG_BIT_SETTER(Rsrc1, FloatRoundMode32)
    REG_BIT_SETTER(Rsrc1, FloatRoundMode16_64)
    REG_BIT_SETTER(Rsrc1, FloatDenormMode32)
    REG_BIT_SETTER(Rsrc1, FloatDenormMode16_64)
    REG_BIT_SETTER(Rsrc1, Priv)
    REG_BIT_SETTER(Rsrc1, EnableDx10Clamp)
    REG_BIT_SETTER(Rsrc1, DebugMode)
    REG_BIT_SETTER(Rsrc1, EnableIEEEMode)
    REG_BIT_SETTER(Rsrc1, EnableBulky)
    REG_BIT_SETTER(Rsrc1, CdbgUser)
    REG_BIT_SETTER(Rsrc1, Reserved1)
    REG_BIT_SETTER(Rsrc2, EnableSgprPrivateSegmentWaveByteOffset)
    REG_BIT_GETTER(Rsrc2, UserSgprCount)
    REG_BIT_GETTER(Rsrc2, EnableTrapHandler)
    REG_BIT_GETTER(Rsrc2, EnableSgprWorkgroupIdX)
    REG_BIT_GETTER(Rsrc2, EnableSgprWorkgroupIdY)
    REG_BIT_GETTER(Rsrc2, EnableSgprWorkgroupIdZ)
    REG_BIT_GETTER(Rsrc2, EnableSgprInfo)
    REG_BIT_GETTER(Rsrc2, EnableVgprWorkitemId)
    REG_BIT_GETTER(Rsrc2, ExceptionAddressWatch)
    REG_BIT_GETTER(Rsrc2, EnableExceptionMemoryViolation)
    REG_BIT_GETTER(Rsrc2, GranulatedLdsSize)
    REG_BIT_GETTER(Rsrc2, EnableExceptionIEEE754FPInvalidOperation)
    REG_BIT_GETTER(Rsrc2, EnableExceptionFPDenormalSource)
    REG_BIT_GETTER(Rsrc2, EnableExceptionIEEE754FPDivisionByZero)
    REG_BIT_GETTER(Rsrc2, EnableExceptionIEEE754FPOverflow)
    REG_BIT_GETTER(Rsrc2, EnableExceptionIEEE754FPUnderflow)
    REG_BIT_GETTER(Rsrc2, EnableExceptionIEEE754FPInexact)
    REG_BIT_GETTER(Rsrc2, EnableExceptionIntDivisionByZero)
    REG_BIT_GETTER(Rsrc2, Reserved1)
    REG_BIT_GETTER(Rsrc2, EnableSgprPrivateSegmentWaveByteOffset)
    REG_BIT_SETTER(Rsrc2, UserSgprCount)
    REG_BIT_SETTER(Rsrc2, EnableTrapHandler)
    REG_BIT_SETTER(Rsrc2, EnableSgprWorkgroupIdX)
    REG_BIT_SETTER(Rsrc2, EnableSgprWorkgroupIdY)
    REG_BIT_SETTER(Rsrc2, EnableSgprWorkgroupIdZ)
    REG_BIT_SETTER(Rsrc2, EnableSgprInfo)
    REG_BIT_SETTER(Rsrc2, EnableVgprWorkitemId)
    REG_BIT_SETTER(Rsrc2, ExceptionAddressWatch)
    REG_BIT_SETTER(Rsrc2, EnableExceptionMemoryViolation)
    REG_BIT_SETTER(Rsrc2, GranulatedLdsSize)
    REG_BIT_SETTER(Rsrc2, EnableExceptionIEEE754FPInvalidOperation)
    REG_BIT_SETTER(Rsrc2, EnableExceptionFPDenormalSource)
    REG_BIT_SETTER(Rsrc2, EnableExceptionIEEE754FPDivisionByZero)
    REG_BIT_SETTER(Rsrc2, EnableExceptionIEEE754FPOverflow)
    REG_BIT_SETTER(Rsrc2, EnableExceptionIEEE754FPUnderflow)
    REG_BIT_SETTER(Rsrc2, EnableExceptionIEEE754FPInexact)
    REG_BIT_SETTER(Rsrc2, EnableExceptionIntDivisionByZero)
    REG_BIT_SETTER(Rsrc2, Reserved1)

    REG_BIT_GETTER(KernelCodeProperties, EnableSgprPrivateSegmentBuffer)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprDispatchPtr)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprQueuePtr)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprKernArgSegmentPtr)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprDispatchId)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprFlatScratchInit)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprPrivateSegmentSize)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprGridWorkgroupCountX)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprGridWorkgroupCountY)
    REG_BIT_GETTER(KernelCodeProperties, EnableSgprGridWorkgroupCountZ)
    REG_BIT_GETTER(KernelCodeProperties, Reserved1)
    REG_BIT_GETTER(KernelCodeProperties, EnableOrderedAppendGds)
    REG_BIT_GETTER(KernelCodeProperties, PrivateElementSize)
    REG_BIT_GETTER(KernelCodeProperties, IsPtr64)
    REG_BIT_GETTER(KernelCodeProperties, IsDynamicCallStack)
    REG_BIT_GETTER(KernelCodeProperties, IsDebugEnabled)
    REG_BIT_GETTER(KernelCodeProperties, IsXnackEnabled)
    REG_BIT_GETTER(KernelCodeProperties, Reserved2)

    REG_BIT_SETTER(KernelCodeProperties, EnableSgprPrivateSegmentBuffer)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprDispatchPtr)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprQueuePtr)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprKernArgSegmentPtr)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprDispatchId)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprFlatScratchInit)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprPrivateSegmentSize)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprGridWorkgroupCountX)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprGridWorkgroupCountY)
    REG_BIT_SETTER(KernelCodeProperties, EnableSgprGridWorkgroupCountZ)
    REG_BIT_SETTER(KernelCodeProperties, Reserved1)
    REG_BIT_SETTER(KernelCodeProperties, EnableOrderedAppendGds)
    REG_BIT_SETTER(KernelCodeProperties, PrivateElementSize)
    REG_BIT_SETTER(KernelCodeProperties, IsPtr64)
    REG_BIT_SETTER(KernelCodeProperties, IsDynamicCallStack)
    REG_BIT_SETTER(KernelCodeProperties, IsDebugEnabled)
    REG_BIT_SETTER(KernelCodeProperties, IsXnackEnabled)
    REG_BIT_SETTER(KernelCodeProperties, Reserved2)

#undef REG_BIT_SETTER
#undef REG_BIT_GETTER

  [[nodiscard]] luthier_address_t getEntryPoint() const;
};
}// namespace luthier::hsa

#endif