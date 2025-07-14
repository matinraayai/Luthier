//===-- KernelDescriptor.h - HSA Kernel Descriptor POD Wrapper --*- C++ -*-===//
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
/// Describes the HSA kernel descriptor POD struct in addition to some
/// convenience methods.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_HSA_KERNEL_DESCRIPTOR_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_HSA_KERNEL_DESCRIPTOR_H

namespace luthier::amdgpu::hsa {

/// \brief POD (plain-old-data) struct to provide an abstraction over the kernel
/// descriptor, plus some convenience methods for inspecting its fields
/// \details As is the case for any POD struct, the contents of a
/// kernel descriptor can be inspected via <tt>reinterpret_cast</tt>-ing its
/// address to <tt>KernelDescriptor</tt>:
/// \code
/// const auto* KD = reinterpret_cast<const KernelDescriptor*>(KDAddress);
/// \endcode
/// \note If the kernel descriptor is loaded onto the device
/// memory, its host-accessible memory must be obtained using
/// \c luthier::hsa::convertToHostEquivalent before its fields are inspected
/// \note This struct is meant for inspecting kernel descriptor fields,
/// and is not meant to carry out any modifications to the kernel descriptor.
struct KernelDescriptor {
  uint32_t GroupSegmentFixedSize;
  uint32_t PrivateSegmentFixedSize;
  uint32_t KernArgSize;
  uint8_t Reserved0[4];
  int64_t KernelCodeEntryByteOffset;
  uint8_t Reserved1[20];
  uint32_t ComputePgmRsrc3; // GFX10+ and GFX90A+
  uint32_t ComputePgmRsrc1;
  uint32_t ComputePgmRsrc2;
  uint16_t KernelCodeProperties;
  uint16_t KernArgPreload;
  uint8_t Reserved2[4];

  /// a struct for easier access to the RSRC1 register without using individual
  /// bit-wise operations
  typedef struct {
    uint32_t GranulatedWorkItemVGPRCount;
    uint32_t GranulatedWaveFrontSGPRCount;
    uint32_t Priority;
    uint32_t FloatRoundMode32;
    uint32_t FloatRoundMode16_64;
    uint32_t FloatDenormMode32;
    uint32_t FloatDenormMode16_64;
    uint32_t Priv;
    uint32_t EnableDx10Clamp;
    uint32_t DebugMode;
    uint32_t EnableIeeeMode;
    uint32_t EnableBulky;
    uint32_t CdbgUser;
    uint32_t Reserved1;
  } Rsrc1Info;

  /// a struct for easier access to the RSRC2 register without using individual
  /// bit-wise operations
  typedef struct {
    uint32_t EnableSgprPrivateSegmentWaveByteOffset;
    uint32_t UserSgprCount;
    uint32_t EnableTrapHandler;
    uint32_t EnableSgprWorkgroupIdX;
    uint32_t EnableSgprWorkgroupIdY;
    uint32_t EnableSgprWorkgroupIdZ;
    uint32_t EnableSgprInfo;
    uint32_t EnableVgprWorkitemId;
    uint32_t ExceptionAddressWatch;
    uint32_t EnableExceptionMemoryViolation;
    uint32_t GranulatedLdsSize;
    uint32_t EnableExceptionIEEE754FPInvalidOperation;
    uint32_t EnableExceptionFPDenormalSource;
    uint32_t EnableExceptionIEEE754FPDivisionByZero;
    uint32_t EnableExceptionIEEE754FPOverflow;
    uint32_t EnableExceptionIEEE754FPUnderflow;
    uint32_t EnableExceptionIEEE754FPInexact;
    uint32_t EnableExceptionIntDivisionByZero;
    uint32_t Reserved1;
  } Rsrc2Info;

  /// a struct for easier access to the kernel code properties register without
  /// using individual bit-wise operations
  typedef struct {
    uint32_t EnableSgprPrivateSegmentBuffer;
    uint32_t EnableSgprDispatchPtr;
    uint32_t EnableSgprQueuePtr;
    uint32_t EnableSgprKernArgSegmentPtr;
    uint32_t EnableSgprDispatchId;
    uint32_t EnableSgprFlatScratchInit;
    uint32_t EnableSgprPrivateSegmentSize;
    uint32_t EnableSgprGridWorkgroupCountX;
    uint32_t EnableSgprGridWorkgroupCountY;
    uint32_t EnableSgprGridWorkgroupCountZ;
    uint32_t Reserved1;
    uint32_t EnableOrderedAppendGds;
    uint32_t PrivateElementSize;
    uint32_t IsPtr64;
    uint32_t IsDynamicCallStack;
    uint32_t IsDebugEnabled;
    uint32_t IsXnackEnabled;
    uint32_t Reserved2;
  } KernelCodePropertiesInfo;

  /// \return parsed Rsrc1 register into the more accessible \c Rsrc1Info format
  [[nodiscard]] Rsrc1Info getRsrc1() const;

  /// \return parsed Rsrc2 register into the more accessible \c Rsrc2Info format
  [[nodiscard]] Rsrc2Info getRsrc2() const;

  /// \return parsed kernel code properties register into the more accessible
  /// \c KernelCodePropertiesInfo format
  [[nodiscard]] KernelCodePropertiesInfo getKernelCodeProperties() const;

  /// \return the entrypoint of the kernel machine code i.e. the address of the
  /// first instruction of the kernel
  [[nodiscard]] uint64_t getEntryPoint() const;

  /// Returns a pointer to the kernel descriptor, given the \p KernelObject
  /// field of the kernel dispatch packet
  /// \param KernelObject obtained from the \c kernel_object field in the
  /// \c hsa_kernel_dispatch_packet_t
  /// \return the kernel descriptor of the <tt>KernelObject</tt>
  static const KernelDescriptor *fromKernelObject(uint64_t KernelObject);
};
} // namespace luthier::amdgpu::hsa

#endif