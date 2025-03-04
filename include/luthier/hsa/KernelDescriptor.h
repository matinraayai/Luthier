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
/// This file describes the HSA kernel descriptor, a POD struct in addition to
/// some convenience methods.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_KERNEL_DESCRIPTOR_HPP
#define LUTHIER_HSA_KERNEL_DESCRIPTOR_HPP
#include "luthier/types.h"
#include <cstdint>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

class LoadedCodeObjectKernel;

/// \brief POD (plain-old-data) struct to provide an abstraction over the kernel
/// descriptor, plus some convenience methods
/// \details To inspect the contents of a kernel descriptor, simply use
/// <tt>reinterpret_cast</tt> over its address:
/// \code
/// const auto* KD = reinterpret_cast<const KernelDescriptor*>(KDAddress);
/// \endcode
/// \warning Kernel descriptors in a \c hsa_loaded_code_object_t should never
/// be modified using this struct (both on host or device memory)
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
    uint32_t EnableSgprPrivateSegmentWaveByteOffset;
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
  [[nodiscard]] luthier::address_t getEntryPoint() const;

  /// Returns a pointer to the kernel descriptor, given the kernel object
  /// field of the kernel dispatch packet
  /// \param KernelObject obtained from the \c kernel_object field in the
  /// \c hsa_kernel_dispatch_packet_t
  /// \return the kernel descriptor of the <tt>KernelObject</tt>
  static const KernelDescriptor *fromKernelObject(uint64_t KernelObject);

  /// \return a const reference to the KD's \c hsa::LoadedCodeObjectKernel on
  /// success, or an \c llvm::Error if the KD is invalid
  [[nodiscard]] llvm::Expected<std::unique_ptr<LoadedCodeObjectKernel>>
  getLoadedCodeObjectKernelSymbol() const;
};
} // namespace luthier::hsa

#endif