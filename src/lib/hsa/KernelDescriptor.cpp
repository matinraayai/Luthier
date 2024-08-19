//===-- KernelDescriptor.cpp - HSA Kernel Descriptor ---------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file implements the HSA kernel descriptor POD struct methods.
//===----------------------------------------------------------------------===//
#include "common/error.hpp"
#include <hsa/amd_hsa_common.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <luthier/hsa/KernelDescriptor.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>

namespace luthier::hsa {

KernelDescriptor::Rsrc1Info KernelDescriptor::getRsrc1() const {
  Rsrc1Info Out;
  Out.GranulatedWaveFrontSGPRCount =
      AMD_HSA_BITS_GET(this->ComputePgmRsrc1,
                       AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT);
  Out.GranulatedWaveFrontSGPRCount = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc1,
      AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT);
  Out.Priority = AMD_HSA_BITS_GET(this->ComputePgmRsrc1,
                                  AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY);
  Out.FloatRoundMode32 = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc1, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32);
  Out.FloatRoundMode16_64 = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc1, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64);
  Out.FloatDenormMode32 = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc1, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32);
  Out.FloatDenormMode16_64 = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc1, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64);
  Out.Priv =
      AMD_HSA_BITS_GET(this->ComputePgmRsrc1, AMD_COMPUTE_PGM_RSRC_ONE_PRIV);
  Out.EnableDx10Clamp = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc1, AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP);
  Out.DebugMode = AMD_HSA_BITS_GET(this->ComputePgmRsrc1,
                                   AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE);
  Out.EnableIeeeMode = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc1, AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE);
  Out.EnableBulky =
      AMD_HSA_BITS_GET(this->ComputePgmRsrc1, AMD_COMPUTE_PGM_RSRC_ONE_BULKY);
  Out.CdbgUser = AMD_HSA_BITS_GET(this->ComputePgmRsrc1,
                                  AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER);
  Out.Reserved1 = AMD_HSA_BITS_GET(this->ComputePgmRsrc1,
                                   AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1);
  return Out;
}

KernelDescriptor::Rsrc2Info KernelDescriptor::getRsrc2() const {
  Rsrc2Info Out;
  Out.EnableSgprPrivateSegmentWaveByteOffset = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);

  Out.UserSgprCount = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT);

  Out.EnableTrapHandler = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER);

  Out.EnableSgprWorkgroupIdX =
      AMD_HSA_BITS_GET(this->ComputePgmRsrc2,
                       AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X);

  Out.EnableSgprWorkgroupIdY =
      AMD_HSA_BITS_GET(this->ComputePgmRsrc2,
                       AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y);

  Out.EnableSgprWorkgroupIdZ =
      AMD_HSA_BITS_GET(this->ComputePgmRsrc2,
                       AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z);

  Out.EnableSgprInfo =
      AMD_HSA_BITS_GET(this->ComputePgmRsrc2,
                       AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO);

  Out.EnableVgprWorkitemId = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID);

  Out.ExceptionAddressWatch =
      AMD_HSA_BITS_GET(this->ComputePgmRsrc2,
                       AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH);

  Out.EnableExceptionMemoryViolation = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION);

  Out.GranulatedLdsSize = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2, AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE);

  Out.EnableExceptionIEEE754FPInvalidOperation = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION);

  Out.EnableExceptionFPDenormalSource = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE);

  Out.EnableExceptionIEEE754FPDivisionByZero = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO);

  Out.EnableExceptionIEEE754FPOverflow = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW);

  Out.EnableExceptionIEEE754FPUnderflow = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW);

  Out.EnableExceptionIEEE754FPInexact = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT);

  Out.EnableExceptionIntDivisionByZero = AMD_HSA_BITS_GET(
      this->ComputePgmRsrc2,
      AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO);

  Out.Reserved1 = AMD_HSA_BITS_GET(this->ComputePgmRsrc2,
                                   AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1);

  return Out;
}

KernelDescriptor::KernelCodePropertiesInfo
KernelDescriptor::getKernelCodeProperties() const {
  KernelCodePropertiesInfo Out;
  Out.EnableSgprPrivateSegmentBuffer = AMD_HSA_BITS_GET(
      this->KernelCodeProperties,
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER);

  Out.EnableSgprDispatchPtr =
      AMD_HSA_BITS_GET(this->KernelCodeProperties,
                       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR);

  Out.EnableSgprQueuePtr =
      AMD_HSA_BITS_GET(this->KernelCodeProperties,
                       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR);

  Out.EnableSgprKernArgSegmentPtr = AMD_HSA_BITS_GET(
      this->KernelCodeProperties,
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR);

  Out.EnableSgprDispatchId =
      AMD_HSA_BITS_GET(this->KernelCodeProperties,
                       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID);

  Out.EnableSgprFlatScratchInit = AMD_HSA_BITS_GET(
      this->KernelCodeProperties,
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT);

  Out.EnableSgprPrivateSegmentSize = AMD_HSA_BITS_GET(
      this->KernelCodeProperties,
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE);

  Out.EnableSgprGridWorkgroupCountX = AMD_HSA_BITS_GET(
      this->KernelCodeProperties,
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X);

  Out.EnableSgprGridWorkgroupCountY = AMD_HSA_BITS_GET(
      this->KernelCodeProperties,
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y);

  Out.EnableSgprGridWorkgroupCountZ = AMD_HSA_BITS_GET(
      this->KernelCodeProperties,
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z);

  Out.Reserved1 = AMD_HSA_BITS_GET(this->KernelCodeProperties,
                                   AMD_KERNEL_CODE_PROPERTIES_RESERVED1);

  Out.EnableOrderedAppendGds =
      AMD_HSA_BITS_GET(this->KernelCodeProperties,
                       AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS);

  Out.PrivateElementSize =
      AMD_HSA_BITS_GET(this->KernelCodeProperties,
                       AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE);

  Out.IsPtr64 = AMD_HSA_BITS_GET(this->KernelCodeProperties,
                                 AMD_KERNEL_CODE_PROPERTIES_IS_PTR64);

  Out.IsDynamicCallStack =
      AMD_HSA_BITS_GET(this->KernelCodeProperties,
                       AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK);

  Out.IsDebugEnabled = AMD_HSA_BITS_GET(
      this->KernelCodeProperties, AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED);

  Out.IsXnackEnabled = AMD_HSA_BITS_GET(
      this->KernelCodeProperties, AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED);

  Out.Reserved2 = AMD_HSA_BITS_GET(this->KernelCodeProperties,
                                   AMD_KERNEL_CODE_PROPERTIES_RESERVED2);

  return Out;
}

address_t KernelDescriptor::getEntryPoint() const {
  return reinterpret_cast<address_t>(this) + this->KernelCodeEntryByteOffset;
}
const KernelDescriptor *
KernelDescriptor::fromKernelObject(uint64_t KernelObject) {
  return reinterpret_cast<KernelDescriptor *>(KernelObject);
}

llvm::Expected<const LoadedCodeObjectKernel &>
KernelDescriptor::getLoadedCodeObjectKernelSymbol() const {
  const auto *Symbol = LoadedCodeObjectKernel::fromLoadedAddress(
      reinterpret_cast<luthier::address_t>(this));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Symbol != nullptr));

  const auto *KernelSymbol = llvm::dyn_cast<LoadedCodeObjectKernel>(Symbol);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(KernelSymbol != nullptr));

  return *KernelSymbol;
}

} // namespace luthier::hsa