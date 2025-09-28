//===-- ApiTable.h - HIP API table Container Logic --------------*- C++ -*-===//
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
/// Defines the \c hip::ApiTableContainer which performs bounds checking on
/// methods dispatched on the HIP API tables.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HIP_API_TABLE_H
#define LUTHIER_HIP_API_TABLE_H
#include <hip/amd_detail/hip_api_trace.hpp>
#include <rocprofiler-sdk/intercept_table.h>

namespace luthier::hip {

//===----------------------------------------------------------------------===//
// Individual Info Structs For Each HIP API function
//===----------------------------------------------------------------------===//

template <auto ApiFunc> struct ApiInfo;

#define DEFINE_HIP_API_INFO_ENTRY(ApiTableName, HipFunc)                       \
  template <> struct ApiInfo<HipFunc> {                                        \
    using ApiTable = ApiTableName;                                             \
    static constexpr auto ApiTablePointerToMember = &ApiTable::HipFunc##_fn;   \
    static constexpr auto ApiTableOffset = offsetof(ApiTable, HipFunc##_fn);   \
  };

#define DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(ApiTableName, HipFunc)        \
  template <> struct ApiInfo<&ApiTableName::HipFunc##_fn> {                    \
    using ApiTable = ApiTableName;                                             \
    static constexpr auto ApiTablePointerToMember = &ApiTable::HipFunc##_fn;   \
    static constexpr auto ApiTableOffset = offsetof(ApiTable, HipFunc##_fn);   \
  };

#define DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(ApiTableName, HipFunc)    \
  DEFINE_HIP_API_INFO_ENTRY(ApiTableName, HipFunc)                             \
  DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(ApiTableName, HipFunc)

//===----------------------------------------------------------------------===//
// HIP Compiler API Entries
//===----------------------------------------------------------------------===//
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipCompilerDispatchTable,
                                             __hipPopCallConfiguration)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipCompilerDispatchTable,
                                             __hipPushCallConfiguration)
/// These functions don't have a public binding in HIP headers, so we only
/// define the member pointer info for them
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipCompilerDispatchTable,
                                         __hipRegisterFatBinary)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipCompilerDispatchTable,
                                         __hipRegisterFunction)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipCompilerDispatchTable,
                                         __hipRegisterManagedVar)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipCompilerDispatchTable,
                                         __hipRegisterSurface)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipCompilerDispatchTable,
                                         __hipRegisterTexture)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipCompilerDispatchTable,
                                         __hipRegisterVar)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipCompilerDispatchTable,
                                         __hipUnregisterFatBinary)

//===----------------------------------------------------------------------===//
// HIP Dispatch API Entries
//===----------------------------------------------------------------------===//
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipApiName)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipArray3DCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipArray3DGetDescriptor)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipArrayCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipArrayDestroy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipArrayGetDescriptor)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipArrayGetInfo)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipBindTexture)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipBindTexture2D)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipBindTextureToArray)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipBindTextureToMipmappedArray)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipChooseDevice)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipChooseDeviceR0000)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipConfigureCall)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipCreateSurfaceObject)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipCreateTextureObject)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxCreate)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxDestroy)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipCtxDisablePeerAccess)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipCtxEnablePeerAccess)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipCtxGetApiVersion)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipCtxGetCacheConfig)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxGetCurrent)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxGetDevice)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxGetFlags)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipCtxGetSharedMemConfig)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxPopCurrent)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxPushCurrent)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipCtxSetCacheConfig)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxSetCurrent)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipCtxSetSharedMemConfig)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipCtxSynchronize)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDestroyExternalMemory)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDestroyExternalSemaphore)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDestroySurfaceObject)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDestroyTextureObject)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceCanAccessPeer)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceComputeCapability)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceDisablePeerAccess)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceEnablePeerAccess)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipDeviceGet)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetByPCIBusId)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetCacheConfig)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetDefaultMemPool)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetGraphMemAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetLimit)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetMemPool)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetName)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetP2PAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetPCIBusId)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetSharedMemConfig)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetStreamPriorityRange)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGetUuid)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceGraphMemTrim)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipDevicePrimaryCtxGetState)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipDevicePrimaryCtxRelease)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipDevicePrimaryCtxReset)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipDevicePrimaryCtxRetain)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipDevicePrimaryCtxSetFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipDeviceReset)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceSetCacheConfig)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceSetGraphMemAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceSetLimit)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceSetMemPool)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceSetSharedMemConfig)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceSynchronize)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDeviceTotalMem)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDriverGetVersion)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGetErrorName)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGetErrorString)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGraphAddMemcpyNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvMemcpy2DUnaligned)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipDrvMemcpy3D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvMemcpy3DAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvPointerGetAttributes)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipEventCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipEventCreateWithFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipEventDestroy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipEventElapsedTime)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipEventQuery)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipEventRecord)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipEventSynchronize)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipExtGetLinkTypeAndHopCount)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipExtLaunchKernel)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipExtLaunchMultiKernelMultiDevice)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipExtMallocWithFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipExtStreamCreateWithCUMask)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipExtStreamGetCUMask)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipExternalMemoryGetMappedBuffer)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipFree)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipFreeArray)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipFreeAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipFreeHost)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipFreeMipmappedArray)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipFuncGetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipFuncGetAttributes)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipFuncSetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipFuncSetCacheConfig)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipFuncSetSharedMemConfig)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGLGetDevices)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetChannelDesc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipGetDevice)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetDeviceCount)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetDeviceFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetDevicePropertiesR0600)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetDevicePropertiesR0000)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetErrorName)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetErrorString)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetLastError)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetMipmappedArrayLevel)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipGetSymbolAddress)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipGetSymbolSize)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipGetTextureAlignmentOffset)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetTextureObjectResourceDesc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGetTextureObjectResourceViewDesc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetTextureObjectTextureDesc)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipGetTextureReference)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddChildGraphNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddDependencies)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddEmptyNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddEventRecordNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddEventWaitNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddHostNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddKernelNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddMemAllocNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddMemFreeNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddMemcpyNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddMemcpyNode1D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddMemcpyNodeFromSymbol)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddMemcpyNodeToSymbol)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddMemsetNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphChildGraphNodeGetGraph)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipGraphClone)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipGraphCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphDebugDotPrint)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphDestroy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphDestroyNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphEventRecordNodeGetEvent)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphEventRecordNodeSetEvent)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphEventWaitNodeGetEvent)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphEventWaitNodeSetEvent)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExecChildGraphNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecDestroy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExecEventRecordNodeSetEvent)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecEventWaitNodeSetEvent)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecHostNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecKernelNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecMemcpyNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecMemcpyNodeSetParams1D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExecMemcpyNodeSetParamsFromSymbol)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExecMemcpyNodeSetParamsToSymbol)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecMemsetNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecUpdate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphGetEdges)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphGetNodes)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphGetRootNodes)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphHostNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphHostNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphInstantiate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphInstantiateWithFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphKernelNodeCopyAttributes)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphKernelNodeGetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphKernelNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphKernelNodeSetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphKernelNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipGraphLaunch)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphMemAllocNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphMemFreeNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphMemcpyNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphMemcpyNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphMemcpyNodeSetParams1D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphMemcpyNodeSetParamsFromSymbol)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphMemcpyNodeSetParamsToSymbol)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphMemsetNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphMemsetNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphNodeFindInClone)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphNodeGetDependencies)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphNodeGetDependentNodes)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphNodeGetEnabled)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphNodeGetType)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphNodeSetEnabled)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphReleaseUserObject)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphRemoveDependencies)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphRetainUserObject)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipGraphUpload)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphicsGLRegisterBuffer)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphicsGLRegisterImage)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphicsMapResources)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphicsResourceGetMappedPointer)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphicsSubResourceGetMappedArray)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphicsUnmapResources)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphicsUnregisterResource)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipHostAlloc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipHostFree)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipHostGetDevicePointer)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipHostGetFlags)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipHostMalloc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipHostRegister)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipHostUnregister)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipImportExternalMemory)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipImportExternalSemaphore)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipInit)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipIpcCloseMemHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipIpcGetEventHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipIpcGetMemHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipIpcOpenEventHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipIpcOpenMemHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipKernelNameRef)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipKernelNameRefByPtr)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipLaunchByPtr)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipLaunchCooperativeKernel)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipLaunchCooperativeKernelMultiDevice)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipLaunchHostFunc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipLaunchKernel)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMalloc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMalloc3D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMalloc3DArray)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMallocArray)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMallocAsync)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipMallocFromPoolAsync)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMallocHost)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMallocManaged)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMallocMipmappedArray)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMallocPitch)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemAddressFree)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemAddressReserve)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemAdvise)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMemAllocHost)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemAllocPitch)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemExportToShareableHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemGetAccess)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemGetAddressRange)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemGetAllocationGranularity)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipMemGetAllocationPropertiesFromHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemGetInfo)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemImportFromShareableHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemMap)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemMapArrayAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolDestroy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolExportPointer)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolExportToShareableHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolGetAccess)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolGetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipMemPoolImportFromShareableHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolImportPointer)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolSetAccess)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolSetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPoolTrimTo)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPrefetchAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemPtrGetInfo)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemRangeGetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemRangeGetAttributes)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemRelease)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemRetainAllocationHandle)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemSetAccess)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemUnmap)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpy2D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DFromArray)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DFromArrayAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DToArray)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DToArrayAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpy3D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy3DAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyAtoH)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyDtoD)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyDtoDAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyDtoH)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyDtoHAsync)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMemcpyFromArray)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipMemcpyFromSymbol)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipMemcpyFromSymbolAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyHtoA)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyHtoD)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyHtoDAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyParam2D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyParam2DAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyPeer)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyPeerAsync)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMemcpyToArray)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipMemcpyToSymbol)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipMemcpyToSymbolAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyWithStream)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemset)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemset2D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemset2DAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemset3D)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemset3DAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemsetAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemsetD16)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemsetD16Async)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemsetD32)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemsetD32Async)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemsetD8)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemsetD8Async)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMipmappedArrayCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMipmappedArrayDestroy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMipmappedArrayGetLevel)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipModuleGetFunction)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipModuleGetGlobal)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipModuleGetTexRef)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipModuleLaunchCooperativeKernel)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipModuleLaunchCooperativeKernelMultiDevice)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipModuleLaunchKernel)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipModuleLoad)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipModuleLoadData)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipModuleLoadDataEx)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable,
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipModuleOccupancyMaxPotentialBlockSize)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipModuleOccupancyMaxPotentialBlockSizeWithFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipModuleUnload)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(
    ::HipDispatchTable, hipOccupancyMaxActiveBlocksPerMultiprocessor)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(
    ::HipDispatchTable, hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipOccupancyMaxPotentialBlockSize)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipPeekAtLastError)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipPointerGetAttribute)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipPointerGetAttributes)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipPointerSetAttribute)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipProfilerStart)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipProfilerStop)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipRuntimeGetVersion)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipSetDevice)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipSetDeviceFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipSetupArgument)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipSignalExternalSemaphoresAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamAddCallback)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamAttachMemAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamBeginCapture)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamCreateWithFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamCreateWithPriority)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamDestroy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamEndCapture)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetCaptureInfo)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetCaptureInfo_v2)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetDevice)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetPriority)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamIsCapturing)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipStreamQuery)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamSynchronize)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamUpdateCaptureDependencies)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamWaitEvent)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamWaitValue32)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamWaitValue64)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamWriteValue32)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamWriteValue64)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipTexObjectCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipTexObjectDestroy)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipTexObjectGetResourceDesc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipTexObjectGetResourceViewDesc)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipTexObjectGetTextureDesc)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetAddress)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetAddressMode)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetFilterMode)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipTexRefGetFlags)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipTexRefGetFormat)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetMaxAnisotropy)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetMipMappedArray)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetMipmapFilterMode)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetMipmapLevelBias)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetMipmapLevelClamp)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetAddress)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetAddress2D)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetAddressMode)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipTexRefSetArray)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetBorderColor)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetFilterMode)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipTexRefSetFlags)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipTexRefSetFormat)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetMaxAnisotropy)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetMipmapFilterMode)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetMipmapLevelBias)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetMipmapLevelClamp)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefSetMipmappedArray)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipThreadExchangeStreamCaptureMode)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipUnbindTexture)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipUserObjectCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipUserObjectRelease)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipUserObjectRetain)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipWaitExternalSemaphoresAsync)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipCreateChannelDesc)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipExtModuleLaunchKernel)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipHccModuleLaunchKernel)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpy_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyToSymbol_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyFromSymbol_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2D_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DFromArray_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy3D_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemset_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemsetAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemset2D_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemset2DAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemset3DAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemset3D_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy3DAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyFromSymbolAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyToSymbolAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyFromArray_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DToArray_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DFromArrayAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DToArrayAsync_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamQuery_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamSynchronize_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetPriority_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamWaitEvent_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetFlags_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamAddCallback_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipEventRecord_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipLaunchCooperativeKernel_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipLaunchKernel_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphLaunch_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamBeginCapture_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamEndCapture_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamIsCapturing_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetCaptureInfo_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamGetCaptureInfo_v2_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipLaunchHostFunc_spt)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetStreamDeviceId)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGraphAddMemsetNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphAddExternalSemaphoresWaitNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphAddExternalSemaphoresSignalNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExternalSemaphoresSignalNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExternalSemaphoresWaitNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExternalSemaphoresSignalNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExternalSemaphoresWaitNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExecExternalSemaphoresSignalNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExecExternalSemaphoresWaitNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphInstantiateWithParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipExtGetLastError)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable,
                                         hipTexRefGetBorderColor)
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipTexRefGetArray)

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 1
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetProcAddress);
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 2
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamBeginCaptureToGraph)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 3
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGetFuncBySymbol)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipSetValidDevices)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyAtoD)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyDtoA)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipMemcpyAtoA)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyAtoHAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpyHtoAAsync)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipMemcpy2DArrayToArray)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 4
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGraphAddMemFreeNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGraphExecMemcpyNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGraphExecMemsetNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecGetFlags)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphExecNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipExternalMemoryGetMappedMipmappedArray)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGraphMemcpyNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipDrvGraphMemcpyNodeSetParams)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 5
DEFINE_HIP_API_INFO_ENTRY_MEMBER_POINTER(::HipDispatchTable, hipExtHostAlloc)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 6
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipDeviceGetTexture1DLinearMaxWidth)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 7
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipStreamBatchMemOp)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 8
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphAddBatchMemOpNode)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphBatchMemOpNodeGetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipGraphBatchMemOpNodeSetParams)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(
    ::HipDispatchTable, hipGraphExecBatchMemOpNodeSetParams)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 9
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipLinkAddData)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipLinkAddFile)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable,
                                             hipLinkComplete)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipLinkCreate)
DEFINE_HIP_AP_INFO_ENTRY_WITH_MEMBER_POINTER(::HipDispatchTable, hipLinkDestroy)
#endif

template <auto Entry, typename HipApiTableType,
          typename = std::enable_if_t<
              std::is_same_v<HipApiTableType, ::HipDispatchTable> ||
              std::is_same_v<HipApiTableType, ::HipCompilerDispatchTable>>>
bool apiTableHasEntry(const HipApiTableType &Table) {
  return ApiInfo<Entry>::ApiTableOffset < Table.size;
}

template <typename HipApiTableType, typename EntryType,
          typename = std::enable_if_t<
              std::is_same_v<HipApiTableType, ::HipDispatchTable> ||
              std::is_same_v<HipApiTableType, ::HipCompilerDispatchTable>>>
bool apiTableHasEntry(const HipApiTableType &Table,
                      const EntryType HipApiTableType::*Entry) {
  return reinterpret_cast<size_t>(&(Table.*Entry)) -
             reinterpret_cast<size_t>(&Table) <
         Table.size;
}

template <rocprofiler_intercept_table_t TableType> struct ApiTableEnumInfo;

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_COMPILER_TABLE> {
  using ApiTableType = ::HipCompilerDispatchTable;
};

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_RUNTIME_TABLE> {
  using ApiTableType = ::HipDispatchTable;
};

template <rocprofiler_intercept_table_t TableType> class ApiTableContainer {
private:
  const typename ApiTableEnumInfo<TableType>::ApiTableType &ApiTable{};

public:
  explicit ApiTableContainer(
      const ApiTableEnumInfo<TableType>::ApiTableType &ApiTable)
      : ApiTable(ApiTable) {};

  /// \brief Checks if the \c Func is present in the API table snapshot
  /// \tparam Func pointer-to-member of the function entry inside the
  /// extension table being queried
  /// \return \c true if the function is available inside the
  /// API table, \c false otherwise. Reports a fatal error
  /// if the snapshot has not been initialized by rocprofiler-sdk
  template <auto Func> [[nodiscard]] bool tableSupportsFunction() const {
    return apiTableHasEntry<Func>(ApiTable);
  }

  /// \returns the function inside the snapshot associated with the
  /// pointer-to-member accessor \c Func
  template <auto Func> const auto &getFunction() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        tableSupportsFunction<Func>(),
        "The passed function is not inside the table."));
    return *(ApiTable.*ApiInfo<Func>::ApiTablePointerToMember);
  }

  /// Obtains the function \c Func from the table snapshot and calls
  /// it with the passed \p Args and returns the results of the function call
  template <auto Func, typename... ArgTypes>
  auto callFunction(ArgTypes... Args) const {
    return getFunction<Func>()(Args...);
  }
};

} // namespace luthier::hip

#endif