#ifndef HIP_ARGS
#define HIP_ARGS

#include <hip/hip_runtime_api.h>
namespace hip {
struct FatBinaryInfo{};
}

typedef struct hip___hipGetPCH_api_args_s {
    const char * * pch;
    unsigned int * size;
} hip___hipGetPCH_api_args_t;

typedef void __hipGetPCH_return_t;


typedef struct hip___hipPopCallConfiguration_api_args_s {
    dim3 * gridDim;
    dim3 * blockDim;
    size_t * sharedMem;
    hipStream_t * stream;
} hip___hipPopCallConfiguration_api_args_t;

typedef hipError_t __hipPopCallConfiguration_return_t;


typedef struct hip___hipPushCallConfiguration_api_args_s {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    hipStream_t stream;
} hip___hipPushCallConfiguration_api_args_t;

typedef hipError_t __hipPushCallConfiguration_return_t;


typedef struct hip___hipRegisterFatBinary_api_args_s {
    const void * data;
} hip___hipRegisterFatBinary_api_args_t;

typedef hip::FatBinaryInfo * * __hipRegisterFatBinary_return_t;


typedef struct hip___hipRegisterFunction_api_args_s {
    hip::FatBinaryInfo * * modules;
    const void * hostFunction;
    char * deviceFunction;
    const char * deviceName;
    unsigned int threadLimit;
    uint3 * tid;
    uint3 * bid;
    dim3 * blockDim;
    dim3 * gridDim;
    int * wSize;
} hip___hipRegisterFunction_api_args_t;

typedef void __hipRegisterFunction_return_t;


typedef struct hip___hipRegisterManagedVar_api_args_s {
    void * hipModule;
    void * * pointer;
    void * init_value;
    const char * name;
    size_t size;
    unsigned align;
} hip___hipRegisterManagedVar_api_args_t;

typedef void __hipRegisterManagedVar_return_t;


typedef struct hip___hipRegisterSurface_api_args_s {
    hip::FatBinaryInfo * * modules;
    void * var;
    char * hostVar;
    char * deviceVar;
    int type;
    int ext;
} hip___hipRegisterSurface_api_args_t;

typedef void __hipRegisterSurface_return_t;


typedef struct hip___hipRegisterTexture_api_args_s {
    hip::FatBinaryInfo * * modules;
    void * var;
    char * hostVar;
    char * deviceVar;
    int type;
    int norm;
    int ext;
} hip___hipRegisterTexture_api_args_t;

typedef void __hipRegisterTexture_return_t;


typedef struct hip___hipRegisterVar_api_args_s {
    hip::FatBinaryInfo * * modules;
    void * var;
    char * hostVar;
    char * deviceVar;
    int ext;
    size_t size;
    int constant;
    int global;
} hip___hipRegisterVar_api_args_t;

typedef void __hipRegisterVar_return_t;


typedef struct hip___hipUnregisterFatBinary_api_args_s {
    hip::FatBinaryInfo * * modules;
} hip___hipUnregisterFatBinary_api_args_t;

typedef void __hipUnregisterFatBinary_return_t;


typedef struct hip_hipApiName_api_args_s {
    uint32_t id;
} hip_hipApiName_api_args_t;

typedef const char * hipApiName_return_t;


typedef struct hip_hipArray3DCreate_api_args_s {
    hipArray_t * array;
    const HIP_ARRAY3D_DESCRIPTOR * pAllocateArray;
} hip_hipArray3DCreate_api_args_t;

typedef hipError_t hipArray3DCreate_return_t;


typedef struct hip_hipArray3DGetDescriptor_api_args_s {
    HIP_ARRAY3D_DESCRIPTOR * pArrayDescriptor;
    hipArray_t array;
} hip_hipArray3DGetDescriptor_api_args_t;

typedef hipError_t hipArray3DGetDescriptor_return_t;


typedef struct hip_hipArrayCreate_api_args_s {
    hipArray_t * pHandle;
    const HIP_ARRAY_DESCRIPTOR * pAllocateArray;
} hip_hipArrayCreate_api_args_t;

typedef hipError_t hipArrayCreate_return_t;


typedef struct hip_hipArrayDestroy_api_args_s {
    hipArray_t array;
} hip_hipArrayDestroy_api_args_t;

typedef hipError_t hipArrayDestroy_return_t;


typedef struct hip_hipArrayGetDescriptor_api_args_s {
    HIP_ARRAY_DESCRIPTOR * pArrayDescriptor;
    hipArray_t array;
} hip_hipArrayGetDescriptor_api_args_t;

typedef hipError_t hipArrayGetDescriptor_return_t;


typedef struct hip_hipArrayGetInfo_api_args_s {
    hipChannelFormatDesc * desc;
    hipExtent * extent;
    unsigned int * flags;
    hipArray_t array;
} hip_hipArrayGetInfo_api_args_t;

typedef hipError_t hipArrayGetInfo_return_t;


typedef struct hip_hipBindTexture_api_args_s {
    size_t * offset;
    const textureReference * tex;
    const void * devPtr;
    const hipChannelFormatDesc * desc;
    size_t size;
} hip_hipBindTexture_api_args_t;

typedef hipError_t hipBindTexture_return_t;


typedef struct hip_hipBindTexture2D_api_args_s {
    size_t * offset;
    const textureReference * tex;
    const void * devPtr;
    const hipChannelFormatDesc * desc;
    size_t width;
    size_t height;
    size_t pitch;
} hip_hipBindTexture2D_api_args_t;

typedef hipError_t hipBindTexture2D_return_t;


typedef struct hip_hipBindTextureToArray_api_args_s {
    const textureReference * tex;
    hipArray_const_t array;
    const hipChannelFormatDesc * desc;
} hip_hipBindTextureToArray_api_args_t;

typedef hipError_t hipBindTextureToArray_return_t;


typedef struct hip_hipBindTextureToMipmappedArray_api_args_s {
    const textureReference * tex;
    hipMipmappedArray_const_t mipmappedArray;
    const hipChannelFormatDesc * desc;
} hip_hipBindTextureToMipmappedArray_api_args_t;

typedef hipError_t hipBindTextureToMipmappedArray_return_t;


typedef struct hip_hipChooseDevice_api_args_s {
    int * device;
    const hipDeviceProp_t * prop;
} hip_hipChooseDevice_api_args_t;

typedef hipError_t hipChooseDevice_return_t;


typedef struct hip_hipConfigureCall_api_args_s {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    hipStream_t stream;
} hip_hipConfigureCall_api_args_t;

typedef hipError_t hipConfigureCall_return_t;


typedef struct hip_hipCreateSurfaceObject_api_args_s {
    hipSurfaceObject_t * pSurfObject;
    const hipResourceDesc * pResDesc;
} hip_hipCreateSurfaceObject_api_args_t;

typedef hipError_t hipCreateSurfaceObject_return_t;


typedef struct hip_hipCreateTextureObject_api_args_s {
    hipTextureObject_t * pTexObject;
    const hipResourceDesc * pResDesc;
    const hipTextureDesc * pTexDesc;
    const struct hipResourceViewDesc * pResViewDesc;
} hip_hipCreateTextureObject_api_args_t;

typedef hipError_t hipCreateTextureObject_return_t;


typedef struct hip_hipCtxCreate_api_args_s {
    hipCtx_t * ctx;
    unsigned int flags;
    hipDevice_t device;
} hip_hipCtxCreate_api_args_t;

typedef hipError_t hipCtxCreate_return_t;


typedef struct hip_hipCtxDestroy_api_args_s {
    hipCtx_t ctx;
} hip_hipCtxDestroy_api_args_t;

typedef hipError_t hipCtxDestroy_return_t;


typedef struct hip_hipCtxDisablePeerAccess_api_args_s {
    hipCtx_t peerCtx;
} hip_hipCtxDisablePeerAccess_api_args_t;

typedef hipError_t hipCtxDisablePeerAccess_return_t;


typedef struct hip_hipCtxEnablePeerAccess_api_args_s {
    hipCtx_t peerCtx;
    unsigned int flags;
} hip_hipCtxEnablePeerAccess_api_args_t;

typedef hipError_t hipCtxEnablePeerAccess_return_t;


typedef struct hip_hipCtxGetApiVersion_api_args_s {
    hipCtx_t ctx;
    int * apiVersion;
} hip_hipCtxGetApiVersion_api_args_t;

typedef hipError_t hipCtxGetApiVersion_return_t;


typedef struct hip_hipCtxGetCacheConfig_api_args_s {
    hipFuncCache_t * cacheConfig;
} hip_hipCtxGetCacheConfig_api_args_t;

typedef hipError_t hipCtxGetCacheConfig_return_t;


typedef struct hip_hipCtxGetCurrent_api_args_s {
    hipCtx_t * ctx;
} hip_hipCtxGetCurrent_api_args_t;

typedef hipError_t hipCtxGetCurrent_return_t;


typedef struct hip_hipCtxGetDevice_api_args_s {
    hipDevice_t * device;
} hip_hipCtxGetDevice_api_args_t;

typedef hipError_t hipCtxGetDevice_return_t;


typedef struct hip_hipCtxGetFlags_api_args_s {
    unsigned int * flags;
} hip_hipCtxGetFlags_api_args_t;

typedef hipError_t hipCtxGetFlags_return_t;


typedef struct hip_hipCtxGetSharedMemConfig_api_args_s {
    hipSharedMemConfig * pConfig;
} hip_hipCtxGetSharedMemConfig_api_args_t;

typedef hipError_t hipCtxGetSharedMemConfig_return_t;


typedef struct hip_hipCtxPopCurrent_api_args_s {
    hipCtx_t * ctx;
} hip_hipCtxPopCurrent_api_args_t;

typedef hipError_t hipCtxPopCurrent_return_t;


typedef struct hip_hipCtxPushCurrent_api_args_s {
    hipCtx_t ctx;
} hip_hipCtxPushCurrent_api_args_t;

typedef hipError_t hipCtxPushCurrent_return_t;


typedef struct hip_hipCtxSetCacheConfig_api_args_s {
    hipFuncCache_t cacheConfig;
} hip_hipCtxSetCacheConfig_api_args_t;

typedef hipError_t hipCtxSetCacheConfig_return_t;


typedef struct hip_hipCtxSetCurrent_api_args_s {
    hipCtx_t ctx;
} hip_hipCtxSetCurrent_api_args_t;

typedef hipError_t hipCtxSetCurrent_return_t;


typedef struct hip_hipCtxSetSharedMemConfig_api_args_s {
    hipSharedMemConfig config;
} hip_hipCtxSetSharedMemConfig_api_args_t;

typedef hipError_t hipCtxSetSharedMemConfig_return_t;


typedef struct hip_hipCtxSynchronize_api_args_s {
} hip_hipCtxSynchronize_api_args_t;

typedef hipError_t hipCtxSynchronize_return_t;


typedef struct hip_hipDestroyExternalMemory_api_args_s {
    hipExternalMemory_t extMem;
} hip_hipDestroyExternalMemory_api_args_t;

typedef hipError_t hipDestroyExternalMemory_return_t;


typedef struct hip_hipDestroyExternalSemaphore_api_args_s {
    hipExternalSemaphore_t extSem;
} hip_hipDestroyExternalSemaphore_api_args_t;

typedef hipError_t hipDestroyExternalSemaphore_return_t;


typedef struct hip_hipDestroySurfaceObject_api_args_s {
    hipSurfaceObject_t surfaceObject;
} hip_hipDestroySurfaceObject_api_args_t;

typedef hipError_t hipDestroySurfaceObject_return_t;


typedef struct hip_hipDestroyTextureObject_api_args_s {
    hipTextureObject_t textureObject;
} hip_hipDestroyTextureObject_api_args_t;

typedef hipError_t hipDestroyTextureObject_return_t;


typedef struct hip_hipDeviceCanAccessPeer_api_args_s {
    int * canAccessPeer;
    int deviceId;
    int peerDeviceId;
} hip_hipDeviceCanAccessPeer_api_args_t;

typedef hipError_t hipDeviceCanAccessPeer_return_t;


typedef struct hip_hipDeviceComputeCapability_api_args_s {
    int * major;
    int * minor;
    hipDevice_t device;
} hip_hipDeviceComputeCapability_api_args_t;

typedef hipError_t hipDeviceComputeCapability_return_t;


typedef struct hip_hipDeviceDisablePeerAccess_api_args_s {
    int peerDeviceId;
} hip_hipDeviceDisablePeerAccess_api_args_t;

typedef hipError_t hipDeviceDisablePeerAccess_return_t;


typedef struct hip_hipDeviceEnablePeerAccess_api_args_s {
    int peerDeviceId;
    unsigned int flags;
} hip_hipDeviceEnablePeerAccess_api_args_t;

typedef hipError_t hipDeviceEnablePeerAccess_return_t;


typedef struct hip_hipDeviceGet_api_args_s {
    hipDevice_t * device;
    int ordinal;
} hip_hipDeviceGet_api_args_t;

typedef hipError_t hipDeviceGet_return_t;


typedef struct hip_hipDeviceGetAttribute_api_args_s {
    int * pi;
    hipDeviceAttribute_t attr;
    int deviceId;
} hip_hipDeviceGetAttribute_api_args_t;

typedef hipError_t hipDeviceGetAttribute_return_t;


typedef struct hip_hipDeviceGetByPCIBusId_api_args_s {
    int * device;
    const char * pciBusId;
} hip_hipDeviceGetByPCIBusId_api_args_t;

typedef hipError_t hipDeviceGetByPCIBusId_return_t;


typedef struct hip_hipDeviceGetCacheConfig_api_args_s {
    hipFuncCache_t * cacheConfig;
} hip_hipDeviceGetCacheConfig_api_args_t;

typedef hipError_t hipDeviceGetCacheConfig_return_t;


typedef struct hip_hipDeviceGetCount_api_args_s {
    int * count;
} hip_hipDeviceGetCount_api_args_t;

typedef hipError_t hipDeviceGetCount_return_t;


typedef struct hip_hipDeviceGetDefaultMemPool_api_args_s {
    hipMemPool_t * mem_pool;
    int device;
} hip_hipDeviceGetDefaultMemPool_api_args_t;

typedef hipError_t hipDeviceGetDefaultMemPool_return_t;


typedef struct hip_hipDeviceGetGraphMemAttribute_api_args_s {
    int device;
    hipGraphMemAttributeType attr;
    void * value;
} hip_hipDeviceGetGraphMemAttribute_api_args_t;

typedef hipError_t hipDeviceGetGraphMemAttribute_return_t;


typedef struct hip_hipDeviceGetLimit_api_args_s {
    size_t * pValue;
    enum hipLimit_t limit;
} hip_hipDeviceGetLimit_api_args_t;

typedef hipError_t hipDeviceGetLimit_return_t;


typedef struct hip_hipDeviceGetMemPool_api_args_s {
    hipMemPool_t * mem_pool;
    int device;
} hip_hipDeviceGetMemPool_api_args_t;

typedef hipError_t hipDeviceGetMemPool_return_t;


typedef struct hip_hipDeviceGetName_api_args_s {
    char * name;
    int len;
    hipDevice_t device;
} hip_hipDeviceGetName_api_args_t;

typedef hipError_t hipDeviceGetName_return_t;


typedef struct hip_hipDeviceGetP2PAttribute_api_args_s {
    int * value;
    hipDeviceP2PAttr attr;
    int srcDevice;
    int dstDevice;
} hip_hipDeviceGetP2PAttribute_api_args_t;

typedef hipError_t hipDeviceGetP2PAttribute_return_t;


typedef struct hip_hipDeviceGetPCIBusId_api_args_s {
    char * pciBusId;
    int len;
    int device;
} hip_hipDeviceGetPCIBusId_api_args_t;

typedef hipError_t hipDeviceGetPCIBusId_return_t;


typedef struct hip_hipDeviceGetSharedMemConfig_api_args_s {
    hipSharedMemConfig * pConfig;
} hip_hipDeviceGetSharedMemConfig_api_args_t;

typedef hipError_t hipDeviceGetSharedMemConfig_return_t;


typedef struct hip_hipDeviceGetStreamPriorityRange_api_args_s {
    int * leastPriority;
    int * greatestPriority;
} hip_hipDeviceGetStreamPriorityRange_api_args_t;

typedef hipError_t hipDeviceGetStreamPriorityRange_return_t;


typedef struct hip_hipDeviceGetUuid_api_args_s {
    hipUUID * uuid;
    hipDevice_t device;
} hip_hipDeviceGetUuid_api_args_t;

typedef hipError_t hipDeviceGetUuid_return_t;


typedef struct hip_hipDeviceGraphMemTrim_api_args_s {
    int device;
} hip_hipDeviceGraphMemTrim_api_args_t;

typedef hipError_t hipDeviceGraphMemTrim_return_t;


typedef struct hip_hipDevicePrimaryCtxGetState_api_args_s {
    hipDevice_t dev;
    unsigned int * flags;
    int * active;
} hip_hipDevicePrimaryCtxGetState_api_args_t;

typedef hipError_t hipDevicePrimaryCtxGetState_return_t;


typedef struct hip_hipDevicePrimaryCtxRelease_api_args_s {
    hipDevice_t dev;
} hip_hipDevicePrimaryCtxRelease_api_args_t;

typedef hipError_t hipDevicePrimaryCtxRelease_return_t;


typedef struct hip_hipDevicePrimaryCtxReset_api_args_s {
    hipDevice_t dev;
} hip_hipDevicePrimaryCtxReset_api_args_t;

typedef hipError_t hipDevicePrimaryCtxReset_return_t;


typedef struct hip_hipDevicePrimaryCtxRetain_api_args_s {
    hipCtx_t * pctx;
    hipDevice_t dev;
} hip_hipDevicePrimaryCtxRetain_api_args_t;

typedef hipError_t hipDevicePrimaryCtxRetain_return_t;


typedef struct hip_hipDevicePrimaryCtxSetFlags_api_args_s {
    hipDevice_t dev;
    unsigned int flags;
} hip_hipDevicePrimaryCtxSetFlags_api_args_t;

typedef hipError_t hipDevicePrimaryCtxSetFlags_return_t;


typedef struct hip_hipDeviceReset_api_args_s {
} hip_hipDeviceReset_api_args_t;

typedef hipError_t hipDeviceReset_return_t;


typedef struct hip_hipDeviceSetCacheConfig_api_args_s {
    hipFuncCache_t cacheConfig;
} hip_hipDeviceSetCacheConfig_api_args_t;

typedef hipError_t hipDeviceSetCacheConfig_return_t;


typedef struct hip_hipDeviceSetGraphMemAttribute_api_args_s {
    int device;
    hipGraphMemAttributeType attr;
    void * value;
} hip_hipDeviceSetGraphMemAttribute_api_args_t;

typedef hipError_t hipDeviceSetGraphMemAttribute_return_t;


typedef struct hip_hipDeviceSetLimit_api_args_s {
    enum hipLimit_t limit;
    size_t value;
} hip_hipDeviceSetLimit_api_args_t;

typedef hipError_t hipDeviceSetLimit_return_t;


typedef struct hip_hipDeviceSetMemPool_api_args_s {
    int device;
    hipMemPool_t mem_pool;
} hip_hipDeviceSetMemPool_api_args_t;

typedef hipError_t hipDeviceSetMemPool_return_t;


typedef struct hip_hipDeviceSetSharedMemConfig_api_args_s {
    hipSharedMemConfig config;
} hip_hipDeviceSetSharedMemConfig_api_args_t;

typedef hipError_t hipDeviceSetSharedMemConfig_return_t;


typedef struct hip_hipDeviceSynchronize_api_args_s {
} hip_hipDeviceSynchronize_api_args_t;

typedef hipError_t hipDeviceSynchronize_return_t;


typedef struct hip_hipDeviceTotalMem_api_args_s {
    size_t * bytes;
    hipDevice_t device;
} hip_hipDeviceTotalMem_api_args_t;

typedef hipError_t hipDeviceTotalMem_return_t;


typedef struct hip_hipDriverGetVersion_api_args_s {
    int * driverVersion;
} hip_hipDriverGetVersion_api_args_t;

typedef hipError_t hipDriverGetVersion_return_t;


typedef struct hip_hipDrvGetErrorName_api_args_s {
    hipError_t hipError;
    const char * * errorString;
} hip_hipDrvGetErrorName_api_args_t;

typedef hipError_t hipDrvGetErrorName_return_t;


typedef struct hip_hipDrvGetErrorString_api_args_s {
    hipError_t hipError;
    const char * * errorString;
} hip_hipDrvGetErrorString_api_args_t;

typedef hipError_t hipDrvGetErrorString_return_t;


typedef struct hip_hipDrvGraphAddMemcpyNode_api_args_s {
    hipGraphNode_t * phGraphNode;
    hipGraph_t hGraph;
    const hipGraphNode_t * dependencies;
    size_t numDependencies;
    const HIP_MEMCPY3D * copyParams;
    hipCtx_t ctx;
} hip_hipDrvGraphAddMemcpyNode_api_args_t;

typedef hipError_t hipDrvGraphAddMemcpyNode_return_t;


typedef struct hip_hipDrvMemcpy2DUnaligned_api_args_s {
    const hip_Memcpy2D * pCopy;
} hip_hipDrvMemcpy2DUnaligned_api_args_t;

typedef hipError_t hipDrvMemcpy2DUnaligned_return_t;


typedef struct hip_hipDrvMemcpy3D_api_args_s {
    const HIP_MEMCPY3D * pCopy;
} hip_hipDrvMemcpy3D_api_args_t;

typedef hipError_t hipDrvMemcpy3D_return_t;


typedef struct hip_hipDrvMemcpy3DAsync_api_args_s {
    const HIP_MEMCPY3D * pCopy;
    hipStream_t stream;
} hip_hipDrvMemcpy3DAsync_api_args_t;

typedef hipError_t hipDrvMemcpy3DAsync_return_t;


typedef struct hip_hipDrvPointerGetAttributes_api_args_s {
    unsigned int numAttributes;
    hipPointer_attribute * attributes;
    void * * data;
    hipDeviceptr_t ptr;
} hip_hipDrvPointerGetAttributes_api_args_t;

typedef hipError_t hipDrvPointerGetAttributes_return_t;


typedef struct hip_hipEventCreate_api_args_s {
    hipEvent_t * event;
} hip_hipEventCreate_api_args_t;

typedef hipError_t hipEventCreate_return_t;


typedef struct hip_hipEventCreateWithFlags_api_args_s {
    hipEvent_t * event;
    unsigned flags;
} hip_hipEventCreateWithFlags_api_args_t;

typedef hipError_t hipEventCreateWithFlags_return_t;


typedef struct hip_hipEventDestroy_api_args_s {
    hipEvent_t event;
} hip_hipEventDestroy_api_args_t;

typedef hipError_t hipEventDestroy_return_t;


typedef struct hip_hipEventElapsedTime_api_args_s {
    float * ms;
    hipEvent_t start;
    hipEvent_t stop;
} hip_hipEventElapsedTime_api_args_t;

typedef hipError_t hipEventElapsedTime_return_t;


typedef struct hip_hipEventQuery_api_args_s {
    hipEvent_t event;
} hip_hipEventQuery_api_args_t;

typedef hipError_t hipEventQuery_return_t;


typedef struct hip_hipEventRecord_api_args_s {
    hipEvent_t event;
    hipStream_t stream;
} hip_hipEventRecord_api_args_t;

typedef hipError_t hipEventRecord_return_t;


typedef struct hip_hipEventSynchronize_api_args_s {
    hipEvent_t event;
} hip_hipEventSynchronize_api_args_t;

typedef hipError_t hipEventSynchronize_return_t;


typedef struct hip_hipExtGetLastError_api_args_s {
} hip_hipExtGetLastError_api_args_t;

typedef hipError_t hipExtGetLastError_return_t;


typedef struct hip_hipExtGetLinkTypeAndHopCount_api_args_s {
    int device1;
    int device2;
    uint32_t * linktype;
    uint32_t * hopcount;
} hip_hipExtGetLinkTypeAndHopCount_api_args_t;

typedef hipError_t hipExtGetLinkTypeAndHopCount_return_t;


typedef struct hip_hipExtLaunchKernel_api_args_s {
    const void * function_address;
    dim3 numBlocks;
    dim3 dimBlocks;
    void * * args;
    size_t sharedMemBytes;
    hipStream_t stream;
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
    int flags;
} hip_hipExtLaunchKernel_api_args_t;

typedef hipError_t hipExtLaunchKernel_return_t;


typedef struct hip_hipExtLaunchMultiKernelMultiDevice_api_args_s {
    hipLaunchParams * launchParamsList;
    int numDevices;
    unsigned int flags;
} hip_hipExtLaunchMultiKernelMultiDevice_api_args_t;

typedef hipError_t hipExtLaunchMultiKernelMultiDevice_return_t;


typedef struct hip_hipExtMallocWithFlags_api_args_s {
    void * * ptr;
    size_t sizeBytes;
    unsigned int flags;
} hip_hipExtMallocWithFlags_api_args_t;

typedef hipError_t hipExtMallocWithFlags_return_t;


typedef struct hip_hipExtModuleLaunchKernel_api_args_s {
    hipFunction_t f;
    uint32_t globalWorkSizeX;
    uint32_t globalWorkSizeY;
    uint32_t globalWorkSizeZ;
    uint32_t localWorkSizeX;
    uint32_t localWorkSizeY;
    uint32_t localWorkSizeZ;
    size_t sharedMemBytes;
    hipStream_t hStream;
    void * * kernelParams;
    void * * extra;
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
    uint32_t flags;
} hip_hipExtModuleLaunchKernel_api_args_t;

typedef hipError_t hipExtModuleLaunchKernel_return_t;


typedef struct hip_hipExtStreamCreateWithCUMask_api_args_s {
    hipStream_t * stream;
    uint32_t cuMaskSize;
    const uint32_t * cuMask;
} hip_hipExtStreamCreateWithCUMask_api_args_t;

typedef hipError_t hipExtStreamCreateWithCUMask_return_t;


typedef struct hip_hipExtStreamGetCUMask_api_args_s {
    hipStream_t stream;
    uint32_t cuMaskSize;
    uint32_t * cuMask;
} hip_hipExtStreamGetCUMask_api_args_t;

typedef hipError_t hipExtStreamGetCUMask_return_t;


typedef struct hip_hipExternalMemoryGetMappedBuffer_api_args_s {
    void * * devPtr;
    hipExternalMemory_t extMem;
    const hipExternalMemoryBufferDesc * bufferDesc;
} hip_hipExternalMemoryGetMappedBuffer_api_args_t;

typedef hipError_t hipExternalMemoryGetMappedBuffer_return_t;


typedef struct hip_hipExternalMemoryGetMappedMipmappedArray_api_args_s {
    hipMipmappedArray_t * mipmap;
    hipExternalMemory_t extMem;
    const hipExternalMemoryMipmappedArrayDesc * mipmapDesc;
} hip_hipExternalMemoryGetMappedMipmappedArray_api_args_t;

typedef hipError_t hipExternalMemoryGetMappedMipmappedArray_return_t;


typedef struct hip_hipFree_api_args_s {
    void * ptr;
} hip_hipFree_api_args_t;

typedef hipError_t hipFree_return_t;


typedef struct hip_hipFreeArray_api_args_s {
    hipArray_t array;
} hip_hipFreeArray_api_args_t;

typedef hipError_t hipFreeArray_return_t;


typedef struct hip_hipFreeAsync_api_args_s {
    void * dev_ptr;
    hipStream_t stream;
} hip_hipFreeAsync_api_args_t;

typedef hipError_t hipFreeAsync_return_t;


typedef struct hip_hipFreeHost_api_args_s {
    void * ptr;
} hip_hipFreeHost_api_args_t;

typedef hipError_t hipFreeHost_return_t;


typedef struct hip_hipFreeMipmappedArray_api_args_s {
    hipMipmappedArray_t mipmappedArray;
} hip_hipFreeMipmappedArray_api_args_t;

typedef hipError_t hipFreeMipmappedArray_return_t;


typedef struct hip_hipFuncGetAttribute_api_args_s {
    int * value;
    hipFunction_attribute attrib;
    hipFunction_t hfunc;
} hip_hipFuncGetAttribute_api_args_t;

typedef hipError_t hipFuncGetAttribute_return_t;


typedef struct hip_hipFuncGetAttributes_api_args_s {
    struct hipFuncAttributes * attr;
    const void * func;
} hip_hipFuncGetAttributes_api_args_t;

typedef hipError_t hipFuncGetAttributes_return_t;


typedef struct hip_hipFuncSetAttribute_api_args_s {
    const void * func;
    hipFuncAttribute attr;
    int value;
} hip_hipFuncSetAttribute_api_args_t;

typedef hipError_t hipFuncSetAttribute_return_t;


typedef struct hip_hipFuncSetCacheConfig_api_args_s {
    const void * func;
    hipFuncCache_t config;
} hip_hipFuncSetCacheConfig_api_args_t;

typedef hipError_t hipFuncSetCacheConfig_return_t;


typedef struct hip_hipFuncSetSharedMemConfig_api_args_s {
    const void * func;
    hipSharedMemConfig config;
} hip_hipFuncSetSharedMemConfig_api_args_t;

typedef hipError_t hipFuncSetSharedMemConfig_return_t;


typedef struct hip_hipGLGetDevices_api_args_s {
    unsigned int * pHipDeviceCount;
    int * pHipDevices;
    unsigned int hipDeviceCount;
    hipGLDeviceList deviceList;
} hip_hipGLGetDevices_api_args_t;

typedef hipError_t hipGLGetDevices_return_t;


typedef struct hip_hipGetChannelDesc_api_args_s {
    hipChannelFormatDesc * desc;
    hipArray_const_t array;
} hip_hipGetChannelDesc_api_args_t;

typedef hipError_t hipGetChannelDesc_return_t;


typedef struct hip_hipGetCmdName_api_args_s {
    unsigned op;
} hip_hipGetCmdName_api_args_t;

typedef const char * hipGetCmdName_return_t;


typedef struct hip_hipGetDevice_api_args_s {
    int * deviceId;
} hip_hipGetDevice_api_args_t;

typedef hipError_t hipGetDevice_return_t;


typedef struct hip_hipGetDeviceCount_api_args_s {
    int * count;
} hip_hipGetDeviceCount_api_args_t;

typedef hipError_t hipGetDeviceCount_return_t;


typedef struct hip_hipGetDeviceFlags_api_args_s {
    unsigned int * flags;
} hip_hipGetDeviceFlags_api_args_t;

typedef hipError_t hipGetDeviceFlags_return_t;


typedef struct hip_hipGetDeviceProperties_api_args_s {
    hipDeviceProp_t * prop;
    int deviceId;
} hip_hipGetDeviceProperties_api_args_t;

typedef hipError_t hipGetDeviceProperties_return_t;


typedef struct hip_hipGetErrorName_api_args_s {
    hipError_t hip_error;
} hip_hipGetErrorName_api_args_t;

typedef const char * hipGetErrorName_return_t;


typedef struct hip_hipGetErrorString_api_args_s {
    hipError_t hipError;
} hip_hipGetErrorString_api_args_t;

typedef const char * hipGetErrorString_return_t;


typedef struct hip_hipGetLastError_api_args_s {
} hip_hipGetLastError_api_args_t;

typedef hipError_t hipGetLastError_return_t;


typedef struct hip_hipGetMipmappedArrayLevel_api_args_s {
    hipArray_t * levelArray;
    hipMipmappedArray_const_t mipmappedArray;
    unsigned int level;
} hip_hipGetMipmappedArrayLevel_api_args_t;

typedef hipError_t hipGetMipmappedArrayLevel_return_t;


typedef struct hip_hipGetStreamDeviceId_api_args_s {
    hipStream_t stream;
} hip_hipGetStreamDeviceId_api_args_t;

typedef int hipGetStreamDeviceId_return_t;


typedef struct hip_hipGetSymbolAddress_api_args_s {
    void * * devPtr;
    const void * symbol;
} hip_hipGetSymbolAddress_api_args_t;

typedef hipError_t hipGetSymbolAddress_return_t;


typedef struct hip_hipGetSymbolSize_api_args_s {
    size_t * size;
    const void * symbol;
} hip_hipGetSymbolSize_api_args_t;

typedef hipError_t hipGetSymbolSize_return_t;


typedef struct hip_hipGetTextureAlignmentOffset_api_args_s {
    size_t * offset;
    const textureReference * texref;
} hip_hipGetTextureAlignmentOffset_api_args_t;

typedef hipError_t hipGetTextureAlignmentOffset_return_t;


typedef struct hip_hipGetTextureObjectResourceDesc_api_args_s {
    hipResourceDesc * pResDesc;
    hipTextureObject_t textureObject;
} hip_hipGetTextureObjectResourceDesc_api_args_t;

typedef hipError_t hipGetTextureObjectResourceDesc_return_t;


typedef struct hip_hipGetTextureObjectResourceViewDesc_api_args_s {
    struct hipResourceViewDesc * pResViewDesc;
    hipTextureObject_t textureObject;
} hip_hipGetTextureObjectResourceViewDesc_api_args_t;

typedef hipError_t hipGetTextureObjectResourceViewDesc_return_t;


typedef struct hip_hipGetTextureObjectTextureDesc_api_args_s {
    hipTextureDesc * pTexDesc;
    hipTextureObject_t textureObject;
} hip_hipGetTextureObjectTextureDesc_api_args_t;

typedef hipError_t hipGetTextureObjectTextureDesc_return_t;


typedef struct hip_hipGetTextureReference_api_args_s {
    const textureReference * * texref;
    const void * symbol;
} hip_hipGetTextureReference_api_args_t;

typedef hipError_t hipGetTextureReference_return_t;


typedef struct hip_hipGraphAddChildGraphNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    hipGraph_t childGraph;
} hip_hipGraphAddChildGraphNode_api_args_t;

typedef hipError_t hipGraphAddChildGraphNode_return_t;


typedef struct hip_hipGraphAddDependencies_api_args_s {
    hipGraph_t graph;
    const hipGraphNode_t * from;
    const hipGraphNode_t * to;
    size_t numDependencies;
} hip_hipGraphAddDependencies_api_args_t;

typedef hipError_t hipGraphAddDependencies_return_t;


typedef struct hip_hipGraphAddEmptyNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
} hip_hipGraphAddEmptyNode_api_args_t;

typedef hipError_t hipGraphAddEmptyNode_return_t;


typedef struct hip_hipGraphAddEventRecordNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    hipEvent_t event;
} hip_hipGraphAddEventRecordNode_api_args_t;

typedef hipError_t hipGraphAddEventRecordNode_return_t;


typedef struct hip_hipGraphAddEventWaitNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    hipEvent_t event;
} hip_hipGraphAddEventWaitNode_api_args_t;

typedef hipError_t hipGraphAddEventWaitNode_return_t;


typedef struct hip_hipGraphAddHostNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    const hipHostNodeParams * pNodeParams;
} hip_hipGraphAddHostNode_api_args_t;

typedef hipError_t hipGraphAddHostNode_return_t;


typedef struct hip_hipGraphAddKernelNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    const hipKernelNodeParams * pNodeParams;
} hip_hipGraphAddKernelNode_api_args_t;

typedef hipError_t hipGraphAddKernelNode_return_t;


typedef struct hip_hipGraphAddMemAllocNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    hipMemAllocNodeParams * pNodeParams;
} hip_hipGraphAddMemAllocNode_api_args_t;

typedef hipError_t hipGraphAddMemAllocNode_return_t;


typedef struct hip_hipGraphAddMemFreeNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    void * dev_ptr;
} hip_hipGraphAddMemFreeNode_api_args_t;

typedef hipError_t hipGraphAddMemFreeNode_return_t;


typedef struct hip_hipGraphAddMemcpyNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    const hipMemcpy3DParms * pCopyParams;
} hip_hipGraphAddMemcpyNode_api_args_t;

typedef hipError_t hipGraphAddMemcpyNode_return_t;


typedef struct hip_hipGraphAddMemcpyNode1D_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    void * dst;
    const void * src;
    size_t count;
    hipMemcpyKind kind;
} hip_hipGraphAddMemcpyNode1D_api_args_t;

typedef hipError_t hipGraphAddMemcpyNode1D_return_t;


typedef struct hip_hipGraphAddMemcpyNodeFromSymbol_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    void * dst;
    const void * symbol;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
} hip_hipGraphAddMemcpyNodeFromSymbol_api_args_t;

typedef hipError_t hipGraphAddMemcpyNodeFromSymbol_return_t;


typedef struct hip_hipGraphAddMemcpyNodeToSymbol_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    const void * symbol;
    const void * src;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
} hip_hipGraphAddMemcpyNodeToSymbol_api_args_t;

typedef hipError_t hipGraphAddMemcpyNodeToSymbol_return_t;


typedef struct hip_hipGraphAddMemsetNode_api_args_s {
    hipGraphNode_t * pGraphNode;
    hipGraph_t graph;
    const hipGraphNode_t * pDependencies;
    size_t numDependencies;
    const hipMemsetParams * pMemsetParams;
} hip_hipGraphAddMemsetNode_api_args_t;

typedef hipError_t hipGraphAddMemsetNode_return_t;


typedef struct hip_hipGraphChildGraphNodeGetGraph_api_args_s {
    hipGraphNode_t node;
    hipGraph_t * pGraph;
} hip_hipGraphChildGraphNodeGetGraph_api_args_t;

typedef hipError_t hipGraphChildGraphNodeGetGraph_return_t;


typedef struct hip_hipGraphClone_api_args_s {
    hipGraph_t * pGraphClone;
    hipGraph_t originalGraph;
} hip_hipGraphClone_api_args_t;

typedef hipError_t hipGraphClone_return_t;


typedef struct hip_hipGraphCreate_api_args_s {
    hipGraph_t * pGraph;
    unsigned int flags;
} hip_hipGraphCreate_api_args_t;

typedef hipError_t hipGraphCreate_return_t;


typedef struct hip_hipGraphDebugDotPrint_api_args_s {
    hipGraph_t graph;
    const char * path;
    unsigned int flags;
} hip_hipGraphDebugDotPrint_api_args_t;

typedef hipError_t hipGraphDebugDotPrint_return_t;


typedef struct hip_hipGraphDestroy_api_args_s {
    hipGraph_t graph;
} hip_hipGraphDestroy_api_args_t;

typedef hipError_t hipGraphDestroy_return_t;


typedef struct hip_hipGraphDestroyNode_api_args_s {
    hipGraphNode_t node;
} hip_hipGraphDestroyNode_api_args_t;

typedef hipError_t hipGraphDestroyNode_return_t;


typedef struct hip_hipGraphEventRecordNodeGetEvent_api_args_s {
    hipGraphNode_t node;
    hipEvent_t * event_out;
} hip_hipGraphEventRecordNodeGetEvent_api_args_t;

typedef hipError_t hipGraphEventRecordNodeGetEvent_return_t;


typedef struct hip_hipGraphEventRecordNodeSetEvent_api_args_s {
    hipGraphNode_t node;
    hipEvent_t event;
} hip_hipGraphEventRecordNodeSetEvent_api_args_t;

typedef hipError_t hipGraphEventRecordNodeSetEvent_return_t;


typedef struct hip_hipGraphEventWaitNodeGetEvent_api_args_s {
    hipGraphNode_t node;
    hipEvent_t * event_out;
} hip_hipGraphEventWaitNodeGetEvent_api_args_t;

typedef hipError_t hipGraphEventWaitNodeGetEvent_return_t;


typedef struct hip_hipGraphEventWaitNodeSetEvent_api_args_s {
    hipGraphNode_t node;
    hipEvent_t event;
} hip_hipGraphEventWaitNodeSetEvent_api_args_t;

typedef hipError_t hipGraphEventWaitNodeSetEvent_return_t;


typedef struct hip_hipGraphExecChildGraphNodeSetParams_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    hipGraph_t childGraph;
} hip_hipGraphExecChildGraphNodeSetParams_api_args_t;

typedef hipError_t hipGraphExecChildGraphNodeSetParams_return_t;


typedef struct hip_hipGraphExecDestroy_api_args_s {
    hipGraphExec_t graphExec;
} hip_hipGraphExecDestroy_api_args_t;

typedef hipError_t hipGraphExecDestroy_return_t;


typedef struct hip_hipGraphExecEventRecordNodeSetEvent_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    hipEvent_t event;
} hip_hipGraphExecEventRecordNodeSetEvent_api_args_t;

typedef hipError_t hipGraphExecEventRecordNodeSetEvent_return_t;


typedef struct hip_hipGraphExecEventWaitNodeSetEvent_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    hipEvent_t event;
} hip_hipGraphExecEventWaitNodeSetEvent_api_args_t;

typedef hipError_t hipGraphExecEventWaitNodeSetEvent_return_t;


typedef struct hip_hipGraphExecHostNodeSetParams_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    const hipHostNodeParams * pNodeParams;
} hip_hipGraphExecHostNodeSetParams_api_args_t;

typedef hipError_t hipGraphExecHostNodeSetParams_return_t;


typedef struct hip_hipGraphExecKernelNodeSetParams_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    const hipKernelNodeParams * pNodeParams;
} hip_hipGraphExecKernelNodeSetParams_api_args_t;

typedef hipError_t hipGraphExecKernelNodeSetParams_return_t;


typedef struct hip_hipGraphExecMemcpyNodeSetParams_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    hipMemcpy3DParms * pNodeParams;
} hip_hipGraphExecMemcpyNodeSetParams_api_args_t;

typedef hipError_t hipGraphExecMemcpyNodeSetParams_return_t;


typedef struct hip_hipGraphExecMemcpyNodeSetParams1D_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    void * dst;
    const void * src;
    size_t count;
    hipMemcpyKind kind;
} hip_hipGraphExecMemcpyNodeSetParams1D_api_args_t;

typedef hipError_t hipGraphExecMemcpyNodeSetParams1D_return_t;


typedef struct hip_hipGraphExecMemcpyNodeSetParamsFromSymbol_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    void * dst;
    const void * symbol;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
} hip_hipGraphExecMemcpyNodeSetParamsFromSymbol_api_args_t;

typedef hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol_return_t;


typedef struct hip_hipGraphExecMemcpyNodeSetParamsToSymbol_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    const void * symbol;
    const void * src;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
} hip_hipGraphExecMemcpyNodeSetParamsToSymbol_api_args_t;

typedef hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol_return_t;


typedef struct hip_hipGraphExecMemsetNodeSetParams_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t node;
    const hipMemsetParams * pNodeParams;
} hip_hipGraphExecMemsetNodeSetParams_api_args_t;

typedef hipError_t hipGraphExecMemsetNodeSetParams_return_t;


typedef struct hip_hipGraphExecUpdate_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraph_t hGraph;
    hipGraphNode_t * hErrorNode_out;
    hipGraphExecUpdateResult * updateResult_out;
} hip_hipGraphExecUpdate_api_args_t;

typedef hipError_t hipGraphExecUpdate_return_t;


typedef struct hip_hipGraphGetEdges_api_args_s {
    hipGraph_t graph;
    hipGraphNode_t * from;
    hipGraphNode_t * to;
    size_t * numEdges;
} hip_hipGraphGetEdges_api_args_t;

typedef hipError_t hipGraphGetEdges_return_t;


typedef struct hip_hipGraphGetNodes_api_args_s {
    hipGraph_t graph;
    hipGraphNode_t * nodes;
    size_t * numNodes;
} hip_hipGraphGetNodes_api_args_t;

typedef hipError_t hipGraphGetNodes_return_t;


typedef struct hip_hipGraphGetRootNodes_api_args_s {
    hipGraph_t graph;
    hipGraphNode_t * pRootNodes;
    size_t * pNumRootNodes;
} hip_hipGraphGetRootNodes_api_args_t;

typedef hipError_t hipGraphGetRootNodes_return_t;


typedef struct hip_hipGraphHostNodeGetParams_api_args_s {
    hipGraphNode_t node;
    hipHostNodeParams * pNodeParams;
} hip_hipGraphHostNodeGetParams_api_args_t;

typedef hipError_t hipGraphHostNodeGetParams_return_t;


typedef struct hip_hipGraphHostNodeSetParams_api_args_s {
    hipGraphNode_t node;
    const hipHostNodeParams * pNodeParams;
} hip_hipGraphHostNodeSetParams_api_args_t;

typedef hipError_t hipGraphHostNodeSetParams_return_t;


typedef struct hip_hipGraphInstantiate_api_args_s {
    hipGraphExec_t * pGraphExec;
    hipGraph_t graph;
    hipGraphNode_t * pErrorNode;
    char * pLogBuffer;
    size_t bufferSize;
} hip_hipGraphInstantiate_api_args_t;

typedef hipError_t hipGraphInstantiate_return_t;


typedef struct hip_hipGraphInstantiateWithFlags_api_args_s {
    hipGraphExec_t * pGraphExec;
    hipGraph_t graph;
    unsigned long long flags;
} hip_hipGraphInstantiateWithFlags_api_args_t;

typedef hipError_t hipGraphInstantiateWithFlags_return_t;


typedef struct hip_hipGraphKernelNodeCopyAttributes_api_args_s {
    hipGraphNode_t hSrc;
    hipGraphNode_t hDst;
} hip_hipGraphKernelNodeCopyAttributes_api_args_t;

typedef hipError_t hipGraphKernelNodeCopyAttributes_return_t;


typedef struct hip_hipGraphKernelNodeGetAttribute_api_args_s {
    hipGraphNode_t hNode;
    hipKernelNodeAttrID attr;
    hipKernelNodeAttrValue * value;
} hip_hipGraphKernelNodeGetAttribute_api_args_t;

typedef hipError_t hipGraphKernelNodeGetAttribute_return_t;


typedef struct hip_hipGraphKernelNodeGetParams_api_args_s {
    hipGraphNode_t node;
    hipKernelNodeParams * pNodeParams;
} hip_hipGraphKernelNodeGetParams_api_args_t;

typedef hipError_t hipGraphKernelNodeGetParams_return_t;


typedef struct hip_hipGraphKernelNodeSetAttribute_api_args_s {
    hipGraphNode_t hNode;
    hipKernelNodeAttrID attr;
    const hipKernelNodeAttrValue * value;
} hip_hipGraphKernelNodeSetAttribute_api_args_t;

typedef hipError_t hipGraphKernelNodeSetAttribute_return_t;


typedef struct hip_hipGraphKernelNodeSetParams_api_args_s {
    hipGraphNode_t node;
    const hipKernelNodeParams * pNodeParams;
} hip_hipGraphKernelNodeSetParams_api_args_t;

typedef hipError_t hipGraphKernelNodeSetParams_return_t;


typedef struct hip_hipGraphLaunch_api_args_s {
    hipGraphExec_t graphExec;
    hipStream_t stream;
} hip_hipGraphLaunch_api_args_t;

typedef hipError_t hipGraphLaunch_return_t;


typedef struct hip_hipGraphMemAllocNodeGetParams_api_args_s {
    hipGraphNode_t node;
    hipMemAllocNodeParams * pNodeParams;
} hip_hipGraphMemAllocNodeGetParams_api_args_t;

typedef hipError_t hipGraphMemAllocNodeGetParams_return_t;


typedef struct hip_hipGraphMemFreeNodeGetParams_api_args_s {
    hipGraphNode_t node;
    void * dev_ptr;
} hip_hipGraphMemFreeNodeGetParams_api_args_t;

typedef hipError_t hipGraphMemFreeNodeGetParams_return_t;


typedef struct hip_hipGraphMemcpyNodeGetParams_api_args_s {
    hipGraphNode_t node;
    hipMemcpy3DParms * pNodeParams;
} hip_hipGraphMemcpyNodeGetParams_api_args_t;

typedef hipError_t hipGraphMemcpyNodeGetParams_return_t;


typedef struct hip_hipGraphMemcpyNodeSetParams_api_args_s {
    hipGraphNode_t node;
    const hipMemcpy3DParms * pNodeParams;
} hip_hipGraphMemcpyNodeSetParams_api_args_t;

typedef hipError_t hipGraphMemcpyNodeSetParams_return_t;


typedef struct hip_hipGraphMemcpyNodeSetParams1D_api_args_s {
    hipGraphNode_t node;
    void * dst;
    const void * src;
    size_t count;
    hipMemcpyKind kind;
} hip_hipGraphMemcpyNodeSetParams1D_api_args_t;

typedef hipError_t hipGraphMemcpyNodeSetParams1D_return_t;


typedef struct hip_hipGraphMemcpyNodeSetParamsFromSymbol_api_args_s {
    hipGraphNode_t node;
    void * dst;
    const void * symbol;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
} hip_hipGraphMemcpyNodeSetParamsFromSymbol_api_args_t;

typedef hipError_t hipGraphMemcpyNodeSetParamsFromSymbol_return_t;


typedef struct hip_hipGraphMemcpyNodeSetParamsToSymbol_api_args_s {
    hipGraphNode_t node;
    const void * symbol;
    const void * src;
    size_t count;
    size_t offset;
    hipMemcpyKind kind;
} hip_hipGraphMemcpyNodeSetParamsToSymbol_api_args_t;

typedef hipError_t hipGraphMemcpyNodeSetParamsToSymbol_return_t;


typedef struct hip_hipGraphMemsetNodeGetParams_api_args_s {
    hipGraphNode_t node;
    hipMemsetParams * pNodeParams;
} hip_hipGraphMemsetNodeGetParams_api_args_t;

typedef hipError_t hipGraphMemsetNodeGetParams_return_t;


typedef struct hip_hipGraphMemsetNodeSetParams_api_args_s {
    hipGraphNode_t node;
    const hipMemsetParams * pNodeParams;
} hip_hipGraphMemsetNodeSetParams_api_args_t;

typedef hipError_t hipGraphMemsetNodeSetParams_return_t;


typedef struct hip_hipGraphNodeFindInClone_api_args_s {
    hipGraphNode_t * pNode;
    hipGraphNode_t originalNode;
    hipGraph_t clonedGraph;
} hip_hipGraphNodeFindInClone_api_args_t;

typedef hipError_t hipGraphNodeFindInClone_return_t;


typedef struct hip_hipGraphNodeGetDependencies_api_args_s {
    hipGraphNode_t node;
    hipGraphNode_t * pDependencies;
    size_t * pNumDependencies;
} hip_hipGraphNodeGetDependencies_api_args_t;

typedef hipError_t hipGraphNodeGetDependencies_return_t;


typedef struct hip_hipGraphNodeGetDependentNodes_api_args_s {
    hipGraphNode_t node;
    hipGraphNode_t * pDependentNodes;
    size_t * pNumDependentNodes;
} hip_hipGraphNodeGetDependentNodes_api_args_t;

typedef hipError_t hipGraphNodeGetDependentNodes_return_t;


typedef struct hip_hipGraphNodeGetEnabled_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    unsigned int * isEnabled;
} hip_hipGraphNodeGetEnabled_api_args_t;

typedef hipError_t hipGraphNodeGetEnabled_return_t;


typedef struct hip_hipGraphNodeGetType_api_args_s {
    hipGraphNode_t node;
    hipGraphNodeType * pType;
} hip_hipGraphNodeGetType_api_args_t;

typedef hipError_t hipGraphNodeGetType_return_t;


typedef struct hip_hipGraphNodeSetEnabled_api_args_s {
    hipGraphExec_t hGraphExec;
    hipGraphNode_t hNode;
    unsigned int isEnabled;
} hip_hipGraphNodeSetEnabled_api_args_t;

typedef hipError_t hipGraphNodeSetEnabled_return_t;


typedef struct hip_hipGraphReleaseUserObject_api_args_s {
    hipGraph_t graph;
    hipUserObject_t object;
    unsigned int count;
} hip_hipGraphReleaseUserObject_api_args_t;

typedef hipError_t hipGraphReleaseUserObject_return_t;


typedef struct hip_hipGraphRemoveDependencies_api_args_s {
    hipGraph_t graph;
    const hipGraphNode_t * from;
    const hipGraphNode_t * to;
    size_t numDependencies;
} hip_hipGraphRemoveDependencies_api_args_t;

typedef hipError_t hipGraphRemoveDependencies_return_t;


typedef struct hip_hipGraphRetainUserObject_api_args_s {
    hipGraph_t graph;
    hipUserObject_t object;
    unsigned int count;
    unsigned int flags;
} hip_hipGraphRetainUserObject_api_args_t;

typedef hipError_t hipGraphRetainUserObject_return_t;


typedef struct hip_hipGraphUpload_api_args_s {
    hipGraphExec_t graphExec;
    hipStream_t stream;
} hip_hipGraphUpload_api_args_t;

typedef hipError_t hipGraphUpload_return_t;


typedef struct hip_hipGraphicsGLRegisterBuffer_api_args_s {
    hipGraphicsResource * * resource;
    GLuint buffer;
    unsigned int flags;
} hip_hipGraphicsGLRegisterBuffer_api_args_t;

typedef hipError_t hipGraphicsGLRegisterBuffer_return_t;


typedef struct hip_hipGraphicsGLRegisterImage_api_args_s {
    hipGraphicsResource * * resource;
    GLuint image;
    GLenum target;
    unsigned int flags;
} hip_hipGraphicsGLRegisterImage_api_args_t;

typedef hipError_t hipGraphicsGLRegisterImage_return_t;


typedef struct hip_hipGraphicsMapResources_api_args_s {
    int count;
    hipGraphicsResource_t * resources;
    hipStream_t stream;
} hip_hipGraphicsMapResources_api_args_t;

typedef hipError_t hipGraphicsMapResources_return_t;


typedef struct hip_hipGraphicsResourceGetMappedPointer_api_args_s {
    void * * devPtr;
    size_t * size;
    hipGraphicsResource_t resource;
} hip_hipGraphicsResourceGetMappedPointer_api_args_t;

typedef hipError_t hipGraphicsResourceGetMappedPointer_return_t;


typedef struct hip_hipGraphicsSubResourceGetMappedArray_api_args_s {
    hipArray_t * array;
    hipGraphicsResource_t resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
} hip_hipGraphicsSubResourceGetMappedArray_api_args_t;

typedef hipError_t hipGraphicsSubResourceGetMappedArray_return_t;


typedef struct hip_hipGraphicsUnmapResources_api_args_s {
    int count;
    hipGraphicsResource_t * resources;
    hipStream_t stream;
} hip_hipGraphicsUnmapResources_api_args_t;

typedef hipError_t hipGraphicsUnmapResources_return_t;


typedef struct hip_hipGraphicsUnregisterResource_api_args_s {
    hipGraphicsResource_t resource;
} hip_hipGraphicsUnregisterResource_api_args_t;

typedef hipError_t hipGraphicsUnregisterResource_return_t;


typedef struct hip_hipHccModuleLaunchKernel_api_args_s {
    hipFunction_t f;
    uint32_t globalWorkSizeX;
    uint32_t globalWorkSizeY;
    uint32_t globalWorkSizeZ;
    uint32_t blockDimX;
    uint32_t blockDimY;
    uint32_t blockDimZ;
    size_t sharedMemBytes;
    hipStream_t hStream;
    void * * kernelParams;
    void * * extra;
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
} hip_hipHccModuleLaunchKernel_api_args_t;

typedef hipError_t hipHccModuleLaunchKernel_return_t;


typedef struct hip_hipHostAlloc_api_args_s {
    void * * ptr;
    size_t size;
    unsigned int flags;
} hip_hipHostAlloc_api_args_t;

typedef hipError_t hipHostAlloc_return_t;


typedef struct hip_hipHostFree_api_args_s {
    void * ptr;
} hip_hipHostFree_api_args_t;

typedef hipError_t hipHostFree_return_t;


typedef struct hip_hipHostGetDevicePointer_api_args_s {
    void * * devPtr;
    void * hstPtr;
    unsigned int flags;
} hip_hipHostGetDevicePointer_api_args_t;

typedef hipError_t hipHostGetDevicePointer_return_t;


typedef struct hip_hipHostGetFlags_api_args_s {
    unsigned int * flagsPtr;
    void * hostPtr;
} hip_hipHostGetFlags_api_args_t;

typedef hipError_t hipHostGetFlags_return_t;


typedef struct hip_hipHostMalloc_api_args_s {
    void * * ptr;
    size_t size;
    unsigned int flags;
} hip_hipHostMalloc_api_args_t;

typedef hipError_t hipHostMalloc_return_t;


typedef struct hip_hipHostRegister_api_args_s {
    void * hostPtr;
    size_t sizeBytes;
    unsigned int flags;
} hip_hipHostRegister_api_args_t;

typedef hipError_t hipHostRegister_return_t;


typedef struct hip_hipHostUnregister_api_args_s {
    void * hostPtr;
} hip_hipHostUnregister_api_args_t;

typedef hipError_t hipHostUnregister_return_t;


typedef struct hip_hipImportExternalMemory_api_args_s {
    hipExternalMemory_t * extMem_out;
    const hipExternalMemoryHandleDesc * memHandleDesc;
} hip_hipImportExternalMemory_api_args_t;

typedef hipError_t hipImportExternalMemory_return_t;


typedef struct hip_hipImportExternalSemaphore_api_args_s {
    hipExternalSemaphore_t * extSem_out;
    const hipExternalSemaphoreHandleDesc * semHandleDesc;
} hip_hipImportExternalSemaphore_api_args_t;

typedef hipError_t hipImportExternalSemaphore_return_t;


typedef struct hip_hipInit_api_args_s {
    unsigned int flags;
} hip_hipInit_api_args_t;

typedef hipError_t hipInit_return_t;


typedef struct hip_hipIpcCloseMemHandle_api_args_s {
    void * devPtr;
} hip_hipIpcCloseMemHandle_api_args_t;

typedef hipError_t hipIpcCloseMemHandle_return_t;


typedef struct hip_hipIpcGetEventHandle_api_args_s {
    hipIpcEventHandle_t * handle;
    hipEvent_t event;
} hip_hipIpcGetEventHandle_api_args_t;

typedef hipError_t hipIpcGetEventHandle_return_t;


typedef struct hip_hipIpcGetMemHandle_api_args_s {
    hipIpcMemHandle_t * handle;
    void * devPtr;
} hip_hipIpcGetMemHandle_api_args_t;

typedef hipError_t hipIpcGetMemHandle_return_t;


typedef struct hip_hipIpcOpenEventHandle_api_args_s {
    hipEvent_t * event;
    hipIpcEventHandle_t handle;
} hip_hipIpcOpenEventHandle_api_args_t;

typedef hipError_t hipIpcOpenEventHandle_return_t;


typedef struct hip_hipIpcOpenMemHandle_api_args_s {
    void * * devPtr;
    hipIpcMemHandle_t handle;
    unsigned int flags;
} hip_hipIpcOpenMemHandle_api_args_t;

typedef hipError_t hipIpcOpenMemHandle_return_t;


typedef struct hip_hipKernelNameRef_api_args_s {
    const hipFunction_t f;
} hip_hipKernelNameRef_api_args_t;

typedef const char * hipKernelNameRef_return_t;


typedef struct hip_hipKernelNameRefByPtr_api_args_s {
    const void * hostFunction;
    hipStream_t stream;
} hip_hipKernelNameRefByPtr_api_args_t;

typedef const char * hipKernelNameRefByPtr_return_t;


typedef struct hip_hipLaunchByPtr_api_args_s {
    const void * func;
} hip_hipLaunchByPtr_api_args_t;

typedef hipError_t hipLaunchByPtr_return_t;


typedef struct hip_hipLaunchCooperativeKernel_api_args_s {
    const void * f;
    dim3 gridDim;
    dim3 blockDimX;
    void * * kernelParams;
    unsigned int sharedMemBytes;
    hipStream_t stream;
} hip_hipLaunchCooperativeKernel_api_args_t;

typedef hipError_t hipLaunchCooperativeKernel_return_t;


typedef struct hip_hipLaunchCooperativeKernelMultiDevice_api_args_s {
    hipLaunchParams * launchParamsList;
    int numDevices;
    unsigned int flags;
} hip_hipLaunchCooperativeKernelMultiDevice_api_args_t;

typedef hipError_t hipLaunchCooperativeKernelMultiDevice_return_t;


typedef struct hip_hipLaunchHostFunc_api_args_s {
    hipStream_t stream;
    hipHostFn_t fn;
    void * userData;
} hip_hipLaunchHostFunc_api_args_t;

typedef hipError_t hipLaunchHostFunc_return_t;


typedef struct hip_hipLaunchKernel_api_args_s {
    const void * function_address;
    dim3 numBlocks;
    dim3 dimBlocks;
    void * * args;
    size_t sharedMemBytes;
    hipStream_t stream;
} hip_hipLaunchKernel_api_args_t;

typedef hipError_t hipLaunchKernel_return_t;


typedef struct hip_hipLaunchKernel_common_api_args_s {
    const void * hostFunction;
    dim3 gridDim;
    dim3 blockDim;
    void * * args;
    size_t sharedMemBytes;
    hipStream_t stream;
} hip_hipLaunchKernel_common_api_args_t;

typedef hipError_t hipLaunchKernel_common_return_t;


typedef struct hip_hipLaunchKernel_spt_api_args_s {
    const void * hostFunction;
    dim3 gridDim;
    dim3 blockDim;
    void * * args;
    size_t sharedMemBytes;
    hipStream_t stream;
} hip_hipLaunchKernel_spt_api_args_t;

typedef hipError_t hipLaunchKernel_spt_return_t;


typedef struct hip_hipMalloc_api_args_s {
    void * * ptr;
    size_t size;
} hip_hipMalloc_api_args_t;

typedef hipError_t hipMalloc_return_t;


typedef struct hip_hipMalloc3D_api_args_s {
    hipPitchedPtr * pitchedDevPtr;
    hipExtent extent;
} hip_hipMalloc3D_api_args_t;

typedef hipError_t hipMalloc3D_return_t;


typedef struct hip_hipMalloc3DArray_api_args_s {
    hipArray_t * array;
    const struct hipChannelFormatDesc * desc;
    struct hipExtent extent;
    unsigned int flags;
} hip_hipMalloc3DArray_api_args_t;

typedef hipError_t hipMalloc3DArray_return_t;


typedef struct hip_hipMallocArray_api_args_s {
    hipArray_t * array;
    const hipChannelFormatDesc * desc;
    size_t width;
    size_t height;
    unsigned int flags;
} hip_hipMallocArray_api_args_t;

typedef hipError_t hipMallocArray_return_t;


typedef struct hip_hipMallocAsync_api_args_s {
    void * * dev_ptr;
    size_t size;
    hipStream_t stream;
} hip_hipMallocAsync_api_args_t;

typedef hipError_t hipMallocAsync_return_t;


typedef struct hip_hipMallocFromPoolAsync_api_args_s {
    void * * dev_ptr;
    size_t size;
    hipMemPool_t mem_pool;
    hipStream_t stream;
} hip_hipMallocFromPoolAsync_api_args_t;

typedef hipError_t hipMallocFromPoolAsync_return_t;


typedef struct hip_hipMallocHost_api_args_s {
    void * * ptr;
    size_t size;
} hip_hipMallocHost_api_args_t;

typedef hipError_t hipMallocHost_return_t;


typedef struct hip_hipMallocManaged_api_args_s {
    void * * dev_ptr;
    size_t size;
    unsigned int flags;
} hip_hipMallocManaged_api_args_t;

typedef hipError_t hipMallocManaged_return_t;


typedef struct hip_hipMallocMipmappedArray_api_args_s {
    hipMipmappedArray_t * mipmappedArray;
    const struct hipChannelFormatDesc * desc;
    struct hipExtent extent;
    unsigned int numLevels;
    unsigned int flags;
} hip_hipMallocMipmappedArray_api_args_t;

typedef hipError_t hipMallocMipmappedArray_return_t;


typedef struct hip_hipMallocPitch_api_args_s {
    void * * ptr;
    size_t * pitch;
    size_t width;
    size_t height;
} hip_hipMallocPitch_api_args_t;

typedef hipError_t hipMallocPitch_return_t;


typedef struct hip_hipMemAddressFree_api_args_s {
    void * devPtr;
    size_t size;
} hip_hipMemAddressFree_api_args_t;

typedef hipError_t hipMemAddressFree_return_t;


typedef struct hip_hipMemAddressReserve_api_args_s {
    void * * ptr;
    size_t size;
    size_t alignment;
    void * addr;
    unsigned long long flags;
} hip_hipMemAddressReserve_api_args_t;

typedef hipError_t hipMemAddressReserve_return_t;


typedef struct hip_hipMemAdvise_api_args_s {
    const void * dev_ptr;
    size_t count;
    hipMemoryAdvise advice;
    int device;
} hip_hipMemAdvise_api_args_t;

typedef hipError_t hipMemAdvise_return_t;


typedef struct hip_hipMemAllocHost_api_args_s {
    void * * ptr;
    size_t size;
} hip_hipMemAllocHost_api_args_t;

typedef hipError_t hipMemAllocHost_return_t;


typedef struct hip_hipMemAllocPitch_api_args_s {
    hipDeviceptr_t * dptr;
    size_t * pitch;
    size_t widthInBytes;
    size_t height;
    unsigned int elementSizeBytes;
} hip_hipMemAllocPitch_api_args_t;

typedef hipError_t hipMemAllocPitch_return_t;


typedef struct hip_hipMemCreate_api_args_s {
    hipMemGenericAllocationHandle_t * handle;
    size_t size;
    const hipMemAllocationProp * prop;
    unsigned long long flags;
} hip_hipMemCreate_api_args_t;

typedef hipError_t hipMemCreate_return_t;


typedef struct hip_hipMemExportToShareableHandle_api_args_s {
    void * shareableHandle;
    hipMemGenericAllocationHandle_t handle;
    hipMemAllocationHandleType handleType;
    unsigned long long flags;
} hip_hipMemExportToShareableHandle_api_args_t;

typedef hipError_t hipMemExportToShareableHandle_return_t;


typedef struct hip_hipMemGetAccess_api_args_s {
    unsigned long long * flags;
    const hipMemLocation * location;
    void * ptr;
} hip_hipMemGetAccess_api_args_t;

typedef hipError_t hipMemGetAccess_return_t;


typedef struct hip_hipMemGetAddressRange_api_args_s {
    hipDeviceptr_t * pbase;
    size_t * psize;
    hipDeviceptr_t dptr;
} hip_hipMemGetAddressRange_api_args_t;

typedef hipError_t hipMemGetAddressRange_return_t;


typedef struct hip_hipMemGetAllocationGranularity_api_args_s {
    size_t * granularity;
    const hipMemAllocationProp * prop;
    hipMemAllocationGranularity_flags option;
} hip_hipMemGetAllocationGranularity_api_args_t;

typedef hipError_t hipMemGetAllocationGranularity_return_t;


typedef struct hip_hipMemGetAllocationPropertiesFromHandle_api_args_s {
    hipMemAllocationProp * prop;
    hipMemGenericAllocationHandle_t handle;
} hip_hipMemGetAllocationPropertiesFromHandle_api_args_t;

typedef hipError_t hipMemGetAllocationPropertiesFromHandle_return_t;


typedef struct hip_hipMemGetInfo_api_args_s {
    size_t * free;
    size_t * total;
} hip_hipMemGetInfo_api_args_t;

typedef hipError_t hipMemGetInfo_return_t;


typedef struct hip_hipMemImportFromShareableHandle_api_args_s {
    hipMemGenericAllocationHandle_t * handle;
    void * osHandle;
    hipMemAllocationHandleType shHandleType;
} hip_hipMemImportFromShareableHandle_api_args_t;

typedef hipError_t hipMemImportFromShareableHandle_return_t;


typedef struct hip_hipMemMap_api_args_s {
    void * ptr;
    size_t size;
    size_t offset;
    hipMemGenericAllocationHandle_t handle;
    unsigned long long flags;
} hip_hipMemMap_api_args_t;

typedef hipError_t hipMemMap_return_t;


typedef struct hip_hipMemMapArrayAsync_api_args_s {
    hipArrayMapInfo * mapInfoList;
    unsigned int count;
    hipStream_t stream;
} hip_hipMemMapArrayAsync_api_args_t;

typedef hipError_t hipMemMapArrayAsync_return_t;


typedef struct hip_hipMemPoolCreate_api_args_s {
    hipMemPool_t * mem_pool;
    const hipMemPoolProps * pool_props;
} hip_hipMemPoolCreate_api_args_t;

typedef hipError_t hipMemPoolCreate_return_t;


typedef struct hip_hipMemPoolDestroy_api_args_s {
    hipMemPool_t mem_pool;
} hip_hipMemPoolDestroy_api_args_t;

typedef hipError_t hipMemPoolDestroy_return_t;


typedef struct hip_hipMemPoolExportPointer_api_args_s {
    hipMemPoolPtrExportData * export_data;
    void * dev_ptr;
} hip_hipMemPoolExportPointer_api_args_t;

typedef hipError_t hipMemPoolExportPointer_return_t;


typedef struct hip_hipMemPoolExportToShareableHandle_api_args_s {
    void * shared_handle;
    hipMemPool_t mem_pool;
    hipMemAllocationHandleType handle_type;
    unsigned int flags;
} hip_hipMemPoolExportToShareableHandle_api_args_t;

typedef hipError_t hipMemPoolExportToShareableHandle_return_t;


typedef struct hip_hipMemPoolGetAccess_api_args_s {
    hipMemAccessFlags * flags;
    hipMemPool_t mem_pool;
    hipMemLocation * location;
} hip_hipMemPoolGetAccess_api_args_t;

typedef hipError_t hipMemPoolGetAccess_return_t;


typedef struct hip_hipMemPoolGetAttribute_api_args_s {
    hipMemPool_t mem_pool;
    hipMemPoolAttr attr;
    void * value;
} hip_hipMemPoolGetAttribute_api_args_t;

typedef hipError_t hipMemPoolGetAttribute_return_t;


typedef struct hip_hipMemPoolImportFromShareableHandle_api_args_s {
    hipMemPool_t * mem_pool;
    void * shared_handle;
    hipMemAllocationHandleType handle_type;
    unsigned int flags;
} hip_hipMemPoolImportFromShareableHandle_api_args_t;

typedef hipError_t hipMemPoolImportFromShareableHandle_return_t;


typedef struct hip_hipMemPoolImportPointer_api_args_s {
    void * * dev_ptr;
    hipMemPool_t mem_pool;
    hipMemPoolPtrExportData * export_data;
} hip_hipMemPoolImportPointer_api_args_t;

typedef hipError_t hipMemPoolImportPointer_return_t;


typedef struct hip_hipMemPoolSetAccess_api_args_s {
    hipMemPool_t mem_pool;
    const hipMemAccessDesc * desc_list;
    size_t count;
} hip_hipMemPoolSetAccess_api_args_t;

typedef hipError_t hipMemPoolSetAccess_return_t;


typedef struct hip_hipMemPoolSetAttribute_api_args_s {
    hipMemPool_t mem_pool;
    hipMemPoolAttr attr;
    void * value;
} hip_hipMemPoolSetAttribute_api_args_t;

typedef hipError_t hipMemPoolSetAttribute_return_t;


typedef struct hip_hipMemPoolTrimTo_api_args_s {
    hipMemPool_t mem_pool;
    size_t min_bytes_to_hold;
} hip_hipMemPoolTrimTo_api_args_t;

typedef hipError_t hipMemPoolTrimTo_return_t;


typedef struct hip_hipMemPrefetchAsync_api_args_s {
    const void * dev_ptr;
    size_t count;
    int device;
    hipStream_t stream;
} hip_hipMemPrefetchAsync_api_args_t;

typedef hipError_t hipMemPrefetchAsync_return_t;


typedef struct hip_hipMemPtrGetInfo_api_args_s {
    void * ptr;
    size_t * size;
} hip_hipMemPtrGetInfo_api_args_t;

typedef hipError_t hipMemPtrGetInfo_return_t;


typedef struct hip_hipMemRangeGetAttribute_api_args_s {
    void * data;
    size_t data_size;
    hipMemRangeAttribute attribute;
    const void * dev_ptr;
    size_t count;
} hip_hipMemRangeGetAttribute_api_args_t;

typedef hipError_t hipMemRangeGetAttribute_return_t;


typedef struct hip_hipMemRangeGetAttributes_api_args_s {
    void * * data;
    size_t * data_sizes;
    hipMemRangeAttribute * attributes;
    size_t num_attributes;
    const void * dev_ptr;
    size_t count;
} hip_hipMemRangeGetAttributes_api_args_t;

typedef hipError_t hipMemRangeGetAttributes_return_t;


typedef struct hip_hipMemRelease_api_args_s {
    hipMemGenericAllocationHandle_t handle;
} hip_hipMemRelease_api_args_t;

typedef hipError_t hipMemRelease_return_t;


typedef struct hip_hipMemRetainAllocationHandle_api_args_s {
    hipMemGenericAllocationHandle_t * handle;
    void * addr;
} hip_hipMemRetainAllocationHandle_api_args_t;

typedef hipError_t hipMemRetainAllocationHandle_return_t;


typedef struct hip_hipMemSetAccess_api_args_s {
    void * ptr;
    size_t size;
    const hipMemAccessDesc * desc;
    size_t count;
} hip_hipMemSetAccess_api_args_t;

typedef hipError_t hipMemSetAccess_return_t;


typedef struct hip_hipMemUnmap_api_args_s {
    void * ptr;
    size_t size;
} hip_hipMemUnmap_api_args_t;

typedef hipError_t hipMemUnmap_return_t;


typedef struct hip_hipMemcpy_api_args_s {
    void * dst;
    const void * src;
    size_t sizeBytes;
    hipMemcpyKind kind;
} hip_hipMemcpy_api_args_t;

typedef hipError_t hipMemcpy_return_t;


typedef struct hip_hipMemcpy2D_api_args_s {
    void * dst;
    size_t dpitch;
    const void * src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
} hip_hipMemcpy2D_api_args_t;

typedef hipError_t hipMemcpy2D_return_t;


typedef struct hip_hipMemcpy2DAsync_api_args_s {
    void * dst;
    size_t dpitch;
    const void * src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
} hip_hipMemcpy2DAsync_api_args_t;

typedef hipError_t hipMemcpy2DAsync_return_t;


typedef struct hip_hipMemcpy2DFromArray_api_args_s {
    void * dst;
    size_t dpitch;
    hipArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
} hip_hipMemcpy2DFromArray_api_args_t;

typedef hipError_t hipMemcpy2DFromArray_return_t;


typedef struct hip_hipMemcpy2DFromArrayAsync_api_args_s {
    void * dst;
    size_t dpitch;
    hipArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
} hip_hipMemcpy2DFromArrayAsync_api_args_t;

typedef hipError_t hipMemcpy2DFromArrayAsync_return_t;


typedef struct hip_hipMemcpy2DToArray_api_args_s {
    hipArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void * src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
} hip_hipMemcpy2DToArray_api_args_t;

typedef hipError_t hipMemcpy2DToArray_return_t;


typedef struct hip_hipMemcpy2DToArrayAsync_api_args_s {
    hipArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void * src;
    size_t spitch;
    size_t width;
    size_t height;
    hipMemcpyKind kind;
    hipStream_t stream;
} hip_hipMemcpy2DToArrayAsync_api_args_t;

typedef hipError_t hipMemcpy2DToArrayAsync_return_t;


typedef struct hip_hipMemcpy3D_api_args_s {
    const struct hipMemcpy3DParms * p;
} hip_hipMemcpy3D_api_args_t;

typedef hipError_t hipMemcpy3D_return_t;


typedef struct hip_hipMemcpy3DAsync_api_args_s {
    const struct hipMemcpy3DParms * p;
    hipStream_t stream;
} hip_hipMemcpy3DAsync_api_args_t;

typedef hipError_t hipMemcpy3DAsync_return_t;


typedef struct hip_hipMemcpyAsync_api_args_s {
    void * dst;
    const void * src;
    size_t sizeBytes;
    hipMemcpyKind kind;
    hipStream_t stream;
} hip_hipMemcpyAsync_api_args_t;

typedef hipError_t hipMemcpyAsync_return_t;


typedef struct hip_hipMemcpyAtoH_api_args_s {
    void * dst;
    hipArray_t srcArray;
    size_t srcOffset;
    size_t count;
} hip_hipMemcpyAtoH_api_args_t;

typedef hipError_t hipMemcpyAtoH_return_t;


typedef struct hip_hipMemcpyDtoD_api_args_s {
    hipDeviceptr_t dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
} hip_hipMemcpyDtoD_api_args_t;

typedef hipError_t hipMemcpyDtoD_return_t;


typedef struct hip_hipMemcpyDtoDAsync_api_args_s {
    hipDeviceptr_t dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
    hipStream_t stream;
} hip_hipMemcpyDtoDAsync_api_args_t;

typedef hipError_t hipMemcpyDtoDAsync_return_t;


typedef struct hip_hipMemcpyDtoH_api_args_s {
    void * dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
} hip_hipMemcpyDtoH_api_args_t;

typedef hipError_t hipMemcpyDtoH_return_t;


typedef struct hip_hipMemcpyDtoHAsync_api_args_s {
    void * dst;
    hipDeviceptr_t src;
    size_t sizeBytes;
    hipStream_t stream;
} hip_hipMemcpyDtoHAsync_api_args_t;

typedef hipError_t hipMemcpyDtoHAsync_return_t;


typedef struct hip_hipMemcpyFromArray_api_args_s {
    void * dst;
    hipArray_const_t srcArray;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    hipMemcpyKind kind;
} hip_hipMemcpyFromArray_api_args_t;

typedef hipError_t hipMemcpyFromArray_return_t;


typedef struct hip_hipMemcpyFromSymbol_api_args_s {
    void * dst;
    const void * symbol;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
} hip_hipMemcpyFromSymbol_api_args_t;

typedef hipError_t hipMemcpyFromSymbol_return_t;


typedef struct hip_hipMemcpyFromSymbolAsync_api_args_s {
    void * dst;
    const void * symbol;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
    hipStream_t stream;
} hip_hipMemcpyFromSymbolAsync_api_args_t;

typedef hipError_t hipMemcpyFromSymbolAsync_return_t;


typedef struct hip_hipMemcpyHtoA_api_args_s {
    hipArray_t dstArray;
    size_t dstOffset;
    const void * srcHost;
    size_t count;
} hip_hipMemcpyHtoA_api_args_t;

typedef hipError_t hipMemcpyHtoA_return_t;


typedef struct hip_hipMemcpyHtoD_api_args_s {
    hipDeviceptr_t dst;
    void * src;
    size_t sizeBytes;
} hip_hipMemcpyHtoD_api_args_t;

typedef hipError_t hipMemcpyHtoD_return_t;


typedef struct hip_hipMemcpyHtoDAsync_api_args_s {
    hipDeviceptr_t dst;
    void * src;
    size_t sizeBytes;
    hipStream_t stream;
} hip_hipMemcpyHtoDAsync_api_args_t;

typedef hipError_t hipMemcpyHtoDAsync_return_t;


typedef struct hip_hipMemcpyParam2D_api_args_s {
    const hip_Memcpy2D * pCopy;
} hip_hipMemcpyParam2D_api_args_t;

typedef hipError_t hipMemcpyParam2D_return_t;


typedef struct hip_hipMemcpyParam2DAsync_api_args_s {
    const hip_Memcpy2D * pCopy;
    hipStream_t stream;
} hip_hipMemcpyParam2DAsync_api_args_t;

typedef hipError_t hipMemcpyParam2DAsync_return_t;


typedef struct hip_hipMemcpyPeer_api_args_s {
    void * dst;
    int dstDeviceId;
    const void * src;
    int srcDeviceId;
    size_t sizeBytes;
} hip_hipMemcpyPeer_api_args_t;

typedef hipError_t hipMemcpyPeer_return_t;


typedef struct hip_hipMemcpyPeerAsync_api_args_s {
    void * dst;
    int dstDeviceId;
    const void * src;
    int srcDevice;
    size_t sizeBytes;
    hipStream_t stream;
} hip_hipMemcpyPeerAsync_api_args_t;

typedef hipError_t hipMemcpyPeerAsync_return_t;


typedef struct hip_hipMemcpyToArray_api_args_s {
    hipArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void * src;
    size_t count;
    hipMemcpyKind kind;
} hip_hipMemcpyToArray_api_args_t;

typedef hipError_t hipMemcpyToArray_return_t;


typedef struct hip_hipMemcpyToSymbol_api_args_s {
    const void * symbol;
    const void * src;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
} hip_hipMemcpyToSymbol_api_args_t;

typedef hipError_t hipMemcpyToSymbol_return_t;


typedef struct hip_hipMemcpyToSymbolAsync_api_args_s {
    const void * symbol;
    const void * src;
    size_t sizeBytes;
    size_t offset;
    hipMemcpyKind kind;
    hipStream_t stream;
} hip_hipMemcpyToSymbolAsync_api_args_t;

typedef hipError_t hipMemcpyToSymbolAsync_return_t;


typedef struct hip_hipMemcpyWithStream_api_args_s {
    void * dst;
    const void * src;
    size_t sizeBytes;
    hipMemcpyKind kind;
    hipStream_t stream;
} hip_hipMemcpyWithStream_api_args_t;

typedef hipError_t hipMemcpyWithStream_return_t;


typedef struct hip_hipMemset_api_args_s {
    void * dst;
    int value;
    size_t sizeBytes;
} hip_hipMemset_api_args_t;

typedef hipError_t hipMemset_return_t;


typedef struct hip_hipMemset2D_api_args_s {
    void * dst;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
} hip_hipMemset2D_api_args_t;

typedef hipError_t hipMemset2D_return_t;


typedef struct hip_hipMemset2DAsync_api_args_s {
    void * dst;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    hipStream_t stream;
} hip_hipMemset2DAsync_api_args_t;

typedef hipError_t hipMemset2DAsync_return_t;


typedef struct hip_hipMemset3D_api_args_s {
    hipPitchedPtr pitchedDevPtr;
    int value;
    hipExtent extent;
} hip_hipMemset3D_api_args_t;

typedef hipError_t hipMemset3D_return_t;


typedef struct hip_hipMemset3DAsync_api_args_s {
    hipPitchedPtr pitchedDevPtr;
    int value;
    hipExtent extent;
    hipStream_t stream;
} hip_hipMemset3DAsync_api_args_t;

typedef hipError_t hipMemset3DAsync_return_t;


typedef struct hip_hipMemsetAsync_api_args_s {
    void * dst;
    int value;
    size_t sizeBytes;
    hipStream_t stream;
} hip_hipMemsetAsync_api_args_t;

typedef hipError_t hipMemsetAsync_return_t;


typedef struct hip_hipMemsetD16_api_args_s {
    hipDeviceptr_t dest;
    unsigned short value;
    size_t count;
} hip_hipMemsetD16_api_args_t;

typedef hipError_t hipMemsetD16_return_t;


typedef struct hip_hipMemsetD16Async_api_args_s {
    hipDeviceptr_t dest;
    unsigned short value;
    size_t count;
    hipStream_t stream;
} hip_hipMemsetD16Async_api_args_t;

typedef hipError_t hipMemsetD16Async_return_t;


typedef struct hip_hipMemsetD32_api_args_s {
    hipDeviceptr_t dest;
    int value;
    size_t count;
} hip_hipMemsetD32_api_args_t;

typedef hipError_t hipMemsetD32_return_t;


typedef struct hip_hipMemsetD32Async_api_args_s {
    hipDeviceptr_t dst;
    int value;
    size_t count;
    hipStream_t stream;
} hip_hipMemsetD32Async_api_args_t;

typedef hipError_t hipMemsetD32Async_return_t;


typedef struct hip_hipMemsetD8_api_args_s {
    hipDeviceptr_t dest;
    unsigned char value;
    size_t count;
} hip_hipMemsetD8_api_args_t;

typedef hipError_t hipMemsetD8_return_t;


typedef struct hip_hipMemsetD8Async_api_args_s {
    hipDeviceptr_t dest;
    unsigned char value;
    size_t count;
    hipStream_t stream;
} hip_hipMemsetD8Async_api_args_t;

typedef hipError_t hipMemsetD8Async_return_t;


typedef struct hip_hipMipmappedArrayCreate_api_args_s {
    hipMipmappedArray_t * pHandle;
    HIP_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc;
    unsigned int numMipmapLevels;
} hip_hipMipmappedArrayCreate_api_args_t;

typedef hipError_t hipMipmappedArrayCreate_return_t;


typedef struct hip_hipMipmappedArrayDestroy_api_args_s {
    hipMipmappedArray_t hMipmappedArray;
} hip_hipMipmappedArrayDestroy_api_args_t;

typedef hipError_t hipMipmappedArrayDestroy_return_t;


typedef struct hip_hipMipmappedArrayGetLevel_api_args_s {
    hipArray_t * pLevelArray;
    hipMipmappedArray_t hMipMappedArray;
    unsigned int level;
} hip_hipMipmappedArrayGetLevel_api_args_t;

typedef hipError_t hipMipmappedArrayGetLevel_return_t;


typedef struct hip_hipModuleGetFunction_api_args_s {
    hipFunction_t * function;
    hipModule_t module;
    const char * kname;
} hip_hipModuleGetFunction_api_args_t;

typedef hipError_t hipModuleGetFunction_return_t;


typedef struct hip_hipModuleGetGlobal_api_args_s {
    hipDeviceptr_t * dptr;
    size_t * bytes;
    hipModule_t hmod;
    const char * name;
} hip_hipModuleGetGlobal_api_args_t;

typedef hipError_t hipModuleGetGlobal_return_t;


typedef struct hip_hipModuleGetTexRef_api_args_s {
    textureReference * * texRef;
    hipModule_t hmod;
    const char * name;
} hip_hipModuleGetTexRef_api_args_t;

typedef hipError_t hipModuleGetTexRef_return_t;


typedef struct hip_hipModuleLaunchCooperativeKernel_api_args_s {
    hipFunction_t f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    hipStream_t stream;
    void * * kernelParams;
} hip_hipModuleLaunchCooperativeKernel_api_args_t;

typedef hipError_t hipModuleLaunchCooperativeKernel_return_t;


typedef struct hip_hipModuleLaunchCooperativeKernelMultiDevice_api_args_s {
    hipFunctionLaunchParams * launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
} hip_hipModuleLaunchCooperativeKernelMultiDevice_api_args_t;

typedef hipError_t hipModuleLaunchCooperativeKernelMultiDevice_return_t;


typedef struct hip_hipModuleLaunchKernel_api_args_s {
    hipFunction_t f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    hipStream_t stream;
    void * * kernelParams;
    void * * extra;
} hip_hipModuleLaunchKernel_api_args_t;

typedef hipError_t hipModuleLaunchKernel_return_t;


typedef struct hip_hipModuleLaunchKernelExt_api_args_s {
    hipFunction_t f;
    uint32_t globalWorkSizeX;
    uint32_t globalWorkSizeY;
    uint32_t globalWorkSizeZ;
    uint32_t blockDimX;
    uint32_t blockDimY;
    uint32_t blockDimZ;
    size_t sharedMemBytes;
    hipStream_t hStream;
    void * * kernelParams;
    void * * extra;
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
} hip_hipModuleLaunchKernelExt_api_args_t;

typedef hipError_t hipModuleLaunchKernelExt_return_t;


typedef struct hip_hipModuleLoad_api_args_s {
    hipModule_t * module;
    const char * fname;
} hip_hipModuleLoad_api_args_t;

typedef hipError_t hipModuleLoad_return_t;


typedef struct hip_hipModuleLoadData_api_args_s {
    hipModule_t * module;
    const void * image;
} hip_hipModuleLoadData_api_args_t;

typedef hipError_t hipModuleLoadData_return_t;


typedef struct hip_hipModuleLoadDataEx_api_args_s {
    hipModule_t * module;
    const void * image;
    unsigned int numOptions;
    hipJitOption * options;
    void * * optionValues;
} hip_hipModuleLoadDataEx_api_args_t;

typedef hipError_t hipModuleLoadDataEx_return_t;


typedef struct hip_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_api_args_s {
    int * numBlocks;
    hipFunction_t f;
    int blockSize;
    size_t dynSharedMemPerBlk;
} hip_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_api_args_t;

typedef hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_return_t;


typedef struct hip_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_api_args_s {
    int * numBlocks;
    hipFunction_t f;
    int blockSize;
    size_t dynSharedMemPerBlk;
    unsigned int flags;
} hip_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_api_args_t;

typedef hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_return_t;


typedef struct hip_hipModuleOccupancyMaxPotentialBlockSize_api_args_s {
    int * gridSize;
    int * blockSize;
    hipFunction_t f;
    size_t dynSharedMemPerBlk;
    int blockSizeLimit;
} hip_hipModuleOccupancyMaxPotentialBlockSize_api_args_t;

typedef hipError_t hipModuleOccupancyMaxPotentialBlockSize_return_t;


typedef struct hip_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_api_args_s {
    int * gridSize;
    int * blockSize;
    hipFunction_t f;
    size_t dynSharedMemPerBlk;
    int blockSizeLimit;
    unsigned int flags;
} hip_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_api_args_t;

typedef hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags_return_t;


typedef struct hip_hipModuleUnload_api_args_s {
    hipModule_t module;
} hip_hipModuleUnload_api_args_t;

typedef hipError_t hipModuleUnload_return_t;


typedef struct hip_hipOccupancyMaxActiveBlocksPerMultiprocessor_api_args_s {
    int * numBlocks;
    const void * f;
    int blockSize;
    size_t dynSharedMemPerBlk;
} hip_hipOccupancyMaxActiveBlocksPerMultiprocessor_api_args_t;

typedef hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor_return_t;


typedef struct hip_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_api_args_s {
    int * numBlocks;
    const void * f;
    int blockSize;
    size_t dynSharedMemPerBlk;
    unsigned int flags;
} hip_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_api_args_t;

typedef hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_return_t;


typedef struct hip_hipOccupancyMaxPotentialBlockSize_api_args_s {
    int * gridSize;
    int * blockSize;
    const void * f;
    size_t dynSharedMemPerBlk;
    int blockSizeLimit;
} hip_hipOccupancyMaxPotentialBlockSize_api_args_t;

typedef hipError_t hipOccupancyMaxPotentialBlockSize_return_t;


typedef struct hip_hipPeekAtLastError_api_args_s {
} hip_hipPeekAtLastError_api_args_t;

typedef hipError_t hipPeekAtLastError_return_t;


typedef struct hip_hipPointerGetAttribute_api_args_s {
    void * data;
    hipPointer_attribute attribute;
    hipDeviceptr_t ptr;
} hip_hipPointerGetAttribute_api_args_t;

typedef hipError_t hipPointerGetAttribute_return_t;


typedef struct hip_hipPointerGetAttributes_api_args_s {
    hipPointerAttribute_t * attributes;
    const void * ptr;
} hip_hipPointerGetAttributes_api_args_t;

typedef hipError_t hipPointerGetAttributes_return_t;


typedef struct hip_hipPointerSetAttribute_api_args_s {
    const void * value;
    hipPointer_attribute attribute;
    hipDeviceptr_t ptr;
} hip_hipPointerSetAttribute_api_args_t;

typedef hipError_t hipPointerSetAttribute_return_t;


typedef struct hip_hipProfilerStart_api_args_s {
} hip_hipProfilerStart_api_args_t;

typedef hipError_t hipProfilerStart_return_t;


typedef struct hip_hipProfilerStop_api_args_s {
} hip_hipProfilerStop_api_args_t;

typedef hipError_t hipProfilerStop_return_t;


typedef struct hip_hipRuntimeGetVersion_api_args_s {
    int * runtimeVersion;
} hip_hipRuntimeGetVersion_api_args_t;

typedef hipError_t hipRuntimeGetVersion_return_t;


typedef struct hip_hipSetDevice_api_args_s {
    int deviceId;
} hip_hipSetDevice_api_args_t;

typedef hipError_t hipSetDevice_return_t;


typedef struct hip_hipSetDeviceFlags_api_args_s {
    unsigned flags;
} hip_hipSetDeviceFlags_api_args_t;

typedef hipError_t hipSetDeviceFlags_return_t;


typedef struct hip_hipSetValidDevices_api_args_s {
    int * device_arr;
    int len;
} hip_hipSetValidDevices_api_args_t;

typedef hipError_t hipSetValidDevices_return_t;


typedef struct hip_hipSetupArgument_api_args_s {
    const void * arg;
    size_t size;
    size_t offset;
} hip_hipSetupArgument_api_args_t;

typedef hipError_t hipSetupArgument_return_t;


typedef struct hip_hipSignalExternalSemaphoresAsync_api_args_s {
    const hipExternalSemaphore_t * extSemArray;
    const hipExternalSemaphoreSignalParams * paramsArray;
    unsigned int numExtSems;
    hipStream_t stream;
} hip_hipSignalExternalSemaphoresAsync_api_args_t;

typedef hipError_t hipSignalExternalSemaphoresAsync_return_t;


typedef struct hip_hipStreamAddCallback_api_args_s {
    hipStream_t stream;
    hipStreamCallback_t callback;
    void * userData;
    unsigned int flags;
} hip_hipStreamAddCallback_api_args_t;

typedef hipError_t hipStreamAddCallback_return_t;


typedef struct hip_hipStreamAttachMemAsync_api_args_s {
    hipStream_t stream;
    void * dev_ptr;
    size_t length;
    unsigned int flags;
} hip_hipStreamAttachMemAsync_api_args_t;

typedef hipError_t hipStreamAttachMemAsync_return_t;


typedef struct hip_hipStreamBeginCapture_api_args_s {
    hipStream_t stream;
    hipStreamCaptureMode mode;
} hip_hipStreamBeginCapture_api_args_t;

typedef hipError_t hipStreamBeginCapture_return_t;


typedef struct hip_hipStreamCreate_api_args_s {
    hipStream_t * stream;
} hip_hipStreamCreate_api_args_t;

typedef hipError_t hipStreamCreate_return_t;


typedef struct hip_hipStreamCreateWithFlags_api_args_s {
    hipStream_t * stream;
    unsigned int flags;
} hip_hipStreamCreateWithFlags_api_args_t;

typedef hipError_t hipStreamCreateWithFlags_return_t;


typedef struct hip_hipStreamCreateWithPriority_api_args_s {
    hipStream_t * stream;
    unsigned int flags;
    int priority;
} hip_hipStreamCreateWithPriority_api_args_t;

typedef hipError_t hipStreamCreateWithPriority_return_t;


typedef struct hip_hipStreamDestroy_api_args_s {
    hipStream_t stream;
} hip_hipStreamDestroy_api_args_t;

typedef hipError_t hipStreamDestroy_return_t;


typedef struct hip_hipStreamEndCapture_api_args_s {
    hipStream_t stream;
    hipGraph_t * pGraph;
} hip_hipStreamEndCapture_api_args_t;

typedef hipError_t hipStreamEndCapture_return_t;


typedef struct hip_hipStreamGetCaptureInfo_api_args_s {
    hipStream_t stream;
    hipStreamCaptureStatus * pCaptureStatus;
    unsigned long long * pId;
} hip_hipStreamGetCaptureInfo_api_args_t;

typedef hipError_t hipStreamGetCaptureInfo_return_t;


typedef struct hip_hipStreamGetCaptureInfo_v2_api_args_s {
    hipStream_t stream;
    hipStreamCaptureStatus * captureStatus_out;
    unsigned long long * id_out;
    hipGraph_t * graph_out;
    const hipGraphNode_t * * dependencies_out;
    size_t * numDependencies_out;
} hip_hipStreamGetCaptureInfo_v2_api_args_t;

typedef hipError_t hipStreamGetCaptureInfo_v2_return_t;


typedef struct hip_hipStreamGetDevice_api_args_s {
    hipStream_t stream;
    hipDevice_t * device;
} hip_hipStreamGetDevice_api_args_t;

typedef hipError_t hipStreamGetDevice_return_t;


typedef struct hip_hipStreamGetFlags_api_args_s {
    hipStream_t stream;
    unsigned int * flags;
} hip_hipStreamGetFlags_api_args_t;

typedef hipError_t hipStreamGetFlags_return_t;


typedef struct hip_hipStreamGetPriority_api_args_s {
    hipStream_t stream;
    int * priority;
} hip_hipStreamGetPriority_api_args_t;

typedef hipError_t hipStreamGetPriority_return_t;


typedef struct hip_hipStreamIsCapturing_api_args_s {
    hipStream_t stream;
    hipStreamCaptureStatus * pCaptureStatus;
} hip_hipStreamIsCapturing_api_args_t;

typedef hipError_t hipStreamIsCapturing_return_t;


typedef struct hip_hipStreamQuery_api_args_s {
    hipStream_t stream;
} hip_hipStreamQuery_api_args_t;

typedef hipError_t hipStreamQuery_return_t;


typedef struct hip_hipStreamSynchronize_api_args_s {
    hipStream_t stream;
} hip_hipStreamSynchronize_api_args_t;

typedef hipError_t hipStreamSynchronize_return_t;


typedef struct hip_hipStreamUpdateCaptureDependencies_api_args_s {
    hipStream_t stream;
    hipGraphNode_t * dependencies;
    size_t numDependencies;
    unsigned int flags;
} hip_hipStreamUpdateCaptureDependencies_api_args_t;

typedef hipError_t hipStreamUpdateCaptureDependencies_return_t;


typedef struct hip_hipStreamWaitEvent_api_args_s {
    hipStream_t stream;
    hipEvent_t event;
    unsigned int flags;
} hip_hipStreamWaitEvent_api_args_t;

typedef hipError_t hipStreamWaitEvent_return_t;


typedef struct hip_hipStreamWaitValue32_api_args_s {
    hipStream_t stream;
    void * ptr;
    uint32_t value;
    unsigned int flags;
    uint32_t mask;
} hip_hipStreamWaitValue32_api_args_t;

typedef hipError_t hipStreamWaitValue32_return_t;


typedef struct hip_hipStreamWaitValue64_api_args_s {
    hipStream_t stream;
    void * ptr;
    uint64_t value;
    unsigned int flags;
    uint64_t mask;
} hip_hipStreamWaitValue64_api_args_t;

typedef hipError_t hipStreamWaitValue64_return_t;


typedef struct hip_hipStreamWriteValue32_api_args_s {
    hipStream_t stream;
    void * ptr;
    uint32_t value;
    unsigned int flags;
} hip_hipStreamWriteValue32_api_args_t;

typedef hipError_t hipStreamWriteValue32_return_t;


typedef struct hip_hipStreamWriteValue64_api_args_s {
    hipStream_t stream;
    void * ptr;
    uint64_t value;
    unsigned int flags;
} hip_hipStreamWriteValue64_api_args_t;

typedef hipError_t hipStreamWriteValue64_return_t;


typedef struct hip_hipTexObjectCreate_api_args_s {
    hipTextureObject_t * pTexObject;
    const HIP_RESOURCE_DESC * pResDesc;
    const HIP_TEXTURE_DESC * pTexDesc;
    const HIP_RESOURCE_VIEW_DESC * pResViewDesc;
} hip_hipTexObjectCreate_api_args_t;

typedef hipError_t hipTexObjectCreate_return_t;


typedef struct hip_hipTexObjectDestroy_api_args_s {
    hipTextureObject_t texObject;
} hip_hipTexObjectDestroy_api_args_t;

typedef hipError_t hipTexObjectDestroy_return_t;


typedef struct hip_hipTexObjectGetResourceDesc_api_args_s {
    HIP_RESOURCE_DESC * pResDesc;
    hipTextureObject_t texObject;
} hip_hipTexObjectGetResourceDesc_api_args_t;

typedef hipError_t hipTexObjectGetResourceDesc_return_t;


typedef struct hip_hipTexObjectGetResourceViewDesc_api_args_s {
    HIP_RESOURCE_VIEW_DESC * pResViewDesc;
    hipTextureObject_t texObject;
} hip_hipTexObjectGetResourceViewDesc_api_args_t;

typedef hipError_t hipTexObjectGetResourceViewDesc_return_t;


typedef struct hip_hipTexObjectGetTextureDesc_api_args_s {
    HIP_TEXTURE_DESC * pTexDesc;
    hipTextureObject_t texObject;
} hip_hipTexObjectGetTextureDesc_api_args_t;

typedef hipError_t hipTexObjectGetTextureDesc_return_t;


typedef struct hip_hipTexRefGetAddress_api_args_s {
    hipDeviceptr_t * dev_ptr;
    const textureReference * texRef;
} hip_hipTexRefGetAddress_api_args_t;

typedef hipError_t hipTexRefGetAddress_return_t;


typedef struct hip_hipTexRefGetAddressMode_api_args_s {
    enum hipTextureAddressMode * pam;
    const textureReference * texRef;
    int dim;
} hip_hipTexRefGetAddressMode_api_args_t;

typedef hipError_t hipTexRefGetAddressMode_return_t;


typedef struct hip_hipTexRefGetArray_api_args_s {
    hipArray_t * pArray;
    const textureReference * texRef;
} hip_hipTexRefGetArray_api_args_t;

typedef hipError_t hipTexRefGetArray_return_t;


typedef struct hip_hipTexRefGetBorderColor_api_args_s {
    float * pBorderColor;
    const textureReference * texRef;
} hip_hipTexRefGetBorderColor_api_args_t;

typedef hipError_t hipTexRefGetBorderColor_return_t;


typedef struct hip_hipTexRefGetFilterMode_api_args_s {
    enum hipTextureFilterMode * pfm;
    const textureReference * texRef;
} hip_hipTexRefGetFilterMode_api_args_t;

typedef hipError_t hipTexRefGetFilterMode_return_t;


typedef struct hip_hipTexRefGetFlags_api_args_s {
    unsigned int * pFlags;
    const textureReference * texRef;
} hip_hipTexRefGetFlags_api_args_t;

typedef hipError_t hipTexRefGetFlags_return_t;


typedef struct hip_hipTexRefGetFormat_api_args_s {
    hipArray_Format * pFormat;
    int * pNumChannels;
    const textureReference * texRef;
} hip_hipTexRefGetFormat_api_args_t;

typedef hipError_t hipTexRefGetFormat_return_t;


typedef struct hip_hipTexRefGetMaxAnisotropy_api_args_s {
    int * pmaxAnsio;
    const textureReference * texRef;
} hip_hipTexRefGetMaxAnisotropy_api_args_t;

typedef hipError_t hipTexRefGetMaxAnisotropy_return_t;


typedef struct hip_hipTexRefGetMipMappedArray_api_args_s {
    hipMipmappedArray_t * pArray;
    const textureReference * texRef;
} hip_hipTexRefGetMipMappedArray_api_args_t;

typedef hipError_t hipTexRefGetMipMappedArray_return_t;


typedef struct hip_hipTexRefGetMipmapFilterMode_api_args_s {
    enum hipTextureFilterMode * pfm;
    const textureReference * texRef;
} hip_hipTexRefGetMipmapFilterMode_api_args_t;

typedef hipError_t hipTexRefGetMipmapFilterMode_return_t;


typedef struct hip_hipTexRefGetMipmapLevelBias_api_args_s {
    float * pbias;
    const textureReference * texRef;
} hip_hipTexRefGetMipmapLevelBias_api_args_t;

typedef hipError_t hipTexRefGetMipmapLevelBias_return_t;


typedef struct hip_hipTexRefGetMipmapLevelClamp_api_args_s {
    float * pminMipmapLevelClamp;
    float * pmaxMipmapLevelClamp;
    const textureReference * texRef;
} hip_hipTexRefGetMipmapLevelClamp_api_args_t;

typedef hipError_t hipTexRefGetMipmapLevelClamp_return_t;


typedef struct hip_hipTexRefGetMipmappedArray_api_args_s {
    hipMipmappedArray_t * pArray;
    const textureReference * texRef;
} hip_hipTexRefGetMipmappedArray_api_args_t;

typedef hipError_t hipTexRefGetMipmappedArray_return_t;


typedef struct hip_hipTexRefSetAddress_api_args_s {
    size_t * ByteOffset;
    textureReference * texRef;
    hipDeviceptr_t dptr;
    size_t bytes;
} hip_hipTexRefSetAddress_api_args_t;

typedef hipError_t hipTexRefSetAddress_return_t;


typedef struct hip_hipTexRefSetAddress2D_api_args_s {
    textureReference * texRef;
    const HIP_ARRAY_DESCRIPTOR * desc;
    hipDeviceptr_t dptr;
    size_t Pitch;
} hip_hipTexRefSetAddress2D_api_args_t;

typedef hipError_t hipTexRefSetAddress2D_return_t;


typedef struct hip_hipTexRefSetAddressMode_api_args_s {
    textureReference * texRef;
    int dim;
    enum hipTextureAddressMode am;
} hip_hipTexRefSetAddressMode_api_args_t;

typedef hipError_t hipTexRefSetAddressMode_return_t;


typedef struct hip_hipTexRefSetArray_api_args_s {
    textureReference * tex;
    hipArray_const_t array;
    unsigned int flags;
} hip_hipTexRefSetArray_api_args_t;

typedef hipError_t hipTexRefSetArray_return_t;


typedef struct hip_hipTexRefSetBorderColor_api_args_s {
    textureReference * texRef;
    float * pBorderColor;
} hip_hipTexRefSetBorderColor_api_args_t;

typedef hipError_t hipTexRefSetBorderColor_return_t;


typedef struct hip_hipTexRefSetFilterMode_api_args_s {
    textureReference * texRef;
    enum hipTextureFilterMode fm;
} hip_hipTexRefSetFilterMode_api_args_t;

typedef hipError_t hipTexRefSetFilterMode_return_t;


typedef struct hip_hipTexRefSetFlags_api_args_s {
    textureReference * texRef;
    unsigned int Flags;
} hip_hipTexRefSetFlags_api_args_t;

typedef hipError_t hipTexRefSetFlags_return_t;


typedef struct hip_hipTexRefSetFormat_api_args_s {
    textureReference * texRef;
    hipArray_Format fmt;
    int NumPackedComponents;
} hip_hipTexRefSetFormat_api_args_t;

typedef hipError_t hipTexRefSetFormat_return_t;


typedef struct hip_hipTexRefSetMaxAnisotropy_api_args_s {
    textureReference * texRef;
    unsigned int maxAniso;
} hip_hipTexRefSetMaxAnisotropy_api_args_t;

typedef hipError_t hipTexRefSetMaxAnisotropy_return_t;


typedef struct hip_hipTexRefSetMipmapFilterMode_api_args_s {
    textureReference * texRef;
    enum hipTextureFilterMode fm;
} hip_hipTexRefSetMipmapFilterMode_api_args_t;

typedef hipError_t hipTexRefSetMipmapFilterMode_return_t;


typedef struct hip_hipTexRefSetMipmapLevelBias_api_args_s {
    textureReference * texRef;
    float bias;
} hip_hipTexRefSetMipmapLevelBias_api_args_t;

typedef hipError_t hipTexRefSetMipmapLevelBias_return_t;


typedef struct hip_hipTexRefSetMipmapLevelClamp_api_args_s {
    textureReference * texRef;
    float minMipMapLevelClamp;
    float maxMipMapLevelClamp;
} hip_hipTexRefSetMipmapLevelClamp_api_args_t;

typedef hipError_t hipTexRefSetMipmapLevelClamp_return_t;


typedef struct hip_hipTexRefSetMipmappedArray_api_args_s {
    textureReference * texRef;
    struct hipMipmappedArray * mipmappedArray;
    unsigned int Flags;
} hip_hipTexRefSetMipmappedArray_api_args_t;

typedef hipError_t hipTexRefSetMipmappedArray_return_t;


typedef struct hip_hipThreadExchangeStreamCaptureMode_api_args_s {
    hipStreamCaptureMode * mode;
} hip_hipThreadExchangeStreamCaptureMode_api_args_t;

typedef hipError_t hipThreadExchangeStreamCaptureMode_return_t;


typedef struct hip_hipUnbindTexture_api_args_s {
    const textureReference * tex;
} hip_hipUnbindTexture_api_args_t;

typedef hipError_t hipUnbindTexture_return_t;


typedef struct hip_hipUserObjectCreate_api_args_s {
    hipUserObject_t * object_out;
    void * ptr;
    hipHostFn_t destroy;
    unsigned int initialRefcount;
    unsigned int flags;
} hip_hipUserObjectCreate_api_args_t;

typedef hipError_t hipUserObjectCreate_return_t;


typedef struct hip_hipUserObjectRelease_api_args_s {
    hipUserObject_t object;
    unsigned int count;
} hip_hipUserObjectRelease_api_args_t;

typedef hipError_t hipUserObjectRelease_return_t;


typedef struct hip_hipUserObjectRetain_api_args_s {
    hipUserObject_t object;
    unsigned int count;
} hip_hipUserObjectRetain_api_args_t;

typedef hipError_t hipUserObjectRetain_return_t;


typedef struct hip_hipWaitExternalSemaphoresAsync_api_args_s {
    const hipExternalSemaphore_t * extSemArray;
    const hipExternalSemaphoreWaitParams * paramsArray;
    unsigned int numExtSems;
    hipStream_t stream;
} hip_hipWaitExternalSemaphoresAsync_api_args_t;

typedef hipError_t hipWaitExternalSemaphoresAsync_return_t;


typedef struct hip_hip_init_api_args_s {
} hip_hip_init_api_args_t;

typedef hipError_t hip_init_return_t;


#endif