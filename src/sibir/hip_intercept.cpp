#include "hip_intercept.h"


__attribute__((visibility("default")))
hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, hipStream_t* stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID___hipPopCallConfiguration;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.__hipPopCallConfiguration.gridDim = gridDim;
    hip_args.__hipPopCallConfiguration.blockDim = blockDim;
    hip_args.__hipPopCallConfiguration.sharedMem = sharedMem;
    hip_args.__hipPopCallConfiguration.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(dim3*,dim3*,size_t*,hipStream_t*)>("__hipPopCallConfiguration");
    hipError_t out = hip_func(hip_args.__hipPopCallConfiguration.gridDim, hip_args.__hipPopCallConfiguration.blockDim, hip_args.__hipPopCallConfiguration.sharedMem, hip_args.__hipPopCallConfiguration.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    gridDim = hip_args.__hipPopCallConfiguration.gridDim;
    blockDim = hip_args.__hipPopCallConfiguration.blockDim;
    sharedMem = hip_args.__hipPopCallConfiguration.sharedMem;
    stream = hip_args.__hipPopCallConfiguration.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID___hipPushCallConfiguration;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.__hipPushCallConfiguration.gridDim = gridDim;
    hip_args.__hipPushCallConfiguration.blockDim = blockDim;
    hip_args.__hipPushCallConfiguration.sharedMem = sharedMem;
    hip_args.__hipPushCallConfiguration.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(dim3,dim3,size_t,hipStream_t)>("__hipPushCallConfiguration");
    hipError_t out = hip_func(hip_args.__hipPushCallConfiguration.gridDim, hip_args.__hipPushCallConfiguration.blockDim, hip_args.__hipPushCallConfiguration.sharedMem, hip_args.__hipPushCallConfiguration.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    gridDim = hip_args.__hipPushCallConfiguration.gridDim;
    blockDim = hip_args.__hipPushCallConfiguration.blockDim;
    sharedMem = hip_args.__hipPushCallConfiguration.sharedMem;
    stream = hip_args.__hipPushCallConfiguration.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipArrayCreate(hipArray** pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipArrayCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipArrayCreate.pHandle = pHandle;
    hip_args.hipArrayCreate.pAllocateArray = pAllocateArray;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray**,const HIP_ARRAY_DESCRIPTOR*)>("hipArrayCreate");
    hipError_t out = hip_func(hip_args.hipArrayCreate.pHandle, hip_args.hipArrayCreate.pAllocateArray);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pHandle = hip_args.hipArrayCreate.pHandle;
    pAllocateArray = hip_args.hipArrayCreate.pAllocateArray;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipArrayDestroy(hipArray* array) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipArrayDestroy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipArrayDestroy.array = array;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray*)>("hipArrayDestroy");
    hipError_t out = hip_func(hip_args.hipArrayDestroy.array);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    array = hip_args.hipArrayDestroy.array;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipChooseDevice;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipChooseDevice.device = device;
    hip_args.hipChooseDevice.prop = prop;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,const hipDeviceProp_t*)>("hipChooseDevice");
    hipError_t out = hip_func(hip_args.hipChooseDevice.device, hip_args.hipChooseDevice.prop);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device = hip_args.hipChooseDevice.device;
    prop = hip_args.hipChooseDevice.prop;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject, const hipResourceDesc* pResDesc) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCreateSurfaceObject;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCreateSurfaceObject.pSurfObject = pSurfObject;
    hip_args.hipCreateSurfaceObject.pResDesc = pResDesc;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSurfaceObject_t*,const hipResourceDesc*)>("hipCreateSurfaceObject");
    hipError_t out = hip_func(hip_args.hipCreateSurfaceObject.pSurfObject, hip_args.hipCreateSurfaceObject.pResDesc);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pSurfObject = hip_args.hipCreateSurfaceObject.pSurfObject;
    pResDesc = hip_args.hipCreateSurfaceObject.pResDesc;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxCreate.ctx = ctx;
    hip_args.hipCtxCreate.flags = flags;
    hip_args.hipCtxCreate.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t*,unsigned int,hipDevice_t)>("hipCtxCreate");
    hipError_t out = hip_func(hip_args.hipCtxCreate.ctx, hip_args.hipCtxCreate.flags, hip_args.hipCtxCreate.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ctx = hip_args.hipCtxCreate.ctx;
    flags = hip_args.hipCtxCreate.flags;
    device = hip_args.hipCtxCreate.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxDestroy(hipCtx_t ctx) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxDestroy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxDestroy.ctx = ctx;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxDestroy");
    hipError_t out = hip_func(hip_args.hipCtxDestroy.ctx);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ctx = hip_args.hipCtxDestroy.ctx;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxDisablePeerAccess;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxDisablePeerAccess.peerCtx = peerCtx;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxDisablePeerAccess");
    hipError_t out = hip_func(hip_args.hipCtxDisablePeerAccess.peerCtx);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    peerCtx = hip_args.hipCtxDisablePeerAccess.peerCtx;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxEnablePeerAccess;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxEnablePeerAccess.peerCtx = peerCtx;
    hip_args.hipCtxEnablePeerAccess.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t,unsigned int)>("hipCtxEnablePeerAccess");
    hipError_t out = hip_func(hip_args.hipCtxEnablePeerAccess.peerCtx, hip_args.hipCtxEnablePeerAccess.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    peerCtx = hip_args.hipCtxEnablePeerAccess.peerCtx;
    flags = hip_args.hipCtxEnablePeerAccess.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxGetApiVersion;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxGetApiVersion.ctx = ctx;
    hip_args.hipCtxGetApiVersion.apiVersion = apiVersion;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t,int*)>("hipCtxGetApiVersion");
    hipError_t out = hip_func(hip_args.hipCtxGetApiVersion.ctx, hip_args.hipCtxGetApiVersion.apiVersion);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ctx = hip_args.hipCtxGetApiVersion.ctx;
    apiVersion = hip_args.hipCtxGetApiVersion.apiVersion;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxGetCacheConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxGetCacheConfig.cacheConfig = cacheConfig;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncCache_t*)>("hipCtxGetCacheConfig");
    hipError_t out = hip_func(hip_args.hipCtxGetCacheConfig.cacheConfig);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    cacheConfig = hip_args.hipCtxGetCacheConfig.cacheConfig;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxGetCurrent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxGetCurrent.ctx = ctx;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t*)>("hipCtxGetCurrent");
    hipError_t out = hip_func(hip_args.hipCtxGetCurrent.ctx);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ctx = hip_args.hipCtxGetCurrent.ctx;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxGetDevice(hipDevice_t* device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxGetDevice;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxGetDevice.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t*)>("hipCtxGetDevice");
    hipError_t out = hip_func(hip_args.hipCtxGetDevice.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device = hip_args.hipCtxGetDevice.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxGetFlags(unsigned int* flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxGetFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxGetFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int*)>("hipCtxGetFlags");
    hipError_t out = hip_func(hip_args.hipCtxGetFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    flags = hip_args.hipCtxGetFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxGetSharedMemConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxGetSharedMemConfig.pConfig = pConfig;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSharedMemConfig*)>("hipCtxGetSharedMemConfig");
    hipError_t out = hip_func(hip_args.hipCtxGetSharedMemConfig.pConfig);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pConfig = hip_args.hipCtxGetSharedMemConfig.pConfig;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxPopCurrent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxPopCurrent.ctx = ctx;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t*)>("hipCtxPopCurrent");
    hipError_t out = hip_func(hip_args.hipCtxPopCurrent.ctx);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ctx = hip_args.hipCtxPopCurrent.ctx;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxPushCurrent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxPushCurrent.ctx = ctx;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxPushCurrent");
    hipError_t out = hip_func(hip_args.hipCtxPushCurrent.ctx);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ctx = hip_args.hipCtxPushCurrent.ctx;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxSetCacheConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxSetCacheConfig.cacheConfig = cacheConfig;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncCache_t)>("hipCtxSetCacheConfig");
    hipError_t out = hip_func(hip_args.hipCtxSetCacheConfig.cacheConfig);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    cacheConfig = hip_args.hipCtxSetCacheConfig.cacheConfig;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxSetCurrent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxSetCurrent.ctx = ctx;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxSetCurrent");
    hipError_t out = hip_func(hip_args.hipCtxSetCurrent.ctx);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ctx = hip_args.hipCtxSetCurrent.ctx;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxSetSharedMemConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipCtxSetSharedMemConfig.config = config;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSharedMemConfig)>("hipCtxSetSharedMemConfig");
    hipError_t out = hip_func(hip_args.hipCtxSetSharedMemConfig.config);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    config = hip_args.hipCtxSetSharedMemConfig.config;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipCtxSynchronize() {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipCtxSynchronize;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipCtxSynchronize");
    hipError_t out = hip_func();
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDestroyExternalMemory;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDestroyExternalMemory.extMem = extMem;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipExternalMemory_t)>("hipDestroyExternalMemory");
    hipError_t out = hip_func(hip_args.hipDestroyExternalMemory.extMem);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    extMem = hip_args.hipDestroyExternalMemory.extMem;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDestroyExternalSemaphore;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDestroyExternalSemaphore.extSem = extSem;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipExternalSemaphore_t)>("hipDestroyExternalSemaphore");
    hipError_t out = hip_func(hip_args.hipDestroyExternalSemaphore.extSem);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    extSem = hip_args.hipDestroyExternalSemaphore.extSem;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDestroySurfaceObject;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDestroySurfaceObject.surfaceObject = surfaceObject;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSurfaceObject_t)>("hipDestroySurfaceObject");
    hipError_t out = hip_func(hip_args.hipDestroySurfaceObject.surfaceObject);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    surfaceObject = hip_args.hipDestroySurfaceObject.surfaceObject;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceComputeCapability;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceComputeCapability.major = major;
    hip_args.hipDeviceComputeCapability.minor = minor;
    hip_args.hipDeviceComputeCapability.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,int*,hipDevice_t)>("hipDeviceComputeCapability");
    hipError_t out = hip_func(hip_args.hipDeviceComputeCapability.major, hip_args.hipDeviceComputeCapability.minor, hip_args.hipDeviceComputeCapability.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    major = hip_args.hipDeviceComputeCapability.major;
    minor = hip_args.hipDeviceComputeCapability.minor;
    device = hip_args.hipDeviceComputeCapability.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceDisablePeerAccess;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceDisablePeerAccess.peerDeviceId = peerDeviceId;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int)>("hipDeviceDisablePeerAccess");
    hipError_t out = hip_func(hip_args.hipDeviceDisablePeerAccess.peerDeviceId);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    peerDeviceId = hip_args.hipDeviceDisablePeerAccess.peerDeviceId;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceEnablePeerAccess;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceEnablePeerAccess.peerDeviceId = peerDeviceId;
    hip_args.hipDeviceEnablePeerAccess.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,unsigned int)>("hipDeviceEnablePeerAccess");
    hipError_t out = hip_func(hip_args.hipDeviceEnablePeerAccess.peerDeviceId, hip_args.hipDeviceEnablePeerAccess.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    peerDeviceId = hip_args.hipDeviceEnablePeerAccess.peerDeviceId;
    flags = hip_args.hipDeviceEnablePeerAccess.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGet;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGet.device = device;
    hip_args.hipDeviceGet.ordinal = ordinal;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t*,int)>("hipDeviceGet");
    hipError_t out = hip_func(hip_args.hipDeviceGet.device, hip_args.hipDeviceGet.ordinal);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device = hip_args.hipDeviceGet.device;
    ordinal = hip_args.hipDeviceGet.ordinal;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetAttribute.pi = pi;
    hip_args.hipDeviceGetAttribute.attr = attr;
    hip_args.hipDeviceGetAttribute.deviceId = deviceId;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,hipDeviceAttribute_t,int)>("hipDeviceGetAttribute");
    hipError_t out = hip_func(hip_args.hipDeviceGetAttribute.pi, hip_args.hipDeviceGetAttribute.attr, hip_args.hipDeviceGetAttribute.deviceId);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pi = hip_args.hipDeviceGetAttribute.pi;
    attr = hip_args.hipDeviceGetAttribute.attr;
    deviceId = hip_args.hipDeviceGetAttribute.deviceId;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetByPCIBusId;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetByPCIBusId.device = device;
    hip_args.hipDeviceGetByPCIBusId.pciBusId = pciBusId;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,const char*)>("hipDeviceGetByPCIBusId");
    hipError_t out = hip_func(hip_args.hipDeviceGetByPCIBusId.device, hip_args.hipDeviceGetByPCIBusId.pciBusId);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device = hip_args.hipDeviceGetByPCIBusId.device;
    pciBusId = hip_args.hipDeviceGetByPCIBusId.pciBusId;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetCacheConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetCacheConfig.cacheConfig = cacheConfig;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncCache_t*)>("hipDeviceGetCacheConfig");
    hipError_t out = hip_func(hip_args.hipDeviceGetCacheConfig.cacheConfig);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    cacheConfig = hip_args.hipDeviceGetCacheConfig.cacheConfig;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetDefaultMemPool;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetDefaultMemPool.mem_pool = mem_pool;
    hip_args.hipDeviceGetDefaultMemPool.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t*,int)>("hipDeviceGetDefaultMemPool");
    hipError_t out = hip_func(hip_args.hipDeviceGetDefaultMemPool.mem_pool, hip_args.hipDeviceGetDefaultMemPool.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipDeviceGetDefaultMemPool.mem_pool;
    device = hip_args.hipDeviceGetDefaultMemPool.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetGraphMemAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetGraphMemAttribute.device = device;
    hip_args.hipDeviceGetGraphMemAttribute.attr = attr;
    hip_args.hipDeviceGetGraphMemAttribute.value = value;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipGraphMemAttributeType,void*)>("hipDeviceGetGraphMemAttribute");
    hipError_t out = hip_func(hip_args.hipDeviceGetGraphMemAttribute.device, hip_args.hipDeviceGetGraphMemAttribute.attr, hip_args.hipDeviceGetGraphMemAttribute.value);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device = hip_args.hipDeviceGetGraphMemAttribute.device;
    attr = hip_args.hipDeviceGetGraphMemAttribute.attr;
    value = hip_args.hipDeviceGetGraphMemAttribute.value;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetLimit;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetLimit.pValue = pValue;
    hip_args.hipDeviceGetLimit.limit = limit;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t*,hipLimit_t)>("hipDeviceGetLimit");
    hipError_t out = hip_func(hip_args.hipDeviceGetLimit.pValue, hip_args.hipDeviceGetLimit.limit);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pValue = hip_args.hipDeviceGetLimit.pValue;
    limit = hip_args.hipDeviceGetLimit.limit;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetMemPool;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetMemPool.mem_pool = mem_pool;
    hip_args.hipDeviceGetMemPool.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t*,int)>("hipDeviceGetMemPool");
    hipError_t out = hip_func(hip_args.hipDeviceGetMemPool.mem_pool, hip_args.hipDeviceGetMemPool.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipDeviceGetMemPool.mem_pool;
    device = hip_args.hipDeviceGetMemPool.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetName;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetName.name = name;
    hip_args.hipDeviceGetName.len = len;
    hip_args.hipDeviceGetName.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(char*,int,hipDevice_t)>("hipDeviceGetName");
    hipError_t out = hip_func(hip_args.hipDeviceGetName.name, hip_args.hipDeviceGetName.len, hip_args.hipDeviceGetName.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    name = hip_args.hipDeviceGetName.name;
    len = hip_args.hipDeviceGetName.len;
    device = hip_args.hipDeviceGetName.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetP2PAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetP2PAttribute.value = value;
    hip_args.hipDeviceGetP2PAttribute.attr = attr;
    hip_args.hipDeviceGetP2PAttribute.srcDevice = srcDevice;
    hip_args.hipDeviceGetP2PAttribute.dstDevice = dstDevice;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,hipDeviceP2PAttr,int,int)>("hipDeviceGetP2PAttribute");
    hipError_t out = hip_func(hip_args.hipDeviceGetP2PAttribute.value, hip_args.hipDeviceGetP2PAttribute.attr, hip_args.hipDeviceGetP2PAttribute.srcDevice, hip_args.hipDeviceGetP2PAttribute.dstDevice);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    value = hip_args.hipDeviceGetP2PAttribute.value;
    attr = hip_args.hipDeviceGetP2PAttribute.attr;
    srcDevice = hip_args.hipDeviceGetP2PAttribute.srcDevice;
    dstDevice = hip_args.hipDeviceGetP2PAttribute.dstDevice;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetPCIBusId;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetPCIBusId.pciBusId = pciBusId;
    hip_args.hipDeviceGetPCIBusId.len = len;
    hip_args.hipDeviceGetPCIBusId.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(char*,int,int)>("hipDeviceGetPCIBusId");
    hipError_t out = hip_func(hip_args.hipDeviceGetPCIBusId.pciBusId, hip_args.hipDeviceGetPCIBusId.len, hip_args.hipDeviceGetPCIBusId.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pciBusId = hip_args.hipDeviceGetPCIBusId.pciBusId;
    len = hip_args.hipDeviceGetPCIBusId.len;
    device = hip_args.hipDeviceGetPCIBusId.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetSharedMemConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetSharedMemConfig.pConfig = pConfig;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSharedMemConfig*)>("hipDeviceGetSharedMemConfig");
    hipError_t out = hip_func(hip_args.hipDeviceGetSharedMemConfig.pConfig);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pConfig = hip_args.hipDeviceGetSharedMemConfig.pConfig;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetStreamPriorityRange;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetStreamPriorityRange.leastPriority = leastPriority;
    hip_args.hipDeviceGetStreamPriorityRange.greatestPriority = greatestPriority;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,int*)>("hipDeviceGetStreamPriorityRange");
    hipError_t out = hip_func(hip_args.hipDeviceGetStreamPriorityRange.leastPriority, hip_args.hipDeviceGetStreamPriorityRange.greatestPriority);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    leastPriority = hip_args.hipDeviceGetStreamPriorityRange.leastPriority;
    greatestPriority = hip_args.hipDeviceGetStreamPriorityRange.greatestPriority;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGetUuid;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGetUuid.uuid = uuid;
    hip_args.hipDeviceGetUuid.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipUUID*,hipDevice_t)>("hipDeviceGetUuid");
    hipError_t out = hip_func(hip_args.hipDeviceGetUuid.uuid, hip_args.hipDeviceGetUuid.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    uuid = hip_args.hipDeviceGetUuid.uuid;
    device = hip_args.hipDeviceGetUuid.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceGraphMemTrim(int device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceGraphMemTrim;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceGraphMemTrim.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int)>("hipDeviceGraphMemTrim");
    hipError_t out = hip_func(hip_args.hipDeviceGraphMemTrim.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device = hip_args.hipDeviceGraphMemTrim.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDevicePrimaryCtxGetState;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDevicePrimaryCtxGetState.dev = dev;
    hip_args.hipDevicePrimaryCtxGetState.flags = flags;
    hip_args.hipDevicePrimaryCtxGetState.active = active;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t,unsigned int*,int*)>("hipDevicePrimaryCtxGetState");
    hipError_t out = hip_func(hip_args.hipDevicePrimaryCtxGetState.dev, hip_args.hipDevicePrimaryCtxGetState.flags, hip_args.hipDevicePrimaryCtxGetState.active);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev = hip_args.hipDevicePrimaryCtxGetState.dev;
    flags = hip_args.hipDevicePrimaryCtxGetState.flags;
    active = hip_args.hipDevicePrimaryCtxGetState.active;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDevicePrimaryCtxRelease;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDevicePrimaryCtxRelease.dev = dev;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t)>("hipDevicePrimaryCtxRelease");
    hipError_t out = hip_func(hip_args.hipDevicePrimaryCtxRelease.dev);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev = hip_args.hipDevicePrimaryCtxRelease.dev;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDevicePrimaryCtxReset;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDevicePrimaryCtxReset.dev = dev;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t)>("hipDevicePrimaryCtxReset");
    hipError_t out = hip_func(hip_args.hipDevicePrimaryCtxReset.dev);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev = hip_args.hipDevicePrimaryCtxReset.dev;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDevicePrimaryCtxRetain;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDevicePrimaryCtxRetain.pctx = pctx;
    hip_args.hipDevicePrimaryCtxRetain.dev = dev;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t*,hipDevice_t)>("hipDevicePrimaryCtxRetain");
    hipError_t out = hip_func(hip_args.hipDevicePrimaryCtxRetain.pctx, hip_args.hipDevicePrimaryCtxRetain.dev);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pctx = hip_args.hipDevicePrimaryCtxRetain.pctx;
    dev = hip_args.hipDevicePrimaryCtxRetain.dev;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDevicePrimaryCtxSetFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDevicePrimaryCtxSetFlags.dev = dev;
    hip_args.hipDevicePrimaryCtxSetFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t,unsigned int)>("hipDevicePrimaryCtxSetFlags");
    hipError_t out = hip_func(hip_args.hipDevicePrimaryCtxSetFlags.dev, hip_args.hipDevicePrimaryCtxSetFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev = hip_args.hipDevicePrimaryCtxSetFlags.dev;
    flags = hip_args.hipDevicePrimaryCtxSetFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceReset() {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceReset;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipDeviceReset");
    hipError_t out = hip_func();
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceSetCacheConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceSetCacheConfig.cacheConfig = cacheConfig;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncCache_t)>("hipDeviceSetCacheConfig");
    hipError_t out = hip_func(hip_args.hipDeviceSetCacheConfig.cacheConfig);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    cacheConfig = hip_args.hipDeviceSetCacheConfig.cacheConfig;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceSetGraphMemAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceSetGraphMemAttribute.device = device;
    hip_args.hipDeviceSetGraphMemAttribute.attr = attr;
    hip_args.hipDeviceSetGraphMemAttribute.value = value;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipGraphMemAttributeType,void*)>("hipDeviceSetGraphMemAttribute");
    hipError_t out = hip_func(hip_args.hipDeviceSetGraphMemAttribute.device, hip_args.hipDeviceSetGraphMemAttribute.attr, hip_args.hipDeviceSetGraphMemAttribute.value);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device = hip_args.hipDeviceSetGraphMemAttribute.device;
    attr = hip_args.hipDeviceSetGraphMemAttribute.attr;
    value = hip_args.hipDeviceSetGraphMemAttribute.value;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceSetLimit;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceSetLimit.limit = limit;
    hip_args.hipDeviceSetLimit.value = value;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipLimit_t,size_t)>("hipDeviceSetLimit");
    hipError_t out = hip_func(hip_args.hipDeviceSetLimit.limit, hip_args.hipDeviceSetLimit.value);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    limit = hip_args.hipDeviceSetLimit.limit;
    value = hip_args.hipDeviceSetLimit.value;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceSetMemPool;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceSetMemPool.device = device;
    hip_args.hipDeviceSetMemPool.mem_pool = mem_pool;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipMemPool_t)>("hipDeviceSetMemPool");
    hipError_t out = hip_func(hip_args.hipDeviceSetMemPool.device, hip_args.hipDeviceSetMemPool.mem_pool);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device = hip_args.hipDeviceSetMemPool.device;
    mem_pool = hip_args.hipDeviceSetMemPool.mem_pool;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceSetSharedMemConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceSetSharedMemConfig.config = config;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSharedMemConfig)>("hipDeviceSetSharedMemConfig");
    hipError_t out = hip_func(hip_args.hipDeviceSetSharedMemConfig.config);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    config = hip_args.hipDeviceSetSharedMemConfig.config;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceSynchronize() {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceSynchronize;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipDeviceSynchronize");
    hipError_t out = hip_func();
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDeviceTotalMem;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDeviceTotalMem.bytes = bytes;
    hip_args.hipDeviceTotalMem.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t*,hipDevice_t)>("hipDeviceTotalMem");
    hipError_t out = hip_func(hip_args.hipDeviceTotalMem.bytes, hip_args.hipDeviceTotalMem.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    bytes = hip_args.hipDeviceTotalMem.bytes;
    device = hip_args.hipDeviceTotalMem.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDriverGetVersion(int* driverVersion) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDriverGetVersion;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDriverGetVersion.driverVersion = driverVersion;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*)>("hipDriverGetVersion");
    hipError_t out = hip_func(hip_args.hipDriverGetVersion.driverVersion);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    driverVersion = hip_args.hipDriverGetVersion.driverVersion;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDrvMemcpy2DUnaligned;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDrvMemcpy2DUnaligned.pCopy = pCopy;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hip_Memcpy2D*)>("hipDrvMemcpy2DUnaligned");
    hipError_t out = hip_func(hip_args.hipDrvMemcpy2DUnaligned.pCopy);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pCopy = hip_args.hipDrvMemcpy2DUnaligned.pCopy;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDrvMemcpy3D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDrvMemcpy3D.pCopy = pCopy;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const HIP_MEMCPY3D*)>("hipDrvMemcpy3D");
    hipError_t out = hip_func(hip_args.hipDrvMemcpy3D.pCopy);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pCopy = hip_args.hipDrvMemcpy3D.pCopy;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDrvMemcpy3DAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDrvMemcpy3DAsync.pCopy = pCopy;
    hip_args.hipDrvMemcpy3DAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const HIP_MEMCPY3D*,hipStream_t)>("hipDrvMemcpy3DAsync");
    hipError_t out = hip_func(hip_args.hipDrvMemcpy3DAsync.pCopy, hip_args.hipDrvMemcpy3DAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pCopy = hip_args.hipDrvMemcpy3DAsync.pCopy;
    stream = hip_args.hipDrvMemcpy3DAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes, void** data, hipDeviceptr_t ptr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipDrvPointerGetAttributes;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipDrvPointerGetAttributes.numAttributes = numAttributes;
    hip_args.hipDrvPointerGetAttributes.attributes = attributes;
    hip_args.hipDrvPointerGetAttributes.data = data;
    hip_args.hipDrvPointerGetAttributes.ptr = ptr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int,hipPointer_attribute*,void**,hipDeviceptr_t)>("hipDrvPointerGetAttributes");
    hipError_t out = hip_func(hip_args.hipDrvPointerGetAttributes.numAttributes, hip_args.hipDrvPointerGetAttributes.attributes, hip_args.hipDrvPointerGetAttributes.data, hip_args.hipDrvPointerGetAttributes.ptr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    numAttributes = hip_args.hipDrvPointerGetAttributes.numAttributes;
    attributes = hip_args.hipDrvPointerGetAttributes.attributes;
    data = hip_args.hipDrvPointerGetAttributes.data;
    ptr = hip_args.hipDrvPointerGetAttributes.ptr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipEventCreate(hipEvent_t* event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipEventCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipEventCreate.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t*)>("hipEventCreate");
    hipError_t out = hip_func(hip_args.hipEventCreate.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    event = hip_args.hipEventCreate.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipEventCreateWithFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipEventCreateWithFlags.event = event;
    hip_args.hipEventCreateWithFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t*,unsigned int)>("hipEventCreateWithFlags");
    hipError_t out = hip_func(hip_args.hipEventCreateWithFlags.event, hip_args.hipEventCreateWithFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    event = hip_args.hipEventCreateWithFlags.event;
    flags = hip_args.hipEventCreateWithFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipEventDestroy(hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipEventDestroy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipEventDestroy.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t)>("hipEventDestroy");
    hipError_t out = hip_func(hip_args.hipEventDestroy.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    event = hip_args.hipEventDestroy.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipEventElapsedTime;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipEventElapsedTime.ms = ms;
    hip_args.hipEventElapsedTime.start = start;
    hip_args.hipEventElapsedTime.stop = stop;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(float*,hipEvent_t,hipEvent_t)>("hipEventElapsedTime");
    hipError_t out = hip_func(hip_args.hipEventElapsedTime.ms, hip_args.hipEventElapsedTime.start, hip_args.hipEventElapsedTime.stop);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ms = hip_args.hipEventElapsedTime.ms;
    start = hip_args.hipEventElapsedTime.start;
    stop = hip_args.hipEventElapsedTime.stop;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipEventQuery(hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipEventQuery;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipEventQuery.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t)>("hipEventQuery");
    hipError_t out = hip_func(hip_args.hipEventQuery.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    event = hip_args.hipEventQuery.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipEventRecord;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipEventRecord.event = event;
    hip_args.hipEventRecord.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t,hipStream_t)>("hipEventRecord");
    hipError_t out = hip_func(hip_args.hipEventRecord.event, hip_args.hipEventRecord.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    event = hip_args.hipEventRecord.event;
    stream = hip_args.hipEventRecord.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipEventSynchronize(hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipEventSynchronize;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipEventSynchronize.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t)>("hipEventSynchronize");
    hipError_t out = hip_func(hip_args.hipEventSynchronize.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    event = hip_args.hipEventSynchronize.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, unsigned int* linktype, unsigned int* hopcount) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipExtGetLinkTypeAndHopCount;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipExtGetLinkTypeAndHopCount.device1 = device1;
    hip_args.hipExtGetLinkTypeAndHopCount.device2 = device2;
    hip_args.hipExtGetLinkTypeAndHopCount.linktype = linktype;
    hip_args.hipExtGetLinkTypeAndHopCount.hopcount = hopcount;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,int,unsigned int*,unsigned int*)>("hipExtGetLinkTypeAndHopCount");
    hipError_t out = hip_func(hip_args.hipExtGetLinkTypeAndHopCount.device1, hip_args.hipExtGetLinkTypeAndHopCount.device2, hip_args.hipExtGetLinkTypeAndHopCount.linktype, hip_args.hipExtGetLinkTypeAndHopCount.hopcount);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    device1 = hip_args.hipExtGetLinkTypeAndHopCount.device1;
    device2 = hip_args.hipExtGetLinkTypeAndHopCount.device2;
    linktype = hip_args.hipExtGetLinkTypeAndHopCount.linktype;
    hopcount = hip_args.hipExtGetLinkTypeAndHopCount.hopcount;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent, hipEvent_t stopEvent, int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipExtLaunchKernel;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipExtLaunchKernel.function_address = function_address;
    hip_args.hipExtLaunchKernel.numBlocks = numBlocks;
    hip_args.hipExtLaunchKernel.dimBlocks = dimBlocks;
    hip_args.hipExtLaunchKernel.args = args;
    hip_args.hipExtLaunchKernel.sharedMemBytes = sharedMemBytes;
    hip_args.hipExtLaunchKernel.stream = stream;
    hip_args.hipExtLaunchKernel.startEvent = startEvent;
    hip_args.hipExtLaunchKernel.stopEvent = stopEvent;
    hip_args.hipExtLaunchKernel.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,dim3,dim3,void**,size_t,hipStream_t,hipEvent_t,hipEvent_t,int)>("hipExtLaunchKernel");
    hipError_t out = hip_func(hip_args.hipExtLaunchKernel.function_address, hip_args.hipExtLaunchKernel.numBlocks, hip_args.hipExtLaunchKernel.dimBlocks, hip_args.hipExtLaunchKernel.args, hip_args.hipExtLaunchKernel.sharedMemBytes, hip_args.hipExtLaunchKernel.stream, hip_args.hipExtLaunchKernel.startEvent, hip_args.hipExtLaunchKernel.stopEvent, hip_args.hipExtLaunchKernel.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    function_address = hip_args.hipExtLaunchKernel.function_address;
    numBlocks = hip_args.hipExtLaunchKernel.numBlocks;
    dimBlocks = hip_args.hipExtLaunchKernel.dimBlocks;
    args = hip_args.hipExtLaunchKernel.args;
    sharedMemBytes = hip_args.hipExtLaunchKernel.sharedMemBytes;
    stream = hip_args.hipExtLaunchKernel.stream;
    startEvent = hip_args.hipExtLaunchKernel.startEvent;
    stopEvent = hip_args.hipExtLaunchKernel.stopEvent;
    flags = hip_args.hipExtLaunchKernel.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipExtLaunchMultiKernelMultiDevice;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipExtLaunchMultiKernelMultiDevice.launchParamsList = launchParamsList;
    hip_args.hipExtLaunchMultiKernelMultiDevice.numDevices = numDevices;
    hip_args.hipExtLaunchMultiKernelMultiDevice.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipLaunchParams*,int,unsigned int)>("hipExtLaunchMultiKernelMultiDevice");
    hipError_t out = hip_func(hip_args.hipExtLaunchMultiKernelMultiDevice.launchParamsList, hip_args.hipExtLaunchMultiKernelMultiDevice.numDevices, hip_args.hipExtLaunchMultiKernelMultiDevice.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    launchParamsList = hip_args.hipExtLaunchMultiKernelMultiDevice.launchParamsList;
    numDevices = hip_args.hipExtLaunchMultiKernelMultiDevice.numDevices;
    flags = hip_args.hipExtLaunchMultiKernelMultiDevice.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipExtMallocWithFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipExtMallocWithFlags.ptr = ptr;
    hip_args.hipExtMallocWithFlags.sizeBytes = sizeBytes;
    hip_args.hipExtMallocWithFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t,unsigned int)>("hipExtMallocWithFlags");
    hipError_t out = hip_func(hip_args.hipExtMallocWithFlags.ptr, hip_args.hipExtMallocWithFlags.sizeBytes, hip_args.hipExtMallocWithFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipExtMallocWithFlags.ptr;
    sizeBytes = hip_args.hipExtMallocWithFlags.sizeBytes;
    flags = hip_args.hipExtMallocWithFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, unsigned int cuMaskSize, const unsigned int* cuMask) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipExtStreamCreateWithCUMask;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipExtStreamCreateWithCUMask.stream = stream;
    hip_args.hipExtStreamCreateWithCUMask.cuMaskSize = cuMaskSize;
    hip_args.hipExtStreamCreateWithCUMask.cuMask = cuMask;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t*,unsigned int,const unsigned int*)>("hipExtStreamCreateWithCUMask");
    hipError_t out = hip_func(hip_args.hipExtStreamCreateWithCUMask.stream, hip_args.hipExtStreamCreateWithCUMask.cuMaskSize, hip_args.hipExtStreamCreateWithCUMask.cuMask);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipExtStreamCreateWithCUMask.stream;
    cuMaskSize = hip_args.hipExtStreamCreateWithCUMask.cuMaskSize;
    cuMask = hip_args.hipExtStreamCreateWithCUMask.cuMask;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipExtStreamGetCUMask(hipStream_t stream, unsigned int cuMaskSize, unsigned int* cuMask) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipExtStreamGetCUMask;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipExtStreamGetCUMask.stream = stream;
    hip_args.hipExtStreamGetCUMask.cuMaskSize = cuMaskSize;
    hip_args.hipExtStreamGetCUMask.cuMask = cuMask;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,unsigned int,unsigned int*)>("hipExtStreamGetCUMask");
    hipError_t out = hip_func(hip_args.hipExtStreamGetCUMask.stream, hip_args.hipExtStreamGetCUMask.cuMaskSize, hip_args.hipExtStreamGetCUMask.cuMask);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipExtStreamGetCUMask.stream;
    cuMaskSize = hip_args.hipExtStreamGetCUMask.cuMaskSize;
    cuMask = hip_args.hipExtStreamGetCUMask.cuMask;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipExternalMemoryGetMappedBuffer(void** devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc* bufferDesc) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipExternalMemoryGetMappedBuffer;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipExternalMemoryGetMappedBuffer.devPtr = devPtr;
    hip_args.hipExternalMemoryGetMappedBuffer.extMem = extMem;
    hip_args.hipExternalMemoryGetMappedBuffer.bufferDesc = bufferDesc;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,hipExternalMemory_t,const hipExternalMemoryBufferDesc*)>("hipExternalMemoryGetMappedBuffer");
    hipError_t out = hip_func(hip_args.hipExternalMemoryGetMappedBuffer.devPtr, hip_args.hipExternalMemoryGetMappedBuffer.extMem, hip_args.hipExternalMemoryGetMappedBuffer.bufferDesc);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    devPtr = hip_args.hipExternalMemoryGetMappedBuffer.devPtr;
    extMem = hip_args.hipExternalMemoryGetMappedBuffer.extMem;
    bufferDesc = hip_args.hipExternalMemoryGetMappedBuffer.bufferDesc;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipFree(void* ptr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipFree;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipFree.ptr = ptr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*)>("hipFree");
    hipError_t out = hip_func(hip_args.hipFree.ptr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipFree.ptr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipFreeArray(hipArray* array) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipFreeArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipFreeArray.array = array;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray*)>("hipFreeArray");
    hipError_t out = hip_func(hip_args.hipFreeArray.array);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    array = hip_args.hipFreeArray.array;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipFreeAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipFreeAsync.dev_ptr = dev_ptr;
    hip_args.hipFreeAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,hipStream_t)>("hipFreeAsync");
    hipError_t out = hip_func(hip_args.hipFreeAsync.dev_ptr, hip_args.hipFreeAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev_ptr = hip_args.hipFreeAsync.dev_ptr;
    stream = hip_args.hipFreeAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipFreeMipmappedArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipFreeMipmappedArray.mipmappedArray = mipmappedArray;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t)>("hipFreeMipmappedArray");
    hipError_t out = hip_func(hip_args.hipFreeMipmappedArray.mipmappedArray);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mipmappedArray = hip_args.hipFreeMipmappedArray.mipmappedArray;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipFuncGetAttributes;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipFuncGetAttributes.attr = attr;
    hip_args.hipFuncGetAttributes.func = func;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncAttributes*,const void*)>("hipFuncGetAttributes");
    hipError_t out = hip_func(hip_args.hipFuncGetAttributes.attr, hip_args.hipFuncGetAttributes.func);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    attr = hip_args.hipFuncGetAttributes.attr;
    func = hip_args.hipFuncGetAttributes.func;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipFuncSetAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipFuncSetAttribute.func = func;
    hip_args.hipFuncSetAttribute.attr = attr;
    hip_args.hipFuncSetAttribute.value = value;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,hipFuncAttribute,int)>("hipFuncSetAttribute");
    hipError_t out = hip_func(hip_args.hipFuncSetAttribute.func, hip_args.hipFuncSetAttribute.attr, hip_args.hipFuncSetAttribute.value);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    func = hip_args.hipFuncSetAttribute.func;
    attr = hip_args.hipFuncSetAttribute.attr;
    value = hip_args.hipFuncSetAttribute.value;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipFuncSetCacheConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipFuncSetCacheConfig.func = func;
    hip_args.hipFuncSetCacheConfig.config = config;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,hipFuncCache_t)>("hipFuncSetCacheConfig");
    hipError_t out = hip_func(hip_args.hipFuncSetCacheConfig.func, hip_args.hipFuncSetCacheConfig.config);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    func = hip_args.hipFuncSetCacheConfig.func;
    config = hip_args.hipFuncSetCacheConfig.config;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipFuncSetSharedMemConfig;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipFuncSetSharedMemConfig.func = func;
    hip_args.hipFuncSetSharedMemConfig.config = config;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,hipSharedMemConfig)>("hipFuncSetSharedMemConfig");
    hipError_t out = hip_func(hip_args.hipFuncSetSharedMemConfig.func, hip_args.hipFuncSetSharedMemConfig.config);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    func = hip_args.hipFuncSetSharedMemConfig.func;
    config = hip_args.hipFuncSetSharedMemConfig.config;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices, unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGLGetDevices;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGLGetDevices.pHipDeviceCount = pHipDeviceCount;
    hip_args.hipGLGetDevices.pHipDevices = pHipDevices;
    hip_args.hipGLGetDevices.hipDeviceCount = hipDeviceCount;
    hip_args.hipGLGetDevices.deviceList = deviceList;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int*,int*,unsigned int,hipGLDeviceList)>("hipGLGetDevices");
    hipError_t out = hip_func(hip_args.hipGLGetDevices.pHipDeviceCount, hip_args.hipGLGetDevices.pHipDevices, hip_args.hipGLGetDevices.hipDeviceCount, hip_args.hipGLGetDevices.deviceList);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pHipDeviceCount = hip_args.hipGLGetDevices.pHipDeviceCount;
    pHipDevices = hip_args.hipGLGetDevices.pHipDevices;
    hipDeviceCount = hip_args.hipGLGetDevices.hipDeviceCount;
    deviceList = hip_args.hipGLGetDevices.deviceList;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGetChannelDesc;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGetChannelDesc.desc = desc;
    hip_args.hipGetChannelDesc.array = array;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipChannelFormatDesc*,hipArray_const_t)>("hipGetChannelDesc");
    hipError_t out = hip_func(hip_args.hipGetChannelDesc.desc, hip_args.hipGetChannelDesc.array);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    desc = hip_args.hipGetChannelDesc.desc;
    array = hip_args.hipGetChannelDesc.array;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGetDevice(int* deviceId) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGetDevice;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGetDevice.deviceId = deviceId;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*)>("hipGetDevice");
    hipError_t out = hip_func(hip_args.hipGetDevice.deviceId);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    deviceId = hip_args.hipGetDevice.deviceId;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGetDeviceCount(int* count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGetDeviceCount;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGetDeviceCount.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*)>("hipGetDeviceCount");
    hipError_t out = hip_func(hip_args.hipGetDeviceCount.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    count = hip_args.hipGetDeviceCount.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGetDeviceFlags(unsigned int* flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGetDeviceFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGetDeviceFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int*)>("hipGetDeviceFlags");
    hipError_t out = hip_func(hip_args.hipGetDeviceFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    flags = hip_args.hipGetDeviceFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGetLastError() {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGetLastError;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipGetLastError");
    hipError_t out = hip_func();
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray, hipMipmappedArray_const_t mipmappedArray, unsigned int level) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGetMipmappedArrayLevel;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGetMipmappedArrayLevel.levelArray = levelArray;
    hip_args.hipGetMipmappedArrayLevel.mipmappedArray = mipmappedArray;
    hip_args.hipGetMipmappedArrayLevel.level = level;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_t*,hipMipmappedArray_const_t,unsigned int)>("hipGetMipmappedArrayLevel");
    hipError_t out = hip_func(hip_args.hipGetMipmappedArrayLevel.levelArray, hip_args.hipGetMipmappedArrayLevel.mipmappedArray, hip_args.hipGetMipmappedArrayLevel.level);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    levelArray = hip_args.hipGetMipmappedArrayLevel.levelArray;
    mipmappedArray = hip_args.hipGetMipmappedArrayLevel.mipmappedArray;
    level = hip_args.hipGetMipmappedArrayLevel.level;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGetSymbolAddress;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGetSymbolAddress.devPtr = devPtr;
    hip_args.hipGetSymbolAddress.symbol = symbol;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,const void*)>("hipGetSymbolAddress");
    hipError_t out = hip_func(hip_args.hipGetSymbolAddress.devPtr, hip_args.hipGetSymbolAddress.symbol);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    devPtr = hip_args.hipGetSymbolAddress.devPtr;
    symbol = hip_args.hipGetSymbolAddress.symbol;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGetSymbolSize(size_t* size, const void* symbol) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGetSymbolSize;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGetSymbolSize.size = size;
    hip_args.hipGetSymbolSize.symbol = symbol;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t*,const void*)>("hipGetSymbolSize");
    hipError_t out = hip_func(hip_args.hipGetSymbolSize.size, hip_args.hipGetSymbolSize.symbol);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    size = hip_args.hipGetSymbolSize.size;
    symbol = hip_args.hipGetSymbolSize.symbol;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipGraph_t childGraph) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddChildGraphNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddChildGraphNode.pGraphNode = pGraphNode;
    hip_args.hipGraphAddChildGraphNode.graph = graph;
    hip_args.hipGraphAddChildGraphNode.pDependencies = pDependencies;
    hip_args.hipGraphAddChildGraphNode.numDependencies = numDependencies;
    hip_args.hipGraphAddChildGraphNode.childGraph = childGraph;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,hipGraph_t)>("hipGraphAddChildGraphNode");
    hipError_t out = hip_func(hip_args.hipGraphAddChildGraphNode.pGraphNode, hip_args.hipGraphAddChildGraphNode.graph, hip_args.hipGraphAddChildGraphNode.pDependencies, hip_args.hipGraphAddChildGraphNode.numDependencies, hip_args.hipGraphAddChildGraphNode.childGraph);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddChildGraphNode.pGraphNode;
    graph = hip_args.hipGraphAddChildGraphNode.graph;
    pDependencies = hip_args.hipGraphAddChildGraphNode.pDependencies;
    numDependencies = hip_args.hipGraphAddChildGraphNode.numDependencies;
    childGraph = hip_args.hipGraphAddChildGraphNode.childGraph;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to, size_t numDependencies) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddDependencies;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddDependencies.graph = graph;
    hip_args.hipGraphAddDependencies.from = from;
    hip_args.hipGraphAddDependencies.to = to;
    hip_args.hipGraphAddDependencies.numDependencies = numDependencies;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,const hipGraphNode_t*,const hipGraphNode_t*,size_t)>("hipGraphAddDependencies");
    hipError_t out = hip_func(hip_args.hipGraphAddDependencies.graph, hip_args.hipGraphAddDependencies.from, hip_args.hipGraphAddDependencies.to, hip_args.hipGraphAddDependencies.numDependencies);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graph = hip_args.hipGraphAddDependencies.graph;
    from = hip_args.hipGraphAddDependencies.from;
    to = hip_args.hipGraphAddDependencies.to;
    numDependencies = hip_args.hipGraphAddDependencies.numDependencies;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddEmptyNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddEmptyNode.pGraphNode = pGraphNode;
    hip_args.hipGraphAddEmptyNode.graph = graph;
    hip_args.hipGraphAddEmptyNode.pDependencies = pDependencies;
    hip_args.hipGraphAddEmptyNode.numDependencies = numDependencies;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t)>("hipGraphAddEmptyNode");
    hipError_t out = hip_func(hip_args.hipGraphAddEmptyNode.pGraphNode, hip_args.hipGraphAddEmptyNode.graph, hip_args.hipGraphAddEmptyNode.pDependencies, hip_args.hipGraphAddEmptyNode.numDependencies);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddEmptyNode.pGraphNode;
    graph = hip_args.hipGraphAddEmptyNode.graph;
    pDependencies = hip_args.hipGraphAddEmptyNode.pDependencies;
    numDependencies = hip_args.hipGraphAddEmptyNode.numDependencies;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddEventRecordNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddEventRecordNode.pGraphNode = pGraphNode;
    hip_args.hipGraphAddEventRecordNode.graph = graph;
    hip_args.hipGraphAddEventRecordNode.pDependencies = pDependencies;
    hip_args.hipGraphAddEventRecordNode.numDependencies = numDependencies;
    hip_args.hipGraphAddEventRecordNode.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,hipEvent_t)>("hipGraphAddEventRecordNode");
    hipError_t out = hip_func(hip_args.hipGraphAddEventRecordNode.pGraphNode, hip_args.hipGraphAddEventRecordNode.graph, hip_args.hipGraphAddEventRecordNode.pDependencies, hip_args.hipGraphAddEventRecordNode.numDependencies, hip_args.hipGraphAddEventRecordNode.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddEventRecordNode.pGraphNode;
    graph = hip_args.hipGraphAddEventRecordNode.graph;
    pDependencies = hip_args.hipGraphAddEventRecordNode.pDependencies;
    numDependencies = hip_args.hipGraphAddEventRecordNode.numDependencies;
    event = hip_args.hipGraphAddEventRecordNode.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddEventWaitNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddEventWaitNode.pGraphNode = pGraphNode;
    hip_args.hipGraphAddEventWaitNode.graph = graph;
    hip_args.hipGraphAddEventWaitNode.pDependencies = pDependencies;
    hip_args.hipGraphAddEventWaitNode.numDependencies = numDependencies;
    hip_args.hipGraphAddEventWaitNode.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,hipEvent_t)>("hipGraphAddEventWaitNode");
    hipError_t out = hip_func(hip_args.hipGraphAddEventWaitNode.pGraphNode, hip_args.hipGraphAddEventWaitNode.graph, hip_args.hipGraphAddEventWaitNode.pDependencies, hip_args.hipGraphAddEventWaitNode.numDependencies, hip_args.hipGraphAddEventWaitNode.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddEventWaitNode.pGraphNode;
    graph = hip_args.hipGraphAddEventWaitNode.graph;
    pDependencies = hip_args.hipGraphAddEventWaitNode.pDependencies;
    numDependencies = hip_args.hipGraphAddEventWaitNode.numDependencies;
    event = hip_args.hipGraphAddEventWaitNode.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipHostNodeParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddHostNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddHostNode.pGraphNode = pGraphNode;
    hip_args.hipGraphAddHostNode.graph = graph;
    hip_args.hipGraphAddHostNode.pDependencies = pDependencies;
    hip_args.hipGraphAddHostNode.numDependencies = numDependencies;
    hip_args.hipGraphAddHostNode.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,const hipHostNodeParams*)>("hipGraphAddHostNode");
    hipError_t out = hip_func(hip_args.hipGraphAddHostNode.pGraphNode, hip_args.hipGraphAddHostNode.graph, hip_args.hipGraphAddHostNode.pDependencies, hip_args.hipGraphAddHostNode.numDependencies, hip_args.hipGraphAddHostNode.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddHostNode.pGraphNode;
    graph = hip_args.hipGraphAddHostNode.graph;
    pDependencies = hip_args.hipGraphAddHostNode.pDependencies;
    numDependencies = hip_args.hipGraphAddHostNode.numDependencies;
    pNodeParams = hip_args.hipGraphAddHostNode.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipKernelNodeParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddKernelNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddKernelNode.pGraphNode = pGraphNode;
    hip_args.hipGraphAddKernelNode.graph = graph;
    hip_args.hipGraphAddKernelNode.pDependencies = pDependencies;
    hip_args.hipGraphAddKernelNode.numDependencies = numDependencies;
    hip_args.hipGraphAddKernelNode.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,const hipKernelNodeParams*)>("hipGraphAddKernelNode");
    hipError_t out = hip_func(hip_args.hipGraphAddKernelNode.pGraphNode, hip_args.hipGraphAddKernelNode.graph, hip_args.hipGraphAddKernelNode.pDependencies, hip_args.hipGraphAddKernelNode.numDependencies, hip_args.hipGraphAddKernelNode.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddKernelNode.pGraphNode;
    graph = hip_args.hipGraphAddKernelNode.graph;
    pDependencies = hip_args.hipGraphAddKernelNode.pDependencies;
    numDependencies = hip_args.hipGraphAddKernelNode.numDependencies;
    pNodeParams = hip_args.hipGraphAddKernelNode.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipMemcpy3DParms* pCopyParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddMemcpyNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddMemcpyNode.pGraphNode = pGraphNode;
    hip_args.hipGraphAddMemcpyNode.graph = graph;
    hip_args.hipGraphAddMemcpyNode.pDependencies = pDependencies;
    hip_args.hipGraphAddMemcpyNode.numDependencies = numDependencies;
    hip_args.hipGraphAddMemcpyNode.pCopyParams = pCopyParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,const hipMemcpy3DParms*)>("hipGraphAddMemcpyNode");
    hipError_t out = hip_func(hip_args.hipGraphAddMemcpyNode.pGraphNode, hip_args.hipGraphAddMemcpyNode.graph, hip_args.hipGraphAddMemcpyNode.pDependencies, hip_args.hipGraphAddMemcpyNode.numDependencies, hip_args.hipGraphAddMemcpyNode.pCopyParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddMemcpyNode.pGraphNode;
    graph = hip_args.hipGraphAddMemcpyNode.graph;
    pDependencies = hip_args.hipGraphAddMemcpyNode.pDependencies;
    numDependencies = hip_args.hipGraphAddMemcpyNode.numDependencies;
    pCopyParams = hip_args.hipGraphAddMemcpyNode.pCopyParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddMemcpyNode1D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddMemcpyNode1D.pGraphNode = pGraphNode;
    hip_args.hipGraphAddMemcpyNode1D.graph = graph;
    hip_args.hipGraphAddMemcpyNode1D.pDependencies = pDependencies;
    hip_args.hipGraphAddMemcpyNode1D.numDependencies = numDependencies;
    hip_args.hipGraphAddMemcpyNode1D.dst = dst;
    hip_args.hipGraphAddMemcpyNode1D.src = src;
    hip_args.hipGraphAddMemcpyNode1D.count = count;
    hip_args.hipGraphAddMemcpyNode1D.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,void*,const void*,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNode1D");
    hipError_t out = hip_func(hip_args.hipGraphAddMemcpyNode1D.pGraphNode, hip_args.hipGraphAddMemcpyNode1D.graph, hip_args.hipGraphAddMemcpyNode1D.pDependencies, hip_args.hipGraphAddMemcpyNode1D.numDependencies, hip_args.hipGraphAddMemcpyNode1D.dst, hip_args.hipGraphAddMemcpyNode1D.src, hip_args.hipGraphAddMemcpyNode1D.count, hip_args.hipGraphAddMemcpyNode1D.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddMemcpyNode1D.pGraphNode;
    graph = hip_args.hipGraphAddMemcpyNode1D.graph;
    pDependencies = hip_args.hipGraphAddMemcpyNode1D.pDependencies;
    numDependencies = hip_args.hipGraphAddMemcpyNode1D.numDependencies;
    dst = hip_args.hipGraphAddMemcpyNode1D.dst;
    src = hip_args.hipGraphAddMemcpyNode1D.src;
    count = hip_args.hipGraphAddMemcpyNode1D.count;
    kind = hip_args.hipGraphAddMemcpyNode1D.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddMemcpyNodeFromSymbol;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddMemcpyNodeFromSymbol.pGraphNode = pGraphNode;
    hip_args.hipGraphAddMemcpyNodeFromSymbol.graph = graph;
    hip_args.hipGraphAddMemcpyNodeFromSymbol.pDependencies = pDependencies;
    hip_args.hipGraphAddMemcpyNodeFromSymbol.numDependencies = numDependencies;
    hip_args.hipGraphAddMemcpyNodeFromSymbol.dst = dst;
    hip_args.hipGraphAddMemcpyNodeFromSymbol.symbol = symbol;
    hip_args.hipGraphAddMemcpyNodeFromSymbol.count = count;
    hip_args.hipGraphAddMemcpyNodeFromSymbol.offset = offset;
    hip_args.hipGraphAddMemcpyNodeFromSymbol.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,void*,const void*,size_t,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNodeFromSymbol");
    hipError_t out = hip_func(hip_args.hipGraphAddMemcpyNodeFromSymbol.pGraphNode, hip_args.hipGraphAddMemcpyNodeFromSymbol.graph, hip_args.hipGraphAddMemcpyNodeFromSymbol.pDependencies, hip_args.hipGraphAddMemcpyNodeFromSymbol.numDependencies, hip_args.hipGraphAddMemcpyNodeFromSymbol.dst, hip_args.hipGraphAddMemcpyNodeFromSymbol.symbol, hip_args.hipGraphAddMemcpyNodeFromSymbol.count, hip_args.hipGraphAddMemcpyNodeFromSymbol.offset, hip_args.hipGraphAddMemcpyNodeFromSymbol.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddMemcpyNodeFromSymbol.pGraphNode;
    graph = hip_args.hipGraphAddMemcpyNodeFromSymbol.graph;
    pDependencies = hip_args.hipGraphAddMemcpyNodeFromSymbol.pDependencies;
    numDependencies = hip_args.hipGraphAddMemcpyNodeFromSymbol.numDependencies;
    dst = hip_args.hipGraphAddMemcpyNodeFromSymbol.dst;
    symbol = hip_args.hipGraphAddMemcpyNodeFromSymbol.symbol;
    count = hip_args.hipGraphAddMemcpyNodeFromSymbol.count;
    offset = hip_args.hipGraphAddMemcpyNodeFromSymbol.offset;
    kind = hip_args.hipGraphAddMemcpyNodeFromSymbol.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddMemcpyNodeToSymbol;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddMemcpyNodeToSymbol.pGraphNode = pGraphNode;
    hip_args.hipGraphAddMemcpyNodeToSymbol.graph = graph;
    hip_args.hipGraphAddMemcpyNodeToSymbol.pDependencies = pDependencies;
    hip_args.hipGraphAddMemcpyNodeToSymbol.numDependencies = numDependencies;
    hip_args.hipGraphAddMemcpyNodeToSymbol.symbol = symbol;
    hip_args.hipGraphAddMemcpyNodeToSymbol.src = src;
    hip_args.hipGraphAddMemcpyNodeToSymbol.count = count;
    hip_args.hipGraphAddMemcpyNodeToSymbol.offset = offset;
    hip_args.hipGraphAddMemcpyNodeToSymbol.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,const void*,const void*,size_t,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNodeToSymbol");
    hipError_t out = hip_func(hip_args.hipGraphAddMemcpyNodeToSymbol.pGraphNode, hip_args.hipGraphAddMemcpyNodeToSymbol.graph, hip_args.hipGraphAddMemcpyNodeToSymbol.pDependencies, hip_args.hipGraphAddMemcpyNodeToSymbol.numDependencies, hip_args.hipGraphAddMemcpyNodeToSymbol.symbol, hip_args.hipGraphAddMemcpyNodeToSymbol.src, hip_args.hipGraphAddMemcpyNodeToSymbol.count, hip_args.hipGraphAddMemcpyNodeToSymbol.offset, hip_args.hipGraphAddMemcpyNodeToSymbol.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddMemcpyNodeToSymbol.pGraphNode;
    graph = hip_args.hipGraphAddMemcpyNodeToSymbol.graph;
    pDependencies = hip_args.hipGraphAddMemcpyNodeToSymbol.pDependencies;
    numDependencies = hip_args.hipGraphAddMemcpyNodeToSymbol.numDependencies;
    symbol = hip_args.hipGraphAddMemcpyNodeToSymbol.symbol;
    src = hip_args.hipGraphAddMemcpyNodeToSymbol.src;
    count = hip_args.hipGraphAddMemcpyNodeToSymbol.count;
    offset = hip_args.hipGraphAddMemcpyNodeToSymbol.offset;
    kind = hip_args.hipGraphAddMemcpyNodeToSymbol.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipMemsetParams* pMemsetParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphAddMemsetNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphAddMemsetNode.pGraphNode = pGraphNode;
    hip_args.hipGraphAddMemsetNode.graph = graph;
    hip_args.hipGraphAddMemsetNode.pDependencies = pDependencies;
    hip_args.hipGraphAddMemsetNode.numDependencies = numDependencies;
    hip_args.hipGraphAddMemsetNode.pMemsetParams = pMemsetParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraph_t,const hipGraphNode_t*,size_t,const hipMemsetParams*)>("hipGraphAddMemsetNode");
    hipError_t out = hip_func(hip_args.hipGraphAddMemsetNode.pGraphNode, hip_args.hipGraphAddMemsetNode.graph, hip_args.hipGraphAddMemsetNode.pDependencies, hip_args.hipGraphAddMemsetNode.numDependencies, hip_args.hipGraphAddMemsetNode.pMemsetParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphNode = hip_args.hipGraphAddMemsetNode.pGraphNode;
    graph = hip_args.hipGraphAddMemsetNode.graph;
    pDependencies = hip_args.hipGraphAddMemsetNode.pDependencies;
    numDependencies = hip_args.hipGraphAddMemsetNode.numDependencies;
    pMemsetParams = hip_args.hipGraphAddMemsetNode.pMemsetParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphChildGraphNodeGetGraph;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphChildGraphNodeGetGraph.node = node;
    hip_args.hipGraphChildGraphNodeGetGraph.pGraph = pGraph;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipGraph_t*)>("hipGraphChildGraphNodeGetGraph");
    hipError_t out = hip_func(hip_args.hipGraphChildGraphNodeGetGraph.node, hip_args.hipGraphChildGraphNodeGetGraph.pGraph);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphChildGraphNodeGetGraph.node;
    pGraph = hip_args.hipGraphChildGraphNodeGetGraph.pGraph;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphClone;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphClone.pGraphClone = pGraphClone;
    hip_args.hipGraphClone.originalGraph = originalGraph;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t*,hipGraph_t)>("hipGraphClone");
    hipError_t out = hip_func(hip_args.hipGraphClone.pGraphClone, hip_args.hipGraphClone.originalGraph);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphClone = hip_args.hipGraphClone.pGraphClone;
    originalGraph = hip_args.hipGraphClone.originalGraph;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphCreate.pGraph = pGraph;
    hip_args.hipGraphCreate.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t*,unsigned int)>("hipGraphCreate");
    hipError_t out = hip_func(hip_args.hipGraphCreate.pGraph, hip_args.hipGraphCreate.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraph = hip_args.hipGraphCreate.pGraph;
    flags = hip_args.hipGraphCreate.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphDestroy(hipGraph_t graph) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphDestroy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphDestroy.graph = graph;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t)>("hipGraphDestroy");
    hipError_t out = hip_func(hip_args.hipGraphDestroy.graph);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graph = hip_args.hipGraphDestroy.graph;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphDestroyNode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphDestroyNode.node = node;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t)>("hipGraphDestroyNode");
    hipError_t out = hip_func(hip_args.hipGraphDestroyNode.node);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphDestroyNode.node;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphEventRecordNodeGetEvent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphEventRecordNodeGetEvent.node = node;
    hip_args.hipGraphEventRecordNodeGetEvent.event_out = event_out;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t*)>("hipGraphEventRecordNodeGetEvent");
    hipError_t out = hip_func(hip_args.hipGraphEventRecordNodeGetEvent.node, hip_args.hipGraphEventRecordNodeGetEvent.event_out);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphEventRecordNodeGetEvent.node;
    event_out = hip_args.hipGraphEventRecordNodeGetEvent.event_out;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphEventRecordNodeSetEvent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphEventRecordNodeSetEvent.node = node;
    hip_args.hipGraphEventRecordNodeSetEvent.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t)>("hipGraphEventRecordNodeSetEvent");
    hipError_t out = hip_func(hip_args.hipGraphEventRecordNodeSetEvent.node, hip_args.hipGraphEventRecordNodeSetEvent.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphEventRecordNodeSetEvent.node;
    event = hip_args.hipGraphEventRecordNodeSetEvent.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphEventWaitNodeGetEvent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphEventWaitNodeGetEvent.node = node;
    hip_args.hipGraphEventWaitNodeGetEvent.event_out = event_out;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t*)>("hipGraphEventWaitNodeGetEvent");
    hipError_t out = hip_func(hip_args.hipGraphEventWaitNodeGetEvent.node, hip_args.hipGraphEventWaitNodeGetEvent.event_out);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphEventWaitNodeGetEvent.node;
    event_out = hip_args.hipGraphEventWaitNodeGetEvent.event_out;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphEventWaitNodeSetEvent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphEventWaitNodeSetEvent.node = node;
    hip_args.hipGraphEventWaitNodeSetEvent.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t)>("hipGraphEventWaitNodeSetEvent");
    hipError_t out = hip_func(hip_args.hipGraphEventWaitNodeSetEvent.node, hip_args.hipGraphEventWaitNodeSetEvent.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphEventWaitNodeSetEvent.node;
    event = hip_args.hipGraphEventWaitNodeSetEvent.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipGraph_t childGraph) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecChildGraphNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecChildGraphNodeSetParams.hGraphExec = hGraphExec;
    hip_args.hipGraphExecChildGraphNodeSetParams.node = node;
    hip_args.hipGraphExecChildGraphNodeSetParams.childGraph = childGraph;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipGraph_t)>("hipGraphExecChildGraphNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphExecChildGraphNodeSetParams.hGraphExec, hip_args.hipGraphExecChildGraphNodeSetParams.node, hip_args.hipGraphExecChildGraphNodeSetParams.childGraph);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecChildGraphNodeSetParams.hGraphExec;
    node = hip_args.hipGraphExecChildGraphNodeSetParams.node;
    childGraph = hip_args.hipGraphExecChildGraphNodeSetParams.childGraph;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecDestroy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecDestroy.graphExec = graphExec;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t)>("hipGraphExecDestroy");
    hipError_t out = hip_func(hip_args.hipGraphExecDestroy.graphExec);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graphExec = hip_args.hipGraphExecDestroy.graphExec;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecEventRecordNodeSetEvent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecEventRecordNodeSetEvent.hGraphExec = hGraphExec;
    hip_args.hipGraphExecEventRecordNodeSetEvent.hNode = hNode;
    hip_args.hipGraphExecEventRecordNodeSetEvent.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t)>("hipGraphExecEventRecordNodeSetEvent");
    hipError_t out = hip_func(hip_args.hipGraphExecEventRecordNodeSetEvent.hGraphExec, hip_args.hipGraphExecEventRecordNodeSetEvent.hNode, hip_args.hipGraphExecEventRecordNodeSetEvent.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecEventRecordNodeSetEvent.hGraphExec;
    hNode = hip_args.hipGraphExecEventRecordNodeSetEvent.hNode;
    event = hip_args.hipGraphExecEventRecordNodeSetEvent.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecEventWaitNodeSetEvent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecEventWaitNodeSetEvent.hGraphExec = hGraphExec;
    hip_args.hipGraphExecEventWaitNodeSetEvent.hNode = hNode;
    hip_args.hipGraphExecEventWaitNodeSetEvent.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t)>("hipGraphExecEventWaitNodeSetEvent");
    hipError_t out = hip_func(hip_args.hipGraphExecEventWaitNodeSetEvent.hGraphExec, hip_args.hipGraphExecEventWaitNodeSetEvent.hNode, hip_args.hipGraphExecEventWaitNodeSetEvent.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecEventWaitNodeSetEvent.hGraphExec;
    hNode = hip_args.hipGraphExecEventWaitNodeSetEvent.hNode;
    event = hip_args.hipGraphExecEventWaitNodeSetEvent.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipHostNodeParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecHostNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecHostNodeSetParams.hGraphExec = hGraphExec;
    hip_args.hipGraphExecHostNodeSetParams.node = node;
    hip_args.hipGraphExecHostNodeSetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipHostNodeParams*)>("hipGraphExecHostNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphExecHostNodeSetParams.hGraphExec, hip_args.hipGraphExecHostNodeSetParams.node, hip_args.hipGraphExecHostNodeSetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecHostNodeSetParams.hGraphExec;
    node = hip_args.hipGraphExecHostNodeSetParams.node;
    pNodeParams = hip_args.hipGraphExecHostNodeSetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipKernelNodeParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecKernelNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecKernelNodeSetParams.hGraphExec = hGraphExec;
    hip_args.hipGraphExecKernelNodeSetParams.node = node;
    hip_args.hipGraphExecKernelNodeSetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipKernelNodeParams*)>("hipGraphExecKernelNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphExecKernelNodeSetParams.hGraphExec, hip_args.hipGraphExecKernelNodeSetParams.node, hip_args.hipGraphExecKernelNodeSetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecKernelNodeSetParams.hGraphExec;
    node = hip_args.hipGraphExecKernelNodeSetParams.node;
    pNodeParams = hip_args.hipGraphExecKernelNodeSetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipMemcpy3DParms* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecMemcpyNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecMemcpyNodeSetParams.hGraphExec = hGraphExec;
    hip_args.hipGraphExecMemcpyNodeSetParams.node = node;
    hip_args.hipGraphExecMemcpyNodeSetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipMemcpy3DParms*)>("hipGraphExecMemcpyNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphExecMemcpyNodeSetParams.hGraphExec, hip_args.hipGraphExecMemcpyNodeSetParams.node, hip_args.hipGraphExecMemcpyNodeSetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecMemcpyNodeSetParams.hGraphExec;
    node = hip_args.hipGraphExecMemcpyNodeSetParams.node;
    pNodeParams = hip_args.hipGraphExecMemcpyNodeSetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node, void* dst, const void* src, size_t count, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecMemcpyNodeSetParams1D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecMemcpyNodeSetParams1D.hGraphExec = hGraphExec;
    hip_args.hipGraphExecMemcpyNodeSetParams1D.node = node;
    hip_args.hipGraphExecMemcpyNodeSetParams1D.dst = dst;
    hip_args.hipGraphExecMemcpyNodeSetParams1D.src = src;
    hip_args.hipGraphExecMemcpyNodeSetParams1D.count = count;
    hip_args.hipGraphExecMemcpyNodeSetParams1D.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,void*,const void*,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParams1D");
    hipError_t out = hip_func(hip_args.hipGraphExecMemcpyNodeSetParams1D.hGraphExec, hip_args.hipGraphExecMemcpyNodeSetParams1D.node, hip_args.hipGraphExecMemcpyNodeSetParams1D.dst, hip_args.hipGraphExecMemcpyNodeSetParams1D.src, hip_args.hipGraphExecMemcpyNodeSetParams1D.count, hip_args.hipGraphExecMemcpyNodeSetParams1D.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecMemcpyNodeSetParams1D.hGraphExec;
    node = hip_args.hipGraphExecMemcpyNodeSetParams1D.node;
    dst = hip_args.hipGraphExecMemcpyNodeSetParams1D.dst;
    src = hip_args.hipGraphExecMemcpyNodeSetParams1D.src;
    count = hip_args.hipGraphExecMemcpyNodeSetParams1D.count;
    kind = hip_args.hipGraphExecMemcpyNodeSetParams1D.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecMemcpyNodeSetParamsFromSymbol;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.hGraphExec = hGraphExec;
    hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.node = node;
    hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.dst = dst;
    hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.symbol = symbol;
    hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.count = count;
    hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.offset = offset;
    hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,void*,const void*,size_t,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParamsFromSymbol");
    hipError_t out = hip_func(hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.hGraphExec, hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.node, hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.dst, hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.symbol, hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.count, hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.offset, hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.hGraphExec;
    node = hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.node;
    dst = hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.dst;
    symbol = hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.symbol;
    count = hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.count;
    offset = hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.offset;
    kind = hip_args.hipGraphExecMemcpyNodeSetParamsFromSymbol.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecMemcpyNodeSetParamsToSymbol;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.hGraphExec = hGraphExec;
    hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.node = node;
    hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.symbol = symbol;
    hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.src = src;
    hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.count = count;
    hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.offset = offset;
    hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const void*,const void*,size_t,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParamsToSymbol");
    hipError_t out = hip_func(hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.hGraphExec, hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.node, hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.symbol, hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.src, hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.count, hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.offset, hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.hGraphExec;
    node = hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.node;
    symbol = hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.symbol;
    src = hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.src;
    count = hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.count;
    offset = hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.offset;
    kind = hip_args.hipGraphExecMemcpyNodeSetParamsToSymbol.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipMemsetParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecMemsetNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecMemsetNodeSetParams.hGraphExec = hGraphExec;
    hip_args.hipGraphExecMemsetNodeSetParams.node = node;
    hip_args.hipGraphExecMemsetNodeSetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipMemsetParams*)>("hipGraphExecMemsetNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphExecMemsetNodeSetParams.hGraphExec, hip_args.hipGraphExecMemsetNodeSetParams.node, hip_args.hipGraphExecMemsetNodeSetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecMemsetNodeSetParams.hGraphExec;
    node = hip_args.hipGraphExecMemsetNodeSetParams.node;
    pNodeParams = hip_args.hipGraphExecMemsetNodeSetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph, hipGraphNode_t* hErrorNode_out, hipGraphExecUpdateResult* updateResult_out) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphExecUpdate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphExecUpdate.hGraphExec = hGraphExec;
    hip_args.hipGraphExecUpdate.hGraph = hGraph;
    hip_args.hipGraphExecUpdate.hErrorNode_out = hErrorNode_out;
    hip_args.hipGraphExecUpdate.updateResult_out = updateResult_out;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraph_t,hipGraphNode_t*,hipGraphExecUpdateResult*)>("hipGraphExecUpdate");
    hipError_t out = hip_func(hip_args.hipGraphExecUpdate.hGraphExec, hip_args.hipGraphExecUpdate.hGraph, hip_args.hipGraphExecUpdate.hErrorNode_out, hip_args.hipGraphExecUpdate.updateResult_out);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hGraphExec = hip_args.hipGraphExecUpdate.hGraphExec;
    hGraph = hip_args.hipGraphExecUpdate.hGraph;
    hErrorNode_out = hip_args.hipGraphExecUpdate.hErrorNode_out;
    updateResult_out = hip_args.hipGraphExecUpdate.updateResult_out;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to, size_t* numEdges) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphGetEdges;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphGetEdges.graph = graph;
    hip_args.hipGraphGetEdges.from = from;
    hip_args.hipGraphGetEdges.to = to;
    hip_args.hipGraphGetEdges.numEdges = numEdges;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t*,hipGraphNode_t*,size_t*)>("hipGraphGetEdges");
    hipError_t out = hip_func(hip_args.hipGraphGetEdges.graph, hip_args.hipGraphGetEdges.from, hip_args.hipGraphGetEdges.to, hip_args.hipGraphGetEdges.numEdges);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graph = hip_args.hipGraphGetEdges.graph;
    from = hip_args.hipGraphGetEdges.from;
    to = hip_args.hipGraphGetEdges.to;
    numEdges = hip_args.hipGraphGetEdges.numEdges;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphGetNodes;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphGetNodes.graph = graph;
    hip_args.hipGraphGetNodes.nodes = nodes;
    hip_args.hipGraphGetNodes.numNodes = numNodes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t*,size_t*)>("hipGraphGetNodes");
    hipError_t out = hip_func(hip_args.hipGraphGetNodes.graph, hip_args.hipGraphGetNodes.nodes, hip_args.hipGraphGetNodes.numNodes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graph = hip_args.hipGraphGetNodes.graph;
    nodes = hip_args.hipGraphGetNodes.nodes;
    numNodes = hip_args.hipGraphGetNodes.numNodes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes, size_t* pNumRootNodes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphGetRootNodes;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphGetRootNodes.graph = graph;
    hip_args.hipGraphGetRootNodes.pRootNodes = pRootNodes;
    hip_args.hipGraphGetRootNodes.pNumRootNodes = pNumRootNodes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t*,size_t*)>("hipGraphGetRootNodes");
    hipError_t out = hip_func(hip_args.hipGraphGetRootNodes.graph, hip_args.hipGraphGetRootNodes.pRootNodes, hip_args.hipGraphGetRootNodes.pNumRootNodes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graph = hip_args.hipGraphGetRootNodes.graph;
    pRootNodes = hip_args.hipGraphGetRootNodes.pRootNodes;
    pNumRootNodes = hip_args.hipGraphGetRootNodes.pNumRootNodes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphHostNodeGetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphHostNodeGetParams.node = node;
    hip_args.hipGraphHostNodeGetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipHostNodeParams*)>("hipGraphHostNodeGetParams");
    hipError_t out = hip_func(hip_args.hipGraphHostNodeGetParams.node, hip_args.hipGraphHostNodeGetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphHostNodeGetParams.node;
    pNodeParams = hip_args.hipGraphHostNodeGetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphHostNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphHostNodeSetParams.node = node;
    hip_args.hipGraphHostNodeSetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const hipHostNodeParams*)>("hipGraphHostNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphHostNodeSetParams.node, hip_args.hipGraphHostNodeSetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphHostNodeSetParams.node;
    pNodeParams = hip_args.hipGraphHostNodeSetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph, hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphInstantiate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphInstantiate.pGraphExec = pGraphExec;
    hip_args.hipGraphInstantiate.graph = graph;
    hip_args.hipGraphInstantiate.pErrorNode = pErrorNode;
    hip_args.hipGraphInstantiate.pLogBuffer = pLogBuffer;
    hip_args.hipGraphInstantiate.bufferSize = bufferSize;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t*,hipGraph_t,hipGraphNode_t*,char*,size_t)>("hipGraphInstantiate");
    hipError_t out = hip_func(hip_args.hipGraphInstantiate.pGraphExec, hip_args.hipGraphInstantiate.graph, hip_args.hipGraphInstantiate.pErrorNode, hip_args.hipGraphInstantiate.pLogBuffer, hip_args.hipGraphInstantiate.bufferSize);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphExec = hip_args.hipGraphInstantiate.pGraphExec;
    graph = hip_args.hipGraphInstantiate.graph;
    pErrorNode = hip_args.hipGraphInstantiate.pErrorNode;
    pLogBuffer = hip_args.hipGraphInstantiate.pLogBuffer;
    bufferSize = hip_args.hipGraphInstantiate.bufferSize;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph, unsigned long long flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphInstantiateWithFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphInstantiateWithFlags.pGraphExec = pGraphExec;
    hip_args.hipGraphInstantiateWithFlags.graph = graph;
    hip_args.hipGraphInstantiateWithFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t*,hipGraph_t,unsigned long long)>("hipGraphInstantiateWithFlags");
    hipError_t out = hip_func(hip_args.hipGraphInstantiateWithFlags.pGraphExec, hip_args.hipGraphInstantiateWithFlags.graph, hip_args.hipGraphInstantiateWithFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pGraphExec = hip_args.hipGraphInstantiateWithFlags.pGraphExec;
    graph = hip_args.hipGraphInstantiateWithFlags.graph;
    flags = hip_args.hipGraphInstantiateWithFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, hipKernelNodeAttrValue* value) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphKernelNodeGetAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphKernelNodeGetAttribute.hNode = hNode;
    hip_args.hipGraphKernelNodeGetAttribute.attr = attr;
    hip_args.hipGraphKernelNodeGetAttribute.value = value;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeAttrID,hipKernelNodeAttrValue*)>("hipGraphKernelNodeGetAttribute");
    hipError_t out = hip_func(hip_args.hipGraphKernelNodeGetAttribute.hNode, hip_args.hipGraphKernelNodeGetAttribute.attr, hip_args.hipGraphKernelNodeGetAttribute.value);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hNode = hip_args.hipGraphKernelNodeGetAttribute.hNode;
    attr = hip_args.hipGraphKernelNodeGetAttribute.attr;
    value = hip_args.hipGraphKernelNodeGetAttribute.value;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphKernelNodeGetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphKernelNodeGetParams.node = node;
    hip_args.hipGraphKernelNodeGetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeParams*)>("hipGraphKernelNodeGetParams");
    hipError_t out = hip_func(hip_args.hipGraphKernelNodeGetParams.node, hip_args.hipGraphKernelNodeGetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphKernelNodeGetParams.node;
    pNodeParams = hip_args.hipGraphKernelNodeGetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, const hipKernelNodeAttrValue* value) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphKernelNodeSetAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphKernelNodeSetAttribute.hNode = hNode;
    hip_args.hipGraphKernelNodeSetAttribute.attr = attr;
    hip_args.hipGraphKernelNodeSetAttribute.value = value;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeAttrID,const hipKernelNodeAttrValue*)>("hipGraphKernelNodeSetAttribute");
    hipError_t out = hip_func(hip_args.hipGraphKernelNodeSetAttribute.hNode, hip_args.hipGraphKernelNodeSetAttribute.attr, hip_args.hipGraphKernelNodeSetAttribute.value);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hNode = hip_args.hipGraphKernelNodeSetAttribute.hNode;
    attr = hip_args.hipGraphKernelNodeSetAttribute.attr;
    value = hip_args.hipGraphKernelNodeSetAttribute.value;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphKernelNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphKernelNodeSetParams.node = node;
    hip_args.hipGraphKernelNodeSetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const hipKernelNodeParams*)>("hipGraphKernelNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphKernelNodeSetParams.node, hip_args.hipGraphKernelNodeSetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphKernelNodeSetParams.node;
    pNodeParams = hip_args.hipGraphKernelNodeSetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphLaunch;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphLaunch.graphExec = graphExec;
    hip_args.hipGraphLaunch.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipStream_t)>("hipGraphLaunch");
    hipError_t out = hip_func(hip_args.hipGraphLaunch.graphExec, hip_args.hipGraphLaunch.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graphExec = hip_args.hipGraphLaunch.graphExec;
    stream = hip_args.hipGraphLaunch.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphMemcpyNodeGetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphMemcpyNodeGetParams.node = node;
    hip_args.hipGraphMemcpyNodeGetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipMemcpy3DParms*)>("hipGraphMemcpyNodeGetParams");
    hipError_t out = hip_func(hip_args.hipGraphMemcpyNodeGetParams.node, hip_args.hipGraphMemcpyNodeGetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphMemcpyNodeGetParams.node;
    pNodeParams = hip_args.hipGraphMemcpyNodeGetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphMemcpyNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphMemcpyNodeSetParams.node = node;
    hip_args.hipGraphMemcpyNodeSetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const hipMemcpy3DParms*)>("hipGraphMemcpyNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphMemcpyNodeSetParams.node, hip_args.hipGraphMemcpyNodeSetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphMemcpyNodeSetParams.node;
    pNodeParams = hip_args.hipGraphMemcpyNodeSetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src, size_t count, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphMemcpyNodeSetParams1D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphMemcpyNodeSetParams1D.node = node;
    hip_args.hipGraphMemcpyNodeSetParams1D.dst = dst;
    hip_args.hipGraphMemcpyNodeSetParams1D.src = src;
    hip_args.hipGraphMemcpyNodeSetParams1D.count = count;
    hip_args.hipGraphMemcpyNodeSetParams1D.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,void*,const void*,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParams1D");
    hipError_t out = hip_func(hip_args.hipGraphMemcpyNodeSetParams1D.node, hip_args.hipGraphMemcpyNodeSetParams1D.dst, hip_args.hipGraphMemcpyNodeSetParams1D.src, hip_args.hipGraphMemcpyNodeSetParams1D.count, hip_args.hipGraphMemcpyNodeSetParams1D.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphMemcpyNodeSetParams1D.node;
    dst = hip_args.hipGraphMemcpyNodeSetParams1D.dst;
    src = hip_args.hipGraphMemcpyNodeSetParams1D.src;
    count = hip_args.hipGraphMemcpyNodeSetParams1D.count;
    kind = hip_args.hipGraphMemcpyNodeSetParams1D.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphMemcpyNodeSetParamsFromSymbol;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.node = node;
    hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.dst = dst;
    hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.symbol = symbol;
    hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.count = count;
    hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.offset = offset;
    hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,void*,const void*,size_t,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParamsFromSymbol");
    hipError_t out = hip_func(hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.node, hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.dst, hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.symbol, hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.count, hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.offset, hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.node;
    dst = hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.dst;
    symbol = hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.symbol;
    count = hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.count;
    offset = hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.offset;
    kind = hip_args.hipGraphMemcpyNodeSetParamsFromSymbol.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphMemcpyNodeSetParamsToSymbol;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphMemcpyNodeSetParamsToSymbol.node = node;
    hip_args.hipGraphMemcpyNodeSetParamsToSymbol.symbol = symbol;
    hip_args.hipGraphMemcpyNodeSetParamsToSymbol.src = src;
    hip_args.hipGraphMemcpyNodeSetParamsToSymbol.count = count;
    hip_args.hipGraphMemcpyNodeSetParamsToSymbol.offset = offset;
    hip_args.hipGraphMemcpyNodeSetParamsToSymbol.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const void*,const void*,size_t,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParamsToSymbol");
    hipError_t out = hip_func(hip_args.hipGraphMemcpyNodeSetParamsToSymbol.node, hip_args.hipGraphMemcpyNodeSetParamsToSymbol.symbol, hip_args.hipGraphMemcpyNodeSetParamsToSymbol.src, hip_args.hipGraphMemcpyNodeSetParamsToSymbol.count, hip_args.hipGraphMemcpyNodeSetParamsToSymbol.offset, hip_args.hipGraphMemcpyNodeSetParamsToSymbol.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphMemcpyNodeSetParamsToSymbol.node;
    symbol = hip_args.hipGraphMemcpyNodeSetParamsToSymbol.symbol;
    src = hip_args.hipGraphMemcpyNodeSetParamsToSymbol.src;
    count = hip_args.hipGraphMemcpyNodeSetParamsToSymbol.count;
    offset = hip_args.hipGraphMemcpyNodeSetParamsToSymbol.offset;
    kind = hip_args.hipGraphMemcpyNodeSetParamsToSymbol.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphMemsetNodeGetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphMemsetNodeGetParams.node = node;
    hip_args.hipGraphMemsetNodeGetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipMemsetParams*)>("hipGraphMemsetNodeGetParams");
    hipError_t out = hip_func(hip_args.hipGraphMemsetNodeGetParams.node, hip_args.hipGraphMemsetNodeGetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphMemsetNodeGetParams.node;
    pNodeParams = hip_args.hipGraphMemsetNodeGetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphMemsetNodeSetParams;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphMemsetNodeSetParams.node = node;
    hip_args.hipGraphMemsetNodeSetParams.pNodeParams = pNodeParams;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const hipMemsetParams*)>("hipGraphMemsetNodeSetParams");
    hipError_t out = hip_func(hip_args.hipGraphMemsetNodeSetParams.node, hip_args.hipGraphMemsetNodeSetParams.pNodeParams);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphMemsetNodeSetParams.node;
    pNodeParams = hip_args.hipGraphMemsetNodeSetParams.pNodeParams;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode, hipGraph_t clonedGraph) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphNodeFindInClone;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphNodeFindInClone.pNode = pNode;
    hip_args.hipGraphNodeFindInClone.originalNode = originalNode;
    hip_args.hipGraphNodeFindInClone.clonedGraph = clonedGraph;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t*,hipGraphNode_t,hipGraph_t)>("hipGraphNodeFindInClone");
    hipError_t out = hip_func(hip_args.hipGraphNodeFindInClone.pNode, hip_args.hipGraphNodeFindInClone.originalNode, hip_args.hipGraphNodeFindInClone.clonedGraph);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pNode = hip_args.hipGraphNodeFindInClone.pNode;
    originalNode = hip_args.hipGraphNodeFindInClone.originalNode;
    clonedGraph = hip_args.hipGraphNodeFindInClone.clonedGraph;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies, size_t* pNumDependencies) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphNodeGetDependencies;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphNodeGetDependencies.node = node;
    hip_args.hipGraphNodeGetDependencies.pDependencies = pDependencies;
    hip_args.hipGraphNodeGetDependencies.pNumDependencies = pNumDependencies;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t*,size_t*)>("hipGraphNodeGetDependencies");
    hipError_t out = hip_func(hip_args.hipGraphNodeGetDependencies.node, hip_args.hipGraphNodeGetDependencies.pDependencies, hip_args.hipGraphNodeGetDependencies.pNumDependencies);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphNodeGetDependencies.node;
    pDependencies = hip_args.hipGraphNodeGetDependencies.pDependencies;
    pNumDependencies = hip_args.hipGraphNodeGetDependencies.pNumDependencies;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphNodeGetDependentNodes;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphNodeGetDependentNodes.node = node;
    hip_args.hipGraphNodeGetDependentNodes.pDependentNodes = pDependentNodes;
    hip_args.hipGraphNodeGetDependentNodes.pNumDependentNodes = pNumDependentNodes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t*,size_t*)>("hipGraphNodeGetDependentNodes");
    hipError_t out = hip_func(hip_args.hipGraphNodeGetDependentNodes.node, hip_args.hipGraphNodeGetDependentNodes.pDependentNodes, hip_args.hipGraphNodeGetDependentNodes.pNumDependentNodes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphNodeGetDependentNodes.node;
    pDependentNodes = hip_args.hipGraphNodeGetDependentNodes.pDependentNodes;
    pNumDependentNodes = hip_args.hipGraphNodeGetDependentNodes.pNumDependentNodes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphNodeGetType;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphNodeGetType.node = node;
    hip_args.hipGraphNodeGetType.pType = pType;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNodeType*)>("hipGraphNodeGetType");
    hipError_t out = hip_func(hip_args.hipGraphNodeGetType.node, hip_args.hipGraphNodeGetType.pType);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    node = hip_args.hipGraphNodeGetType.node;
    pType = hip_args.hipGraphNodeGetType.pType;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphReleaseUserObject;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphReleaseUserObject.graph = graph;
    hip_args.hipGraphReleaseUserObject.object = object;
    hip_args.hipGraphReleaseUserObject.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipUserObject_t,unsigned int)>("hipGraphReleaseUserObject");
    hipError_t out = hip_func(hip_args.hipGraphReleaseUserObject.graph, hip_args.hipGraphReleaseUserObject.object, hip_args.hipGraphReleaseUserObject.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graph = hip_args.hipGraphReleaseUserObject.graph;
    object = hip_args.hipGraphReleaseUserObject.object;
    count = hip_args.hipGraphReleaseUserObject.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to, size_t numDependencies) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphRemoveDependencies;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphRemoveDependencies.graph = graph;
    hip_args.hipGraphRemoveDependencies.from = from;
    hip_args.hipGraphRemoveDependencies.to = to;
    hip_args.hipGraphRemoveDependencies.numDependencies = numDependencies;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,const hipGraphNode_t*,const hipGraphNode_t*,size_t)>("hipGraphRemoveDependencies");
    hipError_t out = hip_func(hip_args.hipGraphRemoveDependencies.graph, hip_args.hipGraphRemoveDependencies.from, hip_args.hipGraphRemoveDependencies.to, hip_args.hipGraphRemoveDependencies.numDependencies);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graph = hip_args.hipGraphRemoveDependencies.graph;
    from = hip_args.hipGraphRemoveDependencies.from;
    to = hip_args.hipGraphRemoveDependencies.to;
    numDependencies = hip_args.hipGraphRemoveDependencies.numDependencies;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphRetainUserObject;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphRetainUserObject.graph = graph;
    hip_args.hipGraphRetainUserObject.object = object;
    hip_args.hipGraphRetainUserObject.count = count;
    hip_args.hipGraphRetainUserObject.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipUserObject_t,unsigned int,unsigned int)>("hipGraphRetainUserObject");
    hipError_t out = hip_func(hip_args.hipGraphRetainUserObject.graph, hip_args.hipGraphRetainUserObject.object, hip_args.hipGraphRetainUserObject.count, hip_args.hipGraphRetainUserObject.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graph = hip_args.hipGraphRetainUserObject.graph;
    object = hip_args.hipGraphRetainUserObject.object;
    count = hip_args.hipGraphRetainUserObject.count;
    flags = hip_args.hipGraphRetainUserObject.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphUpload;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphUpload.graphExec = graphExec;
    hip_args.hipGraphUpload.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipStream_t)>("hipGraphUpload");
    hipError_t out = hip_func(hip_args.hipGraphUpload.graphExec, hip_args.hipGraphUpload.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    graphExec = hip_args.hipGraphUpload.graphExec;
    stream = hip_args.hipGraphUpload.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphicsGLRegisterBuffer;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphicsGLRegisterBuffer.resource = resource;
    hip_args.hipGraphicsGLRegisterBuffer.buffer = buffer;
    hip_args.hipGraphicsGLRegisterBuffer.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphicsResource**,GLuint,unsigned int)>("hipGraphicsGLRegisterBuffer");
    hipError_t out = hip_func(hip_args.hipGraphicsGLRegisterBuffer.resource, hip_args.hipGraphicsGLRegisterBuffer.buffer, hip_args.hipGraphicsGLRegisterBuffer.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    resource = hip_args.hipGraphicsGLRegisterBuffer.resource;
    buffer = hip_args.hipGraphicsGLRegisterBuffer.buffer;
    flags = hip_args.hipGraphicsGLRegisterBuffer.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphicsGLRegisterImage;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphicsGLRegisterImage.resource = resource;
    hip_args.hipGraphicsGLRegisterImage.image = image;
    hip_args.hipGraphicsGLRegisterImage.target = target;
    hip_args.hipGraphicsGLRegisterImage.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphicsResource**,GLuint,GLenum,unsigned int)>("hipGraphicsGLRegisterImage");
    hipError_t out = hip_func(hip_args.hipGraphicsGLRegisterImage.resource, hip_args.hipGraphicsGLRegisterImage.image, hip_args.hipGraphicsGLRegisterImage.target, hip_args.hipGraphicsGLRegisterImage.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    resource = hip_args.hipGraphicsGLRegisterImage.resource;
    image = hip_args.hipGraphicsGLRegisterImage.image;
    target = hip_args.hipGraphicsGLRegisterImage.target;
    flags = hip_args.hipGraphicsGLRegisterImage.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphicsMapResources;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphicsMapResources.count = count;
    hip_args.hipGraphicsMapResources.resources = resources;
    hip_args.hipGraphicsMapResources.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipGraphicsResource_t*,hipStream_t)>("hipGraphicsMapResources");
    hipError_t out = hip_func(hip_args.hipGraphicsMapResources.count, hip_args.hipGraphicsMapResources.resources, hip_args.hipGraphicsMapResources.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    count = hip_args.hipGraphicsMapResources.count;
    resources = hip_args.hipGraphicsMapResources.resources;
    stream = hip_args.hipGraphicsMapResources.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, hipGraphicsResource_t resource) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphicsResourceGetMappedPointer;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphicsResourceGetMappedPointer.devPtr = devPtr;
    hip_args.hipGraphicsResourceGetMappedPointer.size = size;
    hip_args.hipGraphicsResourceGetMappedPointer.resource = resource;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t*,hipGraphicsResource_t)>("hipGraphicsResourceGetMappedPointer");
    hipError_t out = hip_func(hip_args.hipGraphicsResourceGetMappedPointer.devPtr, hip_args.hipGraphicsResourceGetMappedPointer.size, hip_args.hipGraphicsResourceGetMappedPointer.resource);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    devPtr = hip_args.hipGraphicsResourceGetMappedPointer.devPtr;
    size = hip_args.hipGraphicsResourceGetMappedPointer.size;
    resource = hip_args.hipGraphicsResourceGetMappedPointer.resource;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphicsSubResourceGetMappedArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphicsSubResourceGetMappedArray.array = array;
    hip_args.hipGraphicsSubResourceGetMappedArray.resource = resource;
    hip_args.hipGraphicsSubResourceGetMappedArray.arrayIndex = arrayIndex;
    hip_args.hipGraphicsSubResourceGetMappedArray.mipLevel = mipLevel;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_t*,hipGraphicsResource_t,unsigned int,unsigned int)>("hipGraphicsSubResourceGetMappedArray");
    hipError_t out = hip_func(hip_args.hipGraphicsSubResourceGetMappedArray.array, hip_args.hipGraphicsSubResourceGetMappedArray.resource, hip_args.hipGraphicsSubResourceGetMappedArray.arrayIndex, hip_args.hipGraphicsSubResourceGetMappedArray.mipLevel);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    array = hip_args.hipGraphicsSubResourceGetMappedArray.array;
    resource = hip_args.hipGraphicsSubResourceGetMappedArray.resource;
    arrayIndex = hip_args.hipGraphicsSubResourceGetMappedArray.arrayIndex;
    mipLevel = hip_args.hipGraphicsSubResourceGetMappedArray.mipLevel;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphicsUnmapResources;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphicsUnmapResources.count = count;
    hip_args.hipGraphicsUnmapResources.resources = resources;
    hip_args.hipGraphicsUnmapResources.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipGraphicsResource_t*,hipStream_t)>("hipGraphicsUnmapResources");
    hipError_t out = hip_func(hip_args.hipGraphicsUnmapResources.count, hip_args.hipGraphicsUnmapResources.resources, hip_args.hipGraphicsUnmapResources.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    count = hip_args.hipGraphicsUnmapResources.count;
    resources = hip_args.hipGraphicsUnmapResources.resources;
    stream = hip_args.hipGraphicsUnmapResources.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipGraphicsUnregisterResource;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipGraphicsUnregisterResource.resource = resource;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphicsResource_t)>("hipGraphicsUnregisterResource");
    hipError_t out = hip_func(hip_args.hipGraphicsUnregisterResource.resource);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    resource = hip_args.hipGraphicsUnregisterResource.resource;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipHostAlloc;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipHostAlloc.ptr = ptr;
    hip_args.hipHostAlloc.size = size;
    hip_args.hipHostAlloc.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t,unsigned int)>("hipHostAlloc");
    hipError_t out = hip_func(hip_args.hipHostAlloc.ptr, hip_args.hipHostAlloc.size, hip_args.hipHostAlloc.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipHostAlloc.ptr;
    size = hip_args.hipHostAlloc.size;
    flags = hip_args.hipHostAlloc.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipHostFree(void* ptr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipHostFree;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipHostFree.ptr = ptr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*)>("hipHostFree");
    hipError_t out = hip_func(hip_args.hipHostFree.ptr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipHostFree.ptr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipHostGetDevicePointer;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipHostGetDevicePointer.devPtr = devPtr;
    hip_args.hipHostGetDevicePointer.hstPtr = hstPtr;
    hip_args.hipHostGetDevicePointer.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,void*,unsigned int)>("hipHostGetDevicePointer");
    hipError_t out = hip_func(hip_args.hipHostGetDevicePointer.devPtr, hip_args.hipHostGetDevicePointer.hstPtr, hip_args.hipHostGetDevicePointer.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    devPtr = hip_args.hipHostGetDevicePointer.devPtr;
    hstPtr = hip_args.hipHostGetDevicePointer.hstPtr;
    flags = hip_args.hipHostGetDevicePointer.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipHostGetFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipHostGetFlags.flagsPtr = flagsPtr;
    hip_args.hipHostGetFlags.hostPtr = hostPtr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int*,void*)>("hipHostGetFlags");
    hipError_t out = hip_func(hip_args.hipHostGetFlags.flagsPtr, hip_args.hipHostGetFlags.hostPtr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    flagsPtr = hip_args.hipHostGetFlags.flagsPtr;
    hostPtr = hip_args.hipHostGetFlags.hostPtr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipHostMalloc;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipHostMalloc.ptr = ptr;
    hip_args.hipHostMalloc.size = size;
    hip_args.hipHostMalloc.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t,unsigned int)>("hipHostMalloc");
    hipError_t out = hip_func(hip_args.hipHostMalloc.ptr, hip_args.hipHostMalloc.size, hip_args.hipHostMalloc.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipHostMalloc.ptr;
    size = hip_args.hipHostMalloc.size;
    flags = hip_args.hipHostMalloc.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipHostRegister;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipHostRegister.hostPtr = hostPtr;
    hip_args.hipHostRegister.sizeBytes = sizeBytes;
    hip_args.hipHostRegister.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,unsigned int)>("hipHostRegister");
    hipError_t out = hip_func(hip_args.hipHostRegister.hostPtr, hip_args.hipHostRegister.sizeBytes, hip_args.hipHostRegister.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hostPtr = hip_args.hipHostRegister.hostPtr;
    sizeBytes = hip_args.hipHostRegister.sizeBytes;
    flags = hip_args.hipHostRegister.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipHostUnregister(void* hostPtr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipHostUnregister;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipHostUnregister.hostPtr = hostPtr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*)>("hipHostUnregister");
    hipError_t out = hip_func(hip_args.hipHostUnregister.hostPtr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hostPtr = hip_args.hipHostUnregister.hostPtr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out, const hipExternalMemoryHandleDesc* memHandleDesc) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipImportExternalMemory;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipImportExternalMemory.extMem_out = extMem_out;
    hip_args.hipImportExternalMemory.memHandleDesc = memHandleDesc;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipExternalMemory_t*,const hipExternalMemoryHandleDesc*)>("hipImportExternalMemory");
    hipError_t out = hip_func(hip_args.hipImportExternalMemory.extMem_out, hip_args.hipImportExternalMemory.memHandleDesc);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    extMem_out = hip_args.hipImportExternalMemory.extMem_out;
    memHandleDesc = hip_args.hipImportExternalMemory.memHandleDesc;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out, const hipExternalSemaphoreHandleDesc* semHandleDesc) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipImportExternalSemaphore;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipImportExternalSemaphore.extSem_out = extSem_out;
    hip_args.hipImportExternalSemaphore.semHandleDesc = semHandleDesc;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipExternalSemaphore_t*,const hipExternalSemaphoreHandleDesc*)>("hipImportExternalSemaphore");
    hipError_t out = hip_func(hip_args.hipImportExternalSemaphore.extSem_out, hip_args.hipImportExternalSemaphore.semHandleDesc);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    extSem_out = hip_args.hipImportExternalSemaphore.extSem_out;
    semHandleDesc = hip_args.hipImportExternalSemaphore.semHandleDesc;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipInit(unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipInit;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipInit.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int)>("hipInit");
    hipError_t out = hip_func(hip_args.hipInit.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    flags = hip_args.hipInit.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipIpcCloseMemHandle(void* devPtr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipIpcCloseMemHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipIpcCloseMemHandle.devPtr = devPtr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*)>("hipIpcCloseMemHandle");
    hipError_t out = hip_func(hip_args.hipIpcCloseMemHandle.devPtr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    devPtr = hip_args.hipIpcCloseMemHandle.devPtr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipIpcGetEventHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipIpcGetEventHandle.handle = handle;
    hip_args.hipIpcGetEventHandle.event = event;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipIpcEventHandle_t*,hipEvent_t)>("hipIpcGetEventHandle");
    hipError_t out = hip_func(hip_args.hipIpcGetEventHandle.handle, hip_args.hipIpcGetEventHandle.event);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    handle = hip_args.hipIpcGetEventHandle.handle;
    event = hip_args.hipIpcGetEventHandle.event;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipIpcGetMemHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipIpcGetMemHandle.handle = handle;
    hip_args.hipIpcGetMemHandle.devPtr = devPtr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipIpcMemHandle_t*,void*)>("hipIpcGetMemHandle");
    hipError_t out = hip_func(hip_args.hipIpcGetMemHandle.handle, hip_args.hipIpcGetMemHandle.devPtr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    handle = hip_args.hipIpcGetMemHandle.handle;
    devPtr = hip_args.hipIpcGetMemHandle.devPtr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipIpcOpenEventHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipIpcOpenEventHandle.event = event;
    hip_args.hipIpcOpenEventHandle.handle = handle;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t*,hipIpcEventHandle_t)>("hipIpcOpenEventHandle");
    hipError_t out = hip_func(hip_args.hipIpcOpenEventHandle.event, hip_args.hipIpcOpenEventHandle.handle);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    event = hip_args.hipIpcOpenEventHandle.event;
    handle = hip_args.hipIpcOpenEventHandle.handle;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipIpcOpenMemHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipIpcOpenMemHandle.devPtr = devPtr;
    hip_args.hipIpcOpenMemHandle.handle = handle;
    hip_args.hipIpcOpenMemHandle.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,hipIpcMemHandle_t,unsigned int)>("hipIpcOpenMemHandle");
    hipError_t out = hip_func(hip_args.hipIpcOpenMemHandle.devPtr, hip_args.hipIpcOpenMemHandle.handle, hip_args.hipIpcOpenMemHandle.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    devPtr = hip_args.hipIpcOpenMemHandle.devPtr;
    handle = hip_args.hipIpcOpenMemHandle.handle;
    flags = hip_args.hipIpcOpenMemHandle.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipLaunchCooperativeKernel;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipLaunchCooperativeKernel.f = f;
    hip_args.hipLaunchCooperativeKernel.gridDim = gridDim;
    hip_args.hipLaunchCooperativeKernel.blockDimX = blockDimX;
    hip_args.hipLaunchCooperativeKernel.kernelParams = kernelParams;
    hip_args.hipLaunchCooperativeKernel.sharedMemBytes = sharedMemBytes;
    hip_args.hipLaunchCooperativeKernel.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,dim3,dim3,void**,unsigned int,hipStream_t)>("hipLaunchCooperativeKernel");
    hipError_t out = hip_func(hip_args.hipLaunchCooperativeKernel.f, hip_args.hipLaunchCooperativeKernel.gridDim, hip_args.hipLaunchCooperativeKernel.blockDimX, hip_args.hipLaunchCooperativeKernel.kernelParams, hip_args.hipLaunchCooperativeKernel.sharedMemBytes, hip_args.hipLaunchCooperativeKernel.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    f = hip_args.hipLaunchCooperativeKernel.f;
    gridDim = hip_args.hipLaunchCooperativeKernel.gridDim;
    blockDimX = hip_args.hipLaunchCooperativeKernel.blockDimX;
    kernelParams = hip_args.hipLaunchCooperativeKernel.kernelParams;
    sharedMemBytes = hip_args.hipLaunchCooperativeKernel.sharedMemBytes;
    stream = hip_args.hipLaunchCooperativeKernel.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipLaunchCooperativeKernelMultiDevice;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipLaunchCooperativeKernelMultiDevice.launchParamsList = launchParamsList;
    hip_args.hipLaunchCooperativeKernelMultiDevice.numDevices = numDevices;
    hip_args.hipLaunchCooperativeKernelMultiDevice.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipLaunchParams*,int,unsigned int)>("hipLaunchCooperativeKernelMultiDevice");
    hipError_t out = hip_func(hip_args.hipLaunchCooperativeKernelMultiDevice.launchParamsList, hip_args.hipLaunchCooperativeKernelMultiDevice.numDevices, hip_args.hipLaunchCooperativeKernelMultiDevice.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    launchParamsList = hip_args.hipLaunchCooperativeKernelMultiDevice.launchParamsList;
    numDevices = hip_args.hipLaunchCooperativeKernelMultiDevice.numDevices;
    flags = hip_args.hipLaunchCooperativeKernelMultiDevice.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipLaunchHostFunc;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipLaunchHostFunc.stream = stream;
    hip_args.hipLaunchHostFunc.fn = fn;
    hip_args.hipLaunchHostFunc.userData = userData;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipHostFn_t,void*)>("hipLaunchHostFunc");
    hipError_t out = hip_func(hip_args.hipLaunchHostFunc.stream, hip_args.hipLaunchHostFunc.fn, hip_args.hipLaunchHostFunc.userData);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipLaunchHostFunc.stream;
    fn = hip_args.hipLaunchHostFunc.fn;
    userData = hip_args.hipLaunchHostFunc.userData;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipLaunchKernel;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipLaunchKernel.function_address = function_address;
    hip_args.hipLaunchKernel.numBlocks = numBlocks;
    hip_args.hipLaunchKernel.dimBlocks = dimBlocks;
    hip_args.hipLaunchKernel.args = args;
    hip_args.hipLaunchKernel.sharedMemBytes = sharedMemBytes;
    hip_args.hipLaunchKernel.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,dim3,dim3,void**,size_t,hipStream_t)>("hipLaunchKernel");
    hipError_t out = hip_func(hip_args.hipLaunchKernel.function_address, hip_args.hipLaunchKernel.numBlocks, hip_args.hipLaunchKernel.dimBlocks, hip_args.hipLaunchKernel.args, hip_args.hipLaunchKernel.sharedMemBytes, hip_args.hipLaunchKernel.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    function_address = hip_args.hipLaunchKernel.function_address;
    numBlocks = hip_args.hipLaunchKernel.numBlocks;
    dimBlocks = hip_args.hipLaunchKernel.dimBlocks;
    args = hip_args.hipLaunchKernel.args;
    sharedMemBytes = hip_args.hipLaunchKernel.sharedMemBytes;
    stream = hip_args.hipLaunchKernel.stream;
    return out;
}

__attribute__((visibility("default")))
hipError_t hipMalloc(void** ptr, size_t size) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMalloc;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMalloc.ptr = ptr;
    hip_args.hipMalloc.size = size;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t)>("hipMalloc");
    hipError_t out = hip_func(hip_args.hipMalloc.ptr, hip_args.hipMalloc.size);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipMalloc.ptr;
    size = hip_args.hipMalloc.size;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMalloc3D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMalloc3D.pitchedDevPtr = pitchedDevPtr;
    hip_args.hipMalloc3D.extent = extent;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipPitchedPtr*,hipExtent)>("hipMalloc3D");
    hipError_t out = hip_func(hip_args.hipMalloc3D.pitchedDevPtr, hip_args.hipMalloc3D.extent);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pitchedDevPtr = hip_args.hipMalloc3D.pitchedDevPtr;
    extent = hip_args.hipMalloc3D.extent;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMallocArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMallocArray.array = array;
    hip_args.hipMallocArray.desc = desc;
    hip_args.hipMallocArray.width = width;
    hip_args.hipMallocArray.height = height;
    hip_args.hipMallocArray.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray**,const hipChannelFormatDesc*,size_t,size_t,unsigned int)>("hipMallocArray");
    hipError_t out = hip_func(hip_args.hipMallocArray.array, hip_args.hipMallocArray.desc, hip_args.hipMallocArray.width, hip_args.hipMallocArray.height, hip_args.hipMallocArray.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    array = hip_args.hipMallocArray.array;
    desc = hip_args.hipMallocArray.desc;
    width = hip_args.hipMallocArray.width;
    height = hip_args.hipMallocArray.height;
    flags = hip_args.hipMallocArray.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMallocAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMallocAsync.dev_ptr = dev_ptr;
    hip_args.hipMallocAsync.size = size;
    hip_args.hipMallocAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t,hipStream_t)>("hipMallocAsync");
    hipError_t out = hip_func(hip_args.hipMallocAsync.dev_ptr, hip_args.hipMallocAsync.size, hip_args.hipMallocAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev_ptr = hip_args.hipMallocAsync.dev_ptr;
    size = hip_args.hipMallocAsync.size;
    stream = hip_args.hipMallocAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMallocFromPoolAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMallocFromPoolAsync.dev_ptr = dev_ptr;
    hip_args.hipMallocFromPoolAsync.size = size;
    hip_args.hipMallocFromPoolAsync.mem_pool = mem_pool;
    hip_args.hipMallocFromPoolAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t,hipMemPool_t,hipStream_t)>("hipMallocFromPoolAsync");
    hipError_t out = hip_func(hip_args.hipMallocFromPoolAsync.dev_ptr, hip_args.hipMallocFromPoolAsync.size, hip_args.hipMallocFromPoolAsync.mem_pool, hip_args.hipMallocFromPoolAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev_ptr = hip_args.hipMallocFromPoolAsync.dev_ptr;
    size = hip_args.hipMallocFromPoolAsync.size;
    mem_pool = hip_args.hipMallocFromPoolAsync.mem_pool;
    stream = hip_args.hipMallocFromPoolAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray, const hipChannelFormatDesc* desc, hipExtent extent, unsigned int numLevels, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMallocMipmappedArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMallocMipmappedArray.mipmappedArray = mipmappedArray;
    hip_args.hipMallocMipmappedArray.desc = desc;
    hip_args.hipMallocMipmappedArray.extent = extent;
    hip_args.hipMallocMipmappedArray.numLevels = numLevels;
    hip_args.hipMallocMipmappedArray.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t*,const hipChannelFormatDesc*,hipExtent,unsigned int,unsigned int)>("hipMallocMipmappedArray");
    hipError_t out = hip_func(hip_args.hipMallocMipmappedArray.mipmappedArray, hip_args.hipMallocMipmappedArray.desc, hip_args.hipMallocMipmappedArray.extent, hip_args.hipMallocMipmappedArray.numLevels, hip_args.hipMallocMipmappedArray.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mipmappedArray = hip_args.hipMallocMipmappedArray.mipmappedArray;
    desc = hip_args.hipMallocMipmappedArray.desc;
    extent = hip_args.hipMallocMipmappedArray.extent;
    numLevels = hip_args.hipMallocMipmappedArray.numLevels;
    flags = hip_args.hipMallocMipmappedArray.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMallocPitch;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMallocPitch.ptr = ptr;
    hip_args.hipMallocPitch.pitch = pitch;
    hip_args.hipMallocPitch.width = width;
    hip_args.hipMallocPitch.height = height;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t*,size_t,size_t)>("hipMallocPitch");
    hipError_t out = hip_func(hip_args.hipMallocPitch.ptr, hip_args.hipMallocPitch.pitch, hip_args.hipMallocPitch.width, hip_args.hipMallocPitch.height);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipMallocPitch.ptr;
    pitch = hip_args.hipMallocPitch.pitch;
    width = hip_args.hipMallocPitch.width;
    height = hip_args.hipMallocPitch.height;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemAddressFree(void* devPtr, size_t size) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemAddressFree;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemAddressFree.devPtr = devPtr;
    hip_args.hipMemAddressFree.size = size;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t)>("hipMemAddressFree");
    hipError_t out = hip_func(hip_args.hipMemAddressFree.devPtr, hip_args.hipMemAddressFree.size);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    devPtr = hip_args.hipMemAddressFree.devPtr;
    size = hip_args.hipMemAddressFree.size;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemAddressReserve;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemAddressReserve.ptr = ptr;
    hip_args.hipMemAddressReserve.size = size;
    hip_args.hipMemAddressReserve.alignment = alignment;
    hip_args.hipMemAddressReserve.addr = addr;
    hip_args.hipMemAddressReserve.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t,size_t,void*,unsigned long long)>("hipMemAddressReserve");
    hipError_t out = hip_func(hip_args.hipMemAddressReserve.ptr, hip_args.hipMemAddressReserve.size, hip_args.hipMemAddressReserve.alignment, hip_args.hipMemAddressReserve.addr, hip_args.hipMemAddressReserve.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipMemAddressReserve.ptr;
    size = hip_args.hipMemAddressReserve.size;
    alignment = hip_args.hipMemAddressReserve.alignment;
    addr = hip_args.hipMemAddressReserve.addr;
    flags = hip_args.hipMemAddressReserve.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemAdvise;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemAdvise.dev_ptr = dev_ptr;
    hip_args.hipMemAdvise.count = count;
    hip_args.hipMemAdvise.advice = advice;
    hip_args.hipMemAdvise.device = device;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,size_t,hipMemoryAdvise,int)>("hipMemAdvise");
    hipError_t out = hip_func(hip_args.hipMemAdvise.dev_ptr, hip_args.hipMemAdvise.count, hip_args.hipMemAdvise.advice, hip_args.hipMemAdvise.device);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev_ptr = hip_args.hipMemAdvise.dev_ptr;
    count = hip_args.hipMemAdvise.count;
    advice = hip_args.hipMemAdvise.advice;
    device = hip_args.hipMemAdvise.device;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemAllocHost(void** ptr, size_t size) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemAllocHost;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemAllocHost.ptr = ptr;
    hip_args.hipMemAllocHost.size = size;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t)>("hipMemAllocHost");
    hipError_t out = hip_func(hip_args.hipMemAllocHost.ptr, hip_args.hipMemAllocHost.size);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipMemAllocHost.ptr;
    size = hip_args.hipMemAllocHost.size;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemAllocPitch;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemAllocPitch.dptr = dptr;
    hip_args.hipMemAllocPitch.pitch = pitch;
    hip_args.hipMemAllocPitch.widthInBytes = widthInBytes;
    hip_args.hipMemAllocPitch.height = height;
    hip_args.hipMemAllocPitch.elementSizeBytes = elementSizeBytes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t*,size_t*,size_t,size_t,unsigned int)>("hipMemAllocPitch");
    hipError_t out = hip_func(hip_args.hipMemAllocPitch.dptr, hip_args.hipMemAllocPitch.pitch, hip_args.hipMemAllocPitch.widthInBytes, hip_args.hipMemAllocPitch.height, hip_args.hipMemAllocPitch.elementSizeBytes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dptr = hip_args.hipMemAllocPitch.dptr;
    pitch = hip_args.hipMemAllocPitch.pitch;
    widthInBytes = hip_args.hipMemAllocPitch.widthInBytes;
    height = hip_args.hipMemAllocPitch.height;
    elementSizeBytes = hip_args.hipMemAllocPitch.elementSizeBytes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size, const hipMemAllocationProp* prop, unsigned long long flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemCreate.handle = handle;
    hip_args.hipMemCreate.size = size;
    hip_args.hipMemCreate.prop = prop;
    hip_args.hipMemCreate.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t*,size_t,const hipMemAllocationProp*,unsigned long long)>("hipMemCreate");
    hipError_t out = hip_func(hip_args.hipMemCreate.handle, hip_args.hipMemCreate.size, hip_args.hipMemCreate.prop, hip_args.hipMemCreate.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    handle = hip_args.hipMemCreate.handle;
    size = hip_args.hipMemCreate.size;
    prop = hip_args.hipMemCreate.prop;
    flags = hip_args.hipMemCreate.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemExportToShareableHandle(void* shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemExportToShareableHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemExportToShareableHandle.shareableHandle = shareableHandle;
    hip_args.hipMemExportToShareableHandle.handle = handle;
    hip_args.hipMemExportToShareableHandle.handleType = handleType;
    hip_args.hipMemExportToShareableHandle.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,hipMemGenericAllocationHandle_t,hipMemAllocationHandleType,unsigned long long)>("hipMemExportToShareableHandle");
    hipError_t out = hip_func(hip_args.hipMemExportToShareableHandle.shareableHandle, hip_args.hipMemExportToShareableHandle.handle, hip_args.hipMemExportToShareableHandle.handleType, hip_args.hipMemExportToShareableHandle.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    shareableHandle = hip_args.hipMemExportToShareableHandle.shareableHandle;
    handle = hip_args.hipMemExportToShareableHandle.handle;
    handleType = hip_args.hipMemExportToShareableHandle.handleType;
    flags = hip_args.hipMemExportToShareableHandle.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemGetAccess;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemGetAccess.flags = flags;
    hip_args.hipMemGetAccess.location = location;
    hip_args.hipMemGetAccess.ptr = ptr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned long long*,const hipMemLocation*,void*)>("hipMemGetAccess");
    hipError_t out = hip_func(hip_args.hipMemGetAccess.flags, hip_args.hipMemGetAccess.location, hip_args.hipMemGetAccess.ptr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    flags = hip_args.hipMemGetAccess.flags;
    location = hip_args.hipMemGetAccess.location;
    ptr = hip_args.hipMemGetAccess.ptr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemGetAddressRange;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemGetAddressRange.pbase = pbase;
    hip_args.hipMemGetAddressRange.psize = psize;
    hip_args.hipMemGetAddressRange.dptr = dptr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t*,size_t*,hipDeviceptr_t)>("hipMemGetAddressRange");
    hipError_t out = hip_func(hip_args.hipMemGetAddressRange.pbase, hip_args.hipMemGetAddressRange.psize, hip_args.hipMemGetAddressRange.dptr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pbase = hip_args.hipMemGetAddressRange.pbase;
    psize = hip_args.hipMemGetAddressRange.psize;
    dptr = hip_args.hipMemGetAddressRange.dptr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop, hipMemAllocationGranularity_flags option) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemGetAllocationGranularity;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemGetAllocationGranularity.granularity = granularity;
    hip_args.hipMemGetAllocationGranularity.prop = prop;
    hip_args.hipMemGetAllocationGranularity.option = option;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t*,const hipMemAllocationProp*,hipMemAllocationGranularity_flags)>("hipMemGetAllocationGranularity");
    hipError_t out = hip_func(hip_args.hipMemGetAllocationGranularity.granularity, hip_args.hipMemGetAllocationGranularity.prop, hip_args.hipMemGetAllocationGranularity.option);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    granularity = hip_args.hipMemGetAllocationGranularity.granularity;
    prop = hip_args.hipMemGetAllocationGranularity.prop;
    option = hip_args.hipMemGetAllocationGranularity.option;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemGetAllocationPropertiesFromHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemGetAllocationPropertiesFromHandle.prop = prop;
    hip_args.hipMemGetAllocationPropertiesFromHandle.handle = handle;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemAllocationProp*,hipMemGenericAllocationHandle_t)>("hipMemGetAllocationPropertiesFromHandle");
    hipError_t out = hip_func(hip_args.hipMemGetAllocationPropertiesFromHandle.prop, hip_args.hipMemGetAllocationPropertiesFromHandle.handle);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    prop = hip_args.hipMemGetAllocationPropertiesFromHandle.prop;
    handle = hip_args.hipMemGetAllocationPropertiesFromHandle.handle;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemGetInfo(size_t* free, size_t* total) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemGetInfo;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemGetInfo.free = free;
    hip_args.hipMemGetInfo.total = total;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t*,size_t*)>("hipMemGetInfo");
    hipError_t out = hip_func(hip_args.hipMemGetInfo.free, hip_args.hipMemGetInfo.total);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    free = hip_args.hipMemGetInfo.free;
    total = hip_args.hipMemGetInfo.total;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle, hipMemAllocationHandleType shHandleType) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemImportFromShareableHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemImportFromShareableHandle.handle = handle;
    hip_args.hipMemImportFromShareableHandle.osHandle = osHandle;
    hip_args.hipMemImportFromShareableHandle.shHandleType = shHandleType;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t*,void*,hipMemAllocationHandleType)>("hipMemImportFromShareableHandle");
    hipError_t out = hip_func(hip_args.hipMemImportFromShareableHandle.handle, hip_args.hipMemImportFromShareableHandle.osHandle, hip_args.hipMemImportFromShareableHandle.shHandleType);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    handle = hip_args.hipMemImportFromShareableHandle.handle;
    osHandle = hip_args.hipMemImportFromShareableHandle.osHandle;
    shHandleType = hip_args.hipMemImportFromShareableHandle.shHandleType;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemMap;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemMap.ptr = ptr;
    hip_args.hipMemMap.size = size;
    hip_args.hipMemMap.offset = offset;
    hip_args.hipMemMap.handle = handle;
    hip_args.hipMemMap.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,size_t,hipMemGenericAllocationHandle_t,unsigned long long)>("hipMemMap");
    hipError_t out = hip_func(hip_args.hipMemMap.ptr, hip_args.hipMemMap.size, hip_args.hipMemMap.offset, hip_args.hipMemMap.handle, hip_args.hipMemMap.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipMemMap.ptr;
    size = hip_args.hipMemMap.size;
    offset = hip_args.hipMemMap.offset;
    handle = hip_args.hipMemMap.handle;
    flags = hip_args.hipMemMap.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int count, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemMapArrayAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemMapArrayAsync.mapInfoList = mapInfoList;
    hip_args.hipMemMapArrayAsync.count = count;
    hip_args.hipMemMapArrayAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArrayMapInfo*,unsigned int,hipStream_t)>("hipMemMapArrayAsync");
    hipError_t out = hip_func(hip_args.hipMemMapArrayAsync.mapInfoList, hip_args.hipMemMapArrayAsync.count, hip_args.hipMemMapArrayAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mapInfoList = hip_args.hipMemMapArrayAsync.mapInfoList;
    count = hip_args.hipMemMapArrayAsync.count;
    stream = hip_args.hipMemMapArrayAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolCreate.mem_pool = mem_pool;
    hip_args.hipMemPoolCreate.pool_props = pool_props;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t*,const hipMemPoolProps*)>("hipMemPoolCreate");
    hipError_t out = hip_func(hip_args.hipMemPoolCreate.mem_pool, hip_args.hipMemPoolCreate.pool_props);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipMemPoolCreate.mem_pool;
    pool_props = hip_args.hipMemPoolCreate.pool_props;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolDestroy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolDestroy.mem_pool = mem_pool;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t)>("hipMemPoolDestroy");
    hipError_t out = hip_func(hip_args.hipMemPoolDestroy.mem_pool);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipMemPoolDestroy.mem_pool;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolExportPointer;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolExportPointer.export_data = export_data;
    hip_args.hipMemPoolExportPointer.dev_ptr = dev_ptr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPoolPtrExportData*,void*)>("hipMemPoolExportPointer");
    hipError_t out = hip_func(hip_args.hipMemPoolExportPointer.export_data, hip_args.hipMemPoolExportPointer.dev_ptr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    export_data = hip_args.hipMemPoolExportPointer.export_data;
    dev_ptr = hip_args.hipMemPoolExportPointer.dev_ptr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool, hipMemAllocationHandleType handle_type, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolExportToShareableHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolExportToShareableHandle.shared_handle = shared_handle;
    hip_args.hipMemPoolExportToShareableHandle.mem_pool = mem_pool;
    hip_args.hipMemPoolExportToShareableHandle.handle_type = handle_type;
    hip_args.hipMemPoolExportToShareableHandle.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,hipMemPool_t,hipMemAllocationHandleType,unsigned int)>("hipMemPoolExportToShareableHandle");
    hipError_t out = hip_func(hip_args.hipMemPoolExportToShareableHandle.shared_handle, hip_args.hipMemPoolExportToShareableHandle.mem_pool, hip_args.hipMemPoolExportToShareableHandle.handle_type, hip_args.hipMemPoolExportToShareableHandle.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    shared_handle = hip_args.hipMemPoolExportToShareableHandle.shared_handle;
    mem_pool = hip_args.hipMemPoolExportToShareableHandle.mem_pool;
    handle_type = hip_args.hipMemPoolExportToShareableHandle.handle_type;
    flags = hip_args.hipMemPoolExportToShareableHandle.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool, hipMemLocation* location) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolGetAccess;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolGetAccess.flags = flags;
    hip_args.hipMemPoolGetAccess.mem_pool = mem_pool;
    hip_args.hipMemPoolGetAccess.location = location;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemAccessFlags*,hipMemPool_t,hipMemLocation*)>("hipMemPoolGetAccess");
    hipError_t out = hip_func(hip_args.hipMemPoolGetAccess.flags, hip_args.hipMemPoolGetAccess.mem_pool, hip_args.hipMemPoolGetAccess.location);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    flags = hip_args.hipMemPoolGetAccess.flags;
    mem_pool = hip_args.hipMemPoolGetAccess.mem_pool;
    location = hip_args.hipMemPoolGetAccess.location;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolGetAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolGetAttribute.mem_pool = mem_pool;
    hip_args.hipMemPoolGetAttribute.attr = attr;
    hip_args.hipMemPoolGetAttribute.value = value;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t,hipMemPoolAttr,void*)>("hipMemPoolGetAttribute");
    hipError_t out = hip_func(hip_args.hipMemPoolGetAttribute.mem_pool, hip_args.hipMemPoolGetAttribute.attr, hip_args.hipMemPoolGetAttribute.value);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipMemPoolGetAttribute.mem_pool;
    attr = hip_args.hipMemPoolGetAttribute.attr;
    value = hip_args.hipMemPoolGetAttribute.value;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle, hipMemAllocationHandleType handle_type, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolImportFromShareableHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolImportFromShareableHandle.mem_pool = mem_pool;
    hip_args.hipMemPoolImportFromShareableHandle.shared_handle = shared_handle;
    hip_args.hipMemPoolImportFromShareableHandle.handle_type = handle_type;
    hip_args.hipMemPoolImportFromShareableHandle.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t*,void*,hipMemAllocationHandleType,unsigned int)>("hipMemPoolImportFromShareableHandle");
    hipError_t out = hip_func(hip_args.hipMemPoolImportFromShareableHandle.mem_pool, hip_args.hipMemPoolImportFromShareableHandle.shared_handle, hip_args.hipMemPoolImportFromShareableHandle.handle_type, hip_args.hipMemPoolImportFromShareableHandle.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipMemPoolImportFromShareableHandle.mem_pool;
    shared_handle = hip_args.hipMemPoolImportFromShareableHandle.shared_handle;
    handle_type = hip_args.hipMemPoolImportFromShareableHandle.handle_type;
    flags = hip_args.hipMemPoolImportFromShareableHandle.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t mem_pool, hipMemPoolPtrExportData* export_data) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolImportPointer;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolImportPointer.dev_ptr = dev_ptr;
    hip_args.hipMemPoolImportPointer.mem_pool = mem_pool;
    hip_args.hipMemPoolImportPointer.export_data = export_data;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,hipMemPool_t,hipMemPoolPtrExportData*)>("hipMemPoolImportPointer");
    hipError_t out = hip_func(hip_args.hipMemPoolImportPointer.dev_ptr, hip_args.hipMemPoolImportPointer.mem_pool, hip_args.hipMemPoolImportPointer.export_data);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev_ptr = hip_args.hipMemPoolImportPointer.dev_ptr;
    mem_pool = hip_args.hipMemPoolImportPointer.mem_pool;
    export_data = hip_args.hipMemPoolImportPointer.export_data;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolSetAccess;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolSetAccess.mem_pool = mem_pool;
    hip_args.hipMemPoolSetAccess.desc_list = desc_list;
    hip_args.hipMemPoolSetAccess.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t,const hipMemAccessDesc*,size_t)>("hipMemPoolSetAccess");
    hipError_t out = hip_func(hip_args.hipMemPoolSetAccess.mem_pool, hip_args.hipMemPoolSetAccess.desc_list, hip_args.hipMemPoolSetAccess.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipMemPoolSetAccess.mem_pool;
    desc_list = hip_args.hipMemPoolSetAccess.desc_list;
    count = hip_args.hipMemPoolSetAccess.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolSetAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolSetAttribute.mem_pool = mem_pool;
    hip_args.hipMemPoolSetAttribute.attr = attr;
    hip_args.hipMemPoolSetAttribute.value = value;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t,hipMemPoolAttr,void*)>("hipMemPoolSetAttribute");
    hipError_t out = hip_func(hip_args.hipMemPoolSetAttribute.mem_pool, hip_args.hipMemPoolSetAttribute.attr, hip_args.hipMemPoolSetAttribute.value);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipMemPoolSetAttribute.mem_pool;
    attr = hip_args.hipMemPoolSetAttribute.attr;
    value = hip_args.hipMemPoolSetAttribute.value;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPoolTrimTo;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPoolTrimTo.mem_pool = mem_pool;
    hip_args.hipMemPoolTrimTo.min_bytes_to_hold = min_bytes_to_hold;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t,size_t)>("hipMemPoolTrimTo");
    hipError_t out = hip_func(hip_args.hipMemPoolTrimTo.mem_pool, hip_args.hipMemPoolTrimTo.min_bytes_to_hold);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mem_pool = hip_args.hipMemPoolTrimTo.mem_pool;
    min_bytes_to_hold = hip_args.hipMemPoolTrimTo.min_bytes_to_hold;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPrefetchAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPrefetchAsync.dev_ptr = dev_ptr;
    hip_args.hipMemPrefetchAsync.count = count;
    hip_args.hipMemPrefetchAsync.device = device;
    hip_args.hipMemPrefetchAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,size_t,int,hipStream_t)>("hipMemPrefetchAsync");
    hipError_t out = hip_func(hip_args.hipMemPrefetchAsync.dev_ptr, hip_args.hipMemPrefetchAsync.count, hip_args.hipMemPrefetchAsync.device, hip_args.hipMemPrefetchAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev_ptr = hip_args.hipMemPrefetchAsync.dev_ptr;
    count = hip_args.hipMemPrefetchAsync.count;
    device = hip_args.hipMemPrefetchAsync.device;
    stream = hip_args.hipMemPrefetchAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemPtrGetInfo;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemPtrGetInfo.ptr = ptr;
    hip_args.hipMemPtrGetInfo.size = size;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t*)>("hipMemPtrGetInfo");
    hipError_t out = hip_func(hip_args.hipMemPtrGetInfo.ptr, hip_args.hipMemPtrGetInfo.size);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipMemPtrGetInfo.ptr;
    size = hip_args.hipMemPtrGetInfo.size;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute, const void* dev_ptr, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemRangeGetAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemRangeGetAttribute.data = data;
    hip_args.hipMemRangeGetAttribute.data_size = data_size;
    hip_args.hipMemRangeGetAttribute.attribute = attribute;
    hip_args.hipMemRangeGetAttribute.dev_ptr = dev_ptr;
    hip_args.hipMemRangeGetAttribute.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,hipMemRangeAttribute,const void*,size_t)>("hipMemRangeGetAttribute");
    hipError_t out = hip_func(hip_args.hipMemRangeGetAttribute.data, hip_args.hipMemRangeGetAttribute.data_size, hip_args.hipMemRangeGetAttribute.attribute, hip_args.hipMemRangeGetAttribute.dev_ptr, hip_args.hipMemRangeGetAttribute.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    data = hip_args.hipMemRangeGetAttribute.data;
    data_size = hip_args.hipMemRangeGetAttribute.data_size;
    attribute = hip_args.hipMemRangeGetAttribute.attribute;
    dev_ptr = hip_args.hipMemRangeGetAttribute.dev_ptr;
    count = hip_args.hipMemRangeGetAttribute.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes, hipMemRangeAttribute* attributes, size_t num_attributes, const void* dev_ptr, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemRangeGetAttributes;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemRangeGetAttributes.data = data;
    hip_args.hipMemRangeGetAttributes.data_sizes = data_sizes;
    hip_args.hipMemRangeGetAttributes.attributes = attributes;
    hip_args.hipMemRangeGetAttributes.num_attributes = num_attributes;
    hip_args.hipMemRangeGetAttributes.dev_ptr = dev_ptr;
    hip_args.hipMemRangeGetAttributes.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void**,size_t*,hipMemRangeAttribute*,size_t,const void*,size_t)>("hipMemRangeGetAttributes");
    hipError_t out = hip_func(hip_args.hipMemRangeGetAttributes.data, hip_args.hipMemRangeGetAttributes.data_sizes, hip_args.hipMemRangeGetAttributes.attributes, hip_args.hipMemRangeGetAttributes.num_attributes, hip_args.hipMemRangeGetAttributes.dev_ptr, hip_args.hipMemRangeGetAttributes.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    data = hip_args.hipMemRangeGetAttributes.data;
    data_sizes = hip_args.hipMemRangeGetAttributes.data_sizes;
    attributes = hip_args.hipMemRangeGetAttributes.attributes;
    num_attributes = hip_args.hipMemRangeGetAttributes.num_attributes;
    dev_ptr = hip_args.hipMemRangeGetAttributes.dev_ptr;
    count = hip_args.hipMemRangeGetAttributes.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemRelease;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemRelease.handle = handle;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t)>("hipMemRelease");
    hipError_t out = hip_func(hip_args.hipMemRelease.handle);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    handle = hip_args.hipMemRelease.handle;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemRetainAllocationHandle;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemRetainAllocationHandle.handle = handle;
    hip_args.hipMemRetainAllocationHandle.addr = addr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t*,void*)>("hipMemRetainAllocationHandle");
    hipError_t out = hip_func(hip_args.hipMemRetainAllocationHandle.handle, hip_args.hipMemRetainAllocationHandle.addr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    handle = hip_args.hipMemRetainAllocationHandle.handle;
    addr = hip_args.hipMemRetainAllocationHandle.addr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemSetAccess;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemSetAccess.ptr = ptr;
    hip_args.hipMemSetAccess.size = size;
    hip_args.hipMemSetAccess.desc = desc;
    hip_args.hipMemSetAccess.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,const hipMemAccessDesc*,size_t)>("hipMemSetAccess");
    hipError_t out = hip_func(hip_args.hipMemSetAccess.ptr, hip_args.hipMemSetAccess.size, hip_args.hipMemSetAccess.desc, hip_args.hipMemSetAccess.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipMemSetAccess.ptr;
    size = hip_args.hipMemSetAccess.size;
    desc = hip_args.hipMemSetAccess.desc;
    count = hip_args.hipMemSetAccess.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemUnmap(void* ptr, size_t size) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemUnmap;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemUnmap.ptr = ptr;
    hip_args.hipMemUnmap.size = size;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t)>("hipMemUnmap");
    hipError_t out = hip_func(hip_args.hipMemUnmap.ptr, hip_args.hipMemUnmap.size);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ptr = hip_args.hipMemUnmap.ptr;
    size = hip_args.hipMemUnmap.size;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpy.dst = dst;
    hip_args.hipMemcpy.src = src;
    hip_args.hipMemcpy.sizeBytes = sizeBytes;
    hip_args.hipMemcpy.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,const void*,size_t,hipMemcpyKind)>("hipMemcpy");
    hipError_t out = hip_func(hip_args.hipMemcpy.dst, hip_args.hipMemcpy.src, hip_args.hipMemcpy.sizeBytes, hip_args.hipMemcpy.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpy.dst;
    src = hip_args.hipMemcpy.src;
    sizeBytes = hip_args.hipMemcpy.sizeBytes;
    kind = hip_args.hipMemcpy.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpy2D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpy2D.dst = dst;
    hip_args.hipMemcpy2D.dpitch = dpitch;
    hip_args.hipMemcpy2D.src = src;
    hip_args.hipMemcpy2D.spitch = spitch;
    hip_args.hipMemcpy2D.width = width;
    hip_args.hipMemcpy2D.height = height;
    hip_args.hipMemcpy2D.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,const void*,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2D");
    hipError_t out = hip_func(hip_args.hipMemcpy2D.dst, hip_args.hipMemcpy2D.dpitch, hip_args.hipMemcpy2D.src, hip_args.hipMemcpy2D.spitch, hip_args.hipMemcpy2D.width, hip_args.hipMemcpy2D.height, hip_args.hipMemcpy2D.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpy2D.dst;
    dpitch = hip_args.hipMemcpy2D.dpitch;
    src = hip_args.hipMemcpy2D.src;
    spitch = hip_args.hipMemcpy2D.spitch;
    width = hip_args.hipMemcpy2D.width;
    height = hip_args.hipMemcpy2D.height;
    kind = hip_args.hipMemcpy2D.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpy2DAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpy2DAsync.dst = dst;
    hip_args.hipMemcpy2DAsync.dpitch = dpitch;
    hip_args.hipMemcpy2DAsync.src = src;
    hip_args.hipMemcpy2DAsync.spitch = spitch;
    hip_args.hipMemcpy2DAsync.width = width;
    hip_args.hipMemcpy2DAsync.height = height;
    hip_args.hipMemcpy2DAsync.kind = kind;
    hip_args.hipMemcpy2DAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,const void*,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DAsync");
    hipError_t out = hip_func(hip_args.hipMemcpy2DAsync.dst, hip_args.hipMemcpy2DAsync.dpitch, hip_args.hipMemcpy2DAsync.src, hip_args.hipMemcpy2DAsync.spitch, hip_args.hipMemcpy2DAsync.width, hip_args.hipMemcpy2DAsync.height, hip_args.hipMemcpy2DAsync.kind, hip_args.hipMemcpy2DAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpy2DAsync.dst;
    dpitch = hip_args.hipMemcpy2DAsync.dpitch;
    src = hip_args.hipMemcpy2DAsync.src;
    spitch = hip_args.hipMemcpy2DAsync.spitch;
    width = hip_args.hipMemcpy2DAsync.width;
    height = hip_args.hipMemcpy2DAsync.height;
    kind = hip_args.hipMemcpy2DAsync.kind;
    stream = hip_args.hipMemcpy2DAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpy2DFromArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpy2DFromArray.dst = dst;
    hip_args.hipMemcpy2DFromArray.dpitch = dpitch;
    hip_args.hipMemcpy2DFromArray.src = src;
    hip_args.hipMemcpy2DFromArray.wOffset = wOffset;
    hip_args.hipMemcpy2DFromArray.hOffset = hOffset;
    hip_args.hipMemcpy2DFromArray.width = width;
    hip_args.hipMemcpy2DFromArray.height = height;
    hip_args.hipMemcpy2DFromArray.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,hipArray_const_t,size_t,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2DFromArray");
    hipError_t out = hip_func(hip_args.hipMemcpy2DFromArray.dst, hip_args.hipMemcpy2DFromArray.dpitch, hip_args.hipMemcpy2DFromArray.src, hip_args.hipMemcpy2DFromArray.wOffset, hip_args.hipMemcpy2DFromArray.hOffset, hip_args.hipMemcpy2DFromArray.width, hip_args.hipMemcpy2DFromArray.height, hip_args.hipMemcpy2DFromArray.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpy2DFromArray.dst;
    dpitch = hip_args.hipMemcpy2DFromArray.dpitch;
    src = hip_args.hipMemcpy2DFromArray.src;
    wOffset = hip_args.hipMemcpy2DFromArray.wOffset;
    hOffset = hip_args.hipMemcpy2DFromArray.hOffset;
    width = hip_args.hipMemcpy2DFromArray.width;
    height = hip_args.hipMemcpy2DFromArray.height;
    kind = hip_args.hipMemcpy2DFromArray.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpy2DFromArrayAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpy2DFromArrayAsync.dst = dst;
    hip_args.hipMemcpy2DFromArrayAsync.dpitch = dpitch;
    hip_args.hipMemcpy2DFromArrayAsync.src = src;
    hip_args.hipMemcpy2DFromArrayAsync.wOffset = wOffset;
    hip_args.hipMemcpy2DFromArrayAsync.hOffset = hOffset;
    hip_args.hipMemcpy2DFromArrayAsync.width = width;
    hip_args.hipMemcpy2DFromArrayAsync.height = height;
    hip_args.hipMemcpy2DFromArrayAsync.kind = kind;
    hip_args.hipMemcpy2DFromArrayAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,hipArray_const_t,size_t,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DFromArrayAsync");
    hipError_t out = hip_func(hip_args.hipMemcpy2DFromArrayAsync.dst, hip_args.hipMemcpy2DFromArrayAsync.dpitch, hip_args.hipMemcpy2DFromArrayAsync.src, hip_args.hipMemcpy2DFromArrayAsync.wOffset, hip_args.hipMemcpy2DFromArrayAsync.hOffset, hip_args.hipMemcpy2DFromArrayAsync.width, hip_args.hipMemcpy2DFromArrayAsync.height, hip_args.hipMemcpy2DFromArrayAsync.kind, hip_args.hipMemcpy2DFromArrayAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpy2DFromArrayAsync.dst;
    dpitch = hip_args.hipMemcpy2DFromArrayAsync.dpitch;
    src = hip_args.hipMemcpy2DFromArrayAsync.src;
    wOffset = hip_args.hipMemcpy2DFromArrayAsync.wOffset;
    hOffset = hip_args.hipMemcpy2DFromArrayAsync.hOffset;
    width = hip_args.hipMemcpy2DFromArrayAsync.width;
    height = hip_args.hipMemcpy2DFromArrayAsync.height;
    kind = hip_args.hipMemcpy2DFromArrayAsync.kind;
    stream = hip_args.hipMemcpy2DFromArrayAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpy2DToArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpy2DToArray.dst = dst;
    hip_args.hipMemcpy2DToArray.wOffset = wOffset;
    hip_args.hipMemcpy2DToArray.hOffset = hOffset;
    hip_args.hipMemcpy2DToArray.src = src;
    hip_args.hipMemcpy2DToArray.spitch = spitch;
    hip_args.hipMemcpy2DToArray.width = width;
    hip_args.hipMemcpy2DToArray.height = height;
    hip_args.hipMemcpy2DToArray.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray*,size_t,size_t,const void*,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2DToArray");
    hipError_t out = hip_func(hip_args.hipMemcpy2DToArray.dst, hip_args.hipMemcpy2DToArray.wOffset, hip_args.hipMemcpy2DToArray.hOffset, hip_args.hipMemcpy2DToArray.src, hip_args.hipMemcpy2DToArray.spitch, hip_args.hipMemcpy2DToArray.width, hip_args.hipMemcpy2DToArray.height, hip_args.hipMemcpy2DToArray.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpy2DToArray.dst;
    wOffset = hip_args.hipMemcpy2DToArray.wOffset;
    hOffset = hip_args.hipMemcpy2DToArray.hOffset;
    src = hip_args.hipMemcpy2DToArray.src;
    spitch = hip_args.hipMemcpy2DToArray.spitch;
    width = hip_args.hipMemcpy2DToArray.width;
    height = hip_args.hipMemcpy2DToArray.height;
    kind = hip_args.hipMemcpy2DToArray.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpy2DToArrayAsync(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpy2DToArrayAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpy2DToArrayAsync.dst = dst;
    hip_args.hipMemcpy2DToArrayAsync.wOffset = wOffset;
    hip_args.hipMemcpy2DToArrayAsync.hOffset = hOffset;
    hip_args.hipMemcpy2DToArrayAsync.src = src;
    hip_args.hipMemcpy2DToArrayAsync.spitch = spitch;
    hip_args.hipMemcpy2DToArrayAsync.width = width;
    hip_args.hipMemcpy2DToArrayAsync.height = height;
    hip_args.hipMemcpy2DToArrayAsync.kind = kind;
    hip_args.hipMemcpy2DToArrayAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray*,size_t,size_t,const void*,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DToArrayAsync");
    hipError_t out = hip_func(hip_args.hipMemcpy2DToArrayAsync.dst, hip_args.hipMemcpy2DToArrayAsync.wOffset, hip_args.hipMemcpy2DToArrayAsync.hOffset, hip_args.hipMemcpy2DToArrayAsync.src, hip_args.hipMemcpy2DToArrayAsync.spitch, hip_args.hipMemcpy2DToArrayAsync.width, hip_args.hipMemcpy2DToArrayAsync.height, hip_args.hipMemcpy2DToArrayAsync.kind, hip_args.hipMemcpy2DToArrayAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpy2DToArrayAsync.dst;
    wOffset = hip_args.hipMemcpy2DToArrayAsync.wOffset;
    hOffset = hip_args.hipMemcpy2DToArrayAsync.hOffset;
    src = hip_args.hipMemcpy2DToArrayAsync.src;
    spitch = hip_args.hipMemcpy2DToArrayAsync.spitch;
    width = hip_args.hipMemcpy2DToArrayAsync.width;
    height = hip_args.hipMemcpy2DToArrayAsync.height;
    kind = hip_args.hipMemcpy2DToArrayAsync.kind;
    stream = hip_args.hipMemcpy2DToArrayAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpy3D(const hipMemcpy3DParms* p) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpy3D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpy3D.p = p;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hipMemcpy3DParms*)>("hipMemcpy3D");
    hipError_t out = hip_func(hip_args.hipMemcpy3D.p);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    p = hip_args.hipMemcpy3D.p;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyAsync.dst = dst;
    hip_args.hipMemcpyAsync.src = src;
    hip_args.hipMemcpyAsync.sizeBytes = sizeBytes;
    hip_args.hipMemcpyAsync.kind = kind;
    hip_args.hipMemcpyAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,const void*,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyAsync");
    hipError_t out = hip_func(hip_args.hipMemcpyAsync.dst, hip_args.hipMemcpyAsync.src, hip_args.hipMemcpyAsync.sizeBytes, hip_args.hipMemcpyAsync.kind, hip_args.hipMemcpyAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyAsync.dst;
    src = hip_args.hipMemcpyAsync.src;
    sizeBytes = hip_args.hipMemcpyAsync.sizeBytes;
    kind = hip_args.hipMemcpyAsync.kind;
    stream = hip_args.hipMemcpyAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyAtoH;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyAtoH.dst = dst;
    hip_args.hipMemcpyAtoH.srcArray = srcArray;
    hip_args.hipMemcpyAtoH.srcOffset = srcOffset;
    hip_args.hipMemcpyAtoH.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,hipArray*,size_t,size_t)>("hipMemcpyAtoH");
    hipError_t out = hip_func(hip_args.hipMemcpyAtoH.dst, hip_args.hipMemcpyAtoH.srcArray, hip_args.hipMemcpyAtoH.srcOffset, hip_args.hipMemcpyAtoH.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyAtoH.dst;
    srcArray = hip_args.hipMemcpyAtoH.srcArray;
    srcOffset = hip_args.hipMemcpyAtoH.srcOffset;
    count = hip_args.hipMemcpyAtoH.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyDtoD;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyDtoD.dst = dst;
    hip_args.hipMemcpyDtoD.src = src;
    hip_args.hipMemcpyDtoD.sizeBytes = sizeBytes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,hipDeviceptr_t,size_t)>("hipMemcpyDtoD");
    hipError_t out = hip_func(hip_args.hipMemcpyDtoD.dst, hip_args.hipMemcpyDtoD.src, hip_args.hipMemcpyDtoD.sizeBytes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyDtoD.dst;
    src = hip_args.hipMemcpyDtoD.src;
    sizeBytes = hip_args.hipMemcpyDtoD.sizeBytes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyDtoDAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyDtoDAsync.dst = dst;
    hip_args.hipMemcpyDtoDAsync.src = src;
    hip_args.hipMemcpyDtoDAsync.sizeBytes = sizeBytes;
    hip_args.hipMemcpyDtoDAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,hipDeviceptr_t,size_t,hipStream_t)>("hipMemcpyDtoDAsync");
    hipError_t out = hip_func(hip_args.hipMemcpyDtoDAsync.dst, hip_args.hipMemcpyDtoDAsync.src, hip_args.hipMemcpyDtoDAsync.sizeBytes, hip_args.hipMemcpyDtoDAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyDtoDAsync.dst;
    src = hip_args.hipMemcpyDtoDAsync.src;
    sizeBytes = hip_args.hipMemcpyDtoDAsync.sizeBytes;
    stream = hip_args.hipMemcpyDtoDAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyDtoH;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyDtoH.dst = dst;
    hip_args.hipMemcpyDtoH.src = src;
    hip_args.hipMemcpyDtoH.sizeBytes = sizeBytes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,hipDeviceptr_t,size_t)>("hipMemcpyDtoH");
    hipError_t out = hip_func(hip_args.hipMemcpyDtoH.dst, hip_args.hipMemcpyDtoH.src, hip_args.hipMemcpyDtoH.sizeBytes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyDtoH.dst;
    src = hip_args.hipMemcpyDtoH.src;
    sizeBytes = hip_args.hipMemcpyDtoH.sizeBytes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyDtoHAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyDtoHAsync.dst = dst;
    hip_args.hipMemcpyDtoHAsync.src = src;
    hip_args.hipMemcpyDtoHAsync.sizeBytes = sizeBytes;
    hip_args.hipMemcpyDtoHAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,hipDeviceptr_t,size_t,hipStream_t)>("hipMemcpyDtoHAsync");
    hipError_t out = hip_func(hip_args.hipMemcpyDtoHAsync.dst, hip_args.hipMemcpyDtoHAsync.src, hip_args.hipMemcpyDtoHAsync.sizeBytes, hip_args.hipMemcpyDtoHAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyDtoHAsync.dst;
    src = hip_args.hipMemcpyDtoHAsync.src;
    sizeBytes = hip_args.hipMemcpyDtoHAsync.sizeBytes;
    stream = hip_args.hipMemcpyDtoHAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset, size_t count, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyFromArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyFromArray.dst = dst;
    hip_args.hipMemcpyFromArray.srcArray = srcArray;
    hip_args.hipMemcpyFromArray.wOffset = wOffset;
    hip_args.hipMemcpyFromArray.hOffset = hOffset;
    hip_args.hipMemcpyFromArray.count = count;
    hip_args.hipMemcpyFromArray.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,hipArray_const_t,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpyFromArray");
    hipError_t out = hip_func(hip_args.hipMemcpyFromArray.dst, hip_args.hipMemcpyFromArray.srcArray, hip_args.hipMemcpyFromArray.wOffset, hip_args.hipMemcpyFromArray.hOffset, hip_args.hipMemcpyFromArray.count, hip_args.hipMemcpyFromArray.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyFromArray.dst;
    srcArray = hip_args.hipMemcpyFromArray.srcArray;
    wOffset = hip_args.hipMemcpyFromArray.wOffset;
    hOffset = hip_args.hipMemcpyFromArray.hOffset;
    count = hip_args.hipMemcpyFromArray.count;
    kind = hip_args.hipMemcpyFromArray.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyFromSymbol;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyFromSymbol.dst = dst;
    hip_args.hipMemcpyFromSymbol.symbol = symbol;
    hip_args.hipMemcpyFromSymbol.sizeBytes = sizeBytes;
    hip_args.hipMemcpyFromSymbol.offset = offset;
    hip_args.hipMemcpyFromSymbol.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,const void*,size_t,size_t,hipMemcpyKind)>("hipMemcpyFromSymbol");
    hipError_t out = hip_func(hip_args.hipMemcpyFromSymbol.dst, hip_args.hipMemcpyFromSymbol.symbol, hip_args.hipMemcpyFromSymbol.sizeBytes, hip_args.hipMemcpyFromSymbol.offset, hip_args.hipMemcpyFromSymbol.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyFromSymbol.dst;
    symbol = hip_args.hipMemcpyFromSymbol.symbol;
    sizeBytes = hip_args.hipMemcpyFromSymbol.sizeBytes;
    offset = hip_args.hipMemcpyFromSymbol.offset;
    kind = hip_args.hipMemcpyFromSymbol.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyFromSymbolAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyFromSymbolAsync.dst = dst;
    hip_args.hipMemcpyFromSymbolAsync.symbol = symbol;
    hip_args.hipMemcpyFromSymbolAsync.sizeBytes = sizeBytes;
    hip_args.hipMemcpyFromSymbolAsync.offset = offset;
    hip_args.hipMemcpyFromSymbolAsync.kind = kind;
    hip_args.hipMemcpyFromSymbolAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,const void*,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyFromSymbolAsync");
    hipError_t out = hip_func(hip_args.hipMemcpyFromSymbolAsync.dst, hip_args.hipMemcpyFromSymbolAsync.symbol, hip_args.hipMemcpyFromSymbolAsync.sizeBytes, hip_args.hipMemcpyFromSymbolAsync.offset, hip_args.hipMemcpyFromSymbolAsync.kind, hip_args.hipMemcpyFromSymbolAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyFromSymbolAsync.dst;
    symbol = hip_args.hipMemcpyFromSymbolAsync.symbol;
    sizeBytes = hip_args.hipMemcpyFromSymbolAsync.sizeBytes;
    offset = hip_args.hipMemcpyFromSymbolAsync.offset;
    kind = hip_args.hipMemcpyFromSymbolAsync.kind;
    stream = hip_args.hipMemcpyFromSymbolAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyHtoA;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyHtoA.dstArray = dstArray;
    hip_args.hipMemcpyHtoA.dstOffset = dstOffset;
    hip_args.hipMemcpyHtoA.srcHost = srcHost;
    hip_args.hipMemcpyHtoA.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray*,size_t,const void*,size_t)>("hipMemcpyHtoA");
    hipError_t out = hip_func(hip_args.hipMemcpyHtoA.dstArray, hip_args.hipMemcpyHtoA.dstOffset, hip_args.hipMemcpyHtoA.srcHost, hip_args.hipMemcpyHtoA.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dstArray = hip_args.hipMemcpyHtoA.dstArray;
    dstOffset = hip_args.hipMemcpyHtoA.dstOffset;
    srcHost = hip_args.hipMemcpyHtoA.srcHost;
    count = hip_args.hipMemcpyHtoA.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyHtoD;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyHtoD.dst = dst;
    hip_args.hipMemcpyHtoD.src = src;
    hip_args.hipMemcpyHtoD.sizeBytes = sizeBytes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,void*,size_t)>("hipMemcpyHtoD");
    hipError_t out = hip_func(hip_args.hipMemcpyHtoD.dst, hip_args.hipMemcpyHtoD.src, hip_args.hipMemcpyHtoD.sizeBytes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyHtoD.dst;
    src = hip_args.hipMemcpyHtoD.src;
    sizeBytes = hip_args.hipMemcpyHtoD.sizeBytes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyHtoDAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyHtoDAsync.dst = dst;
    hip_args.hipMemcpyHtoDAsync.src = src;
    hip_args.hipMemcpyHtoDAsync.sizeBytes = sizeBytes;
    hip_args.hipMemcpyHtoDAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,void*,size_t,hipStream_t)>("hipMemcpyHtoDAsync");
    hipError_t out = hip_func(hip_args.hipMemcpyHtoDAsync.dst, hip_args.hipMemcpyHtoDAsync.src, hip_args.hipMemcpyHtoDAsync.sizeBytes, hip_args.hipMemcpyHtoDAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyHtoDAsync.dst;
    src = hip_args.hipMemcpyHtoDAsync.src;
    sizeBytes = hip_args.hipMemcpyHtoDAsync.sizeBytes;
    stream = hip_args.hipMemcpyHtoDAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyParam2D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyParam2D.pCopy = pCopy;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hip_Memcpy2D*)>("hipMemcpyParam2D");
    hipError_t out = hip_func(hip_args.hipMemcpyParam2D.pCopy);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pCopy = hip_args.hipMemcpyParam2D.pCopy;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t count, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyToArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyToArray.dst = dst;
    hip_args.hipMemcpyToArray.wOffset = wOffset;
    hip_args.hipMemcpyToArray.hOffset = hOffset;
    hip_args.hipMemcpyToArray.src = src;
    hip_args.hipMemcpyToArray.count = count;
    hip_args.hipMemcpyToArray.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray*,size_t,size_t,const void*,size_t,hipMemcpyKind)>("hipMemcpyToArray");
    hipError_t out = hip_func(hip_args.hipMemcpyToArray.dst, hip_args.hipMemcpyToArray.wOffset, hip_args.hipMemcpyToArray.hOffset, hip_args.hipMemcpyToArray.src, hip_args.hipMemcpyToArray.count, hip_args.hipMemcpyToArray.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyToArray.dst;
    wOffset = hip_args.hipMemcpyToArray.wOffset;
    hOffset = hip_args.hipMemcpyToArray.hOffset;
    src = hip_args.hipMemcpyToArray.src;
    count = hip_args.hipMemcpyToArray.count;
    kind = hip_args.hipMemcpyToArray.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset, hipMemcpyKind kind) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyToSymbol;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyToSymbol.symbol = symbol;
    hip_args.hipMemcpyToSymbol.src = src;
    hip_args.hipMemcpyToSymbol.sizeBytes = sizeBytes;
    hip_args.hipMemcpyToSymbol.offset = offset;
    hip_args.hipMemcpyToSymbol.kind = kind;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,const void*,size_t,size_t,hipMemcpyKind)>("hipMemcpyToSymbol");
    hipError_t out = hip_func(hip_args.hipMemcpyToSymbol.symbol, hip_args.hipMemcpyToSymbol.src, hip_args.hipMemcpyToSymbol.sizeBytes, hip_args.hipMemcpyToSymbol.offset, hip_args.hipMemcpyToSymbol.kind);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    symbol = hip_args.hipMemcpyToSymbol.symbol;
    src = hip_args.hipMemcpyToSymbol.src;
    sizeBytes = hip_args.hipMemcpyToSymbol.sizeBytes;
    offset = hip_args.hipMemcpyToSymbol.offset;
    kind = hip_args.hipMemcpyToSymbol.kind;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyToSymbolAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyToSymbolAsync.symbol = symbol;
    hip_args.hipMemcpyToSymbolAsync.src = src;
    hip_args.hipMemcpyToSymbolAsync.sizeBytes = sizeBytes;
    hip_args.hipMemcpyToSymbolAsync.offset = offset;
    hip_args.hipMemcpyToSymbolAsync.kind = kind;
    hip_args.hipMemcpyToSymbolAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void*,const void*,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyToSymbolAsync");
    hipError_t out = hip_func(hip_args.hipMemcpyToSymbolAsync.symbol, hip_args.hipMemcpyToSymbolAsync.src, hip_args.hipMemcpyToSymbolAsync.sizeBytes, hip_args.hipMemcpyToSymbolAsync.offset, hip_args.hipMemcpyToSymbolAsync.kind, hip_args.hipMemcpyToSymbolAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    symbol = hip_args.hipMemcpyToSymbolAsync.symbol;
    src = hip_args.hipMemcpyToSymbolAsync.src;
    sizeBytes = hip_args.hipMemcpyToSymbolAsync.sizeBytes;
    offset = hip_args.hipMemcpyToSymbolAsync.offset;
    kind = hip_args.hipMemcpyToSymbolAsync.kind;
    stream = hip_args.hipMemcpyToSymbolAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemcpyWithStream;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemcpyWithStream.dst = dst;
    hip_args.hipMemcpyWithStream.src = src;
    hip_args.hipMemcpyWithStream.sizeBytes = sizeBytes;
    hip_args.hipMemcpyWithStream.kind = kind;
    hip_args.hipMemcpyWithStream.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,const void*,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyWithStream");
    hipError_t out = hip_func(hip_args.hipMemcpyWithStream.dst, hip_args.hipMemcpyWithStream.src, hip_args.hipMemcpyWithStream.sizeBytes, hip_args.hipMemcpyWithStream.kind, hip_args.hipMemcpyWithStream.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemcpyWithStream.dst;
    src = hip_args.hipMemcpyWithStream.src;
    sizeBytes = hip_args.hipMemcpyWithStream.sizeBytes;
    kind = hip_args.hipMemcpyWithStream.kind;
    stream = hip_args.hipMemcpyWithStream.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemset;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemset.dst = dst;
    hip_args.hipMemset.value = value;
    hip_args.hipMemset.sizeBytes = sizeBytes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,int,size_t)>("hipMemset");
    hipError_t out = hip_func(hip_args.hipMemset.dst, hip_args.hipMemset.value, hip_args.hipMemset.sizeBytes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemset.dst;
    value = hip_args.hipMemset.value;
    sizeBytes = hip_args.hipMemset.sizeBytes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemset2D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemset2D.dst = dst;
    hip_args.hipMemset2D.pitch = pitch;
    hip_args.hipMemset2D.value = value;
    hip_args.hipMemset2D.width = width;
    hip_args.hipMemset2D.height = height;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,int,size_t,size_t)>("hipMemset2D");
    hipError_t out = hip_func(hip_args.hipMemset2D.dst, hip_args.hipMemset2D.pitch, hip_args.hipMemset2D.value, hip_args.hipMemset2D.width, hip_args.hipMemset2D.height);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemset2D.dst;
    pitch = hip_args.hipMemset2D.pitch;
    value = hip_args.hipMemset2D.value;
    width = hip_args.hipMemset2D.width;
    height = hip_args.hipMemset2D.height;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemset2DAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemset2DAsync.dst = dst;
    hip_args.hipMemset2DAsync.pitch = pitch;
    hip_args.hipMemset2DAsync.value = value;
    hip_args.hipMemset2DAsync.width = width;
    hip_args.hipMemset2DAsync.height = height;
    hip_args.hipMemset2DAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,size_t,int,size_t,size_t,hipStream_t)>("hipMemset2DAsync");
    hipError_t out = hip_func(hip_args.hipMemset2DAsync.dst, hip_args.hipMemset2DAsync.pitch, hip_args.hipMemset2DAsync.value, hip_args.hipMemset2DAsync.width, hip_args.hipMemset2DAsync.height, hip_args.hipMemset2DAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemset2DAsync.dst;
    pitch = hip_args.hipMemset2DAsync.pitch;
    value = hip_args.hipMemset2DAsync.value;
    width = hip_args.hipMemset2DAsync.width;
    height = hip_args.hipMemset2DAsync.height;
    stream = hip_args.hipMemset2DAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemset3D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemset3D.pitchedDevPtr = pitchedDevPtr;
    hip_args.hipMemset3D.value = value;
    hip_args.hipMemset3D.extent = extent;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipPitchedPtr,int,hipExtent)>("hipMemset3D");
    hipError_t out = hip_func(hip_args.hipMemset3D.pitchedDevPtr, hip_args.hipMemset3D.value, hip_args.hipMemset3D.extent);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pitchedDevPtr = hip_args.hipMemset3D.pitchedDevPtr;
    value = hip_args.hipMemset3D.value;
    extent = hip_args.hipMemset3D.extent;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemsetAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemsetAsync.dst = dst;
    hip_args.hipMemsetAsync.value = value;
    hip_args.hipMemsetAsync.sizeBytes = sizeBytes;
    hip_args.hipMemsetAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,int,size_t,hipStream_t)>("hipMemsetAsync");
    hipError_t out = hip_func(hip_args.hipMemsetAsync.dst, hip_args.hipMemsetAsync.value, hip_args.hipMemsetAsync.sizeBytes, hip_args.hipMemsetAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemsetAsync.dst;
    value = hip_args.hipMemsetAsync.value;
    sizeBytes = hip_args.hipMemsetAsync.sizeBytes;
    stream = hip_args.hipMemsetAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemsetD16;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemsetD16.dest = dest;
    hip_args.hipMemsetD16.value = value;
    hip_args.hipMemsetD16.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned short,size_t)>("hipMemsetD16");
    hipError_t out = hip_func(hip_args.hipMemsetD16.dest, hip_args.hipMemsetD16.value, hip_args.hipMemsetD16.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dest = hip_args.hipMemsetD16.dest;
    value = hip_args.hipMemsetD16.value;
    count = hip_args.hipMemsetD16.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemsetD16Async;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemsetD16Async.dest = dest;
    hip_args.hipMemsetD16Async.value = value;
    hip_args.hipMemsetD16Async.count = count;
    hip_args.hipMemsetD16Async.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned short,size_t,hipStream_t)>("hipMemsetD16Async");
    hipError_t out = hip_func(hip_args.hipMemsetD16Async.dest, hip_args.hipMemsetD16Async.value, hip_args.hipMemsetD16Async.count, hip_args.hipMemsetD16Async.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dest = hip_args.hipMemsetD16Async.dest;
    value = hip_args.hipMemsetD16Async.value;
    count = hip_args.hipMemsetD16Async.count;
    stream = hip_args.hipMemsetD16Async.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemsetD32;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemsetD32.dest = dest;
    hip_args.hipMemsetD32.value = value;
    hip_args.hipMemsetD32.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,int,size_t)>("hipMemsetD32");
    hipError_t out = hip_func(hip_args.hipMemsetD32.dest, hip_args.hipMemsetD32.value, hip_args.hipMemsetD32.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dest = hip_args.hipMemsetD32.dest;
    value = hip_args.hipMemsetD32.value;
    count = hip_args.hipMemsetD32.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemsetD32Async;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemsetD32Async.dst = dst;
    hip_args.hipMemsetD32Async.value = value;
    hip_args.hipMemsetD32Async.count = count;
    hip_args.hipMemsetD32Async.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,int,size_t,hipStream_t)>("hipMemsetD32Async");
    hipError_t out = hip_func(hip_args.hipMemsetD32Async.dst, hip_args.hipMemsetD32Async.value, hip_args.hipMemsetD32Async.count, hip_args.hipMemsetD32Async.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dst = hip_args.hipMemsetD32Async.dst;
    value = hip_args.hipMemsetD32Async.value;
    count = hip_args.hipMemsetD32Async.count;
    stream = hip_args.hipMemsetD32Async.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemsetD8;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemsetD8.dest = dest;
    hip_args.hipMemsetD8.value = value;
    hip_args.hipMemsetD8.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned char,size_t)>("hipMemsetD8");
    hipError_t out = hip_func(hip_args.hipMemsetD8.dest, hip_args.hipMemsetD8.value, hip_args.hipMemsetD8.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dest = hip_args.hipMemsetD8.dest;
    value = hip_args.hipMemsetD8.value;
    count = hip_args.hipMemsetD8.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMemsetD8Async;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMemsetD8Async.dest = dest;
    hip_args.hipMemsetD8Async.value = value;
    hip_args.hipMemsetD8Async.count = count;
    hip_args.hipMemsetD8Async.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned char,size_t,hipStream_t)>("hipMemsetD8Async");
    hipError_t out = hip_func(hip_args.hipMemsetD8Async.dest, hip_args.hipMemsetD8Async.value, hip_args.hipMemsetD8Async.count, hip_args.hipMemsetD8Async.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dest = hip_args.hipMemsetD8Async.dest;
    value = hip_args.hipMemsetD8Async.value;
    count = hip_args.hipMemsetD8Async.count;
    stream = hip_args.hipMemsetD8Async.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle, HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMipmappedArrayCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMipmappedArrayCreate.pHandle = pHandle;
    hip_args.hipMipmappedArrayCreate.pMipmappedArrayDesc = pMipmappedArrayDesc;
    hip_args.hipMipmappedArrayCreate.numMipmapLevels = numMipmapLevels;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t*,HIP_ARRAY3D_DESCRIPTOR*,unsigned int)>("hipMipmappedArrayCreate");
    hipError_t out = hip_func(hip_args.hipMipmappedArrayCreate.pHandle, hip_args.hipMipmappedArrayCreate.pMipmappedArrayDesc, hip_args.hipMipmappedArrayCreate.numMipmapLevels);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pHandle = hip_args.hipMipmappedArrayCreate.pHandle;
    pMipmappedArrayDesc = hip_args.hipMipmappedArrayCreate.pMipmappedArrayDesc;
    numMipmapLevels = hip_args.hipMipmappedArrayCreate.numMipmapLevels;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMipmappedArrayDestroy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMipmappedArrayDestroy.hMipmappedArray = hMipmappedArray;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t)>("hipMipmappedArrayDestroy");
    hipError_t out = hip_func(hip_args.hipMipmappedArrayDestroy.hMipmappedArray);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    hMipmappedArray = hip_args.hipMipmappedArrayDestroy.hMipmappedArray;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray, hipMipmappedArray_t hMipMappedArray, unsigned int level) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipMipmappedArrayGetLevel;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipMipmappedArrayGetLevel.pLevelArray = pLevelArray;
    hip_args.hipMipmappedArrayGetLevel.hMipMappedArray = hMipMappedArray;
    hip_args.hipMipmappedArrayGetLevel.level = level;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_t*,hipMipmappedArray_t,unsigned int)>("hipMipmappedArrayGetLevel");
    hipError_t out = hip_func(hip_args.hipMipmappedArrayGetLevel.pLevelArray, hip_args.hipMipmappedArrayGetLevel.hMipMappedArray, hip_args.hipMipmappedArrayGetLevel.level);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pLevelArray = hip_args.hipMipmappedArrayGetLevel.pLevelArray;
    hMipMappedArray = hip_args.hipMipmappedArrayGetLevel.hMipMappedArray;
    level = hip_args.hipMipmappedArrayGetLevel.level;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleGetFunction;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleGetFunction.function = function;
    hip_args.hipModuleGetFunction.module = module;
    hip_args.hipModuleGetFunction.kname = kname;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFunction_t*,hipModule_t,const char*)>("hipModuleGetFunction");
    hipError_t out = hip_func(hip_args.hipModuleGetFunction.function, hip_args.hipModuleGetFunction.module, hip_args.hipModuleGetFunction.kname);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    function = hip_args.hipModuleGetFunction.function;
    module = hip_args.hipModuleGetFunction.module;
    kname = hip_args.hipModuleGetFunction.kname;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod, const char* name) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleGetGlobal;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleGetGlobal.dptr = dptr;
    hip_args.hipModuleGetGlobal.bytes = bytes;
    hip_args.hipModuleGetGlobal.hmod = hmod;
    hip_args.hipModuleGetGlobal.name = name;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t*,size_t*,hipModule_t,const char*)>("hipModuleGetGlobal");
    hipError_t out = hip_func(hip_args.hipModuleGetGlobal.dptr, hip_args.hipModuleGetGlobal.bytes, hip_args.hipModuleGetGlobal.hmod, hip_args.hipModuleGetGlobal.name);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dptr = hip_args.hipModuleGetGlobal.dptr;
    bytes = hip_args.hipModuleGetGlobal.bytes;
    hmod = hip_args.hipModuleGetGlobal.hmod;
    name = hip_args.hipModuleGetGlobal.name;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleGetTexRef;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleGetTexRef.texRef = texRef;
    hip_args.hipModuleGetTexRef.hmod = hmod;
    hip_args.hipModuleGetTexRef.name = name;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference**,hipModule_t,const char*)>("hipModuleGetTexRef");
    hipError_t out = hip_func(hip_args.hipModuleGetTexRef.texRef, hip_args.hipModuleGetTexRef.hmod, hip_args.hipModuleGetTexRef.name);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipModuleGetTexRef.texRef;
    hmod = hip_args.hipModuleGetTexRef.hmod;
    name = hip_args.hipModuleGetTexRef.name;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleLaunchKernel;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleLaunchKernel.f = f;
    hip_args.hipModuleLaunchKernel.gridDimX = gridDimX;
    hip_args.hipModuleLaunchKernel.gridDimY = gridDimY;
    hip_args.hipModuleLaunchKernel.gridDimZ = gridDimZ;
    hip_args.hipModuleLaunchKernel.blockDimX = blockDimX;
    hip_args.hipModuleLaunchKernel.blockDimY = blockDimY;
    hip_args.hipModuleLaunchKernel.blockDimZ = blockDimZ;
    hip_args.hipModuleLaunchKernel.sharedMemBytes = sharedMemBytes;
    hip_args.hipModuleLaunchKernel.stream = stream;
    hip_args.hipModuleLaunchKernel.kernelParams = kernelParams;
    hip_args.hipModuleLaunchKernel.extra = extra;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFunction_t,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,hipStream_t,void**,void**)>("hipModuleLaunchKernel");
    hipError_t out = hip_func(hip_args.hipModuleLaunchKernel.f, hip_args.hipModuleLaunchKernel.gridDimX, hip_args.hipModuleLaunchKernel.gridDimY, hip_args.hipModuleLaunchKernel.gridDimZ, hip_args.hipModuleLaunchKernel.blockDimX, hip_args.hipModuleLaunchKernel.blockDimY, hip_args.hipModuleLaunchKernel.blockDimZ, hip_args.hipModuleLaunchKernel.sharedMemBytes, hip_args.hipModuleLaunchKernel.stream, hip_args.hipModuleLaunchKernel.kernelParams, hip_args.hipModuleLaunchKernel.extra);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    f = hip_args.hipModuleLaunchKernel.f;
    gridDimX = hip_args.hipModuleLaunchKernel.gridDimX;
    gridDimY = hip_args.hipModuleLaunchKernel.gridDimY;
    gridDimZ = hip_args.hipModuleLaunchKernel.gridDimZ;
    blockDimX = hip_args.hipModuleLaunchKernel.blockDimX;
    blockDimY = hip_args.hipModuleLaunchKernel.blockDimY;
    blockDimZ = hip_args.hipModuleLaunchKernel.blockDimZ;
    sharedMemBytes = hip_args.hipModuleLaunchKernel.sharedMemBytes;
    stream = hip_args.hipModuleLaunchKernel.stream;
    kernelParams = hip_args.hipModuleLaunchKernel.kernelParams;
    extra = hip_args.hipModuleLaunchKernel.extra;
    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleLoad;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleLoad.module = module;
    hip_args.hipModuleLoad.fname = fname;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipModule_t*,const char*)>("hipModuleLoad");
    hipError_t out = hip_func(hip_args.hipModuleLoad.module, hip_args.hipModuleLoad.fname);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    module = hip_args.hipModuleLoad.module;
    fname = hip_args.hipModuleLoad.fname;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleLoadData;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleLoadData.module = module;
    hip_args.hipModuleLoadData.image = image;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipModule_t*,const void*)>("hipModuleLoadData");
    hipError_t out = hip_func(hip_args.hipModuleLoadData.module, hip_args.hipModuleLoadData.image);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    module = hip_args.hipModuleLoadData.module;
    image = hip_args.hipModuleLoadData.image;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks = numBlocks;
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.f = f;
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.blockSize = blockSize;
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.dynSharedMemPerBlk = dynSharedMemPerBlk;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,hipFunction_t,int,size_t)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
    hipError_t out = hip_func(hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks, hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.f, hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.blockSize, hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.dynSharedMemPerBlk);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    numBlocks = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks;
    f = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.f;
    blockSize = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.blockSize;
    dynSharedMemPerBlk = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.dynSharedMemPerBlk;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks = numBlocks;
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.f = f;
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.blockSize = blockSize;
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.dynSharedMemPerBlk = dynSharedMemPerBlk;
    hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,hipFunction_t,int,size_t,unsigned int)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    hipError_t out = hip_func(hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks, hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.f, hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.blockSize, hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.dynSharedMemPerBlk, hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    numBlocks = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks;
    f = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.f;
    blockSize = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.blockSize;
    dynSharedMemPerBlk = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.dynSharedMemPerBlk;
    flags = hip_args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleOccupancyMaxPotentialBlockSize.gridSize = gridSize;
    hip_args.hipModuleOccupancyMaxPotentialBlockSize.blockSize = blockSize;
    hip_args.hipModuleOccupancyMaxPotentialBlockSize.f = f;
    hip_args.hipModuleOccupancyMaxPotentialBlockSize.dynSharedMemPerBlk = dynSharedMemPerBlk;
    hip_args.hipModuleOccupancyMaxPotentialBlockSize.blockSizeLimit = blockSizeLimit;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,int*,hipFunction_t,size_t,int)>("hipModuleOccupancyMaxPotentialBlockSize");
    hipError_t out = hip_func(hip_args.hipModuleOccupancyMaxPotentialBlockSize.gridSize, hip_args.hipModuleOccupancyMaxPotentialBlockSize.blockSize, hip_args.hipModuleOccupancyMaxPotentialBlockSize.f, hip_args.hipModuleOccupancyMaxPotentialBlockSize.dynSharedMemPerBlk, hip_args.hipModuleOccupancyMaxPotentialBlockSize.blockSizeLimit);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    gridSize = hip_args.hipModuleOccupancyMaxPotentialBlockSize.gridSize;
    blockSize = hip_args.hipModuleOccupancyMaxPotentialBlockSize.blockSize;
    f = hip_args.hipModuleOccupancyMaxPotentialBlockSize.f;
    dynSharedMemPerBlk = hip_args.hipModuleOccupancyMaxPotentialBlockSize.dynSharedMemPerBlk;
    blockSizeLimit = hip_args.hipModuleOccupancyMaxPotentialBlockSize.blockSizeLimit;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize = gridSize;
    hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize = blockSize;
    hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.f = f;
    hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.dynSharedMemPerBlk = dynSharedMemPerBlk;
    hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSizeLimit = blockSizeLimit;
    hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,int*,hipFunction_t,size_t,int,unsigned int)>("hipModuleOccupancyMaxPotentialBlockSizeWithFlags");
    hipError_t out = hip_func(hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize, hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize, hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.f, hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.dynSharedMemPerBlk, hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSizeLimit, hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    gridSize = hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize;
    blockSize = hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize;
    f = hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.f;
    dynSharedMemPerBlk = hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.dynSharedMemPerBlk;
    blockSizeLimit = hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSizeLimit;
    flags = hip_args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipModuleUnload(hipModule_t module) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipModuleUnload;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipModuleUnload.module = module;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipModule_t)>("hipModuleUnload");
    hipError_t out = hip_func(hip_args.hipModuleUnload.module);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    module = hip_args.hipModuleUnload.module;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipPeekAtLastError() {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipPeekAtLastError;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipPeekAtLastError");
    hipError_t out = hip_func();
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);

    return out;
}

__attribute__((visibility("default")))
hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipPointerGetAttribute;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipPointerGetAttribute.data = data;
    hip_args.hipPointerGetAttribute.attribute = attribute;
    hip_args.hipPointerGetAttribute.ptr = ptr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void*,hipPointer_attribute,hipDeviceptr_t)>("hipPointerGetAttribute");
    hipError_t out = hip_func(hip_args.hipPointerGetAttribute.data, hip_args.hipPointerGetAttribute.attribute, hip_args.hipPointerGetAttribute.ptr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    data = hip_args.hipPointerGetAttribute.data;
    attribute = hip_args.hipPointerGetAttribute.attribute;
    ptr = hip_args.hipPointerGetAttribute.ptr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipPointerGetAttributes;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipPointerGetAttributes.attributes = attributes;
    hip_args.hipPointerGetAttributes.ptr = ptr;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipPointerAttribute_t*,const void*)>("hipPointerGetAttributes");
    hipError_t out = hip_func(hip_args.hipPointerGetAttributes.attributes, hip_args.hipPointerGetAttributes.ptr);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    attributes = hip_args.hipPointerGetAttributes.attributes;
    ptr = hip_args.hipPointerGetAttributes.ptr;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipProfilerStart() {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipProfilerStart;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipProfilerStart");
    hipError_t out = hip_func();
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);

    return out;
}

__attribute__((visibility("default")))
hipError_t hipProfilerStop() {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipProfilerStop;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipProfilerStop");
    hipError_t out = hip_func();
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);

    return out;
}

__attribute__((visibility("default")))
hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipRuntimeGetVersion;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipRuntimeGetVersion.runtimeVersion = runtimeVersion;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*)>("hipRuntimeGetVersion");
    hipError_t out = hip_func(hip_args.hipRuntimeGetVersion.runtimeVersion);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    runtimeVersion = hip_args.hipRuntimeGetVersion.runtimeVersion;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipSetDevice(int deviceId) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipSetDevice;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipSetDevice.deviceId = deviceId;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int)>("hipSetDevice");
    hipError_t out = hip_func(hip_args.hipSetDevice.deviceId);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    deviceId = hip_args.hipSetDevice.deviceId;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipSetDeviceFlags(unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipSetDeviceFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipSetDeviceFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int)>("hipSetDeviceFlags");
    hipError_t out = hip_func(hip_args.hipSetDeviceFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    flags = hip_args.hipSetDeviceFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipSignalExternalSemaphoresAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipSignalExternalSemaphoresAsync.extSemArray = extSemArray;
    hip_args.hipSignalExternalSemaphoresAsync.paramsArray = paramsArray;
    hip_args.hipSignalExternalSemaphoresAsync.numExtSems = numExtSems;
    hip_args.hipSignalExternalSemaphoresAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hipExternalSemaphore_t*,const hipExternalSemaphoreSignalParams*,unsigned int,hipStream_t)>("hipSignalExternalSemaphoresAsync");
    hipError_t out = hip_func(hip_args.hipSignalExternalSemaphoresAsync.extSemArray, hip_args.hipSignalExternalSemaphoresAsync.paramsArray, hip_args.hipSignalExternalSemaphoresAsync.numExtSems, hip_args.hipSignalExternalSemaphoresAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    extSemArray = hip_args.hipSignalExternalSemaphoresAsync.extSemArray;
    paramsArray = hip_args.hipSignalExternalSemaphoresAsync.paramsArray;
    numExtSems = hip_args.hipSignalExternalSemaphoresAsync.numExtSems;
    stream = hip_args.hipSignalExternalSemaphoresAsync.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamAddCallback;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamAddCallback.stream = stream;
    hip_args.hipStreamAddCallback.callback = callback;
    hip_args.hipStreamAddCallback.userData = userData;
    hip_args.hipStreamAddCallback.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCallback_t,void*,unsigned int)>("hipStreamAddCallback");
    hipError_t out = hip_func(hip_args.hipStreamAddCallback.stream, hip_args.hipStreamAddCallback.callback, hip_args.hipStreamAddCallback.userData, hip_args.hipStreamAddCallback.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamAddCallback.stream;
    callback = hip_args.hipStreamAddCallback.callback;
    userData = hip_args.hipStreamAddCallback.userData;
    flags = hip_args.hipStreamAddCallback.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamAttachMemAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamAttachMemAsync.stream = stream;
    hip_args.hipStreamAttachMemAsync.dev_ptr = dev_ptr;
    hip_args.hipStreamAttachMemAsync.length = length;
    hip_args.hipStreamAttachMemAsync.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void*,size_t,unsigned int)>("hipStreamAttachMemAsync");
    hipError_t out = hip_func(hip_args.hipStreamAttachMemAsync.stream, hip_args.hipStreamAttachMemAsync.dev_ptr, hip_args.hipStreamAttachMemAsync.length, hip_args.hipStreamAttachMemAsync.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamAttachMemAsync.stream;
    dev_ptr = hip_args.hipStreamAttachMemAsync.dev_ptr;
    length = hip_args.hipStreamAttachMemAsync.length;
    flags = hip_args.hipStreamAttachMemAsync.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamBeginCapture;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamBeginCapture.stream = stream;
    hip_args.hipStreamBeginCapture.mode = mode;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureMode)>("hipStreamBeginCapture");
    hipError_t out = hip_func(hip_args.hipStreamBeginCapture.stream, hip_args.hipStreamBeginCapture.mode);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamBeginCapture.stream;
    mode = hip_args.hipStreamBeginCapture.mode;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamCreate(hipStream_t* stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamCreate.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t*)>("hipStreamCreate");
    hipError_t out = hip_func(hip_args.hipStreamCreate.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamCreate.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamCreateWithFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamCreateWithFlags.stream = stream;
    hip_args.hipStreamCreateWithFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t*,unsigned int)>("hipStreamCreateWithFlags");
    hipError_t out = hip_func(hip_args.hipStreamCreateWithFlags.stream, hip_args.hipStreamCreateWithFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamCreateWithFlags.stream;
    flags = hip_args.hipStreamCreateWithFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamCreateWithPriority;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamCreateWithPriority.stream = stream;
    hip_args.hipStreamCreateWithPriority.flags = flags;
    hip_args.hipStreamCreateWithPriority.priority = priority;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t*,unsigned int,int)>("hipStreamCreateWithPriority");
    hipError_t out = hip_func(hip_args.hipStreamCreateWithPriority.stream, hip_args.hipStreamCreateWithPriority.flags, hip_args.hipStreamCreateWithPriority.priority);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamCreateWithPriority.stream;
    flags = hip_args.hipStreamCreateWithPriority.flags;
    priority = hip_args.hipStreamCreateWithPriority.priority;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamDestroy(hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamDestroy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamDestroy.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t)>("hipStreamDestroy");
    hipError_t out = hip_func(hip_args.hipStreamDestroy.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamDestroy.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamEndCapture;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamEndCapture.stream = stream;
    hip_args.hipStreamEndCapture.pGraph = pGraph;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipGraph_t*)>("hipStreamEndCapture");
    hipError_t out = hip_func(hip_args.hipStreamEndCapture.stream, hip_args.hipStreamEndCapture.pGraph);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamEndCapture.stream;
    pGraph = hip_args.hipStreamEndCapture.pGraph;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus, unsigned long long* pId) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamGetCaptureInfo;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamGetCaptureInfo.stream = stream;
    hip_args.hipStreamGetCaptureInfo.pCaptureStatus = pCaptureStatus;
    hip_args.hipStreamGetCaptureInfo.pId = pId;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus*,unsigned long long*)>("hipStreamGetCaptureInfo");
    hipError_t out = hip_func(hip_args.hipStreamGetCaptureInfo.stream, hip_args.hipStreamGetCaptureInfo.pCaptureStatus, hip_args.hipStreamGetCaptureInfo.pId);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamGetCaptureInfo.stream;
    pCaptureStatus = hip_args.hipStreamGetCaptureInfo.pCaptureStatus;
    pId = hip_args.hipStreamGetCaptureInfo.pId;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamGetCaptureInfo_v2;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamGetCaptureInfo_v2.stream = stream;
    hip_args.hipStreamGetCaptureInfo_v2.captureStatus_out = captureStatus_out;
    hip_args.hipStreamGetCaptureInfo_v2.id_out = id_out;
    hip_args.hipStreamGetCaptureInfo_v2.graph_out = graph_out;
    hip_args.hipStreamGetCaptureInfo_v2.dependencies_out = dependencies_out;
    hip_args.hipStreamGetCaptureInfo_v2.numDependencies_out = numDependencies_out;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus*,unsigned long long*,hipGraph_t*,const hipGraphNode_t**,size_t*)>("hipStreamGetCaptureInfo_v2");
    hipError_t out = hip_func(hip_args.hipStreamGetCaptureInfo_v2.stream, hip_args.hipStreamGetCaptureInfo_v2.captureStatus_out, hip_args.hipStreamGetCaptureInfo_v2.id_out, hip_args.hipStreamGetCaptureInfo_v2.graph_out, hip_args.hipStreamGetCaptureInfo_v2.dependencies_out, hip_args.hipStreamGetCaptureInfo_v2.numDependencies_out);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamGetCaptureInfo_v2.stream;
    captureStatus_out = hip_args.hipStreamGetCaptureInfo_v2.captureStatus_out;
    id_out = hip_args.hipStreamGetCaptureInfo_v2.id_out;
    graph_out = hip_args.hipStreamGetCaptureInfo_v2.graph_out;
    dependencies_out = hip_args.hipStreamGetCaptureInfo_v2.dependencies_out;
    numDependencies_out = hip_args.hipStreamGetCaptureInfo_v2.numDependencies_out;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamGetFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamGetFlags.stream = stream;
    hip_args.hipStreamGetFlags.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,unsigned int*)>("hipStreamGetFlags");
    hipError_t out = hip_func(hip_args.hipStreamGetFlags.stream, hip_args.hipStreamGetFlags.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamGetFlags.stream;
    flags = hip_args.hipStreamGetFlags.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamIsCapturing;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamIsCapturing.stream = stream;
    hip_args.hipStreamIsCapturing.pCaptureStatus = pCaptureStatus;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus*)>("hipStreamIsCapturing");
    hipError_t out = hip_func(hip_args.hipStreamIsCapturing.stream, hip_args.hipStreamIsCapturing.pCaptureStatus);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamIsCapturing.stream;
    pCaptureStatus = hip_args.hipStreamIsCapturing.pCaptureStatus;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamQuery(hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamQuery;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamQuery.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t)>("hipStreamQuery");
    hipError_t out = hip_func(hip_args.hipStreamQuery.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamQuery.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamSynchronize(hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamSynchronize;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamSynchronize.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t)>("hipStreamSynchronize");
    hipError_t out = hip_func(hip_args.hipStreamSynchronize.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamSynchronize.stream;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies, size_t numDependencies, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamUpdateCaptureDependencies;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamUpdateCaptureDependencies.stream = stream;
    hip_args.hipStreamUpdateCaptureDependencies.dependencies = dependencies;
    hip_args.hipStreamUpdateCaptureDependencies.numDependencies = numDependencies;
    hip_args.hipStreamUpdateCaptureDependencies.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipGraphNode_t*,size_t,unsigned int)>("hipStreamUpdateCaptureDependencies");
    hipError_t out = hip_func(hip_args.hipStreamUpdateCaptureDependencies.stream, hip_args.hipStreamUpdateCaptureDependencies.dependencies, hip_args.hipStreamUpdateCaptureDependencies.numDependencies, hip_args.hipStreamUpdateCaptureDependencies.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamUpdateCaptureDependencies.stream;
    dependencies = hip_args.hipStreamUpdateCaptureDependencies.dependencies;
    numDependencies = hip_args.hipStreamUpdateCaptureDependencies.numDependencies;
    flags = hip_args.hipStreamUpdateCaptureDependencies.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamWaitEvent;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamWaitEvent.stream = stream;
    hip_args.hipStreamWaitEvent.event = event;
    hip_args.hipStreamWaitEvent.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipEvent_t,unsigned int)>("hipStreamWaitEvent");
    hipError_t out = hip_func(hip_args.hipStreamWaitEvent.stream, hip_args.hipStreamWaitEvent.event, hip_args.hipStreamWaitEvent.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamWaitEvent.stream;
    event = hip_args.hipStreamWaitEvent.event;
    flags = hip_args.hipStreamWaitEvent.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, unsigned int value, unsigned int flags, unsigned int mask) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamWaitValue32;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamWaitValue32.stream = stream;
    hip_args.hipStreamWaitValue32.ptr = ptr;
    hip_args.hipStreamWaitValue32.value = value;
    hip_args.hipStreamWaitValue32.flags = flags;
    hip_args.hipStreamWaitValue32.mask = mask;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void*,unsigned int,unsigned int,unsigned int)>("hipStreamWaitValue32");
    hipError_t out = hip_func(hip_args.hipStreamWaitValue32.stream, hip_args.hipStreamWaitValue32.ptr, hip_args.hipStreamWaitValue32.value, hip_args.hipStreamWaitValue32.flags, hip_args.hipStreamWaitValue32.mask);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamWaitValue32.stream;
    ptr = hip_args.hipStreamWaitValue32.ptr;
    value = hip_args.hipStreamWaitValue32.value;
    flags = hip_args.hipStreamWaitValue32.flags;
    mask = hip_args.hipStreamWaitValue32.mask;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags, uint64_t mask) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamWaitValue64;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamWaitValue64.stream = stream;
    hip_args.hipStreamWaitValue64.ptr = ptr;
    hip_args.hipStreamWaitValue64.value = value;
    hip_args.hipStreamWaitValue64.flags = flags;
    hip_args.hipStreamWaitValue64.mask = mask;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void*,uint64_t,unsigned int,uint64_t)>("hipStreamWaitValue64");
    hipError_t out = hip_func(hip_args.hipStreamWaitValue64.stream, hip_args.hipStreamWaitValue64.ptr, hip_args.hipStreamWaitValue64.value, hip_args.hipStreamWaitValue64.flags, hip_args.hipStreamWaitValue64.mask);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamWaitValue64.stream;
    ptr = hip_args.hipStreamWaitValue64.ptr;
    value = hip_args.hipStreamWaitValue64.value;
    flags = hip_args.hipStreamWaitValue64.flags;
    mask = hip_args.hipStreamWaitValue64.mask;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, unsigned int value, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamWriteValue32;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamWriteValue32.stream = stream;
    hip_args.hipStreamWriteValue32.ptr = ptr;
    hip_args.hipStreamWriteValue32.value = value;
    hip_args.hipStreamWriteValue32.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void*,unsigned int,unsigned int)>("hipStreamWriteValue32");
    hipError_t out = hip_func(hip_args.hipStreamWriteValue32.stream, hip_args.hipStreamWriteValue32.ptr, hip_args.hipStreamWriteValue32.value, hip_args.hipStreamWriteValue32.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamWriteValue32.stream;
    ptr = hip_args.hipStreamWriteValue32.ptr;
    value = hip_args.hipStreamWriteValue32.value;
    flags = hip_args.hipStreamWriteValue32.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipStreamWriteValue64;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipStreamWriteValue64.stream = stream;
    hip_args.hipStreamWriteValue64.ptr = ptr;
    hip_args.hipStreamWriteValue64.value = value;
    hip_args.hipStreamWriteValue64.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void*,uint64_t,unsigned int)>("hipStreamWriteValue64");
    hipError_t out = hip_func(hip_args.hipStreamWriteValue64.stream, hip_args.hipStreamWriteValue64.ptr, hip_args.hipStreamWriteValue64.value, hip_args.hipStreamWriteValue64.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    stream = hip_args.hipStreamWriteValue64.stream;
    ptr = hip_args.hipStreamWriteValue64.ptr;
    value = hip_args.hipStreamWriteValue64.value;
    flags = hip_args.hipStreamWriteValue64.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr, const textureReference* texRef) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefGetAddress;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefGetAddress.dev_ptr = dev_ptr;
    hip_args.hipTexRefGetAddress.texRef = texRef;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t*,const textureReference*)>("hipTexRefGetAddress");
    hipError_t out = hip_func(hip_args.hipTexRefGetAddress.dev_ptr, hip_args.hipTexRefGetAddress.texRef);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    dev_ptr = hip_args.hipTexRefGetAddress.dev_ptr;
    texRef = hip_args.hipTexRefGetAddress.texRef;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefGetFlags(unsigned int* pFlags, const textureReference* texRef) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefGetFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefGetFlags.pFlags = pFlags;
    hip_args.hipTexRefGetFlags.texRef = texRef;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int*,const textureReference*)>("hipTexRefGetFlags");
    hipError_t out = hip_func(hip_args.hipTexRefGetFlags.pFlags, hip_args.hipTexRefGetFlags.texRef);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pFlags = hip_args.hipTexRefGetFlags.pFlags;
    texRef = hip_args.hipTexRefGetFlags.texRef;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefGetFormat(hipArray_Format* pFormat, int* pNumChannels, const textureReference* texRef) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefGetFormat;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefGetFormat.pFormat = pFormat;
    hip_args.hipTexRefGetFormat.pNumChannels = pNumChannels;
    hip_args.hipTexRefGetFormat.texRef = texRef;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_Format*,int*,const textureReference*)>("hipTexRefGetFormat");
    hipError_t out = hip_func(hip_args.hipTexRefGetFormat.pFormat, hip_args.hipTexRefGetFormat.pNumChannels, hip_args.hipTexRefGetFormat.texRef);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pFormat = hip_args.hipTexRefGetFormat.pFormat;
    pNumChannels = hip_args.hipTexRefGetFormat.pNumChannels;
    texRef = hip_args.hipTexRefGetFormat.texRef;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio, const textureReference* texRef) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefGetMaxAnisotropy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefGetMaxAnisotropy.pmaxAnsio = pmaxAnsio;
    hip_args.hipTexRefGetMaxAnisotropy.texRef = texRef;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int*,const textureReference*)>("hipTexRefGetMaxAnisotropy");
    hipError_t out = hip_func(hip_args.hipTexRefGetMaxAnisotropy.pmaxAnsio, hip_args.hipTexRefGetMaxAnisotropy.texRef);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pmaxAnsio = hip_args.hipTexRefGetMaxAnisotropy.pmaxAnsio;
    texRef = hip_args.hipTexRefGetMaxAnisotropy.texRef;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray, const textureReference* texRef) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefGetMipMappedArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefGetMipMappedArray.pArray = pArray;
    hip_args.hipTexRefGetMipMappedArray.texRef = texRef;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t*,const textureReference*)>("hipTexRefGetMipMappedArray");
    hipError_t out = hip_func(hip_args.hipTexRefGetMipMappedArray.pArray, hip_args.hipTexRefGetMipMappedArray.texRef);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pArray = hip_args.hipTexRefGetMipMappedArray.pArray;
    texRef = hip_args.hipTexRefGetMipMappedArray.texRef;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefGetMipmapLevelBias(float* pbias, const textureReference* texRef) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefGetMipmapLevelBias;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefGetMipmapLevelBias.pbias = pbias;
    hip_args.hipTexRefGetMipmapLevelBias.texRef = texRef;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(float*,const textureReference*)>("hipTexRefGetMipmapLevelBias");
    hipError_t out = hip_func(hip_args.hipTexRefGetMipmapLevelBias.pbias, hip_args.hipTexRefGetMipmapLevelBias.texRef);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pbias = hip_args.hipTexRefGetMipmapLevelBias.pbias;
    texRef = hip_args.hipTexRefGetMipmapLevelBias.texRef;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, const textureReference* texRef) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefGetMipmapLevelClamp;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefGetMipmapLevelClamp.pminMipmapLevelClamp = pminMipmapLevelClamp;
    hip_args.hipTexRefGetMipmapLevelClamp.pmaxMipmapLevelClamp = pmaxMipmapLevelClamp;
    hip_args.hipTexRefGetMipmapLevelClamp.texRef = texRef;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(float*,float*,const textureReference*)>("hipTexRefGetMipmapLevelClamp");
    hipError_t out = hip_func(hip_args.hipTexRefGetMipmapLevelClamp.pminMipmapLevelClamp, hip_args.hipTexRefGetMipmapLevelClamp.pmaxMipmapLevelClamp, hip_args.hipTexRefGetMipmapLevelClamp.texRef);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    pminMipmapLevelClamp = hip_args.hipTexRefGetMipmapLevelClamp.pminMipmapLevelClamp;
    pmaxMipmapLevelClamp = hip_args.hipTexRefGetMipmapLevelClamp.pmaxMipmapLevelClamp;
    texRef = hip_args.hipTexRefGetMipmapLevelClamp.texRef;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetAddress(size_t* ByteOffset, textureReference* texRef, hipDeviceptr_t dptr, size_t bytes) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetAddress;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetAddress.ByteOffset = ByteOffset;
    hip_args.hipTexRefSetAddress.texRef = texRef;
    hip_args.hipTexRefSetAddress.dptr = dptr;
    hip_args.hipTexRefSetAddress.bytes = bytes;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t*,textureReference*,hipDeviceptr_t,size_t)>("hipTexRefSetAddress");
    hipError_t out = hip_func(hip_args.hipTexRefSetAddress.ByteOffset, hip_args.hipTexRefSetAddress.texRef, hip_args.hipTexRefSetAddress.dptr, hip_args.hipTexRefSetAddress.bytes);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    ByteOffset = hip_args.hipTexRefSetAddress.ByteOffset;
    texRef = hip_args.hipTexRefSetAddress.texRef;
    dptr = hip_args.hipTexRefSetAddress.dptr;
    bytes = hip_args.hipTexRefSetAddress.bytes;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc, hipDeviceptr_t dptr, size_t Pitch) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetAddress2D;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetAddress2D.texRef = texRef;
    hip_args.hipTexRefSetAddress2D.desc = desc;
    hip_args.hipTexRefSetAddress2D.dptr = dptr;
    hip_args.hipTexRefSetAddress2D.Pitch = Pitch;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,const HIP_ARRAY_DESCRIPTOR*,hipDeviceptr_t,size_t)>("hipTexRefSetAddress2D");
    hipError_t out = hip_func(hip_args.hipTexRefSetAddress2D.texRef, hip_args.hipTexRefSetAddress2D.desc, hip_args.hipTexRefSetAddress2D.dptr, hip_args.hipTexRefSetAddress2D.Pitch);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipTexRefSetAddress2D.texRef;
    desc = hip_args.hipTexRefSetAddress2D.desc;
    dptr = hip_args.hipTexRefSetAddress2D.dptr;
    Pitch = hip_args.hipTexRefSetAddress2D.Pitch;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetArray.tex = tex;
    hip_args.hipTexRefSetArray.array = array;
    hip_args.hipTexRefSetArray.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,hipArray_const_t,unsigned int)>("hipTexRefSetArray");
    hipError_t out = hip_func(hip_args.hipTexRefSetArray.tex, hip_args.hipTexRefSetArray.array, hip_args.hipTexRefSetArray.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    tex = hip_args.hipTexRefSetArray.tex;
    array = hip_args.hipTexRefSetArray.array;
    flags = hip_args.hipTexRefSetArray.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetBorderColor(textureReference* texRef, float* pBorderColor) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetBorderColor;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetBorderColor.texRef = texRef;
    hip_args.hipTexRefSetBorderColor.pBorderColor = pBorderColor;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,float*)>("hipTexRefSetBorderColor");
    hipError_t out = hip_func(hip_args.hipTexRefSetBorderColor.texRef, hip_args.hipTexRefSetBorderColor.pBorderColor);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipTexRefSetBorderColor.texRef;
    pBorderColor = hip_args.hipTexRefSetBorderColor.pBorderColor;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetFlags(textureReference* texRef, unsigned int Flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetFlags;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetFlags.texRef = texRef;
    hip_args.hipTexRefSetFlags.Flags = Flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,unsigned int)>("hipTexRefSetFlags");
    hipError_t out = hip_func(hip_args.hipTexRefSetFlags.texRef, hip_args.hipTexRefSetFlags.Flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipTexRefSetFlags.texRef;
    Flags = hip_args.hipTexRefSetFlags.Flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetFormat(textureReference* texRef, hipArray_Format fmt, int NumPackedComponents) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetFormat;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetFormat.texRef = texRef;
    hip_args.hipTexRefSetFormat.fmt = fmt;
    hip_args.hipTexRefSetFormat.NumPackedComponents = NumPackedComponents;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,hipArray_Format,int)>("hipTexRefSetFormat");
    hipError_t out = hip_func(hip_args.hipTexRefSetFormat.texRef, hip_args.hipTexRefSetFormat.fmt, hip_args.hipTexRefSetFormat.NumPackedComponents);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipTexRefSetFormat.texRef;
    fmt = hip_args.hipTexRefSetFormat.fmt;
    NumPackedComponents = hip_args.hipTexRefSetFormat.NumPackedComponents;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef, unsigned int maxAniso) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetMaxAnisotropy;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetMaxAnisotropy.texRef = texRef;
    hip_args.hipTexRefSetMaxAnisotropy.maxAniso = maxAniso;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,unsigned int)>("hipTexRefSetMaxAnisotropy");
    hipError_t out = hip_func(hip_args.hipTexRefSetMaxAnisotropy.texRef, hip_args.hipTexRefSetMaxAnisotropy.maxAniso);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipTexRefSetMaxAnisotropy.texRef;
    maxAniso = hip_args.hipTexRefSetMaxAnisotropy.maxAniso;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef, float bias) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetMipmapLevelBias;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetMipmapLevelBias.texRef = texRef;
    hip_args.hipTexRefSetMipmapLevelBias.bias = bias;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,float)>("hipTexRefSetMipmapLevelBias");
    hipError_t out = hip_func(hip_args.hipTexRefSetMipmapLevelBias.texRef, hip_args.hipTexRefSetMipmapLevelBias.bias);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipTexRefSetMipmapLevelBias.texRef;
    bias = hip_args.hipTexRefSetMipmapLevelBias.bias;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef, float minMipMapLevelClamp, float maxMipMapLevelClamp) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetMipmapLevelClamp;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetMipmapLevelClamp.texRef = texRef;
    hip_args.hipTexRefSetMipmapLevelClamp.minMipMapLevelClamp = minMipMapLevelClamp;
    hip_args.hipTexRefSetMipmapLevelClamp.maxMipMapLevelClamp = maxMipMapLevelClamp;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,float,float)>("hipTexRefSetMipmapLevelClamp");
    hipError_t out = hip_func(hip_args.hipTexRefSetMipmapLevelClamp.texRef, hip_args.hipTexRefSetMipmapLevelClamp.minMipMapLevelClamp, hip_args.hipTexRefSetMipmapLevelClamp.maxMipMapLevelClamp);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipTexRefSetMipmapLevelClamp.texRef;
    minMipMapLevelClamp = hip_args.hipTexRefSetMipmapLevelClamp.minMipMapLevelClamp;
    maxMipMapLevelClamp = hip_args.hipTexRefSetMipmapLevelClamp.maxMipMapLevelClamp;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipTexRefSetMipmappedArray(textureReference* texRef, hipMipmappedArray* mipmappedArray, unsigned int Flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipTexRefSetMipmappedArray;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipTexRefSetMipmappedArray.texRef = texRef;
    hip_args.hipTexRefSetMipmappedArray.mipmappedArray = mipmappedArray;
    hip_args.hipTexRefSetMipmappedArray.Flags = Flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference*,hipMipmappedArray*,unsigned int)>("hipTexRefSetMipmappedArray");
    hipError_t out = hip_func(hip_args.hipTexRefSetMipmappedArray.texRef, hip_args.hipTexRefSetMipmappedArray.mipmappedArray, hip_args.hipTexRefSetMipmappedArray.Flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    texRef = hip_args.hipTexRefSetMipmappedArray.texRef;
    mipmappedArray = hip_args.hipTexRefSetMipmappedArray.mipmappedArray;
    Flags = hip_args.hipTexRefSetMipmappedArray.Flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipThreadExchangeStreamCaptureMode;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipThreadExchangeStreamCaptureMode.mode = mode;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStreamCaptureMode*)>("hipThreadExchangeStreamCaptureMode");
    hipError_t out = hip_func(hip_args.hipThreadExchangeStreamCaptureMode.mode);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    mode = hip_args.hipThreadExchangeStreamCaptureMode.mode;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipUserObjectCreate;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipUserObjectCreate.object_out = object_out;
    hip_args.hipUserObjectCreate.ptr = ptr;
    hip_args.hipUserObjectCreate.destroy = destroy;
    hip_args.hipUserObjectCreate.initialRefcount = initialRefcount;
    hip_args.hipUserObjectCreate.flags = flags;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipUserObject_t*,void*,hipHostFn_t,unsigned int,unsigned int)>("hipUserObjectCreate");
    hipError_t out = hip_func(hip_args.hipUserObjectCreate.object_out, hip_args.hipUserObjectCreate.ptr, hip_args.hipUserObjectCreate.destroy, hip_args.hipUserObjectCreate.initialRefcount, hip_args.hipUserObjectCreate.flags);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    object_out = hip_args.hipUserObjectCreate.object_out;
    ptr = hip_args.hipUserObjectCreate.ptr;
    destroy = hip_args.hipUserObjectCreate.destroy;
    initialRefcount = hip_args.hipUserObjectCreate.initialRefcount;
    flags = hip_args.hipUserObjectCreate.flags;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipUserObjectRelease;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipUserObjectRelease.object = object;
    hip_args.hipUserObjectRelease.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipUserObject_t,unsigned int)>("hipUserObjectRelease");
    hipError_t out = hip_func(hip_args.hipUserObjectRelease.object, hip_args.hipUserObjectRelease.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    object = hip_args.hipUserObjectRelease.object;
    count = hip_args.hipUserObjectRelease.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipUserObjectRetain;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipUserObjectRetain.object = object;
    hip_args.hipUserObjectRetain.count = count;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipUserObject_t,unsigned int)>("hipUserObjectRetain");
    hipError_t out = hip_func(hip_args.hipUserObjectRetain.object, hip_args.hipUserObjectRetain.count);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    object = hip_args.hipUserObjectRetain.object;
    count = hip_args.hipUserObjectRetain.count;

    return out;
}

__attribute__((visibility("default")))
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, hipStream_t stream) {
    auto& hipInterceptor = SibirHipInterceptor::Instance();
    auto& hipCallback = hipInterceptor.getCallback();
    auto api_id = HIP_API_ID_hipWaitExternalSemaphoresAsync;
    // Copy Arguments for PHASE_ENTER
    hip_api_args_t hip_args{};
    hip_args.hipWaitExternalSemaphoresAsync.extSemArray = extSemArray;
    hip_args.hipWaitExternalSemaphoresAsync.paramsArray = paramsArray;
    hip_args.hipWaitExternalSemaphoresAsync.numExtSems = numExtSems;
    hip_args.hipWaitExternalSemaphoresAsync.stream = stream;
    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);
    static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hipExternalSemaphore_t*,const hipExternalSemaphoreWaitParams*,unsigned int,hipStream_t)>("hipWaitExternalSemaphoresAsync");
    hipError_t out = hip_func(hip_args.hipWaitExternalSemaphoresAsync.extSemArray, hip_args.hipWaitExternalSemaphoresAsync.paramsArray, hip_args.hipWaitExternalSemaphoresAsync.numExtSems, hip_args.hipWaitExternalSemaphoresAsync.stream);
    // Exit Callback
    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);
    // Copy the modified arguments back to the original arguments
    extSemArray = hip_args.hipWaitExternalSemaphoresAsync.extSemArray;
    paramsArray = hip_args.hipWaitExternalSemaphoresAsync.paramsArray;
    numExtSems = hip_args.hipWaitExternalSemaphoresAsync.numExtSems;
    stream = hip_args.hipWaitExternalSemaphoresAsync.stream;

    return out;
}

