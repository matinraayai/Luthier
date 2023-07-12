#ifndef HIP_PRIVATE_API
#define HIP_PRIVATE_API

enum hip_private_api_id_t {
	HIP_PRIVATE_API_ID_NONE = 0,
	HIP_PRIVATE_API_ID_FIRST = 1000,
	HIP_PRIVATE_API_ID___hipGetPCH = 1000,
	HIP_PRIVATE_API_ID___hipRegisterFatBinary = 1001,
	HIP_PRIVATE_API_ID___hipRegisterFunction = 1002,
	HIP_PRIVATE_API_ID___hipRegisterManagedVar = 1003,
	HIP_PRIVATE_API_ID___hipRegisterSurface = 1004,
	HIP_PRIVATE_API_ID___hipRegisterTexture = 1005,
	HIP_PRIVATE_API_ID___hipRegisterVar = 1006,
	HIP_PRIVATE_API_ID___hipUnregisterFatBinary = 1007,
	HIP_PRIVATE_API_ID_hipApiName = 1008,
	HIP_PRIVATE_API_ID_hipDrvGetErrorName = 1009,
	HIP_PRIVATE_API_ID_hipDrvGetErrorString = 1010,
	HIP_PRIVATE_API_ID_hipGetCmdName = 1011,
	HIP_PRIVATE_API_ID_hipGetErrorName = 1012,
	HIP_PRIVATE_API_ID_hipGetStreamDeviceId = 1013,
	HIP_PRIVATE_API_ID_hipKernelNameRef = 1014,
	HIP_PRIVATE_API_ID_hipKernelNameRefByPtr = 1015,
	HIP_PRIVATE_API_ID_hipLaunchKernel_common = 1016,
	HIP_PRIVATE_API_ID_hipLaunchKernel_spt = 1017,
	HIP_PRIVATE_API_ID_hip_init = 1018,
	HIP_PRIVATE_API_ID_LAST = 1018
};

#endif