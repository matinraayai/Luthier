#include "hip_intercept.hpp"


extern "C" __attribute__((visibility("default")))
void __hipGetPCH(const char * * pch, unsigned int * size) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID___hipGetPCH;
	// Copy Arguments for PHASE_ENTER
	hip___hipGetPCH_api_args_t hip_func_args{pch, size};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	static auto hip_func = hipInterceptor.GetHipFunction<void(*)(const char * *,unsigned int *)>("__hipGetPCH");
	hip_func(hip_func_args.pch, hip_func_args.size);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	size = hip_func_args.size;
};

extern "C" __attribute__((visibility("default")))
hipError_t __hipPopCallConfiguration(dim3 * gridDim, dim3 * blockDim, size_t * sharedMem, hipStream_t * stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID___hipPopCallConfiguration;
	// Copy Arguments for PHASE_ENTER
	hip___hipPopCallConfiguration_api_args_t hip_func_args{gridDim, blockDim, sharedMem, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(dim3 *,dim3 *,size_t *,hipStream_t *)>("__hipPopCallConfiguration");
	out = hip_func(hip_func_args.gridDim, hip_func_args.blockDim, hip_func_args.sharedMem, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridDim = hip_func_args.gridDim;
	blockDim = hip_func_args.blockDim;
	sharedMem = hip_func_args.sharedMem;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID___hipPushCallConfiguration;
	// Copy Arguments for PHASE_ENTER
	hip___hipPushCallConfiguration_api_args_t hip_func_args{gridDim, blockDim, sharedMem, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(dim3,dim3,size_t,hipStream_t)>("__hipPushCallConfiguration");
	out = hip_func(hip_func_args.gridDim, hip_func_args.blockDim, hip_func_args.sharedMem, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridDim = hip_func_args.gridDim;
	blockDim = hip_func_args.blockDim;
	sharedMem = hip_func_args.sharedMem;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hip::FatBinaryInfo * * __hipRegisterFatBinary(const void * data) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID___hipRegisterFatBinary;
	// Copy Arguments for PHASE_ENTER
	hip___hipRegisterFatBinary_api_args_t hip_func_args{data};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hip::FatBinaryInfo * * out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hip::FatBinaryInfo * *(*)(const void *)>("__hipRegisterFatBinary");
	out = hip_func(hip_func_args.data);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
void __hipRegisterFunction(hip::FatBinaryInfo * * modules, const void * hostFunction, char * deviceFunction, const char * deviceName, unsigned int threadLimit, uint3 * tid, uint3 * bid, dim3 * blockDim, dim3 * gridDim, int * wSize) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID___hipRegisterFunction;
	// Copy Arguments for PHASE_ENTER
	hip___hipRegisterFunction_api_args_t hip_func_args{modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim, wSize};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	static auto hip_func = hipInterceptor.GetHipFunction<void(*)(hip::FatBinaryInfo * *,const void *,char *,const char *,unsigned int,uint3 *,uint3 *,dim3 *,dim3 *,int *)>("__hipRegisterFunction");
	hip_func(hip_func_args.modules, hip_func_args.hostFunction, hip_func_args.deviceFunction, hip_func_args.deviceName, hip_func_args.threadLimit, hip_func_args.tid, hip_func_args.bid, hip_func_args.blockDim, hip_func_args.gridDim, hip_func_args.wSize);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	modules = hip_func_args.modules;
	deviceFunction = hip_func_args.deviceFunction;
	threadLimit = hip_func_args.threadLimit;
	tid = hip_func_args.tid;
	bid = hip_func_args.bid;
	blockDim = hip_func_args.blockDim;
	gridDim = hip_func_args.gridDim;
	wSize = hip_func_args.wSize;
};

extern "C" __attribute__((visibility("default")))
void __hipRegisterManagedVar(void * hipModule, void * * pointer, void * init_value, const char * name, size_t size, unsigned align) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID___hipRegisterManagedVar;
	// Copy Arguments for PHASE_ENTER
	hip___hipRegisterManagedVar_api_args_t hip_func_args{hipModule, pointer, init_value, name, size, align};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	static auto hip_func = hipInterceptor.GetHipFunction<void(*)(void *,void * *,void *,const char *,size_t,unsigned)>("__hipRegisterManagedVar");
	hip_func(hip_func_args.hipModule, hip_func_args.pointer, hip_func_args.init_value, hip_func_args.name, hip_func_args.size, hip_func_args.align);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hipModule = hip_func_args.hipModule;
	pointer = hip_func_args.pointer;
	init_value = hip_func_args.init_value;
	size = hip_func_args.size;
	align = hip_func_args.align;
};

extern "C" __attribute__((visibility("default")))
void __hipRegisterSurface(hip::FatBinaryInfo * * modules, void * var, char * hostVar, char * deviceVar, int type, int ext) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID___hipRegisterSurface;
	// Copy Arguments for PHASE_ENTER
	hip___hipRegisterSurface_api_args_t hip_func_args{modules, var, hostVar, deviceVar, type, ext};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	static auto hip_func = hipInterceptor.GetHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,int)>("__hipRegisterSurface");
	hip_func(hip_func_args.modules, hip_func_args.var, hip_func_args.hostVar, hip_func_args.deviceVar, hip_func_args.type, hip_func_args.ext);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	modules = hip_func_args.modules;
	var = hip_func_args.var;
	hostVar = hip_func_args.hostVar;
	deviceVar = hip_func_args.deviceVar;
	type = hip_func_args.type;
	ext = hip_func_args.ext;
};

extern "C" __attribute__((visibility("default")))
void __hipRegisterTexture(hip::FatBinaryInfo * * modules, void * var, char * hostVar, char * deviceVar, int type, int norm, int ext) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID___hipRegisterTexture;
	// Copy Arguments for PHASE_ENTER
	hip___hipRegisterTexture_api_args_t hip_func_args{modules, var, hostVar, deviceVar, type, norm, ext};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	static auto hip_func = hipInterceptor.GetHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,int,int)>("__hipRegisterTexture");
	hip_func(hip_func_args.modules, hip_func_args.var, hip_func_args.hostVar, hip_func_args.deviceVar, hip_func_args.type, hip_func_args.norm, hip_func_args.ext);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	modules = hip_func_args.modules;
	var = hip_func_args.var;
	hostVar = hip_func_args.hostVar;
	deviceVar = hip_func_args.deviceVar;
	type = hip_func_args.type;
	norm = hip_func_args.norm;
	ext = hip_func_args.ext;
};

extern "C" __attribute__((visibility("default")))
void __hipRegisterVar(hip::FatBinaryInfo * * modules, void * var, char * hostVar, char * deviceVar, int ext, size_t size, int constant, int global) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID___hipRegisterVar;
	// Copy Arguments for PHASE_ENTER
	hip___hipRegisterVar_api_args_t hip_func_args{modules, var, hostVar, deviceVar, ext, size, constant, global};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	static auto hip_func = hipInterceptor.GetHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,size_t,int,int)>("__hipRegisterVar");
	hip_func(hip_func_args.modules, hip_func_args.var, hip_func_args.hostVar, hip_func_args.deviceVar, hip_func_args.ext, hip_func_args.size, hip_func_args.constant, hip_func_args.global);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	modules = hip_func_args.modules;
	var = hip_func_args.var;
	hostVar = hip_func_args.hostVar;
	deviceVar = hip_func_args.deviceVar;
	ext = hip_func_args.ext;
	size = hip_func_args.size;
	constant = hip_func_args.constant;
	global = hip_func_args.global;
};

extern "C" __attribute__((visibility("default")))
void __hipUnregisterFatBinary(hip::FatBinaryInfo * * modules) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID___hipUnregisterFatBinary;
	// Copy Arguments for PHASE_ENTER
	hip___hipUnregisterFatBinary_api_args_t hip_func_args{modules};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	static auto hip_func = hipInterceptor.GetHipFunction<void(*)(hip::FatBinaryInfo * *)>("__hipUnregisterFatBinary");
	hip_func(hip_func_args.modules);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	modules = hip_func_args.modules;
};

extern "C" __attribute__((visibility("default")))
const char * hipApiName(uint32_t id) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipApiName;
	// Copy Arguments for PHASE_ENTER
	hip_hipApiName_api_args_t hip_func_args{id};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	const char * out{};
	static auto hip_func = hipInterceptor.GetHipFunction<const char *(*)(uint32_t)>("hipApiName");
	out = hip_func(hip_func_args.id);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	id = hip_func_args.id;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipArray3DCreate(hipArray * * array, const HIP_ARRAY3D_DESCRIPTOR * pAllocateArray) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipArray3DCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipArray3DCreate_api_args_t hip_func_args{array, pAllocateArray};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray * *,const HIP_ARRAY3D_DESCRIPTOR *)>("hipArray3DCreate");
	out = hip_func(hip_func_args.array, hip_func_args.pAllocateArray);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	array = hip_func_args.array;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipArrayCreate(hipArray * * pHandle, const HIP_ARRAY_DESCRIPTOR * pAllocateArray) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipArrayCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipArrayCreate_api_args_t hip_func_args{pHandle, pAllocateArray};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray * *,const HIP_ARRAY_DESCRIPTOR *)>("hipArrayCreate");
	out = hip_func(hip_func_args.pHandle, hip_func_args.pAllocateArray);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pHandle = hip_func_args.pHandle;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipArrayDestroy(hipArray * array) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipArrayDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipArrayDestroy_api_args_t hip_func_args{array};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray *)>("hipArrayDestroy");
	out = hip_func(hip_func_args.array);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	array = hip_func_args.array;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipBindTexture(size_t * offset, const textureReference * tex, const void * devPtr, const hipChannelFormatDesc * desc, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipBindTexture;
	// Copy Arguments for PHASE_ENTER
	hip_hipBindTexture_api_args_t hip_func_args{offset, tex, devPtr, desc, size};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,const textureReference *,const void *,const hipChannelFormatDesc *,size_t)>("hipBindTexture");
	out = hip_func(hip_func_args.offset, hip_func_args.tex, hip_func_args.devPtr, hip_func_args.desc, hip_func_args.size);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	offset = hip_func_args.offset;
	size = hip_func_args.size;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipBindTexture2D(size_t * offset, const textureReference * tex, const void * devPtr, const hipChannelFormatDesc * desc, size_t width, size_t height, size_t pitch) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipBindTexture2D;
	// Copy Arguments for PHASE_ENTER
	hip_hipBindTexture2D_api_args_t hip_func_args{offset, tex, devPtr, desc, width, height, pitch};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,const textureReference *,const void *,const hipChannelFormatDesc *,size_t,size_t,size_t)>("hipBindTexture2D");
	out = hip_func(hip_func_args.offset, hip_func_args.tex, hip_func_args.devPtr, hip_func_args.desc, hip_func_args.width, hip_func_args.height, hip_func_args.pitch);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	offset = hip_func_args.offset;
	width = hip_func_args.width;
	height = hip_func_args.height;
	pitch = hip_func_args.pitch;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipBindTextureToArray(const textureReference * tex, hipArray_const_t array, const hipChannelFormatDesc * desc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipBindTextureToArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipBindTextureToArray_api_args_t hip_func_args{tex, array, desc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const textureReference *,hipArray_const_t,const hipChannelFormatDesc *)>("hipBindTextureToArray");
	out = hip_func(hip_func_args.tex, hip_func_args.array, hip_func_args.desc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipBindTextureToMipmappedArray(const textureReference * tex, hipMipmappedArray_const_t mipmappedArray, const hipChannelFormatDesc * desc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipBindTextureToMipmappedArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipBindTextureToMipmappedArray_api_args_t hip_func_args{tex, mipmappedArray, desc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const textureReference *,hipMipmappedArray_const_t,const hipChannelFormatDesc *)>("hipBindTextureToMipmappedArray");
	out = hip_func(hip_func_args.tex, hip_func_args.mipmappedArray, hip_func_args.desc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipChooseDevice(int * device, const hipDeviceProp_t * prop) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipChooseDevice;
	// Copy Arguments for PHASE_ENTER
	hip_hipChooseDevice_api_args_t hip_func_args{device, prop};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,const hipDeviceProp_t *)>("hipChooseDevice");
	out = hip_func(hip_func_args.device, hip_func_args.prop);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipConfigureCall;
	// Copy Arguments for PHASE_ENTER
	hip_hipConfigureCall_api_args_t hip_func_args{gridDim, blockDim, sharedMem, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(dim3,dim3,size_t,hipStream_t)>("hipConfigureCall");
	out = hip_func(hip_func_args.gridDim, hip_func_args.blockDim, hip_func_args.sharedMem, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridDim = hip_func_args.gridDim;
	blockDim = hip_func_args.blockDim;
	sharedMem = hip_func_args.sharedMem;
	stream = hip_func_args.stream;

	return out;
};

__attribute__((visibility("default")))
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t * pSurfObject, const hipResourceDesc * pResDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCreateSurfaceObject;
	// Copy Arguments for PHASE_ENTER
	hip_hipCreateSurfaceObject_api_args_t hip_func_args{pSurfObject, pResDesc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSurfaceObject_t *,const hipResourceDesc *)>("hipCreateSurfaceObject");
	out = hip_func(hip_func_args.pSurfObject, hip_func_args.pResDesc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pSurfObject = hip_func_args.pSurfObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCreateTextureObject(hipTextureObject_t * pTexObject, const hipResourceDesc * pResDesc, const hipTextureDesc * pTexDesc, const struct hipResourceViewDesc * pResViewDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCreateTextureObject;
	// Copy Arguments for PHASE_ENTER
	hip_hipCreateTextureObject_api_args_t hip_func_args{pTexObject, pResDesc, pTexDesc, pResViewDesc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipTextureObject_t *,const hipResourceDesc *,const hipTextureDesc *,const struct hipResourceViewDesc *)>("hipCreateTextureObject");
	out = hip_func(hip_func_args.pTexObject, hip_func_args.pResDesc, hip_func_args.pTexDesc, hip_func_args.pResViewDesc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pTexObject = hip_func_args.pTexObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxCreate(hipCtx_t * ctx, unsigned int flags, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxCreate_api_args_t hip_func_args{ctx, flags, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t *,unsigned int,hipDevice_t)>("hipCtxCreate");
	out = hip_func(hip_func_args.ctx, hip_func_args.flags, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ctx = hip_func_args.ctx;
	flags = hip_func_args.flags;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxDestroy(hipCtx_t ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxDestroy_api_args_t hip_func_args{ctx};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxDestroy");
	out = hip_func(hip_func_args.ctx);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ctx = hip_func_args.ctx;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxDisablePeerAccess;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxDisablePeerAccess_api_args_t hip_func_args{peerCtx};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxDisablePeerAccess");
	out = hip_func(hip_func_args.peerCtx);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	peerCtx = hip_func_args.peerCtx;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxEnablePeerAccess;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxEnablePeerAccess_api_args_t hip_func_args{peerCtx, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t,unsigned int)>("hipCtxEnablePeerAccess");
	out = hip_func(hip_func_args.peerCtx, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	peerCtx = hip_func_args.peerCtx;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int * apiVersion) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxGetApiVersion;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxGetApiVersion_api_args_t hip_func_args{ctx, apiVersion};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t,int *)>("hipCtxGetApiVersion");
	out = hip_func(hip_func_args.ctx, hip_func_args.apiVersion);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ctx = hip_func_args.ctx;
	apiVersion = hip_func_args.apiVersion;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetCacheConfig(hipFuncCache_t * cacheConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxGetCacheConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxGetCacheConfig_api_args_t hip_func_args{cacheConfig};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncCache_t *)>("hipCtxGetCacheConfig");
	out = hip_func(hip_func_args.cacheConfig);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	cacheConfig = hip_func_args.cacheConfig;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetCurrent(hipCtx_t * ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxGetCurrent;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxGetCurrent_api_args_t hip_func_args{ctx};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t *)>("hipCtxGetCurrent");
	out = hip_func(hip_func_args.ctx);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ctx = hip_func_args.ctx;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetDevice(hipDevice_t * device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxGetDevice;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxGetDevice_api_args_t hip_func_args{device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t *)>("hipCtxGetDevice");
	out = hip_func(hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetFlags(unsigned int * flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxGetFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxGetFlags_api_args_t hip_func_args{flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int *)>("hipCtxGetFlags");
	out = hip_func(hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig * pConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxGetSharedMemConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxGetSharedMemConfig_api_args_t hip_func_args{pConfig};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSharedMemConfig *)>("hipCtxGetSharedMemConfig");
	out = hip_func(hip_func_args.pConfig);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pConfig = hip_func_args.pConfig;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxPopCurrent(hipCtx_t * ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxPopCurrent;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxPopCurrent_api_args_t hip_func_args{ctx};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t *)>("hipCtxPopCurrent");
	out = hip_func(hip_func_args.ctx);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ctx = hip_func_args.ctx;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxPushCurrent;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxPushCurrent_api_args_t hip_func_args{ctx};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxPushCurrent");
	out = hip_func(hip_func_args.ctx);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ctx = hip_func_args.ctx;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxSetCacheConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxSetCacheConfig_api_args_t hip_func_args{cacheConfig};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncCache_t)>("hipCtxSetCacheConfig");
	out = hip_func(hip_func_args.cacheConfig);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	cacheConfig = hip_func_args.cacheConfig;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxSetCurrent;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxSetCurrent_api_args_t hip_func_args{ctx};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxSetCurrent");
	out = hip_func(hip_func_args.ctx);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ctx = hip_func_args.ctx;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxSetSharedMemConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipCtxSetSharedMemConfig_api_args_t hip_func_args{config};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSharedMemConfig)>("hipCtxSetSharedMemConfig");
	out = hip_func(hip_func_args.config);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	config = hip_func_args.config;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxSynchronize() {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipCtxSynchronize;
	// Copy Arguments for PHASE_ENTER
	hipUserCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipCtxSynchronize");
	out = hip_func();
	// Exit Callback
	hipUserCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDestroyExternalMemory;
	// Copy Arguments for PHASE_ENTER
	hip_hipDestroyExternalMemory_api_args_t hip_func_args{extMem};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipExternalMemory_t)>("hipDestroyExternalMemory");
	out = hip_func(hip_func_args.extMem);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	extMem = hip_func_args.extMem;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDestroyExternalSemaphore;
	// Copy Arguments for PHASE_ENTER
	hip_hipDestroyExternalSemaphore_api_args_t hip_func_args{extSem};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipExternalSemaphore_t)>("hipDestroyExternalSemaphore");
	out = hip_func(hip_func_args.extSem);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	extSem = hip_func_args.extSem;

	return out;
};

__attribute__((visibility("default")))
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDestroySurfaceObject;
	// Copy Arguments for PHASE_ENTER
	hip_hipDestroySurfaceObject_api_args_t hip_func_args{surfaceObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSurfaceObject_t)>("hipDestroySurfaceObject");
	out = hip_func(hip_func_args.surfaceObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	surfaceObject = hip_func_args.surfaceObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDestroyTextureObject;
	// Copy Arguments for PHASE_ENTER
	hip_hipDestroyTextureObject_api_args_t hip_func_args{textureObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipTextureObject_t)>("hipDestroyTextureObject");
	out = hip_func(hip_func_args.textureObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	textureObject = hip_func_args.textureObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceCanAccessPeer(int * canAccessPeer, int deviceId, int peerDeviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceCanAccessPeer;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceCanAccessPeer_api_args_t hip_func_args{canAccessPeer, deviceId, peerDeviceId};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,int,int)>("hipDeviceCanAccessPeer");
	out = hip_func(hip_func_args.canAccessPeer, hip_func_args.deviceId, hip_func_args.peerDeviceId);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	canAccessPeer = hip_func_args.canAccessPeer;
	deviceId = hip_func_args.deviceId;
	peerDeviceId = hip_func_args.peerDeviceId;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceComputeCapability(int * major, int * minor, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceComputeCapability;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceComputeCapability_api_args_t hip_func_args{major, minor, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,int *,hipDevice_t)>("hipDeviceComputeCapability");
	out = hip_func(hip_func_args.major, hip_func_args.minor, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	major = hip_func_args.major;
	minor = hip_func_args.minor;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceDisablePeerAccess;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceDisablePeerAccess_api_args_t hip_func_args{peerDeviceId};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int)>("hipDeviceDisablePeerAccess");
	out = hip_func(hip_func_args.peerDeviceId);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	peerDeviceId = hip_func_args.peerDeviceId;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceEnablePeerAccess;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceEnablePeerAccess_api_args_t hip_func_args{peerDeviceId, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,unsigned int)>("hipDeviceEnablePeerAccess");
	out = hip_func(hip_func_args.peerDeviceId, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	peerDeviceId = hip_func_args.peerDeviceId;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGet(hipDevice_t * device, int ordinal) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGet;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGet_api_args_t hip_func_args{device, ordinal};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t *,int)>("hipDeviceGet");
	out = hip_func(hip_func_args.device, hip_func_args.ordinal);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device = hip_func_args.device;
	ordinal = hip_func_args.ordinal;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetAttribute(int * pi, hipDeviceAttribute_t attr, int deviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetAttribute_api_args_t hip_func_args{pi, attr, deviceId};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,hipDeviceAttribute_t,int)>("hipDeviceGetAttribute");
	out = hip_func(hip_func_args.pi, hip_func_args.attr, hip_func_args.deviceId);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pi = hip_func_args.pi;
	attr = hip_func_args.attr;
	deviceId = hip_func_args.deviceId;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetByPCIBusId(int * device, const char * pciBusId) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetByPCIBusId;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetByPCIBusId_api_args_t hip_func_args{device, pciBusId};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,const char *)>("hipDeviceGetByPCIBusId");
	out = hip_func(hip_func_args.device, hip_func_args.pciBusId);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t * cacheConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetCacheConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetCacheConfig_api_args_t hip_func_args{cacheConfig};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncCache_t *)>("hipDeviceGetCacheConfig");
	out = hip_func(hip_func_args.cacheConfig);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	cacheConfig = hip_func_args.cacheConfig;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetCount(int * count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetCount;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetCount_api_args_t hip_func_args{count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *)>("hipDeviceGetCount");
	out = hip_func(hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t * mem_pool, int device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetDefaultMemPool;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetDefaultMemPool_api_args_t hip_func_args{mem_pool, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t *,int)>("hipDeviceGetDefaultMemPool");
	out = hip_func(hip_func_args.mem_pool, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void * value) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetGraphMemAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetGraphMemAttribute_api_args_t hip_func_args{device, attr, value};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipGraphMemAttributeType,void *)>("hipDeviceGetGraphMemAttribute");
	out = hip_func(hip_func_args.device, hip_func_args.attr, hip_func_args.value);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device = hip_func_args.device;
	attr = hip_func_args.attr;
	value = hip_func_args.value;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetLimit(size_t * pValue, enum hipLimit_t limit) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetLimit;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetLimit_api_args_t hip_func_args{pValue, limit};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,enum hipLimit_t)>("hipDeviceGetLimit");
	out = hip_func(hip_func_args.pValue, hip_func_args.limit);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pValue = hip_func_args.pValue;
	limit = hip_func_args.limit;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetMemPool(hipMemPool_t * mem_pool, int device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetMemPool;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetMemPool_api_args_t hip_func_args{mem_pool, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t *,int)>("hipDeviceGetMemPool");
	out = hip_func(hip_func_args.mem_pool, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetName(char * name, int len, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetName;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetName_api_args_t hip_func_args{name, len, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(char *,int,hipDevice_t)>("hipDeviceGetName");
	out = hip_func(hip_func_args.name, hip_func_args.len, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	name = hip_func_args.name;
	len = hip_func_args.len;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetP2PAttribute(int * value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetP2PAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetP2PAttribute_api_args_t hip_func_args{value, attr, srcDevice, dstDevice};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,hipDeviceP2PAttr,int,int)>("hipDeviceGetP2PAttribute");
	out = hip_func(hip_func_args.value, hip_func_args.attr, hip_func_args.srcDevice, hip_func_args.dstDevice);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	value = hip_func_args.value;
	attr = hip_func_args.attr;
	srcDevice = hip_func_args.srcDevice;
	dstDevice = hip_func_args.dstDevice;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetPCIBusId(char * pciBusId, int len, int device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetPCIBusId;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetPCIBusId_api_args_t hip_func_args{pciBusId, len, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(char *,int,int)>("hipDeviceGetPCIBusId");
	out = hip_func(hip_func_args.pciBusId, hip_func_args.len, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pciBusId = hip_func_args.pciBusId;
	len = hip_func_args.len;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig * pConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetSharedMemConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetSharedMemConfig_api_args_t hip_func_args{pConfig};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSharedMemConfig *)>("hipDeviceGetSharedMemConfig");
	out = hip_func(hip_func_args.pConfig);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pConfig = hip_func_args.pConfig;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetStreamPriorityRange;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetStreamPriorityRange_api_args_t hip_func_args{leastPriority, greatestPriority};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,int *)>("hipDeviceGetStreamPriorityRange");
	out = hip_func(hip_func_args.leastPriority, hip_func_args.greatestPriority);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	leastPriority = hip_func_args.leastPriority;
	greatestPriority = hip_func_args.greatestPriority;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetUuid(hipUUID * uuid, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGetUuid;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGetUuid_api_args_t hip_func_args{uuid, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipUUID *,hipDevice_t)>("hipDeviceGetUuid");
	out = hip_func(hip_func_args.uuid, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	uuid = hip_func_args.uuid;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGraphMemTrim(int device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceGraphMemTrim;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceGraphMemTrim_api_args_t hip_func_args{device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int)>("hipDeviceGraphMemTrim");
	out = hip_func(hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int * flags, int * active) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDevicePrimaryCtxGetState;
	// Copy Arguments for PHASE_ENTER
	hip_hipDevicePrimaryCtxGetState_api_args_t hip_func_args{dev, flags, active};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t,unsigned int *,int *)>("hipDevicePrimaryCtxGetState");
	out = hip_func(hip_func_args.dev, hip_func_args.flags, hip_func_args.active);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev = hip_func_args.dev;
	flags = hip_func_args.flags;
	active = hip_func_args.active;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDevicePrimaryCtxRelease;
	// Copy Arguments for PHASE_ENTER
	hip_hipDevicePrimaryCtxRelease_api_args_t hip_func_args{dev};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t)>("hipDevicePrimaryCtxRelease");
	out = hip_func(hip_func_args.dev);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev = hip_func_args.dev;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDevicePrimaryCtxReset;
	// Copy Arguments for PHASE_ENTER
	hip_hipDevicePrimaryCtxReset_api_args_t hip_func_args{dev};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t)>("hipDevicePrimaryCtxReset");
	out = hip_func(hip_func_args.dev);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev = hip_func_args.dev;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t * pctx, hipDevice_t dev) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDevicePrimaryCtxRetain;
	// Copy Arguments for PHASE_ENTER
	hip_hipDevicePrimaryCtxRetain_api_args_t hip_func_args{pctx, dev};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipCtx_t *,hipDevice_t)>("hipDevicePrimaryCtxRetain");
	out = hip_func(hip_func_args.pctx, hip_func_args.dev);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pctx = hip_func_args.pctx;
	dev = hip_func_args.dev;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDevicePrimaryCtxSetFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipDevicePrimaryCtxSetFlags_api_args_t hip_func_args{dev, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDevice_t,unsigned int)>("hipDevicePrimaryCtxSetFlags");
	out = hip_func(hip_func_args.dev, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev = hip_func_args.dev;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceReset() {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceReset;
	// Copy Arguments for PHASE_ENTER
	hipUserCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipDeviceReset");
	out = hip_func();
	// Exit Callback
	hipUserCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceSetCacheConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceSetCacheConfig_api_args_t hip_func_args{cacheConfig};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFuncCache_t)>("hipDeviceSetCacheConfig");
	out = hip_func(hip_func_args.cacheConfig);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	cacheConfig = hip_func_args.cacheConfig;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void * value) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceSetGraphMemAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceSetGraphMemAttribute_api_args_t hip_func_args{device, attr, value};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipGraphMemAttributeType,void *)>("hipDeviceSetGraphMemAttribute");
	out = hip_func(hip_func_args.device, hip_func_args.attr, hip_func_args.value);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device = hip_func_args.device;
	attr = hip_func_args.attr;
	value = hip_func_args.value;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceSetLimit;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceSetLimit_api_args_t hip_func_args{limit, value};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(enum hipLimit_t,size_t)>("hipDeviceSetLimit");
	out = hip_func(hip_func_args.limit, hip_func_args.value);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	limit = hip_func_args.limit;
	value = hip_func_args.value;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceSetMemPool;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceSetMemPool_api_args_t hip_func_args{device, mem_pool};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipMemPool_t)>("hipDeviceSetMemPool");
	out = hip_func(hip_func_args.device, hip_func_args.mem_pool);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device = hip_func_args.device;
	mem_pool = hip_func_args.mem_pool;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceSetSharedMemConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceSetSharedMemConfig_api_args_t hip_func_args{config};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipSharedMemConfig)>("hipDeviceSetSharedMemConfig");
	out = hip_func(hip_func_args.config);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	config = hip_func_args.config;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSynchronize() {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceSynchronize;
	// Copy Arguments for PHASE_ENTER
	hipUserCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipDeviceSynchronize");
	out = hip_func();
	// Exit Callback
	hipUserCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceTotalMem(size_t * bytes, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDeviceTotalMem;
	// Copy Arguments for PHASE_ENTER
	hip_hipDeviceTotalMem_api_args_t hip_func_args{bytes, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,hipDevice_t)>("hipDeviceTotalMem");
	out = hip_func(hip_func_args.bytes, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	bytes = hip_func_args.bytes;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDriverGetVersion(int * driverVersion) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDriverGetVersion;
	// Copy Arguments for PHASE_ENTER
	hip_hipDriverGetVersion_api_args_t hip_func_args{driverVersion};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *)>("hipDriverGetVersion");
	out = hip_func(hip_func_args.driverVersion);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	driverVersion = hip_func_args.driverVersion;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvGetErrorName(hipError_t hipError, const char * * errorString) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipDrvGetErrorName;
	// Copy Arguments for PHASE_ENTER
	hip_hipDrvGetErrorName_api_args_t hip_func_args{hipError, errorString};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipError_t,const char * *)>("hipDrvGetErrorName");
	out = hip_func(hip_func_args.hipError, hip_func_args.errorString);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hipError = hip_func_args.hipError;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvGetErrorString(hipError_t hipError, const char * * errorString) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipDrvGetErrorString;
	// Copy Arguments for PHASE_ENTER
	hip_hipDrvGetErrorString_api_args_t hip_func_args{hipError, errorString};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipError_t,const char * *)>("hipDrvGetErrorString");
	out = hip_func(hip_func_args.hipError, hip_func_args.errorString);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hipError = hip_func_args.hipError;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D * pCopy) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDrvMemcpy2DUnaligned;
	// Copy Arguments for PHASE_ENTER
	hip_hipDrvMemcpy2DUnaligned_api_args_t hip_func_args{pCopy};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hip_Memcpy2D *)>("hipDrvMemcpy2DUnaligned");
	out = hip_func(hip_func_args.pCopy);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D * pCopy) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDrvMemcpy3D;
	// Copy Arguments for PHASE_ENTER
	hip_hipDrvMemcpy3D_api_args_t hip_func_args{pCopy};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const HIP_MEMCPY3D *)>("hipDrvMemcpy3D");
	out = hip_func(hip_func_args.pCopy);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D * pCopy, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDrvMemcpy3DAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipDrvMemcpy3DAsync_api_args_t hip_func_args{pCopy, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const HIP_MEMCPY3D *,hipStream_t)>("hipDrvMemcpy3DAsync");
	out = hip_func(hip_func_args.pCopy, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute * attributes, void * * data, hipDeviceptr_t ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipDrvPointerGetAttributes;
	// Copy Arguments for PHASE_ENTER
	hip_hipDrvPointerGetAttributes_api_args_t hip_func_args{numAttributes, attributes, data, ptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int,hipPointer_attribute *,void * *,hipDeviceptr_t)>("hipDrvPointerGetAttributes");
	out = hip_func(hip_func_args.numAttributes, hip_func_args.attributes, hip_func_args.data, hip_func_args.ptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numAttributes = hip_func_args.numAttributes;
	attributes = hip_func_args.attributes;
	data = hip_func_args.data;
	ptr = hip_func_args.ptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipEventCreate(hipEvent_t * event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipEventCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipEventCreate_api_args_t hip_func_args{event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t *)>("hipEventCreate");
	out = hip_func(hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipEventCreateWithFlags(hipEvent_t * event, unsigned flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipEventCreateWithFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipEventCreateWithFlags_api_args_t hip_func_args{event, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t *,unsigned)>("hipEventCreateWithFlags");
	out = hip_func(hip_func_args.event, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	event = hip_func_args.event;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipEventDestroy(hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipEventDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipEventDestroy_api_args_t hip_func_args{event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t)>("hipEventDestroy");
	out = hip_func(hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipEventElapsedTime(float * ms, hipEvent_t start, hipEvent_t stop) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipEventElapsedTime;
	// Copy Arguments for PHASE_ENTER
	hip_hipEventElapsedTime_api_args_t hip_func_args{ms, start, stop};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(float *,hipEvent_t,hipEvent_t)>("hipEventElapsedTime");
	out = hip_func(hip_func_args.ms, hip_func_args.start, hip_func_args.stop);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ms = hip_func_args.ms;
	start = hip_func_args.start;
	stop = hip_func_args.stop;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipEventQuery(hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipEventQuery;
	// Copy Arguments for PHASE_ENTER
	hip_hipEventQuery_api_args_t hip_func_args{event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t)>("hipEventQuery");
	out = hip_func(hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipEventRecord;
	// Copy Arguments for PHASE_ENTER
	hip_hipEventRecord_api_args_t hip_func_args{event, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t,hipStream_t)>("hipEventRecord");
	out = hip_func(hip_func_args.event, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	event = hip_func_args.event;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipEventSynchronize(hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipEventSynchronize;
	// Copy Arguments for PHASE_ENTER
	hip_hipEventSynchronize_api_args_t hip_func_args{event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t)>("hipEventSynchronize");
	out = hip_func(hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t * linktype, uint32_t * hopcount) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipExtGetLinkTypeAndHopCount;
	// Copy Arguments for PHASE_ENTER
	hip_hipExtGetLinkTypeAndHopCount_api_args_t hip_func_args{device1, device2, linktype, hopcount};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,int,uint32_t *,uint32_t *)>("hipExtGetLinkTypeAndHopCount");
	out = hip_func(hip_func_args.device1, hip_func_args.device2, hip_func_args.linktype, hip_func_args.hopcount);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device1 = hip_func_args.device1;
	device2 = hip_func_args.device2;
	linktype = hip_func_args.linktype;
	hopcount = hip_func_args.hopcount;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipExtLaunchKernel(const void * function_address, dim3 numBlocks, dim3 dimBlocks, void * * args, size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent, hipEvent_t stopEvent, int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipExtLaunchKernel;
	// Copy Arguments for PHASE_ENTER
	hip_hipExtLaunchKernel_api_args_t hip_func_args{function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t,hipEvent_t,hipEvent_t,int)>("hipExtLaunchKernel");
	out = hip_func(hip_func_args.function_address, hip_func_args.numBlocks, hip_func_args.dimBlocks, hip_func_args.args, hip_func_args.sharedMemBytes, hip_func_args.stream, hip_func_args.startEvent, hip_func_args.stopEvent, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numBlocks = hip_func_args.numBlocks;
	dimBlocks = hip_func_args.dimBlocks;
	args = hip_func_args.args;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	stream = hip_func_args.stream;
	startEvent = hip_func_args.startEvent;
	stopEvent = hip_func_args.stopEvent;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams * launchParamsList, int numDevices, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipExtLaunchMultiKernelMultiDevice;
	// Copy Arguments for PHASE_ENTER
	hip_hipExtLaunchMultiKernelMultiDevice_api_args_t hip_func_args{launchParamsList, numDevices, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipLaunchParams *,int,unsigned int)>("hipExtLaunchMultiKernelMultiDevice");
	out = hip_func(hip_func_args.launchParamsList, hip_func_args.numDevices, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	launchParamsList = hip_func_args.launchParamsList;
	numDevices = hip_func_args.numDevices;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipExtMallocWithFlags(void * * ptr, size_t sizeBytes, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipExtMallocWithFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipExtMallocWithFlags_api_args_t hip_func_args{ptr, sizeBytes, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipExtMallocWithFlags");
	out = hip_func(hip_func_args.ptr, hip_func_args.sizeBytes, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	sizeBytes = hip_func_args.sizeBytes;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ, size_t sharedMemBytes, hipStream_t hStream, void * * kernelParams, void * * extra, hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipExtModuleLaunchKernel;
	// Copy Arguments for PHASE_ENTER
	hip_hipExtModuleLaunchKernel_api_args_t hip_func_args{f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t,uint32_t)>("hipExtModuleLaunchKernel");
	out = hip_func(hip_func_args.f, hip_func_args.globalWorkSizeX, hip_func_args.globalWorkSizeY, hip_func_args.globalWorkSizeZ, hip_func_args.localWorkSizeX, hip_func_args.localWorkSizeY, hip_func_args.localWorkSizeZ, hip_func_args.sharedMemBytes, hip_func_args.hStream, hip_func_args.kernelParams, hip_func_args.extra, hip_func_args.startEvent, hip_func_args.stopEvent, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	f = hip_func_args.f;
	globalWorkSizeX = hip_func_args.globalWorkSizeX;
	globalWorkSizeY = hip_func_args.globalWorkSizeY;
	globalWorkSizeZ = hip_func_args.globalWorkSizeZ;
	localWorkSizeX = hip_func_args.localWorkSizeX;
	localWorkSizeY = hip_func_args.localWorkSizeY;
	localWorkSizeZ = hip_func_args.localWorkSizeZ;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	hStream = hip_func_args.hStream;
	kernelParams = hip_func_args.kernelParams;
	extra = hip_func_args.extra;
	startEvent = hip_func_args.startEvent;
	stopEvent = hip_func_args.stopEvent;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipExtStreamCreateWithCUMask(hipStream_t * stream, uint32_t cuMaskSize, const uint32_t * cuMask) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipExtStreamCreateWithCUMask;
	// Copy Arguments for PHASE_ENTER
	hip_hipExtStreamCreateWithCUMask_api_args_t hip_func_args{stream, cuMaskSize, cuMask};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t *,uint32_t,const uint32_t *)>("hipExtStreamCreateWithCUMask");
	out = hip_func(hip_func_args.stream, hip_func_args.cuMaskSize, hip_func_args.cuMask);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	cuMaskSize = hip_func_args.cuMaskSize;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t * cuMask) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipExtStreamGetCUMask;
	// Copy Arguments for PHASE_ENTER
	hip_hipExtStreamGetCUMask_api_args_t hip_func_args{stream, cuMaskSize, cuMask};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,uint32_t,uint32_t *)>("hipExtStreamGetCUMask");
	out = hip_func(hip_func_args.stream, hip_func_args.cuMaskSize, hip_func_args.cuMask);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	cuMaskSize = hip_func_args.cuMaskSize;
	cuMask = hip_func_args.cuMask;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipExternalMemoryGetMappedBuffer(void * * devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc * bufferDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipExternalMemoryGetMappedBuffer;
	// Copy Arguments for PHASE_ENTER
	hip_hipExternalMemoryGetMappedBuffer_api_args_t hip_func_args{devPtr, extMem, bufferDesc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,hipExternalMemory_t,const hipExternalMemoryBufferDesc *)>("hipExternalMemoryGetMappedBuffer");
	out = hip_func(hip_func_args.devPtr, hip_func_args.extMem, hip_func_args.bufferDesc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	devPtr = hip_func_args.devPtr;
	extMem = hip_func_args.extMem;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFree(void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFree;
	// Copy Arguments for PHASE_ENTER
	hip_hipFree_api_args_t hip_func_args{ptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *)>("hipFree");
	out = hip_func(hip_func_args.ptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFreeArray(hipArray * array) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFreeArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipFreeArray_api_args_t hip_func_args{array};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray *)>("hipFreeArray");
	out = hip_func(hip_func_args.array);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	array = hip_func_args.array;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFreeAsync(void * dev_ptr, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFreeAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipFreeAsync_api_args_t hip_func_args{dev_ptr, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,hipStream_t)>("hipFreeAsync");
	out = hip_func(hip_func_args.dev_ptr, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev_ptr = hip_func_args.dev_ptr;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFreeHost(void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFreeHost;
	// Copy Arguments for PHASE_ENTER
	hip_hipFreeHost_api_args_t hip_func_args{ptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *)>("hipFreeHost");
	out = hip_func(hip_func_args.ptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFreeMipmappedArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipFreeMipmappedArray_api_args_t hip_func_args{mipmappedArray};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t)>("hipFreeMipmappedArray");
	out = hip_func(hip_func_args.mipmappedArray);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mipmappedArray = hip_func_args.mipmappedArray;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncGetAttribute(int * value, hipFunction_attribute attrib, hipFunction_t hfunc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFuncGetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipFuncGetAttribute_api_args_t hip_func_args{value, attrib, hfunc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,hipFunction_attribute,hipFunction_t)>("hipFuncGetAttribute");
	out = hip_func(hip_func_args.value, hip_func_args.attrib, hip_func_args.hfunc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	value = hip_func_args.value;
	attrib = hip_func_args.attrib;
	hfunc = hip_func_args.hfunc;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncGetAttributes(struct hipFuncAttributes * attr, const void * func) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFuncGetAttributes;
	// Copy Arguments for PHASE_ENTER
	hip_hipFuncGetAttributes_api_args_t hip_func_args{attr, func};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(struct hipFuncAttributes *,const void *)>("hipFuncGetAttributes");
	out = hip_func(hip_func_args.attr, hip_func_args.func);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	attr = hip_func_args.attr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncSetAttribute(const void * func, hipFuncAttribute attr, int value) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFuncSetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipFuncSetAttribute_api_args_t hip_func_args{func, attr, value};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,hipFuncAttribute,int)>("hipFuncSetAttribute");
	out = hip_func(hip_func_args.func, hip_func_args.attr, hip_func_args.value);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	attr = hip_func_args.attr;
	value = hip_func_args.value;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncSetCacheConfig(const void * func, hipFuncCache_t config) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFuncSetCacheConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipFuncSetCacheConfig_api_args_t hip_func_args{func, config};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,hipFuncCache_t)>("hipFuncSetCacheConfig");
	out = hip_func(hip_func_args.func, hip_func_args.config);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	config = hip_func_args.config;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncSetSharedMemConfig(const void * func, hipSharedMemConfig config) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipFuncSetSharedMemConfig;
	// Copy Arguments for PHASE_ENTER
	hip_hipFuncSetSharedMemConfig_api_args_t hip_func_args{func, config};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,hipSharedMemConfig)>("hipFuncSetSharedMemConfig");
	out = hip_func(hip_func_args.func, hip_func_args.config);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	config = hip_func_args.config;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGLGetDevices(unsigned int * pHipDeviceCount, int * pHipDevices, unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGLGetDevices;
	// Copy Arguments for PHASE_ENTER
	hip_hipGLGetDevices_api_args_t hip_func_args{pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int *,int *,unsigned int,hipGLDeviceList)>("hipGLGetDevices");
	out = hip_func(hip_func_args.pHipDeviceCount, hip_func_args.pHipDevices, hip_func_args.hipDeviceCount, hip_func_args.deviceList);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pHipDeviceCount = hip_func_args.pHipDeviceCount;
	pHipDevices = hip_func_args.pHipDevices;
	hipDeviceCount = hip_func_args.hipDeviceCount;
	deviceList = hip_func_args.deviceList;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetChannelDesc(hipChannelFormatDesc * desc, hipArray_const_t array) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetChannelDesc;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetChannelDesc_api_args_t hip_func_args{desc, array};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipChannelFormatDesc *,hipArray_const_t)>("hipGetChannelDesc");
	out = hip_func(hip_func_args.desc, hip_func_args.array);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	desc = hip_func_args.desc;

	return out;
};

extern "C" __attribute__((visibility("default")))
const char * hipGetCmdName(unsigned op) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipGetCmdName;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetCmdName_api_args_t hip_func_args{op};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	const char * out{};
	static auto hip_func = hipInterceptor.GetHipFunction<const char *(*)(unsigned)>("hipGetCmdName");
	out = hip_func(hip_func_args.op);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	op = hip_func_args.op;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetDevice(int * deviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetDevice;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetDevice_api_args_t hip_func_args{deviceId};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *)>("hipGetDevice");
	out = hip_func(hip_func_args.deviceId);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	deviceId = hip_func_args.deviceId;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetDeviceCount(int * count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetDeviceCount;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetDeviceCount_api_args_t hip_func_args{count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *)>("hipGetDeviceCount");
	out = hip_func(hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetDeviceFlags(unsigned int * flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetDeviceFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetDeviceFlags_api_args_t hip_func_args{flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int *)>("hipGetDeviceFlags");
	out = hip_func(hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetDeviceProperties(hipDeviceProp_t * prop, int deviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetDeviceProperties;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetDeviceProperties_api_args_t hip_func_args{prop, deviceId};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceProp_t *,int)>("hipGetDeviceProperties");
	out = hip_func(hip_func_args.prop, hip_func_args.deviceId);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	prop = hip_func_args.prop;
	deviceId = hip_func_args.deviceId;

	return out;
};

extern "C" __attribute__((visibility("default")))
const char * hipGetErrorName(hipError_t hip_error) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipGetErrorName;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetErrorName_api_args_t hip_func_args{hip_error};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	const char * out{};
	static auto hip_func = hipInterceptor.GetHipFunction<const char *(*)(hipError_t)>("hipGetErrorName");
	out = hip_func(hip_func_args.hip_error);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hip_error = hip_func_args.hip_error;

	return out;
};

extern "C" __attribute__((visibility("default")))
const char * hipGetErrorString(hipError_t hipError) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetErrorString;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetErrorString_api_args_t hip_func_args{hipError};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	const char * out{};
	static auto hip_func = hipInterceptor.GetHipFunction<const char *(*)(hipError_t)>("hipGetErrorString");
	out = hip_func(hip_func_args.hipError);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hipError = hip_func_args.hipError;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetLastError() {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetLastError;
	// Copy Arguments for PHASE_ENTER
	hipUserCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipGetLastError");
	out = hip_func();
	// Exit Callback
	hipUserCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetMipmappedArrayLevel(hipArray_t * levelArray, hipMipmappedArray_const_t mipmappedArray, unsigned int level) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetMipmappedArrayLevel;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetMipmappedArrayLevel_api_args_t hip_func_args{levelArray, mipmappedArray, level};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_t *,hipMipmappedArray_const_t,unsigned int)>("hipGetMipmappedArrayLevel");
	out = hip_func(hip_func_args.levelArray, hip_func_args.mipmappedArray, hip_func_args.level);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	levelArray = hip_func_args.levelArray;
	level = hip_func_args.level;

	return out;
};

extern "C" __attribute__((visibility("default")))
int hipGetStreamDeviceId(hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipGetStreamDeviceId;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetStreamDeviceId_api_args_t hip_func_args{stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	int out{};
	static auto hip_func = hipInterceptor.GetHipFunction<int(*)(hipStream_t)>("hipGetStreamDeviceId");
	out = hip_func(hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetSymbolAddress(void * * devPtr, const void * symbol) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetSymbolAddress;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetSymbolAddress_api_args_t hip_func_args{devPtr, symbol};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,const void *)>("hipGetSymbolAddress");
	out = hip_func(hip_func_args.devPtr, hip_func_args.symbol);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	devPtr = hip_func_args.devPtr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetSymbolSize(size_t * size, const void * symbol) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetSymbolSize;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetSymbolSize_api_args_t hip_func_args{size, symbol};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,const void *)>("hipGetSymbolSize");
	out = hip_func(hip_func_args.size, hip_func_args.symbol);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	size = hip_func_args.size;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureAlignmentOffset(size_t * offset, const textureReference * texref) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetTextureAlignmentOffset;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetTextureAlignmentOffset_api_args_t hip_func_args{offset, texref};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,const textureReference *)>("hipGetTextureAlignmentOffset");
	out = hip_func(hip_func_args.offset, hip_func_args.texref);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	offset = hip_func_args.offset;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc * pResDesc, hipTextureObject_t textureObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetTextureObjectResourceDesc;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetTextureObjectResourceDesc_api_args_t hip_func_args{pResDesc, textureObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipResourceDesc *,hipTextureObject_t)>("hipGetTextureObjectResourceDesc");
	out = hip_func(hip_func_args.pResDesc, hip_func_args.textureObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pResDesc = hip_func_args.pResDesc;
	textureObject = hip_func_args.textureObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc * pResViewDesc, hipTextureObject_t textureObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetTextureObjectResourceViewDesc;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetTextureObjectResourceViewDesc_api_args_t hip_func_args{pResViewDesc, textureObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(struct hipResourceViewDesc *,hipTextureObject_t)>("hipGetTextureObjectResourceViewDesc");
	out = hip_func(hip_func_args.pResViewDesc, hip_func_args.textureObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pResViewDesc = hip_func_args.pResViewDesc;
	textureObject = hip_func_args.textureObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc * pTexDesc, hipTextureObject_t textureObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetTextureObjectTextureDesc;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetTextureObjectTextureDesc_api_args_t hip_func_args{pTexDesc, textureObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipTextureDesc *,hipTextureObject_t)>("hipGetTextureObjectTextureDesc");
	out = hip_func(hip_func_args.pTexDesc, hip_func_args.textureObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pTexDesc = hip_func_args.pTexDesc;
	textureObject = hip_func_args.textureObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureReference(const textureReference * * texref, const void * symbol) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGetTextureReference;
	// Copy Arguments for PHASE_ENTER
	hip_hipGetTextureReference_api_args_t hip_func_args{texref, symbol};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const textureReference * *,const void *)>("hipGetTextureReference");
	out = hip_func(hip_func_args.texref, hip_func_args.symbol);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipGraph_t childGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddChildGraphNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddChildGraphNode_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, childGraph};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipGraph_t)>("hipGraphAddChildGraphNode");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.childGraph);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;
	childGraph = hip_func_args.childGraph;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t * from, const hipGraphNode_t * to, size_t numDependencies) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddDependencies;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddDependencies_api_args_t hip_func_args{graph, from, to, numDependencies};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,const hipGraphNode_t *,const hipGraphNode_t *,size_t)>("hipGraphAddDependencies");
	out = hip_func(hip_func_args.graph, hip_func_args.from, hip_func_args.to, hip_func_args.numDependencies);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddEmptyNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddEmptyNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddEmptyNode_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t)>("hipGraphAddEmptyNode");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddEventRecordNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddEventRecordNode_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipEvent_t)>("hipGraphAddEventRecordNode");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddEventWaitNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddEventWaitNode_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipEvent_t)>("hipGraphAddEventWaitNode");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddHostNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipHostNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddHostNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddHostNode_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipHostNodeParams *)>("hipGraphAddHostNode");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddKernelNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipKernelNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddKernelNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddKernelNode_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipKernelNodeParams *)>("hipGraphAddKernelNode");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipMemcpy3DParms * pCopyParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddMemcpyNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddMemcpyNode_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, pCopyParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipMemcpy3DParms *)>("hipGraphAddMemcpyNode");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.pCopyParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddMemcpyNode1D;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddMemcpyNode1D_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNode1D");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.dst, hip_func_args.src, hip_func_args.count, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;
	dst = hip_func_args.dst;
	count = hip_func_args.count;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddMemcpyNodeFromSymbol;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddMemcpyNodeFromSymbol_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNodeFromSymbol");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.dst, hip_func_args.symbol, hip_func_args.count, hip_func_args.offset, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;
	dst = hip_func_args.dst;
	count = hip_func_args.count;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddMemcpyNodeToSymbol;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddMemcpyNodeToSymbol_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNodeToSymbol");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.symbol, hip_func_args.src, hip_func_args.count, hip_func_args.offset, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;
	count = hip_func_args.count;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemsetNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipMemsetParams * pMemsetParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphAddMemsetNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphAddMemsetNode_api_args_t hip_func_args{pGraphNode, graph, pDependencies, numDependencies, pMemsetParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipMemsetParams *)>("hipGraphAddMemsetNode");
	out = hip_func(hip_func_args.pGraphNode, hip_func_args.graph, hip_func_args.pDependencies, hip_func_args.numDependencies, hip_func_args.pMemsetParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphNode = hip_func_args.pGraphNode;
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t * pGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphChildGraphNodeGetGraph;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphChildGraphNodeGetGraph_api_args_t hip_func_args{node, pGraph};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipGraph_t *)>("hipGraphChildGraphNodeGetGraph");
	out = hip_func(hip_func_args.node, hip_func_args.pGraph);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	pGraph = hip_func_args.pGraph;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphClone(hipGraph_t * pGraphClone, hipGraph_t originalGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphClone;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphClone_api_args_t hip_func_args{pGraphClone, originalGraph};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t *,hipGraph_t)>("hipGraphClone");
	out = hip_func(hip_func_args.pGraphClone, hip_func_args.originalGraph);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphClone = hip_func_args.pGraphClone;
	originalGraph = hip_func_args.originalGraph;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphCreate(hipGraph_t * pGraph, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphCreate_api_args_t hip_func_args{pGraph, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t *,unsigned int)>("hipGraphCreate");
	out = hip_func(hip_func_args.pGraph, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraph = hip_func_args.pGraph;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphDestroy(hipGraph_t graph) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphDestroy_api_args_t hip_func_args{graph};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t)>("hipGraphDestroy");
	out = hip_func(hip_func_args.graph);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graph = hip_func_args.graph;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphDestroyNode;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphDestroyNode_api_args_t hip_func_args{node};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t)>("hipGraphDestroyNode");
	out = hip_func(hip_func_args.node);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t * event_out) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphEventRecordNodeGetEvent;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphEventRecordNodeGetEvent_api_args_t hip_func_args{node, event_out};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t *)>("hipGraphEventRecordNodeGetEvent");
	out = hip_func(hip_func_args.node, hip_func_args.event_out);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	event_out = hip_func_args.event_out;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphEventRecordNodeSetEvent;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphEventRecordNodeSetEvent_api_args_t hip_func_args{node, event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t)>("hipGraphEventRecordNodeSetEvent");
	out = hip_func(hip_func_args.node, hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t * event_out) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphEventWaitNodeGetEvent;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphEventWaitNodeGetEvent_api_args_t hip_func_args{node, event_out};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t *)>("hipGraphEventWaitNodeGetEvent");
	out = hip_func(hip_func_args.node, hip_func_args.event_out);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	event_out = hip_func_args.event_out;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphEventWaitNodeSetEvent;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphEventWaitNodeSetEvent_api_args_t hip_func_args{node, event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t)>("hipGraphEventWaitNodeSetEvent");
	out = hip_func(hip_func_args.node, hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipGraph_t childGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecChildGraphNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecChildGraphNodeSetParams_api_args_t hip_func_args{hGraphExec, node, childGraph};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipGraph_t)>("hipGraphExecChildGraphNodeSetParams");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.node, hip_func_args.childGraph);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	node = hip_func_args.node;
	childGraph = hip_func_args.childGraph;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecDestroy_api_args_t hip_func_args{graphExec};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t)>("hipGraphExecDestroy");
	out = hip_func(hip_func_args.graphExec);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graphExec = hip_func_args.graphExec;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecEventRecordNodeSetEvent;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecEventRecordNodeSetEvent_api_args_t hip_func_args{hGraphExec, hNode, event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t)>("hipGraphExecEventRecordNodeSetEvent");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.hNode, hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	hNode = hip_func_args.hNode;
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecEventWaitNodeSetEvent;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecEventWaitNodeSetEvent_api_args_t hip_func_args{hGraphExec, hNode, event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t)>("hipGraphExecEventWaitNodeSetEvent");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.hNode, hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	hNode = hip_func_args.hNode;
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipHostNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecHostNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecHostNodeSetParams_api_args_t hip_func_args{hGraphExec, node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipHostNodeParams *)>("hipGraphExecHostNodeSetParams");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	node = hip_func_args.node;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipKernelNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecKernelNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecKernelNodeSetParams_api_args_t hip_func_args{hGraphExec, node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipKernelNodeParams *)>("hipGraphExecKernelNodeSetParams");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	node = hip_func_args.node;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipMemcpy3DParms * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecMemcpyNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecMemcpyNodeSetParams_api_args_t hip_func_args{hGraphExec, node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipMemcpy3DParms *)>("hipGraphExecMemcpyNodeSetParams");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	node = hip_func_args.node;
	pNodeParams = hip_func_args.pNodeParams;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node, void * dst, const void * src, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecMemcpyNodeSetParams1D;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecMemcpyNodeSetParams1D_api_args_t hip_func_args{hGraphExec, node, dst, src, count, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParams1D");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.node, hip_func_args.dst, hip_func_args.src, hip_func_args.count, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	node = hip_func_args.node;
	dst = hip_func_args.dst;
	count = hip_func_args.count;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecMemcpyNodeSetParamsFromSymbol;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecMemcpyNodeSetParamsFromSymbol_api_args_t hip_func_args{hGraphExec, node, dst, symbol, count, offset, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParamsFromSymbol");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.node, hip_func_args.dst, hip_func_args.symbol, hip_func_args.count, hip_func_args.offset, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	node = hip_func_args.node;
	dst = hip_func_args.dst;
	count = hip_func_args.count;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecMemcpyNodeSetParamsToSymbol;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecMemcpyNodeSetParamsToSymbol_api_args_t hip_func_args{hGraphExec, node, symbol, src, count, offset, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParamsToSymbol");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.node, hip_func_args.symbol, hip_func_args.src, hip_func_args.count, hip_func_args.offset, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	node = hip_func_args.node;
	count = hip_func_args.count;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipMemsetParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecMemsetNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecMemsetNodeSetParams_api_args_t hip_func_args{hGraphExec, node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipMemsetParams *)>("hipGraphExecMemsetNodeSetParams");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	node = hip_func_args.node;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph, hipGraphNode_t * hErrorNode_out, hipGraphExecUpdateResult * updateResult_out) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphExecUpdate;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphExecUpdate_api_args_t hip_func_args{hGraphExec, hGraph, hErrorNode_out, updateResult_out};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipGraph_t,hipGraphNode_t *,hipGraphExecUpdateResult *)>("hipGraphExecUpdate");
	out = hip_func(hip_func_args.hGraphExec, hip_func_args.hGraph, hip_func_args.hErrorNode_out, hip_func_args.updateResult_out);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hGraphExec = hip_func_args.hGraphExec;
	hGraph = hip_func_args.hGraph;
	hErrorNode_out = hip_func_args.hErrorNode_out;
	updateResult_out = hip_func_args.updateResult_out;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t * from, hipGraphNode_t * to, size_t * numEdges) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphGetEdges;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphGetEdges_api_args_t hip_func_args{graph, from, to, numEdges};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,hipGraphNode_t *,size_t *)>("hipGraphGetEdges");
	out = hip_func(hip_func_args.graph, hip_func_args.from, hip_func_args.to, hip_func_args.numEdges);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graph = hip_func_args.graph;
	from = hip_func_args.from;
	to = hip_func_args.to;
	numEdges = hip_func_args.numEdges;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t * nodes, size_t * numNodes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphGetNodes;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphGetNodes_api_args_t hip_func_args{graph, nodes, numNodes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,size_t *)>("hipGraphGetNodes");
	out = hip_func(hip_func_args.graph, hip_func_args.nodes, hip_func_args.numNodes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graph = hip_func_args.graph;
	nodes = hip_func_args.nodes;
	numNodes = hip_func_args.numNodes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t * pRootNodes, size_t * pNumRootNodes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphGetRootNodes;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphGetRootNodes_api_args_t hip_func_args{graph, pRootNodes, pNumRootNodes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,size_t *)>("hipGraphGetRootNodes");
	out = hip_func(hip_func_args.graph, hip_func_args.pRootNodes, hip_func_args.pNumRootNodes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graph = hip_func_args.graph;
	pRootNodes = hip_func_args.pRootNodes;
	pNumRootNodes = hip_func_args.pNumRootNodes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphHostNodeGetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphHostNodeGetParams_api_args_t hip_func_args{node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipHostNodeParams *)>("hipGraphHostNodeGetParams");
	out = hip_func(hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	pNodeParams = hip_func_args.pNodeParams;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphHostNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphHostNodeSetParams_api_args_t hip_func_args{node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const hipHostNodeParams *)>("hipGraphHostNodeSetParams");
	out = hip_func(hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphInstantiate(hipGraphExec_t * pGraphExec, hipGraph_t graph, hipGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphInstantiate;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphInstantiate_api_args_t hip_func_args{pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t *,hipGraph_t,hipGraphNode_t *,char *,size_t)>("hipGraphInstantiate");
	out = hip_func(hip_func_args.pGraphExec, hip_func_args.graph, hip_func_args.pErrorNode, hip_func_args.pLogBuffer, hip_func_args.bufferSize);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphExec = hip_func_args.pGraphExec;
	graph = hip_func_args.graph;
	pErrorNode = hip_func_args.pErrorNode;
	pLogBuffer = hip_func_args.pLogBuffer;
	bufferSize = hip_func_args.bufferSize;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t * pGraphExec, hipGraph_t graph, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphInstantiateWithFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphInstantiateWithFlags_api_args_t hip_func_args{pGraphExec, graph, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t *,hipGraph_t,unsigned long long)>("hipGraphInstantiateWithFlags");
	out = hip_func(hip_func_args.pGraphExec, hip_func_args.graph, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pGraphExec = hip_func_args.pGraphExec;
	graph = hip_func_args.graph;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, hipKernelNodeAttrValue * value) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphKernelNodeGetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphKernelNodeGetAttribute_api_args_t hip_func_args{hNode, attr, value};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeAttrID,hipKernelNodeAttrValue *)>("hipGraphKernelNodeGetAttribute");
	out = hip_func(hip_func_args.hNode, hip_func_args.attr, hip_func_args.value);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hNode = hip_func_args.hNode;
	attr = hip_func_args.attr;
	value = hip_func_args.value;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphKernelNodeGetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphKernelNodeGetParams_api_args_t hip_func_args{node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeParams *)>("hipGraphKernelNodeGetParams");
	out = hip_func(hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	pNodeParams = hip_func_args.pNodeParams;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, const hipKernelNodeAttrValue * value) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphKernelNodeSetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphKernelNodeSetAttribute_api_args_t hip_func_args{hNode, attr, value};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeAttrID,const hipKernelNodeAttrValue *)>("hipGraphKernelNodeSetAttribute");
	out = hip_func(hip_func_args.hNode, hip_func_args.attr, hip_func_args.value);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hNode = hip_func_args.hNode;
	attr = hip_func_args.attr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphKernelNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphKernelNodeSetParams_api_args_t hip_func_args{node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const hipKernelNodeParams *)>("hipGraphKernelNodeSetParams");
	out = hip_func(hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphLaunch;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphLaunch_api_args_t hip_func_args{graphExec, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipStream_t)>("hipGraphLaunch");
	out = hip_func(hip_func_args.graphExec, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graphExec = hip_func_args.graphExec;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphMemcpyNodeGetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphMemcpyNodeGetParams_api_args_t hip_func_args{node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipMemcpy3DParms *)>("hipGraphMemcpyNodeGetParams");
	out = hip_func(hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	pNodeParams = hip_func_args.pNodeParams;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphMemcpyNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphMemcpyNodeSetParams_api_args_t hip_func_args{node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const hipMemcpy3DParms *)>("hipGraphMemcpyNodeSetParams");
	out = hip_func(hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void * dst, const void * src, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphMemcpyNodeSetParams1D;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphMemcpyNodeSetParams1D_api_args_t hip_func_args{node, dst, src, count, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParams1D");
	out = hip_func(hip_func_args.node, hip_func_args.dst, hip_func_args.src, hip_func_args.count, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	dst = hip_func_args.dst;
	count = hip_func_args.count;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphMemcpyNodeSetParamsFromSymbol;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphMemcpyNodeSetParamsFromSymbol_api_args_t hip_func_args{node, dst, symbol, count, offset, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParamsFromSymbol");
	out = hip_func(hip_func_args.node, hip_func_args.dst, hip_func_args.symbol, hip_func_args.count, hip_func_args.offset, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	dst = hip_func_args.dst;
	count = hip_func_args.count;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphMemcpyNodeSetParamsToSymbol;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphMemcpyNodeSetParamsToSymbol_api_args_t hip_func_args{node, symbol, src, count, offset, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParamsToSymbol");
	out = hip_func(hip_func_args.node, hip_func_args.symbol, hip_func_args.src, hip_func_args.count, hip_func_args.offset, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	count = hip_func_args.count;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphMemsetNodeGetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphMemsetNodeGetParams_api_args_t hip_func_args{node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipMemsetParams *)>("hipGraphMemsetNodeGetParams");
	out = hip_func(hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	pNodeParams = hip_func_args.pNodeParams;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphMemsetNodeSetParams;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphMemsetNodeSetParams_api_args_t hip_func_args{node, pNodeParams};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,const hipMemsetParams *)>("hipGraphMemsetNodeSetParams");
	out = hip_func(hip_func_args.node, hip_func_args.pNodeParams);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeFindInClone(hipGraphNode_t * pNode, hipGraphNode_t originalNode, hipGraph_t clonedGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphNodeFindInClone;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphNodeFindInClone_api_args_t hip_func_args{pNode, originalNode, clonedGraph};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraphNode_t,hipGraph_t)>("hipGraphNodeFindInClone");
	out = hip_func(hip_func_args.pNode, hip_func_args.originalNode, hip_func_args.clonedGraph);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pNode = hip_func_args.pNode;
	originalNode = hip_func_args.originalNode;
	clonedGraph = hip_func_args.clonedGraph;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t * pDependencies, size_t * pNumDependencies) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphNodeGetDependencies;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphNodeGetDependencies_api_args_t hip_func_args{node, pDependencies, pNumDependencies};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t *,size_t *)>("hipGraphNodeGetDependencies");
	out = hip_func(hip_func_args.node, hip_func_args.pDependencies, hip_func_args.pNumDependencies);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	pDependencies = hip_func_args.pDependencies;
	pNumDependencies = hip_func_args.pNumDependencies;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t * pDependentNodes, size_t * pNumDependentNodes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphNodeGetDependentNodes;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphNodeGetDependentNodes_api_args_t hip_func_args{node, pDependentNodes, pNumDependentNodes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t *,size_t *)>("hipGraphNodeGetDependentNodes");
	out = hip_func(hip_func_args.node, hip_func_args.pDependentNodes, hip_func_args.pNumDependentNodes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	pDependentNodes = hip_func_args.pDependentNodes;
	pNumDependentNodes = hip_func_args.pNumDependentNodes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType * pType) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphNodeGetType;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphNodeGetType_api_args_t hip_func_args{node, pType};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNodeType *)>("hipGraphNodeGetType");
	out = hip_func(hip_func_args.node, hip_func_args.pType);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	node = hip_func_args.node;
	pType = hip_func_args.pType;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphReleaseUserObject;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphReleaseUserObject_api_args_t hip_func_args{graph, object, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipUserObject_t,unsigned int)>("hipGraphReleaseUserObject");
	out = hip_func(hip_func_args.graph, hip_func_args.object, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graph = hip_func_args.graph;
	object = hip_func_args.object;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t * from, const hipGraphNode_t * to, size_t numDependencies) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphRemoveDependencies;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphRemoveDependencies_api_args_t hip_func_args{graph, from, to, numDependencies};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,const hipGraphNode_t *,const hipGraphNode_t *,size_t)>("hipGraphRemoveDependencies");
	out = hip_func(hip_func_args.graph, hip_func_args.from, hip_func_args.to, hip_func_args.numDependencies);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graph = hip_func_args.graph;
	numDependencies = hip_func_args.numDependencies;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphRetainUserObject;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphRetainUserObject_api_args_t hip_func_args{graph, object, count, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraph_t,hipUserObject_t,unsigned int,unsigned int)>("hipGraphRetainUserObject");
	out = hip_func(hip_func_args.graph, hip_func_args.object, hip_func_args.count, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graph = hip_func_args.graph;
	object = hip_func_args.object;
	count = hip_func_args.count;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphUpload;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphUpload_api_args_t hip_func_args{graphExec, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphExec_t,hipStream_t)>("hipGraphUpload");
	out = hip_func(hip_func_args.graphExec, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	graphExec = hip_func_args.graphExec;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource * * resource, GLuint buffer, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphicsGLRegisterBuffer;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphicsGLRegisterBuffer_api_args_t hip_func_args{resource, buffer, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphicsResource * *,GLuint,unsigned int)>("hipGraphicsGLRegisterBuffer");
	out = hip_func(hip_func_args.resource, hip_func_args.buffer, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	resource = hip_func_args.resource;
	buffer = hip_func_args.buffer;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource * * resource, GLuint image, GLenum target, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphicsGLRegisterImage;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphicsGLRegisterImage_api_args_t hip_func_args{resource, image, target, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphicsResource * *,GLuint,GLenum,unsigned int)>("hipGraphicsGLRegisterImage");
	out = hip_func(hip_func_args.resource, hip_func_args.image, hip_func_args.target, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	resource = hip_func_args.resource;
	image = hip_func_args.image;
	target = hip_func_args.target;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t * resources, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphicsMapResources;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphicsMapResources_api_args_t hip_func_args{count, resources, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipGraphicsResource_t *,hipStream_t)>("hipGraphicsMapResources");
	out = hip_func(hip_func_args.count, hip_func_args.resources, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	count = hip_func_args.count;
	resources = hip_func_args.resources;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsResourceGetMappedPointer(void * * devPtr, size_t * size, hipGraphicsResource_t resource) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphicsResourceGetMappedPointer;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphicsResourceGetMappedPointer_api_args_t hip_func_args{devPtr, size, resource};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t *,hipGraphicsResource_t)>("hipGraphicsResourceGetMappedPointer");
	out = hip_func(hip_func_args.devPtr, hip_func_args.size, hip_func_args.resource);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	devPtr = hip_func_args.devPtr;
	size = hip_func_args.size;
	resource = hip_func_args.resource;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t * array, hipGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphicsSubResourceGetMappedArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphicsSubResourceGetMappedArray_api_args_t hip_func_args{array, resource, arrayIndex, mipLevel};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_t *,hipGraphicsResource_t,unsigned int,unsigned int)>("hipGraphicsSubResourceGetMappedArray");
	out = hip_func(hip_func_args.array, hip_func_args.resource, hip_func_args.arrayIndex, hip_func_args.mipLevel);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	array = hip_func_args.array;
	resource = hip_func_args.resource;
	arrayIndex = hip_func_args.arrayIndex;
	mipLevel = hip_func_args.mipLevel;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t * resources, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphicsUnmapResources;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphicsUnmapResources_api_args_t hip_func_args{count, resources, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int,hipGraphicsResource_t *,hipStream_t)>("hipGraphicsUnmapResources");
	out = hip_func(hip_func_args.count, hip_func_args.resources, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	count = hip_func_args.count;
	resources = hip_func_args.resources;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipGraphicsUnregisterResource;
	// Copy Arguments for PHASE_ENTER
	hip_hipGraphicsUnregisterResource_api_args_t hip_func_args{resource};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipGraphicsResource_t)>("hipGraphicsUnregisterResource");
	out = hip_func(hip_func_args.resource);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	resource = hip_func_args.resource;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, size_t sharedMemBytes, hipStream_t hStream, void * * kernelParams, void * * extra, hipEvent_t startEvent, hipEvent_t stopEvent) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipHccModuleLaunchKernel;
	// Copy Arguments for PHASE_ENTER
	hip_hipHccModuleLaunchKernel_api_args_t hip_func_args{f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t)>("hipHccModuleLaunchKernel");
	out = hip_func(hip_func_args.f, hip_func_args.globalWorkSizeX, hip_func_args.globalWorkSizeY, hip_func_args.globalWorkSizeZ, hip_func_args.blockDimX, hip_func_args.blockDimY, hip_func_args.blockDimZ, hip_func_args.sharedMemBytes, hip_func_args.hStream, hip_func_args.kernelParams, hip_func_args.extra, hip_func_args.startEvent, hip_func_args.stopEvent);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	f = hip_func_args.f;
	globalWorkSizeX = hip_func_args.globalWorkSizeX;
	globalWorkSizeY = hip_func_args.globalWorkSizeY;
	globalWorkSizeZ = hip_func_args.globalWorkSizeZ;
	blockDimX = hip_func_args.blockDimX;
	blockDimY = hip_func_args.blockDimY;
	blockDimZ = hip_func_args.blockDimZ;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	hStream = hip_func_args.hStream;
	kernelParams = hip_func_args.kernelParams;
	extra = hip_func_args.extra;
	startEvent = hip_func_args.startEvent;
	stopEvent = hip_func_args.stopEvent;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipHostAlloc(void * * ptr, size_t size, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipHostAlloc;
	// Copy Arguments for PHASE_ENTER
	hip_hipHostAlloc_api_args_t hip_func_args{ptr, size, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipHostAlloc");
	out = hip_func(hip_func_args.ptr, hip_func_args.size, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipHostFree(void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipHostFree;
	// Copy Arguments for PHASE_ENTER
	hip_hipHostFree_api_args_t hip_func_args{ptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *)>("hipHostFree");
	out = hip_func(hip_func_args.ptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipHostGetDevicePointer(void * * devPtr, void * hstPtr, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipHostGetDevicePointer;
	// Copy Arguments for PHASE_ENTER
	hip_hipHostGetDevicePointer_api_args_t hip_func_args{devPtr, hstPtr, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,void *,unsigned int)>("hipHostGetDevicePointer");
	out = hip_func(hip_func_args.devPtr, hip_func_args.hstPtr, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	devPtr = hip_func_args.devPtr;
	hstPtr = hip_func_args.hstPtr;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipHostGetFlags(unsigned int * flagsPtr, void * hostPtr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipHostGetFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipHostGetFlags_api_args_t hip_func_args{flagsPtr, hostPtr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int *,void *)>("hipHostGetFlags");
	out = hip_func(hip_func_args.flagsPtr, hip_func_args.hostPtr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	flagsPtr = hip_func_args.flagsPtr;
	hostPtr = hip_func_args.hostPtr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipHostMalloc(void * * ptr, size_t size, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipHostMalloc;
	// Copy Arguments for PHASE_ENTER
	hip_hipHostMalloc_api_args_t hip_func_args{ptr, size, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipHostMalloc");
	out = hip_func(hip_func_args.ptr, hip_func_args.size, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipHostRegister(void * hostPtr, size_t sizeBytes, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipHostRegister;
	// Copy Arguments for PHASE_ENTER
	hip_hipHostRegister_api_args_t hip_func_args{hostPtr, sizeBytes, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,unsigned int)>("hipHostRegister");
	out = hip_func(hip_func_args.hostPtr, hip_func_args.sizeBytes, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hostPtr = hip_func_args.hostPtr;
	sizeBytes = hip_func_args.sizeBytes;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipHostUnregister(void * hostPtr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipHostUnregister;
	// Copy Arguments for PHASE_ENTER
	hip_hipHostUnregister_api_args_t hip_func_args{hostPtr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *)>("hipHostUnregister");
	out = hip_func(hip_func_args.hostPtr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hostPtr = hip_func_args.hostPtr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipImportExternalMemory(hipExternalMemory_t * extMem_out, const hipExternalMemoryHandleDesc * memHandleDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipImportExternalMemory;
	// Copy Arguments for PHASE_ENTER
	hip_hipImportExternalMemory_api_args_t hip_func_args{extMem_out, memHandleDesc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipExternalMemory_t *,const hipExternalMemoryHandleDesc *)>("hipImportExternalMemory");
	out = hip_func(hip_func_args.extMem_out, hip_func_args.memHandleDesc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	extMem_out = hip_func_args.extMem_out;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t * extSem_out, const hipExternalSemaphoreHandleDesc * semHandleDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipImportExternalSemaphore;
	// Copy Arguments for PHASE_ENTER
	hip_hipImportExternalSemaphore_api_args_t hip_func_args{extSem_out, semHandleDesc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipExternalSemaphore_t *,const hipExternalSemaphoreHandleDesc *)>("hipImportExternalSemaphore");
	out = hip_func(hip_func_args.extSem_out, hip_func_args.semHandleDesc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	extSem_out = hip_func_args.extSem_out;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipInit(unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipInit;
	// Copy Arguments for PHASE_ENTER
	hip_hipInit_api_args_t hip_func_args{flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int)>("hipInit");
	out = hip_func(hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcCloseMemHandle(void * devPtr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipIpcCloseMemHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipIpcCloseMemHandle_api_args_t hip_func_args{devPtr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *)>("hipIpcCloseMemHandle");
	out = hip_func(hip_func_args.devPtr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	devPtr = hip_func_args.devPtr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t * handle, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipIpcGetEventHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipIpcGetEventHandle_api_args_t hip_func_args{handle, event};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipIpcEventHandle_t *,hipEvent_t)>("hipIpcGetEventHandle");
	out = hip_func(hip_func_args.handle, hip_func_args.event);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	handle = hip_func_args.handle;
	event = hip_func_args.event;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t * handle, void * devPtr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipIpcGetMemHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipIpcGetMemHandle_api_args_t hip_func_args{handle, devPtr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipIpcMemHandle_t *,void *)>("hipIpcGetMemHandle");
	out = hip_func(hip_func_args.handle, hip_func_args.devPtr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	handle = hip_func_args.handle;
	devPtr = hip_func_args.devPtr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcOpenEventHandle(hipEvent_t * event, hipIpcEventHandle_t handle) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipIpcOpenEventHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipIpcOpenEventHandle_api_args_t hip_func_args{event, handle};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipEvent_t *,hipIpcEventHandle_t)>("hipIpcOpenEventHandle");
	out = hip_func(hip_func_args.event, hip_func_args.handle);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	event = hip_func_args.event;
	handle = hip_func_args.handle;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcOpenMemHandle(void * * devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipIpcOpenMemHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipIpcOpenMemHandle_api_args_t hip_func_args{devPtr, handle, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,hipIpcMemHandle_t,unsigned int)>("hipIpcOpenMemHandle");
	out = hip_func(hip_func_args.devPtr, hip_func_args.handle, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	devPtr = hip_func_args.devPtr;
	handle = hip_func_args.handle;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
const char * hipKernelNameRef(const hipFunction_t f) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipKernelNameRef;
	// Copy Arguments for PHASE_ENTER
	hip_hipKernelNameRef_api_args_t hip_func_args{f};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	const char * out{};
	static auto hip_func = hipInterceptor.GetHipFunction<const char *(*)(const hipFunction_t)>("hipKernelNameRef");
	out = hip_func(hip_func_args.f);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
const char * hipKernelNameRefByPtr(const void * hostFunction, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipKernelNameRefByPtr;
	// Copy Arguments for PHASE_ENTER
	hip_hipKernelNameRefByPtr_api_args_t hip_func_args{hostFunction, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	const char * out{};
	static auto hip_func = hipInterceptor.GetHipFunction<const char *(*)(const void *,hipStream_t)>("hipKernelNameRefByPtr");
	out = hip_func(hip_func_args.hostFunction, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchByPtr(const void * func) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipLaunchByPtr;
	// Copy Arguments for PHASE_ENTER
	hip_hipLaunchByPtr_api_args_t hip_func_args{func};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *)>("hipLaunchByPtr");
	out = hip_func(hip_func_args.func);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchCooperativeKernel(const void * f, dim3 gridDim, dim3 blockDimX, void * * kernelParams, unsigned int sharedMemBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipLaunchCooperativeKernel;
	// Copy Arguments for PHASE_ENTER
	hip_hipLaunchCooperativeKernel_api_args_t hip_func_args{f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,unsigned int,hipStream_t)>("hipLaunchCooperativeKernel");
	out = hip_func(hip_func_args.f, hip_func_args.gridDim, hip_func_args.blockDimX, hip_func_args.kernelParams, hip_func_args.sharedMemBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridDim = hip_func_args.gridDim;
	blockDimX = hip_func_args.blockDimX;
	kernelParams = hip_func_args.kernelParams;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams * launchParamsList, int numDevices, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipLaunchCooperativeKernelMultiDevice;
	// Copy Arguments for PHASE_ENTER
	hip_hipLaunchCooperativeKernelMultiDevice_api_args_t hip_func_args{launchParamsList, numDevices, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipLaunchParams *,int,unsigned int)>("hipLaunchCooperativeKernelMultiDevice");
	out = hip_func(hip_func_args.launchParamsList, hip_func_args.numDevices, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	launchParamsList = hip_func_args.launchParamsList;
	numDevices = hip_func_args.numDevices;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void * userData) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipLaunchHostFunc;
	// Copy Arguments for PHASE_ENTER
	hip_hipLaunchHostFunc_api_args_t hip_func_args{stream, fn, userData};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipHostFn_t,void *)>("hipLaunchHostFunc");
	out = hip_func(hip_func_args.stream, hip_func_args.fn, hip_func_args.userData);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	fn = hip_func_args.fn;
	userData = hip_func_args.userData;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchKernel(const void * function_address, dim3 numBlocks, dim3 dimBlocks, void * * args, size_t sharedMemBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipLaunchKernel;
	// Copy Arguments for PHASE_ENTER
	hip_hipLaunchKernel_api_args_t hip_func_args{function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel");
	out = hip_func(hip_func_args.function_address, hip_func_args.numBlocks, hip_func_args.dimBlocks, hip_func_args.args, hip_func_args.sharedMemBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numBlocks = hip_func_args.numBlocks;
	dimBlocks = hip_func_args.dimBlocks;
	args = hip_func_args.args;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchKernel_common(const void * hostFunction, dim3 gridDim, dim3 blockDim, void * * args, size_t sharedMemBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipLaunchKernel_common;
	// Copy Arguments for PHASE_ENTER
	hip_hipLaunchKernel_common_api_args_t hip_func_args{hostFunction, gridDim, blockDim, args, sharedMemBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel_common");
	out = hip_func(hip_func_args.hostFunction, hip_func_args.gridDim, hip_func_args.blockDim, hip_func_args.args, hip_func_args.sharedMemBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridDim = hip_func_args.gridDim;
	blockDim = hip_func_args.blockDim;
	args = hip_func_args.args;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchKernel_spt(const void * hostFunction, dim3 gridDim, dim3 blockDim, void * * args, size_t sharedMemBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hipLaunchKernel_spt;
	// Copy Arguments for PHASE_ENTER
	hip_hipLaunchKernel_spt_api_args_t hip_func_args{hostFunction, gridDim, blockDim, args, sharedMemBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel_spt");
	out = hip_func(hip_func_args.hostFunction, hip_func_args.gridDim, hip_func_args.blockDim, hip_func_args.args, hip_func_args.sharedMemBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridDim = hip_func_args.gridDim;
	blockDim = hip_func_args.blockDim;
	args = hip_func_args.args;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMalloc(void * * ptr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMalloc;
	// Copy Arguments for PHASE_ENTER
	hip_hipMalloc_api_args_t hip_func_args{ptr, size};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t)>("hipMalloc");
	out = hip_func(hip_func_args.ptr, hip_func_args.size);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMalloc3D(hipPitchedPtr * pitchedDevPtr, hipExtent extent) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMalloc3D;
	// Copy Arguments for PHASE_ENTER
	hip_hipMalloc3D_api_args_t hip_func_args{pitchedDevPtr, extent};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipPitchedPtr *,hipExtent)>("hipMalloc3D");
	out = hip_func(hip_func_args.pitchedDevPtr, hip_func_args.extent);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pitchedDevPtr = hip_func_args.pitchedDevPtr;
	extent = hip_func_args.extent;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMalloc3DArray(hipArray * * array, const struct hipChannelFormatDesc * desc, struct hipExtent extent, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMalloc3DArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipMalloc3DArray_api_args_t hip_func_args{array, desc, extent, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray * *,const struct hipChannelFormatDesc *,struct hipExtent,unsigned int)>("hipMalloc3DArray");
	out = hip_func(hip_func_args.array, hip_func_args.desc, hip_func_args.extent, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	array = hip_func_args.array;
	extent = hip_func_args.extent;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocArray(hipArray * * array, const hipChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMallocArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipMallocArray_api_args_t hip_func_args{array, desc, width, height, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray * *,const hipChannelFormatDesc *,size_t,size_t,unsigned int)>("hipMallocArray");
	out = hip_func(hip_func_args.array, hip_func_args.desc, hip_func_args.width, hip_func_args.height, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	array = hip_func_args.array;
	width = hip_func_args.width;
	height = hip_func_args.height;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocAsync(void * * dev_ptr, size_t size, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMallocAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMallocAsync_api_args_t hip_func_args{dev_ptr, size, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t,hipStream_t)>("hipMallocAsync");
	out = hip_func(hip_func_args.dev_ptr, hip_func_args.size, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev_ptr = hip_func_args.dev_ptr;
	size = hip_func_args.size;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocFromPoolAsync(void * * dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMallocFromPoolAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMallocFromPoolAsync_api_args_t hip_func_args{dev_ptr, size, mem_pool, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t,hipMemPool_t,hipStream_t)>("hipMallocFromPoolAsync");
	out = hip_func(hip_func_args.dev_ptr, hip_func_args.size, hip_func_args.mem_pool, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev_ptr = hip_func_args.dev_ptr;
	size = hip_func_args.size;
	mem_pool = hip_func_args.mem_pool;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocHost(void * * ptr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMallocHost;
	// Copy Arguments for PHASE_ENTER
	hip_hipMallocHost_api_args_t hip_func_args{ptr, size};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t)>("hipMallocHost");
	out = hip_func(hip_func_args.ptr, hip_func_args.size);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocManaged(void * * dev_ptr, size_t size, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMallocManaged;
	// Copy Arguments for PHASE_ENTER
	hip_hipMallocManaged_api_args_t hip_func_args{dev_ptr, size, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipMallocManaged");
	out = hip_func(hip_func_args.dev_ptr, hip_func_args.size, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev_ptr = hip_func_args.dev_ptr;
	size = hip_func_args.size;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocMipmappedArray(hipMipmappedArray_t * mipmappedArray, const struct hipChannelFormatDesc * desc, struct hipExtent extent, unsigned int numLevels, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMallocMipmappedArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipMallocMipmappedArray_api_args_t hip_func_args{mipmappedArray, desc, extent, numLevels, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t *,const struct hipChannelFormatDesc *,struct hipExtent,unsigned int,unsigned int)>("hipMallocMipmappedArray");
	out = hip_func(hip_func_args.mipmappedArray, hip_func_args.desc, hip_func_args.extent, hip_func_args.numLevels, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mipmappedArray = hip_func_args.mipmappedArray;
	extent = hip_func_args.extent;
	numLevels = hip_func_args.numLevels;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocPitch(void * * ptr, size_t * pitch, size_t width, size_t height) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMallocPitch;
	// Copy Arguments for PHASE_ENTER
	hip_hipMallocPitch_api_args_t hip_func_args{ptr, pitch, width, height};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t *,size_t,size_t)>("hipMallocPitch");
	out = hip_func(hip_func_args.ptr, hip_func_args.pitch, hip_func_args.width, hip_func_args.height);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	pitch = hip_func_args.pitch;
	width = hip_func_args.width;
	height = hip_func_args.height;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAddressFree(void * devPtr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemAddressFree;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemAddressFree_api_args_t hip_func_args{devPtr, size};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t)>("hipMemAddressFree");
	out = hip_func(hip_func_args.devPtr, hip_func_args.size);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	devPtr = hip_func_args.devPtr;
	size = hip_func_args.size;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAddressReserve(void * * ptr, size_t size, size_t alignment, void * addr, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemAddressReserve;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemAddressReserve_api_args_t hip_func_args{ptr, size, alignment, addr, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t,size_t,void *,unsigned long long)>("hipMemAddressReserve");
	out = hip_func(hip_func_args.ptr, hip_func_args.size, hip_func_args.alignment, hip_func_args.addr, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;
	alignment = hip_func_args.alignment;
	addr = hip_func_args.addr;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAdvise(const void * dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemAdvise;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemAdvise_api_args_t hip_func_args{dev_ptr, count, advice, device};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,size_t,hipMemoryAdvise,int)>("hipMemAdvise");
	out = hip_func(hip_func_args.dev_ptr, hip_func_args.count, hip_func_args.advice, hip_func_args.device);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	count = hip_func_args.count;
	advice = hip_func_args.advice;
	device = hip_func_args.device;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAllocHost(void * * ptr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemAllocHost;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemAllocHost_api_args_t hip_func_args{ptr, size};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t)>("hipMemAllocHost");
	out = hip_func(hip_func_args.ptr, hip_func_args.size);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAllocPitch(hipDeviceptr_t * dptr, size_t * pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemAllocPitch;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemAllocPitch_api_args_t hip_func_args{dptr, pitch, widthInBytes, height, elementSizeBytes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,size_t,size_t,unsigned int)>("hipMemAllocPitch");
	out = hip_func(hip_func_args.dptr, hip_func_args.pitch, hip_func_args.widthInBytes, hip_func_args.height, hip_func_args.elementSizeBytes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dptr = hip_func_args.dptr;
	pitch = hip_func_args.pitch;
	widthInBytes = hip_func_args.widthInBytes;
	height = hip_func_args.height;
	elementSizeBytes = hip_func_args.elementSizeBytes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t * handle, size_t size, const hipMemAllocationProp * prop, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemCreate_api_args_t hip_func_args{handle, size, prop, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,size_t,const hipMemAllocationProp *,unsigned long long)>("hipMemCreate");
	out = hip_func(hip_func_args.handle, hip_func_args.size, hip_func_args.prop, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	handle = hip_func_args.handle;
	size = hip_func_args.size;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemExportToShareableHandle(void * shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemExportToShareableHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemExportToShareableHandle_api_args_t hip_func_args{shareableHandle, handle, handleType, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,hipMemGenericAllocationHandle_t,hipMemAllocationHandleType,unsigned long long)>("hipMemExportToShareableHandle");
	out = hip_func(hip_func_args.shareableHandle, hip_func_args.handle, hip_func_args.handleType, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	shareableHandle = hip_func_args.shareableHandle;
	handle = hip_func_args.handle;
	handleType = hip_func_args.handleType;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetAccess(unsigned long long * flags, const hipMemLocation * location, void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemGetAccess;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemGetAccess_api_args_t hip_func_args{flags, location, ptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned long long *,const hipMemLocation *,void *)>("hipMemGetAccess");
	out = hip_func(hip_func_args.flags, hip_func_args.location, hip_func_args.ptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	flags = hip_func_args.flags;
	ptr = hip_func_args.ptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetAddressRange(hipDeviceptr_t * pbase, size_t * psize, hipDeviceptr_t dptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemGetAddressRange;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemGetAddressRange_api_args_t hip_func_args{pbase, psize, dptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,hipDeviceptr_t)>("hipMemGetAddressRange");
	out = hip_func(hip_func_args.pbase, hip_func_args.psize, hip_func_args.dptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pbase = hip_func_args.pbase;
	psize = hip_func_args.psize;
	dptr = hip_func_args.dptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetAllocationGranularity(size_t * granularity, const hipMemAllocationProp * prop, hipMemAllocationGranularity_flags option) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemGetAllocationGranularity;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemGetAllocationGranularity_api_args_t hip_func_args{granularity, prop, option};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,const hipMemAllocationProp *,hipMemAllocationGranularity_flags)>("hipMemGetAllocationGranularity");
	out = hip_func(hip_func_args.granularity, hip_func_args.prop, hip_func_args.option);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	granularity = hip_func_args.granularity;
	option = hip_func_args.option;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp * prop, hipMemGenericAllocationHandle_t handle) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemGetAllocationPropertiesFromHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemGetAllocationPropertiesFromHandle_api_args_t hip_func_args{prop, handle};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemAllocationProp *,hipMemGenericAllocationHandle_t)>("hipMemGetAllocationPropertiesFromHandle");
	out = hip_func(hip_func_args.prop, hip_func_args.handle);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	prop = hip_func_args.prop;
	handle = hip_func_args.handle;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetInfo(size_t * free, size_t * total) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemGetInfo;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemGetInfo_api_args_t hip_func_args{free, total};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,size_t *)>("hipMemGetInfo");
	out = hip_func(hip_func_args.free, hip_func_args.total);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	free = hip_func_args.free;
	total = hip_func_args.total;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t * handle, void * osHandle, hipMemAllocationHandleType shHandleType) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemImportFromShareableHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemImportFromShareableHandle_api_args_t hip_func_args{handle, osHandle, shHandleType};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,void *,hipMemAllocationHandleType)>("hipMemImportFromShareableHandle");
	out = hip_func(hip_func_args.handle, hip_func_args.osHandle, hip_func_args.shHandleType);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	handle = hip_func_args.handle;
	osHandle = hip_func_args.osHandle;
	shHandleType = hip_func_args.shHandleType;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemMap(void * ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemMap;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemMap_api_args_t hip_func_args{ptr, size, offset, handle, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,size_t,hipMemGenericAllocationHandle_t,unsigned long long)>("hipMemMap");
	out = hip_func(hip_func_args.ptr, hip_func_args.size, hip_func_args.offset, hip_func_args.handle, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;
	offset = hip_func_args.offset;
	handle = hip_func_args.handle;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemMapArrayAsync(hipArrayMapInfo * mapInfoList, unsigned int count, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemMapArrayAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemMapArrayAsync_api_args_t hip_func_args{mapInfoList, count, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArrayMapInfo *,unsigned int,hipStream_t)>("hipMemMapArrayAsync");
	out = hip_func(hip_func_args.mapInfoList, hip_func_args.count, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mapInfoList = hip_func_args.mapInfoList;
	count = hip_func_args.count;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolCreate(hipMemPool_t * mem_pool, const hipMemPoolProps * pool_props) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolCreate_api_args_t hip_func_args{mem_pool, pool_props};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t *,const hipMemPoolProps *)>("hipMemPoolCreate");
	out = hip_func(hip_func_args.mem_pool, hip_func_args.pool_props);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolDestroy_api_args_t hip_func_args{mem_pool};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t)>("hipMemPoolDestroy");
	out = hip_func(hip_func_args.mem_pool);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData * export_data, void * dev_ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolExportPointer;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolExportPointer_api_args_t hip_func_args{export_data, dev_ptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPoolPtrExportData *,void *)>("hipMemPoolExportPointer");
	out = hip_func(hip_func_args.export_data, hip_func_args.dev_ptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	export_data = hip_func_args.export_data;
	dev_ptr = hip_func_args.dev_ptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolExportToShareableHandle(void * shared_handle, hipMemPool_t mem_pool, hipMemAllocationHandleType handle_type, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolExportToShareableHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolExportToShareableHandle_api_args_t hip_func_args{shared_handle, mem_pool, handle_type, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,hipMemPool_t,hipMemAllocationHandleType,unsigned int)>("hipMemPoolExportToShareableHandle");
	out = hip_func(hip_func_args.shared_handle, hip_func_args.mem_pool, hip_func_args.handle_type, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	shared_handle = hip_func_args.shared_handle;
	mem_pool = hip_func_args.mem_pool;
	handle_type = hip_func_args.handle_type;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolGetAccess(hipMemAccessFlags * flags, hipMemPool_t mem_pool, hipMemLocation * location) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolGetAccess;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolGetAccess_api_args_t hip_func_args{flags, mem_pool, location};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemAccessFlags *,hipMemPool_t,hipMemLocation *)>("hipMemPoolGetAccess");
	out = hip_func(hip_func_args.flags, hip_func_args.mem_pool, hip_func_args.location);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	flags = hip_func_args.flags;
	mem_pool = hip_func_args.mem_pool;
	location = hip_func_args.location;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void * value) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolGetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolGetAttribute_api_args_t hip_func_args{mem_pool, attr, value};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t,hipMemPoolAttr,void *)>("hipMemPoolGetAttribute");
	out = hip_func(hip_func_args.mem_pool, hip_func_args.attr, hip_func_args.value);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;
	attr = hip_func_args.attr;
	value = hip_func_args.value;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t * mem_pool, void * shared_handle, hipMemAllocationHandleType handle_type, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolImportFromShareableHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolImportFromShareableHandle_api_args_t hip_func_args{mem_pool, shared_handle, handle_type, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t *,void *,hipMemAllocationHandleType,unsigned int)>("hipMemPoolImportFromShareableHandle");
	out = hip_func(hip_func_args.mem_pool, hip_func_args.shared_handle, hip_func_args.handle_type, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;
	shared_handle = hip_func_args.shared_handle;
	handle_type = hip_func_args.handle_type;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolImportPointer(void * * dev_ptr, hipMemPool_t mem_pool, hipMemPoolPtrExportData * export_data) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolImportPointer;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolImportPointer_api_args_t hip_func_args{dev_ptr, mem_pool, export_data};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,hipMemPool_t,hipMemPoolPtrExportData *)>("hipMemPoolImportPointer");
	out = hip_func(hip_func_args.dev_ptr, hip_func_args.mem_pool, hip_func_args.export_data);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev_ptr = hip_func_args.dev_ptr;
	mem_pool = hip_func_args.mem_pool;
	export_data = hip_func_args.export_data;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc * desc_list, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolSetAccess;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolSetAccess_api_args_t hip_func_args{mem_pool, desc_list, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t,const hipMemAccessDesc *,size_t)>("hipMemPoolSetAccess");
	out = hip_func(hip_func_args.mem_pool, hip_func_args.desc_list, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void * value) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolSetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolSetAttribute_api_args_t hip_func_args{mem_pool, attr, value};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t,hipMemPoolAttr,void *)>("hipMemPoolSetAttribute");
	out = hip_func(hip_func_args.mem_pool, hip_func_args.attr, hip_func_args.value);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;
	attr = hip_func_args.attr;
	value = hip_func_args.value;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPoolTrimTo;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPoolTrimTo_api_args_t hip_func_args{mem_pool, min_bytes_to_hold};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemPool_t,size_t)>("hipMemPoolTrimTo");
	out = hip_func(hip_func_args.mem_pool, hip_func_args.min_bytes_to_hold);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mem_pool = hip_func_args.mem_pool;
	min_bytes_to_hold = hip_func_args.min_bytes_to_hold;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPrefetchAsync(const void * dev_ptr, size_t count, int device, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPrefetchAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPrefetchAsync_api_args_t hip_func_args{dev_ptr, count, device, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,size_t,int,hipStream_t)>("hipMemPrefetchAsync");
	out = hip_func(hip_func_args.dev_ptr, hip_func_args.count, hip_func_args.device, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	count = hip_func_args.count;
	device = hip_func_args.device;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPtrGetInfo(void * ptr, size_t * size) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemPtrGetInfo;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemPtrGetInfo_api_args_t hip_func_args{ptr, size};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t *)>("hipMemPtrGetInfo");
	out = hip_func(hip_func_args.ptr, hip_func_args.size);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemRangeGetAttribute(void * data, size_t data_size, hipMemRangeAttribute attribute, const void * dev_ptr, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemRangeGetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemRangeGetAttribute_api_args_t hip_func_args{data, data_size, attribute, dev_ptr, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,hipMemRangeAttribute,const void *,size_t)>("hipMemRangeGetAttribute");
	out = hip_func(hip_func_args.data, hip_func_args.data_size, hip_func_args.attribute, hip_func_args.dev_ptr, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	data = hip_func_args.data;
	data_size = hip_func_args.data_size;
	attribute = hip_func_args.attribute;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemRangeGetAttributes(void * * data, size_t * data_sizes, hipMemRangeAttribute * attributes, size_t num_attributes, const void * dev_ptr, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemRangeGetAttributes;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemRangeGetAttributes_api_args_t hip_func_args{data, data_sizes, attributes, num_attributes, dev_ptr, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void * *,size_t *,hipMemRangeAttribute *,size_t,const void *,size_t)>("hipMemRangeGetAttributes");
	out = hip_func(hip_func_args.data, hip_func_args.data_sizes, hip_func_args.attributes, hip_func_args.num_attributes, hip_func_args.dev_ptr, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	data = hip_func_args.data;
	data_sizes = hip_func_args.data_sizes;
	attributes = hip_func_args.attributes;
	num_attributes = hip_func_args.num_attributes;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemRelease;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemRelease_api_args_t hip_func_args{handle};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t)>("hipMemRelease");
	out = hip_func(hip_func_args.handle);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	handle = hip_func_args.handle;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t * handle, void * addr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemRetainAllocationHandle;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemRetainAllocationHandle_api_args_t hip_func_args{handle, addr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,void *)>("hipMemRetainAllocationHandle");
	out = hip_func(hip_func_args.handle, hip_func_args.addr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	handle = hip_func_args.handle;
	addr = hip_func_args.addr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemSetAccess(void * ptr, size_t size, const hipMemAccessDesc * desc, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemSetAccess;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemSetAccess_api_args_t hip_func_args{ptr, size, desc, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,const hipMemAccessDesc *,size_t)>("hipMemSetAccess");
	out = hip_func(hip_func_args.ptr, hip_func_args.size, hip_func_args.desc, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemUnmap(void * ptr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemUnmap;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemUnmap_api_args_t hip_func_args{ptr, size};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t)>("hipMemUnmap");
	out = hip_func(hip_func_args.ptr, hip_func_args.size);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ptr = hip_func_args.ptr;
	size = hip_func_args.size;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy_api_args_t hip_func_args{dst, src, sizeBytes, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind)>("hipMemcpy");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	sizeBytes = hip_func_args.sizeBytes;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy2D;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy2D_api_args_t hip_func_args{dst, dpitch, src, spitch, width, height, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2D");
	out = hip_func(hip_func_args.dst, hip_func_args.dpitch, hip_func_args.src, hip_func_args.spitch, hip_func_args.width, hip_func_args.height, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	dpitch = hip_func_args.dpitch;
	spitch = hip_func_args.spitch;
	width = hip_func_args.width;
	height = hip_func_args.height;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy2DAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy2DAsync_api_args_t hip_func_args{dst, dpitch, src, spitch, width, height, kind, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.dpitch, hip_func_args.src, hip_func_args.spitch, hip_func_args.width, hip_func_args.height, hip_func_args.kind, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	dpitch = hip_func_args.dpitch;
	spitch = hip_func_args.spitch;
	width = hip_func_args.width;
	height = hip_func_args.height;
	kind = hip_func_args.kind;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DFromArray(void * dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy2DFromArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy2DFromArray_api_args_t hip_func_args{dst, dpitch, src, wOffset, hOffset, width, height, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,hipArray_const_t,size_t,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2DFromArray");
	out = hip_func(hip_func_args.dst, hip_func_args.dpitch, hip_func_args.src, hip_func_args.wOffset, hip_func_args.hOffset, hip_func_args.width, hip_func_args.height, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	dpitch = hip_func_args.dpitch;
	wOffset = hip_func_args.wOffset;
	hOffset = hip_func_args.hOffset;
	width = hip_func_args.width;
	height = hip_func_args.height;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DFromArrayAsync(void * dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy2DFromArrayAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy2DFromArrayAsync_api_args_t hip_func_args{dst, dpitch, src, wOffset, hOffset, width, height, kind, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,hipArray_const_t,size_t,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DFromArrayAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.dpitch, hip_func_args.src, hip_func_args.wOffset, hip_func_args.hOffset, hip_func_args.width, hip_func_args.height, hip_func_args.kind, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	dpitch = hip_func_args.dpitch;
	wOffset = hip_func_args.wOffset;
	hOffset = hip_func_args.hOffset;
	width = hip_func_args.width;
	height = hip_func_args.height;
	kind = hip_func_args.kind;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DToArray(hipArray * dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy2DToArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy2DToArray_api_args_t hip_func_args{dst, wOffset, hOffset, src, spitch, width, height, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray *,size_t,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2DToArray");
	out = hip_func(hip_func_args.dst, hip_func_args.wOffset, hip_func_args.hOffset, hip_func_args.src, hip_func_args.spitch, hip_func_args.width, hip_func_args.height, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	wOffset = hip_func_args.wOffset;
	hOffset = hip_func_args.hOffset;
	spitch = hip_func_args.spitch;
	width = hip_func_args.width;
	height = hip_func_args.height;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DToArrayAsync(hipArray * dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy2DToArrayAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy2DToArrayAsync_api_args_t hip_func_args{dst, wOffset, hOffset, src, spitch, width, height, kind, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray *,size_t,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DToArrayAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.wOffset, hip_func_args.hOffset, hip_func_args.src, hip_func_args.spitch, hip_func_args.width, hip_func_args.height, hip_func_args.kind, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	wOffset = hip_func_args.wOffset;
	hOffset = hip_func_args.hOffset;
	spitch = hip_func_args.spitch;
	width = hip_func_args.width;
	height = hip_func_args.height;
	kind = hip_func_args.kind;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy3D(const struct hipMemcpy3DParms * p) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy3D;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy3D_api_args_t hip_func_args{p};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const struct hipMemcpy3DParms *)>("hipMemcpy3D");
	out = hip_func(hip_func_args.p);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms * p, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpy3DAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpy3DAsync_api_args_t hip_func_args{p, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const struct hipMemcpy3DParms *,hipStream_t)>("hipMemcpy3DAsync");
	out = hip_func(hip_func_args.p, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyAsync(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyAsync_api_args_t hip_func_args{dst, src, sizeBytes, kind, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes, hip_func_args.kind, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	sizeBytes = hip_func_args.sizeBytes;
	kind = hip_func_args.kind;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyAtoH(void * dst, hipArray * srcArray, size_t srcOffset, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyAtoH;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyAtoH_api_args_t hip_func_args{dst, srcArray, srcOffset, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,hipArray *,size_t,size_t)>("hipMemcpyAtoH");
	out = hip_func(hip_func_args.dst, hip_func_args.srcArray, hip_func_args.srcOffset, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	srcArray = hip_func_args.srcArray;
	srcOffset = hip_func_args.srcOffset;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyDtoD;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyDtoD_api_args_t hip_func_args{dst, src, sizeBytes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,hipDeviceptr_t,size_t)>("hipMemcpyDtoD");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	src = hip_func_args.src;
	sizeBytes = hip_func_args.sizeBytes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyDtoDAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyDtoDAsync_api_args_t hip_func_args{dst, src, sizeBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,hipDeviceptr_t,size_t,hipStream_t)>("hipMemcpyDtoDAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	src = hip_func_args.src;
	sizeBytes = hip_func_args.sizeBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyDtoH(void * dst, hipDeviceptr_t src, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyDtoH;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyDtoH_api_args_t hip_func_args{dst, src, sizeBytes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,hipDeviceptr_t,size_t)>("hipMemcpyDtoH");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	src = hip_func_args.src;
	sizeBytes = hip_func_args.sizeBytes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyDtoHAsync(void * dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyDtoHAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyDtoHAsync_api_args_t hip_func_args{dst, src, sizeBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,hipDeviceptr_t,size_t,hipStream_t)>("hipMemcpyDtoHAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	src = hip_func_args.src;
	sizeBytes = hip_func_args.sizeBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyFromArray(void * dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyFromArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyFromArray_api_args_t hip_func_args{dst, srcArray, wOffset, hOffset, count, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,hipArray_const_t,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpyFromArray");
	out = hip_func(hip_func_args.dst, hip_func_args.srcArray, hip_func_args.wOffset, hip_func_args.hOffset, hip_func_args.count, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	wOffset = hip_func_args.wOffset;
	hOffset = hip_func_args.hOffset;
	count = hip_func_args.count;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyFromSymbol(void * dst, const void * symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyFromSymbol;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyFromSymbol_api_args_t hip_func_args{dst, symbol, sizeBytes, offset, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,const void *,size_t,size_t,hipMemcpyKind)>("hipMemcpyFromSymbol");
	out = hip_func(hip_func_args.dst, hip_func_args.symbol, hip_func_args.sizeBytes, hip_func_args.offset, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	sizeBytes = hip_func_args.sizeBytes;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyFromSymbolAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyFromSymbolAsync_api_args_t hip_func_args{dst, symbol, sizeBytes, offset, kind, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,const void *,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyFromSymbolAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.symbol, hip_func_args.sizeBytes, hip_func_args.offset, hip_func_args.kind, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	sizeBytes = hip_func_args.sizeBytes;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyHtoA(hipArray * dstArray, size_t dstOffset, const void * srcHost, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyHtoA;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyHtoA_api_args_t hip_func_args{dstArray, dstOffset, srcHost, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray *,size_t,const void *,size_t)>("hipMemcpyHtoA");
	out = hip_func(hip_func_args.dstArray, hip_func_args.dstOffset, hip_func_args.srcHost, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dstArray = hip_func_args.dstArray;
	dstOffset = hip_func_args.dstOffset;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void * src, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyHtoD;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyHtoD_api_args_t hip_func_args{dst, src, sizeBytes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,void *,size_t)>("hipMemcpyHtoD");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	src = hip_func_args.src;
	sizeBytes = hip_func_args.sizeBytes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void * src, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyHtoDAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyHtoDAsync_api_args_t hip_func_args{dst, src, sizeBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,void *,size_t,hipStream_t)>("hipMemcpyHtoDAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	src = hip_func_args.src;
	sizeBytes = hip_func_args.sizeBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyParam2D(const hip_Memcpy2D * pCopy) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyParam2D;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyParam2D_api_args_t hip_func_args{pCopy};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hip_Memcpy2D *)>("hipMemcpyParam2D");
	out = hip_func(hip_func_args.pCopy);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D * pCopy, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyParam2DAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyParam2DAsync_api_args_t hip_func_args{pCopy, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hip_Memcpy2D *,hipStream_t)>("hipMemcpyParam2DAsync");
	out = hip_func(hip_func_args.pCopy, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyPeer(void * dst, int dstDeviceId, const void * src, int srcDeviceId, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyPeer;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyPeer_api_args_t hip_func_args{dst, dstDeviceId, src, srcDeviceId, sizeBytes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,int,const void *,int,size_t)>("hipMemcpyPeer");
	out = hip_func(hip_func_args.dst, hip_func_args.dstDeviceId, hip_func_args.src, hip_func_args.srcDeviceId, hip_func_args.sizeBytes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	dstDeviceId = hip_func_args.dstDeviceId;
	srcDeviceId = hip_func_args.srcDeviceId;
	sizeBytes = hip_func_args.sizeBytes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyPeerAsync(void * dst, int dstDeviceId, const void * src, int srcDevice, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyPeerAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyPeerAsync_api_args_t hip_func_args{dst, dstDeviceId, src, srcDevice, sizeBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,int,const void *,int,size_t,hipStream_t)>("hipMemcpyPeerAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.dstDeviceId, hip_func_args.src, hip_func_args.srcDevice, hip_func_args.sizeBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	dstDeviceId = hip_func_args.dstDeviceId;
	srcDevice = hip_func_args.srcDevice;
	sizeBytes = hip_func_args.sizeBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyToArray(hipArray * dst, size_t wOffset, size_t hOffset, const void * src, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyToArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyToArray_api_args_t hip_func_args{dst, wOffset, hOffset, src, count, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray *,size_t,size_t,const void *,size_t,hipMemcpyKind)>("hipMemcpyToArray");
	out = hip_func(hip_func_args.dst, hip_func_args.wOffset, hip_func_args.hOffset, hip_func_args.src, hip_func_args.count, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	wOffset = hip_func_args.wOffset;
	hOffset = hip_func_args.hOffset;
	count = hip_func_args.count;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyToSymbol(const void * symbol, const void * src, size_t sizeBytes, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyToSymbol;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyToSymbol_api_args_t hip_func_args{symbol, src, sizeBytes, offset, kind};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipMemcpyToSymbol");
	out = hip_func(hip_func_args.symbol, hip_func_args.src, hip_func_args.sizeBytes, hip_func_args.offset, hip_func_args.kind);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	sizeBytes = hip_func_args.sizeBytes;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyToSymbolAsync(const void * symbol, const void * src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyToSymbolAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyToSymbolAsync_api_args_t hip_func_args{symbol, src, sizeBytes, offset, kind, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,const void *,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyToSymbolAsync");
	out = hip_func(hip_func_args.symbol, hip_func_args.src, hip_func_args.sizeBytes, hip_func_args.offset, hip_func_args.kind, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	sizeBytes = hip_func_args.sizeBytes;
	offset = hip_func_args.offset;
	kind = hip_func_args.kind;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyWithStream(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemcpyWithStream;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemcpyWithStream_api_args_t hip_func_args{dst, src, sizeBytes, kind, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyWithStream");
	out = hip_func(hip_func_args.dst, hip_func_args.src, hip_func_args.sizeBytes, hip_func_args.kind, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	sizeBytes = hip_func_args.sizeBytes;
	kind = hip_func_args.kind;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset(void * dst, int value, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemset;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemset_api_args_t hip_func_args{dst, value, sizeBytes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,int,size_t)>("hipMemset");
	out = hip_func(hip_func_args.dst, hip_func_args.value, hip_func_args.sizeBytes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	value = hip_func_args.value;
	sizeBytes = hip_func_args.sizeBytes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset2D(void * dst, size_t pitch, int value, size_t width, size_t height) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemset2D;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemset2D_api_args_t hip_func_args{dst, pitch, value, width, height};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,int,size_t,size_t)>("hipMemset2D");
	out = hip_func(hip_func_args.dst, hip_func_args.pitch, hip_func_args.value, hip_func_args.width, hip_func_args.height);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	pitch = hip_func_args.pitch;
	value = hip_func_args.value;
	width = hip_func_args.width;
	height = hip_func_args.height;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset2DAsync(void * dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemset2DAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemset2DAsync_api_args_t hip_func_args{dst, pitch, value, width, height, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,size_t,int,size_t,size_t,hipStream_t)>("hipMemset2DAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.pitch, hip_func_args.value, hip_func_args.width, hip_func_args.height, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	pitch = hip_func_args.pitch;
	value = hip_func_args.value;
	width = hip_func_args.width;
	height = hip_func_args.height;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemset3D;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemset3D_api_args_t hip_func_args{pitchedDevPtr, value, extent};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipPitchedPtr,int,hipExtent)>("hipMemset3D");
	out = hip_func(hip_func_args.pitchedDevPtr, hip_func_args.value, hip_func_args.extent);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pitchedDevPtr = hip_func_args.pitchedDevPtr;
	value = hip_func_args.value;
	extent = hip_func_args.extent;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemset3DAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemset3DAsync_api_args_t hip_func_args{pitchedDevPtr, value, extent, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipPitchedPtr,int,hipExtent,hipStream_t)>("hipMemset3DAsync");
	out = hip_func(hip_func_args.pitchedDevPtr, hip_func_args.value, hip_func_args.extent, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pitchedDevPtr = hip_func_args.pitchedDevPtr;
	value = hip_func_args.value;
	extent = hip_func_args.extent;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetAsync(void * dst, int value, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemsetAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemsetAsync_api_args_t hip_func_args{dst, value, sizeBytes, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,int,size_t,hipStream_t)>("hipMemsetAsync");
	out = hip_func(hip_func_args.dst, hip_func_args.value, hip_func_args.sizeBytes, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	value = hip_func_args.value;
	sizeBytes = hip_func_args.sizeBytes;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemsetD16;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemsetD16_api_args_t hip_func_args{dest, value, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned short,size_t)>("hipMemsetD16");
	out = hip_func(hip_func_args.dest, hip_func_args.value, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dest = hip_func_args.dest;
	value = hip_func_args.value;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemsetD16Async;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemsetD16Async_api_args_t hip_func_args{dest, value, count, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned short,size_t,hipStream_t)>("hipMemsetD16Async");
	out = hip_func(hip_func_args.dest, hip_func_args.value, hip_func_args.count, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dest = hip_func_args.dest;
	value = hip_func_args.value;
	count = hip_func_args.count;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemsetD32;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemsetD32_api_args_t hip_func_args{dest, value, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,int,size_t)>("hipMemsetD32");
	out = hip_func(hip_func_args.dest, hip_func_args.value, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dest = hip_func_args.dest;
	value = hip_func_args.value;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemsetD32Async;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemsetD32Async_api_args_t hip_func_args{dst, value, count, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,int,size_t,hipStream_t)>("hipMemsetD32Async");
	out = hip_func(hip_func_args.dst, hip_func_args.value, hip_func_args.count, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dst = hip_func_args.dst;
	value = hip_func_args.value;
	count = hip_func_args.count;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemsetD8;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemsetD8_api_args_t hip_func_args{dest, value, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned char,size_t)>("hipMemsetD8");
	out = hip_func(hip_func_args.dest, hip_func_args.value, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dest = hip_func_args.dest;
	value = hip_func_args.value;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMemsetD8Async;
	// Copy Arguments for PHASE_ENTER
	hip_hipMemsetD8Async_api_args_t hip_func_args{dest, value, count, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned char,size_t,hipStream_t)>("hipMemsetD8Async");
	out = hip_func(hip_func_args.dest, hip_func_args.value, hip_func_args.count, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dest = hip_func_args.dest;
	value = hip_func_args.value;
	count = hip_func_args.count;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t * pHandle, HIP_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int numMipmapLevels) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMipmappedArrayCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipMipmappedArrayCreate_api_args_t hip_func_args{pHandle, pMipmappedArrayDesc, numMipmapLevels};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t *,HIP_ARRAY3D_DESCRIPTOR *,unsigned int)>("hipMipmappedArrayCreate");
	out = hip_func(hip_func_args.pHandle, hip_func_args.pMipmappedArrayDesc, hip_func_args.numMipmapLevels);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pHandle = hip_func_args.pHandle;
	pMipmappedArrayDesc = hip_func_args.pMipmappedArrayDesc;
	numMipmapLevels = hip_func_args.numMipmapLevels;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMipmappedArrayDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipMipmappedArrayDestroy_api_args_t hip_func_args{hMipmappedArray};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t)>("hipMipmappedArrayDestroy");
	out = hip_func(hip_func_args.hMipmappedArray);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	hMipmappedArray = hip_func_args.hMipmappedArray;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipMipmappedArrayGetLevel(hipArray_t * pLevelArray, hipMipmappedArray_t hMipMappedArray, unsigned int level) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipMipmappedArrayGetLevel;
	// Copy Arguments for PHASE_ENTER
	hip_hipMipmappedArrayGetLevel_api_args_t hip_func_args{pLevelArray, hMipMappedArray, level};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_t *,hipMipmappedArray_t,unsigned int)>("hipMipmappedArrayGetLevel");
	out = hip_func(hip_func_args.pLevelArray, hip_func_args.hMipMappedArray, hip_func_args.level);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pLevelArray = hip_func_args.pLevelArray;
	hMipMappedArray = hip_func_args.hMipMappedArray;
	level = hip_func_args.level;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleGetFunction(hipFunction_t * function, hipModule_t module, const char * kname) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleGetFunction;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleGetFunction_api_args_t hip_func_args{function, module, kname};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFunction_t *,hipModule_t,const char *)>("hipModuleGetFunction");
	out = hip_func(hip_func_args.function, hip_func_args.module, hip_func_args.kname);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	function = hip_func_args.function;
	module = hip_func_args.module;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleGetGlobal(hipDeviceptr_t * dptr, size_t * bytes, hipModule_t hmod, const char * name) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleGetGlobal;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleGetGlobal_api_args_t hip_func_args{dptr, bytes, hmod, name};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,hipModule_t,const char *)>("hipModuleGetGlobal");
	out = hip_func(hip_func_args.dptr, hip_func_args.bytes, hip_func_args.hmod, hip_func_args.name);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dptr = hip_func_args.dptr;
	bytes = hip_func_args.bytes;
	hmod = hip_func_args.hmod;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleGetTexRef(textureReference * * texRef, hipModule_t hmod, const char * name) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleGetTexRef;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleGetTexRef_api_args_t hip_func_args{texRef, hmod, name};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference * *,hipModule_t,const char *)>("hipModuleGetTexRef");
	out = hip_func(hip_func_args.texRef, hip_func_args.hmod, hip_func_args.name);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	hmod = hip_func_args.hmod;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void * * kernelParams, void * * extra) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleLaunchKernel;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleLaunchKernel_api_args_t hip_func_args{f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFunction_t,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,hipStream_t,void * *,void * *)>("hipModuleLaunchKernel");
	out = hip_func(hip_func_args.f, hip_func_args.gridDimX, hip_func_args.gridDimY, hip_func_args.gridDimZ, hip_func_args.blockDimX, hip_func_args.blockDimY, hip_func_args.blockDimZ, hip_func_args.sharedMemBytes, hip_func_args.stream, hip_func_args.kernelParams, hip_func_args.extra);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	f = hip_func_args.f;
	gridDimX = hip_func_args.gridDimX;
	gridDimY = hip_func_args.gridDimY;
	gridDimZ = hip_func_args.gridDimZ;
	blockDimX = hip_func_args.blockDimX;
	blockDimY = hip_func_args.blockDimY;
	blockDimZ = hip_func_args.blockDimZ;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	stream = hip_func_args.stream;
	kernelParams = hip_func_args.kernelParams;
	extra = hip_func_args.extra;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLaunchKernelExt(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, size_t sharedMemBytes, hipStream_t hStream, void * * kernelParams, void * * extra, hipEvent_t startEvent, hipEvent_t stopEvent) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleLaunchKernelExt;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleLaunchKernelExt_api_args_t hip_func_args{f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t)>("hipModuleLaunchKernelExt");
	out = hip_func(hip_func_args.f, hip_func_args.globalWorkSizeX, hip_func_args.globalWorkSizeY, hip_func_args.globalWorkSizeZ, hip_func_args.blockDimX, hip_func_args.blockDimY, hip_func_args.blockDimZ, hip_func_args.sharedMemBytes, hip_func_args.hStream, hip_func_args.kernelParams, hip_func_args.extra, hip_func_args.startEvent, hip_func_args.stopEvent);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	f = hip_func_args.f;
	globalWorkSizeX = hip_func_args.globalWorkSizeX;
	globalWorkSizeY = hip_func_args.globalWorkSizeY;
	globalWorkSizeZ = hip_func_args.globalWorkSizeZ;
	blockDimX = hip_func_args.blockDimX;
	blockDimY = hip_func_args.blockDimY;
	blockDimZ = hip_func_args.blockDimZ;
	sharedMemBytes = hip_func_args.sharedMemBytes;
	hStream = hip_func_args.hStream;
	kernelParams = hip_func_args.kernelParams;
	extra = hip_func_args.extra;
	startEvent = hip_func_args.startEvent;
	stopEvent = hip_func_args.stopEvent;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLoad(hipModule_t * module, const char * fname) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleLoad;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleLoad_api_args_t hip_func_args{module, fname};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipModule_t *,const char *)>("hipModuleLoad");
	out = hip_func(hip_func_args.module, hip_func_args.fname);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	module = hip_func_args.module;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLoadData(hipModule_t * module, const void * image) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleLoadData;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleLoadData_api_args_t hip_func_args{module, image};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipModule_t *,const void *)>("hipModuleLoadData");
	out = hip_func(hip_func_args.module, hip_func_args.image);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	module = hip_func_args.module;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLoadDataEx(hipModule_t * module, const void * image, unsigned int numOptions, hipJitOption * options, void * * optionValues) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleLoadDataEx;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleLoadDataEx_api_args_t hip_func_args{module, image, numOptions, options, optionValues};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipModule_t *,const void *,unsigned int,hipJitOption *,void * *)>("hipModuleLoadDataEx");
	out = hip_func(hip_func_args.module, hip_func_args.image, hip_func_args.numOptions, hip_func_args.options, hip_func_args.optionValues);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	module = hip_func_args.module;
	numOptions = hip_func_args.numOptions;
	options = hip_func_args.options;
	optionValues = hip_func_args.optionValues;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_api_args_t hip_func_args{numBlocks, f, blockSize, dynSharedMemPerBlk};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,hipFunction_t,int,size_t)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
	out = hip_func(hip_func_args.numBlocks, hip_func_args.f, hip_func_args.blockSize, hip_func_args.dynSharedMemPerBlk);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numBlocks = hip_func_args.numBlocks;
	f = hip_func_args.f;
	blockSize = hip_func_args.blockSize;
	dynSharedMemPerBlk = hip_func_args.dynSharedMemPerBlk;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_api_args_t hip_func_args{numBlocks, f, blockSize, dynSharedMemPerBlk, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,hipFunction_t,int,size_t,unsigned int)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
	out = hip_func(hip_func_args.numBlocks, hip_func_args.f, hip_func_args.blockSize, hip_func_args.dynSharedMemPerBlk, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numBlocks = hip_func_args.numBlocks;
	f = hip_func_args.f;
	blockSize = hip_func_args.blockSize;
	dynSharedMemPerBlk = hip_func_args.dynSharedMemPerBlk;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int * gridSize, int * blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleOccupancyMaxPotentialBlockSize_api_args_t hip_func_args{gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,int *,hipFunction_t,size_t,int)>("hipModuleOccupancyMaxPotentialBlockSize");
	out = hip_func(hip_func_args.gridSize, hip_func_args.blockSize, hip_func_args.f, hip_func_args.dynSharedMemPerBlk, hip_func_args.blockSizeLimit);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridSize = hip_func_args.gridSize;
	blockSize = hip_func_args.blockSize;
	f = hip_func_args.f;
	dynSharedMemPerBlk = hip_func_args.dynSharedMemPerBlk;
	blockSizeLimit = hip_func_args.blockSizeLimit;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int * gridSize, int * blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_api_args_t hip_func_args{gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,int *,hipFunction_t,size_t,int,unsigned int)>("hipModuleOccupancyMaxPotentialBlockSizeWithFlags");
	out = hip_func(hip_func_args.gridSize, hip_func_args.blockSize, hip_func_args.f, hip_func_args.dynSharedMemPerBlk, hip_func_args.blockSizeLimit, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridSize = hip_func_args.gridSize;
	blockSize = hip_func_args.blockSize;
	f = hip_func_args.f;
	dynSharedMemPerBlk = hip_func_args.dynSharedMemPerBlk;
	blockSizeLimit = hip_func_args.blockSizeLimit;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleUnload(hipModule_t module) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipModuleUnload;
	// Copy Arguments for PHASE_ENTER
	hip_hipModuleUnload_api_args_t hip_func_args{module};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipModule_t)>("hipModuleUnload");
	out = hip_func(hip_func_args.module);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	module = hip_func_args.module;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * f, int blockSize, size_t dynSharedMemPerBlk) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor;
	// Copy Arguments for PHASE_ENTER
	hip_hipOccupancyMaxActiveBlocksPerMultiprocessor_api_args_t hip_func_args{numBlocks, f, blockSize, dynSharedMemPerBlk};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,const void *,int,size_t)>("hipOccupancyMaxActiveBlocksPerMultiprocessor");
	out = hip_func(hip_func_args.numBlocks, hip_func_args.f, hip_func_args.blockSize, hip_func_args.dynSharedMemPerBlk);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numBlocks = hip_func_args.numBlocks;
	blockSize = hip_func_args.blockSize;
	dynSharedMemPerBlk = hip_func_args.dynSharedMemPerBlk;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_api_args_t hip_func_args{numBlocks, f, blockSize, dynSharedMemPerBlk, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,const void *,int,size_t,unsigned int)>("hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
	out = hip_func(hip_func_args.numBlocks, hip_func_args.f, hip_func_args.blockSize, hip_func_args.dynSharedMemPerBlk, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numBlocks = hip_func_args.numBlocks;
	blockSize = hip_func_args.blockSize;
	dynSharedMemPerBlk = hip_func_args.dynSharedMemPerBlk;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipOccupancyMaxPotentialBlockSize(int * gridSize, int * blockSize, const void * f, size_t dynSharedMemPerBlk, int blockSizeLimit) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipOccupancyMaxPotentialBlockSize;
	// Copy Arguments for PHASE_ENTER
	hip_hipOccupancyMaxPotentialBlockSize_api_args_t hip_func_args{gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,int *,const void *,size_t,int)>("hipOccupancyMaxPotentialBlockSize");
	out = hip_func(hip_func_args.gridSize, hip_func_args.blockSize, hip_func_args.f, hip_func_args.dynSharedMemPerBlk, hip_func_args.blockSizeLimit);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	gridSize = hip_func_args.gridSize;
	blockSize = hip_func_args.blockSize;
	dynSharedMemPerBlk = hip_func_args.dynSharedMemPerBlk;
	blockSizeLimit = hip_func_args.blockSizeLimit;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipPeekAtLastError() {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipPeekAtLastError;
	// Copy Arguments for PHASE_ENTER
	hipUserCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipPeekAtLastError");
	out = hip_func();
	// Exit Callback
	hipUserCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipPointerGetAttribute(void * data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipPointerGetAttribute;
	// Copy Arguments for PHASE_ENTER
	hip_hipPointerGetAttribute_api_args_t hip_func_args{data, attribute, ptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(void *,hipPointer_attribute,hipDeviceptr_t)>("hipPointerGetAttribute");
	out = hip_func(hip_func_args.data, hip_func_args.attribute, hip_func_args.ptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	data = hip_func_args.data;
	attribute = hip_func_args.attribute;
	ptr = hip_func_args.ptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipPointerGetAttributes(hipPointerAttribute_t * attributes, const void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipPointerGetAttributes;
	// Copy Arguments for PHASE_ENTER
	hip_hipPointerGetAttributes_api_args_t hip_func_args{attributes, ptr};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipPointerAttribute_t *,const void *)>("hipPointerGetAttributes");
	out = hip_func(hip_func_args.attributes, hip_func_args.ptr);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	attributes = hip_func_args.attributes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipProfilerStart() {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipProfilerStart;
	// Copy Arguments for PHASE_ENTER
	hipUserCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipProfilerStart");
	out = hip_func();
	// Exit Callback
	hipUserCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipProfilerStop() {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipProfilerStop;
	// Copy Arguments for PHASE_ENTER
	hipUserCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hipProfilerStop");
	out = hip_func();
	// Exit Callback
	hipUserCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipRuntimeGetVersion(int * runtimeVersion) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipRuntimeGetVersion;
	// Copy Arguments for PHASE_ENTER
	hip_hipRuntimeGetVersion_api_args_t hip_func_args{runtimeVersion};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *)>("hipRuntimeGetVersion");
	out = hip_func(hip_func_args.runtimeVersion);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	runtimeVersion = hip_func_args.runtimeVersion;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipSetDevice(int deviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipSetDevice;
	// Copy Arguments for PHASE_ENTER
	hip_hipSetDevice_api_args_t hip_func_args{deviceId};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int)>("hipSetDevice");
	out = hip_func(hip_func_args.deviceId);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	deviceId = hip_func_args.deviceId;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipSetDeviceFlags(unsigned flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipSetDeviceFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipSetDeviceFlags_api_args_t hip_func_args{flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned)>("hipSetDeviceFlags");
	out = hip_func(hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipSetValidDevices(int * device_arr, int len) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipSetValidDevices;
	// Copy Arguments for PHASE_ENTER
	hip_hipSetValidDevices_api_args_t hip_func_args{device_arr, len};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,int)>("hipSetValidDevices");
	out = hip_func(hip_func_args.device_arr, hip_func_args.len);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	device_arr = hip_func_args.device_arr;
	len = hip_func_args.len;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipSetupArgument(const void * arg, size_t size, size_t offset) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipSetupArgument;
	// Copy Arguments for PHASE_ENTER
	hip_hipSetupArgument_api_args_t hip_func_args{arg, size, offset};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const void *,size_t,size_t)>("hipSetupArgument");
	out = hip_func(hip_func_args.arg, hip_func_args.size, hip_func_args.offset);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	size = hip_func_args.size;
	offset = hip_func_args.offset;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t * extSemArray, const hipExternalSemaphoreSignalParams * paramsArray, unsigned int numExtSems, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipSignalExternalSemaphoresAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipSignalExternalSemaphoresAsync_api_args_t hip_func_args{extSemArray, paramsArray, numExtSems, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hipExternalSemaphore_t *,const hipExternalSemaphoreSignalParams *,unsigned int,hipStream_t)>("hipSignalExternalSemaphoresAsync");
	out = hip_func(hip_func_args.extSemArray, hip_func_args.paramsArray, hip_func_args.numExtSems, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numExtSems = hip_func_args.numExtSems;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void * userData, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamAddCallback;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamAddCallback_api_args_t hip_func_args{stream, callback, userData, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCallback_t,void *,unsigned int)>("hipStreamAddCallback");
	out = hip_func(hip_func_args.stream, hip_func_args.callback, hip_func_args.userData, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	callback = hip_func_args.callback;
	userData = hip_func_args.userData;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void * dev_ptr, size_t length, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamAttachMemAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamAttachMemAsync_api_args_t hip_func_args{stream, dev_ptr, length, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void *,size_t,unsigned int)>("hipStreamAttachMemAsync");
	out = hip_func(hip_func_args.stream, hip_func_args.dev_ptr, hip_func_args.length, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	dev_ptr = hip_func_args.dev_ptr;
	length = hip_func_args.length;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamBeginCapture;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamBeginCapture_api_args_t hip_func_args{stream, mode};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureMode)>("hipStreamBeginCapture");
	out = hip_func(hip_func_args.stream, hip_func_args.mode);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	mode = hip_func_args.mode;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamCreate(hipStream_t * stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamCreate_api_args_t hip_func_args{stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t *)>("hipStreamCreate");
	out = hip_func(hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamCreateWithFlags(hipStream_t * stream, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamCreateWithFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamCreateWithFlags_api_args_t hip_func_args{stream, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t *,unsigned int)>("hipStreamCreateWithFlags");
	out = hip_func(hip_func_args.stream, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamCreateWithPriority(hipStream_t * stream, unsigned int flags, int priority) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamCreateWithPriority;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamCreateWithPriority_api_args_t hip_func_args{stream, flags, priority};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t *,unsigned int,int)>("hipStreamCreateWithPriority");
	out = hip_func(hip_func_args.stream, hip_func_args.flags, hip_func_args.priority);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	flags = hip_func_args.flags;
	priority = hip_func_args.priority;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamDestroy(hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamDestroy_api_args_t hip_func_args{stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t)>("hipStreamDestroy");
	out = hip_func(hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t * pGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamEndCapture;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamEndCapture_api_args_t hip_func_args{stream, pGraph};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipGraph_t *)>("hipStreamEndCapture");
	out = hip_func(hip_func_args.stream, hip_func_args.pGraph);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	pGraph = hip_func_args.pGraph;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus * pCaptureStatus, unsigned long long * pId) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamGetCaptureInfo;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamGetCaptureInfo_api_args_t hip_func_args{stream, pCaptureStatus, pId};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *)>("hipStreamGetCaptureInfo");
	out = hip_func(hip_func_args.stream, hip_func_args.pCaptureStatus, hip_func_args.pId);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	pCaptureStatus = hip_func_args.pCaptureStatus;
	pId = hip_func_args.pId;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, hipGraph_t * graph_out, const hipGraphNode_t * * dependencies_out, size_t * numDependencies_out) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamGetCaptureInfo_v2;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamGetCaptureInfo_v2_api_args_t hip_func_args{stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *,hipGraph_t *,const hipGraphNode_t * *,size_t *)>("hipStreamGetCaptureInfo_v2");
	out = hip_func(hip_func_args.stream, hip_func_args.captureStatus_out, hip_func_args.id_out, hip_func_args.graph_out, hip_func_args.dependencies_out, hip_func_args.numDependencies_out);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	captureStatus_out = hip_func_args.captureStatus_out;
	id_out = hip_func_args.id_out;
	graph_out = hip_func_args.graph_out;
	numDependencies_out = hip_func_args.numDependencies_out;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int * flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamGetFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamGetFlags_api_args_t hip_func_args{stream, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,unsigned int *)>("hipStreamGetFlags");
	out = hip_func(hip_func_args.stream, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetPriority(hipStream_t stream, int * priority) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamGetPriority;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamGetPriority_api_args_t hip_func_args{stream, priority};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,int *)>("hipStreamGetPriority");
	out = hip_func(hip_func_args.stream, hip_func_args.priority);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	priority = hip_func_args.priority;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus * pCaptureStatus) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamIsCapturing;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamIsCapturing_api_args_t hip_func_args{stream, pCaptureStatus};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *)>("hipStreamIsCapturing");
	out = hip_func(hip_func_args.stream, hip_func_args.pCaptureStatus);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	pCaptureStatus = hip_func_args.pCaptureStatus;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamQuery(hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamQuery;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamQuery_api_args_t hip_func_args{stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t)>("hipStreamQuery");
	out = hip_func(hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamSynchronize(hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamSynchronize;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamSynchronize_api_args_t hip_func_args{stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t)>("hipStreamSynchronize");
	out = hip_func(hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t * dependencies, size_t numDependencies, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamUpdateCaptureDependencies;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamUpdateCaptureDependencies_api_args_t hip_func_args{stream, dependencies, numDependencies, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipGraphNode_t *,size_t,unsigned int)>("hipStreamUpdateCaptureDependencies");
	out = hip_func(hip_func_args.stream, hip_func_args.dependencies, hip_func_args.numDependencies, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	dependencies = hip_func_args.dependencies;
	numDependencies = hip_func_args.numDependencies;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamWaitEvent;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamWaitEvent_api_args_t hip_func_args{stream, event, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,hipEvent_t,unsigned int)>("hipStreamWaitEvent");
	out = hip_func(hip_func_args.stream, hip_func_args.event, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	event = hip_func_args.event;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWaitValue32(hipStream_t stream, void * ptr, uint32_t value, unsigned int flags, uint32_t mask) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamWaitValue32;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamWaitValue32_api_args_t hip_func_args{stream, ptr, value, flags, mask};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void *,uint32_t,unsigned int,uint32_t)>("hipStreamWaitValue32");
	out = hip_func(hip_func_args.stream, hip_func_args.ptr, hip_func_args.value, hip_func_args.flags, hip_func_args.mask);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	ptr = hip_func_args.ptr;
	value = hip_func_args.value;
	flags = hip_func_args.flags;
	mask = hip_func_args.mask;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWaitValue64(hipStream_t stream, void * ptr, uint64_t value, unsigned int flags, uint64_t mask) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamWaitValue64;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamWaitValue64_api_args_t hip_func_args{stream, ptr, value, flags, mask};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void *,uint64_t,unsigned int,uint64_t)>("hipStreamWaitValue64");
	out = hip_func(hip_func_args.stream, hip_func_args.ptr, hip_func_args.value, hip_func_args.flags, hip_func_args.mask);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	ptr = hip_func_args.ptr;
	value = hip_func_args.value;
	flags = hip_func_args.flags;
	mask = hip_func_args.mask;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWriteValue32(hipStream_t stream, void * ptr, uint32_t value, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamWriteValue32;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamWriteValue32_api_args_t hip_func_args{stream, ptr, value, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void *,uint32_t,unsigned int)>("hipStreamWriteValue32");
	out = hip_func(hip_func_args.stream, hip_func_args.ptr, hip_func_args.value, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	ptr = hip_func_args.ptr;
	value = hip_func_args.value;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWriteValue64(hipStream_t stream, void * ptr, uint64_t value, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipStreamWriteValue64;
	// Copy Arguments for PHASE_ENTER
	hip_hipStreamWriteValue64_api_args_t hip_func_args{stream, ptr, value, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStream_t,void *,uint64_t,unsigned int)>("hipStreamWriteValue64");
	out = hip_func(hip_func_args.stream, hip_func_args.ptr, hip_func_args.value, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	stream = hip_func_args.stream;
	ptr = hip_func_args.ptr;
	value = hip_func_args.value;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectCreate(hipTextureObject_t * pTexObject, const HIP_RESOURCE_DESC * pResDesc, const HIP_TEXTURE_DESC * pTexDesc, const HIP_RESOURCE_VIEW_DESC * pResViewDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexObjectCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexObjectCreate_api_args_t hip_func_args{pTexObject, pResDesc, pTexDesc, pResViewDesc};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipTextureObject_t *,const HIP_RESOURCE_DESC *,const HIP_TEXTURE_DESC *,const HIP_RESOURCE_VIEW_DESC *)>("hipTexObjectCreate");
	out = hip_func(hip_func_args.pTexObject, hip_func_args.pResDesc, hip_func_args.pTexDesc, hip_func_args.pResViewDesc);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pTexObject = hip_func_args.pTexObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexObjectDestroy;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexObjectDestroy_api_args_t hip_func_args{texObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipTextureObject_t)>("hipTexObjectDestroy");
	out = hip_func(hip_func_args.texObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texObject = hip_func_args.texObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC * pResDesc, hipTextureObject_t texObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexObjectGetResourceDesc;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexObjectGetResourceDesc_api_args_t hip_func_args{pResDesc, texObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(HIP_RESOURCE_DESC *,hipTextureObject_t)>("hipTexObjectGetResourceDesc");
	out = hip_func(hip_func_args.pResDesc, hip_func_args.texObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pResDesc = hip_func_args.pResDesc;
	texObject = hip_func_args.texObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC * pResViewDesc, hipTextureObject_t texObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexObjectGetResourceViewDesc;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexObjectGetResourceViewDesc_api_args_t hip_func_args{pResViewDesc, texObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(HIP_RESOURCE_VIEW_DESC *,hipTextureObject_t)>("hipTexObjectGetResourceViewDesc");
	out = hip_func(hip_func_args.pResViewDesc, hip_func_args.texObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pResViewDesc = hip_func_args.pResViewDesc;
	texObject = hip_func_args.texObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC * pTexDesc, hipTextureObject_t texObject) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexObjectGetTextureDesc;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexObjectGetTextureDesc_api_args_t hip_func_args{pTexDesc, texObject};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(HIP_TEXTURE_DESC *,hipTextureObject_t)>("hipTexObjectGetTextureDesc");
	out = hip_func(hip_func_args.pTexDesc, hip_func_args.texObject);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pTexDesc = hip_func_args.pTexDesc;
	texObject = hip_func_args.texObject;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetAddress(hipDeviceptr_t * dev_ptr, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetAddress;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetAddress_api_args_t hip_func_args{dev_ptr, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipDeviceptr_t *,const textureReference *)>("hipTexRefGetAddress");
	out = hip_func(hip_func_args.dev_ptr, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	dev_ptr = hip_func_args.dev_ptr;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode * pam, const textureReference * texRef, int dim) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetAddressMode;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetAddressMode_api_args_t hip_func_args{pam, texRef, dim};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(enum hipTextureAddressMode *,const textureReference *,int)>("hipTexRefGetAddressMode");
	out = hip_func(hip_func_args.pam, hip_func_args.texRef, hip_func_args.dim);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pam = hip_func_args.pam;
	dim = hip_func_args.dim;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetArray(hipArray_t * pArray, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetArray_api_args_t hip_func_args{pArray, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_t *,const textureReference *)>("hipTexRefGetArray");
	out = hip_func(hip_func_args.pArray, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pArray = hip_func_args.pArray;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetBorderColor(float * pBorderColor, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetBorderColor;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetBorderColor_api_args_t hip_func_args{pBorderColor, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(float *,const textureReference *)>("hipTexRefGetBorderColor");
	out = hip_func(hip_func_args.pBorderColor, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pBorderColor = hip_func_args.pBorderColor;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode * pfm, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetFilterMode;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetFilterMode_api_args_t hip_func_args{pfm, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(enum hipTextureFilterMode *,const textureReference *)>("hipTexRefGetFilterMode");
	out = hip_func(hip_func_args.pfm, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pfm = hip_func_args.pfm;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetFlags(unsigned int * pFlags, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetFlags_api_args_t hip_func_args{pFlags, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(unsigned int *,const textureReference *)>("hipTexRefGetFlags");
	out = hip_func(hip_func_args.pFlags, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pFlags = hip_func_args.pFlags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetFormat(hipArray_Format * pFormat, int * pNumChannels, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetFormat;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetFormat_api_args_t hip_func_args{pFormat, pNumChannels, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipArray_Format *,int *,const textureReference *)>("hipTexRefGetFormat");
	out = hip_func(hip_func_args.pFormat, hip_func_args.pNumChannels, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pFormat = hip_func_args.pFormat;
	pNumChannels = hip_func_args.pNumChannels;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMaxAnisotropy(int * pmaxAnsio, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetMaxAnisotropy;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetMaxAnisotropy_api_args_t hip_func_args{pmaxAnsio, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(int *,const textureReference *)>("hipTexRefGetMaxAnisotropy");
	out = hip_func(hip_func_args.pmaxAnsio, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pmaxAnsio = hip_func_args.pmaxAnsio;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t * pArray, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetMipMappedArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetMipMappedArray_api_args_t hip_func_args{pArray, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t *,const textureReference *)>("hipTexRefGetMipMappedArray");
	out = hip_func(hip_func_args.pArray, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pArray = hip_func_args.pArray;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode * pfm, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetMipmapFilterMode;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetMipmapFilterMode_api_args_t hip_func_args{pfm, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(enum hipTextureFilterMode *,const textureReference *)>("hipTexRefGetMipmapFilterMode");
	out = hip_func(hip_func_args.pfm, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pfm = hip_func_args.pfm;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipmapLevelBias(float * pbias, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetMipmapLevelBias;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetMipmapLevelBias_api_args_t hip_func_args{pbias, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(float *,const textureReference *)>("hipTexRefGetMipmapLevelBias");
	out = hip_func(hip_func_args.pbias, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pbias = hip_func_args.pbias;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetMipmapLevelClamp;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetMipmapLevelClamp_api_args_t hip_func_args{pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(float *,float *,const textureReference *)>("hipTexRefGetMipmapLevelClamp");
	out = hip_func(hip_func_args.pminMipmapLevelClamp, hip_func_args.pmaxMipmapLevelClamp, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pminMipmapLevelClamp = hip_func_args.pminMipmapLevelClamp;
	pmaxMipmapLevelClamp = hip_func_args.pmaxMipmapLevelClamp;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipmappedArray(hipMipmappedArray_t * pArray, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefGetMipmappedArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefGetMipmappedArray_api_args_t hip_func_args{pArray, texRef};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipMipmappedArray_t *,const textureReference *)>("hipTexRefGetMipmappedArray");
	out = hip_func(hip_func_args.pArray, hip_func_args.texRef);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	pArray = hip_func_args.pArray;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetAddress(size_t * ByteOffset, textureReference * texRef, hipDeviceptr_t dptr, size_t bytes) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetAddress;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetAddress_api_args_t hip_func_args{ByteOffset, texRef, dptr, bytes};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(size_t *,textureReference *,hipDeviceptr_t,size_t)>("hipTexRefSetAddress");
	out = hip_func(hip_func_args.ByteOffset, hip_func_args.texRef, hip_func_args.dptr, hip_func_args.bytes);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	ByteOffset = hip_func_args.ByteOffset;
	texRef = hip_func_args.texRef;
	dptr = hip_func_args.dptr;
	bytes = hip_func_args.bytes;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetAddress2D(textureReference * texRef, const HIP_ARRAY_DESCRIPTOR * desc, hipDeviceptr_t dptr, size_t Pitch) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetAddress2D;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetAddress2D_api_args_t hip_func_args{texRef, desc, dptr, Pitch};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,const HIP_ARRAY_DESCRIPTOR *,hipDeviceptr_t,size_t)>("hipTexRefSetAddress2D");
	out = hip_func(hip_func_args.texRef, hip_func_args.desc, hip_func_args.dptr, hip_func_args.Pitch);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	dptr = hip_func_args.dptr;
	Pitch = hip_func_args.Pitch;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetAddressMode(textureReference * texRef, int dim, enum hipTextureAddressMode am) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetAddressMode;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetAddressMode_api_args_t hip_func_args{texRef, dim, am};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,int,enum hipTextureAddressMode)>("hipTexRefSetAddressMode");
	out = hip_func(hip_func_args.texRef, hip_func_args.dim, hip_func_args.am);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	dim = hip_func_args.dim;
	am = hip_func_args.am;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetArray(textureReference * tex, hipArray_const_t array, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetArray_api_args_t hip_func_args{tex, array, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,hipArray_const_t,unsigned int)>("hipTexRefSetArray");
	out = hip_func(hip_func_args.tex, hip_func_args.array, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	tex = hip_func_args.tex;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetBorderColor(textureReference * texRef, float * pBorderColor) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetBorderColor;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetBorderColor_api_args_t hip_func_args{texRef, pBorderColor};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,float *)>("hipTexRefSetBorderColor");
	out = hip_func(hip_func_args.texRef, hip_func_args.pBorderColor);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	pBorderColor = hip_func_args.pBorderColor;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetFilterMode(textureReference * texRef, enum hipTextureFilterMode fm) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetFilterMode;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetFilterMode_api_args_t hip_func_args{texRef, fm};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,enum hipTextureFilterMode)>("hipTexRefSetFilterMode");
	out = hip_func(hip_func_args.texRef, hip_func_args.fm);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	fm = hip_func_args.fm;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetFlags(textureReference * texRef, unsigned int Flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetFlags;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetFlags_api_args_t hip_func_args{texRef, Flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,unsigned int)>("hipTexRefSetFlags");
	out = hip_func(hip_func_args.texRef, hip_func_args.Flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	Flags = hip_func_args.Flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetFormat(textureReference * texRef, hipArray_Format fmt, int NumPackedComponents) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetFormat;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetFormat_api_args_t hip_func_args{texRef, fmt, NumPackedComponents};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,hipArray_Format,int)>("hipTexRefSetFormat");
	out = hip_func(hip_func_args.texRef, hip_func_args.fmt, hip_func_args.NumPackedComponents);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	fmt = hip_func_args.fmt;
	NumPackedComponents = hip_func_args.NumPackedComponents;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMaxAnisotropy(textureReference * texRef, unsigned int maxAniso) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetMaxAnisotropy;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetMaxAnisotropy_api_args_t hip_func_args{texRef, maxAniso};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,unsigned int)>("hipTexRefSetMaxAnisotropy");
	out = hip_func(hip_func_args.texRef, hip_func_args.maxAniso);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	maxAniso = hip_func_args.maxAniso;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMipmapFilterMode(textureReference * texRef, enum hipTextureFilterMode fm) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetMipmapFilterMode;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetMipmapFilterMode_api_args_t hip_func_args{texRef, fm};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,enum hipTextureFilterMode)>("hipTexRefSetMipmapFilterMode");
	out = hip_func(hip_func_args.texRef, hip_func_args.fm);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	fm = hip_func_args.fm;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMipmapLevelBias(textureReference * texRef, float bias) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetMipmapLevelBias;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetMipmapLevelBias_api_args_t hip_func_args{texRef, bias};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,float)>("hipTexRefSetMipmapLevelBias");
	out = hip_func(hip_func_args.texRef, hip_func_args.bias);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	bias = hip_func_args.bias;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMipmapLevelClamp(textureReference * texRef, float minMipMapLevelClamp, float maxMipMapLevelClamp) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetMipmapLevelClamp;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetMipmapLevelClamp_api_args_t hip_func_args{texRef, minMipMapLevelClamp, maxMipMapLevelClamp};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,float,float)>("hipTexRefSetMipmapLevelClamp");
	out = hip_func(hip_func_args.texRef, hip_func_args.minMipMapLevelClamp, hip_func_args.maxMipMapLevelClamp);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	minMipMapLevelClamp = hip_func_args.minMipMapLevelClamp;
	maxMipMapLevelClamp = hip_func_args.maxMipMapLevelClamp;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMipmappedArray(textureReference * texRef, struct hipMipmappedArray * mipmappedArray, unsigned int Flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipTexRefSetMipmappedArray;
	// Copy Arguments for PHASE_ENTER
	hip_hipTexRefSetMipmappedArray_api_args_t hip_func_args{texRef, mipmappedArray, Flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(textureReference *,struct hipMipmappedArray *,unsigned int)>("hipTexRefSetMipmappedArray");
	out = hip_func(hip_func_args.texRef, hip_func_args.mipmappedArray, hip_func_args.Flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	texRef = hip_func_args.texRef;
	mipmappedArray = hip_func_args.mipmappedArray;
	Flags = hip_func_args.Flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode * mode) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipThreadExchangeStreamCaptureMode;
	// Copy Arguments for PHASE_ENTER
	hip_hipThreadExchangeStreamCaptureMode_api_args_t hip_func_args{mode};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipStreamCaptureMode *)>("hipThreadExchangeStreamCaptureMode");
	out = hip_func(hip_func_args.mode);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	mode = hip_func_args.mode;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipUnbindTexture(const textureReference * tex) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipUnbindTexture;
	// Copy Arguments for PHASE_ENTER
	hip_hipUnbindTexture_api_args_t hip_func_args{tex};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const textureReference *)>("hipUnbindTexture");
	out = hip_func(hip_func_args.tex);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipUserObjectCreate(hipUserObject_t * object_out, void * ptr, hipHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipUserObjectCreate;
	// Copy Arguments for PHASE_ENTER
	hip_hipUserObjectCreate_api_args_t hip_func_args{object_out, ptr, destroy, initialRefcount, flags};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipUserObject_t *,void *,hipHostFn_t,unsigned int,unsigned int)>("hipUserObjectCreate");
	out = hip_func(hip_func_args.object_out, hip_func_args.ptr, hip_func_args.destroy, hip_func_args.initialRefcount, hip_func_args.flags);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	object_out = hip_func_args.object_out;
	ptr = hip_func_args.ptr;
	destroy = hip_func_args.destroy;
	initialRefcount = hip_func_args.initialRefcount;
	flags = hip_func_args.flags;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipUserObjectRelease;
	// Copy Arguments for PHASE_ENTER
	hip_hipUserObjectRelease_api_args_t hip_func_args{object, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipUserObject_t,unsigned int)>("hipUserObjectRelease");
	out = hip_func(hip_func_args.object, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	object = hip_func_args.object;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipUserObjectRetain;
	// Copy Arguments for PHASE_ENTER
	hip_hipUserObjectRetain_api_args_t hip_func_args{object, count};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(hipUserObject_t,unsigned int)>("hipUserObjectRetain");
	out = hip_func(hip_func_args.object, hip_func_args.count);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	object = hip_func_args.object;
	count = hip_func_args.count;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t * extSemArray, const hipExternalSemaphoreWaitParams * paramsArray, unsigned int numExtSems, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_API_ID_hipWaitExternalSemaphoresAsync;
	// Copy Arguments for PHASE_ENTER
	hip_hipWaitExternalSemaphoresAsync_api_args_t hip_func_args{extSemArray, paramsArray, numExtSems, stream};
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)(const hipExternalSemaphore_t *,const hipExternalSemaphoreWaitParams *,unsigned int,hipStream_t)>("hipWaitExternalSemaphoresAsync");
	out = hip_func(hip_func_args.extSemArray, hip_func_args.paramsArray, hip_func_args.numExtSems, hip_func_args.stream);
	// Exit Callback
	hipUserCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(static_cast<void*>(&hip_func_args), LUTHIER_API_PHASE_EXIT, api_id);
	// Copy the modified arguments back to the original arguments (if non-const)
	numExtSems = hip_func_args.numExtSems;
	stream = hip_func_args.stream;

	return out;
};

extern "C" __attribute__((visibility("default")))
hipError_t hip_init() {
	auto& hipInterceptor = luthier::HipInterceptor::Instance();
	auto& hipUserCallback = hipInterceptor.getUserCallback();
	auto& hipInternalCallback = hipInterceptor.getInternalCallback();
	auto api_id = HIP_PRIVATE_API_ID_hip_init;
	// Copy Arguments for PHASE_ENTER
	hipUserCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_ENTER, api_id);
	hipError_t out{};
	static auto hip_func = hipInterceptor.GetHipFunction<hipError_t(*)()>("hip_init");
	out = hip_func();
	// Exit Callback
	hipUserCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);
	hipInternalCallback(nullptr, LUTHIER_API_PHASE_EXIT, api_id);

	return out;
};

