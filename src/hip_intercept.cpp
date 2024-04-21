#include <link.h>
#include "hip_intercept.hpp"


luthier::HipInterceptor::HipInterceptor() {
    // Iterate through the process' loaded shared objects and try to dlopen the first entry with a
    // file name starting with the given 'pattern'. This allows the loader to acquire a handle
    // to the target library iff it is already loaded. The handle is used to query symbols
    // exported by that library.
    auto callback = [this](dl_phdr_info *info) {
    if (Handle == nullptr && fs::path(info->dlpi_name).filename().string().rfind("libamdhip64.so", 0) == 0)
      Handle = ::dlopen(info->dlpi_name, RTLD_LAZY);
    };
    dl_iterate_phdr(
        [](dl_phdr_info *info, size_t size, void *data) {
            (*reinterpret_cast<decltype(callback) *>(data))(info);
            return 0;
        }, &callback);
};

extern "C" __attribute__((visibility("default")))
void __hipGetPCH(const char * * pch, unsigned int * size) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID___hipGetPCH;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipGetPCH_api_args_t hipFuncArgs{pch, size};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<void(*)(const char * *,unsigned int *)>("__hipGetPCH");
			hipFunc(hipFuncArgs.pch, hipFuncArgs.size);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		size = hipFuncArgs.size;
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<void(*)(const char * *,unsigned int *)>("__hipGetPCH");
		hipFunc(pch, size);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t __hipPopCallConfiguration(dim3 * gridDim, dim3 * blockDim, size_t * sharedMem, hipStream_t * stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID___hipPopCallConfiguration;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipPopCallConfiguration_api_args_t hipFuncArgs{gridDim, blockDim, sharedMem, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(dim3 *,dim3 *,size_t *,hipStream_t *)>("__hipPopCallConfiguration");
			out = hipFunc(hipFuncArgs.gridDim, hipFuncArgs.blockDim, hipFuncArgs.sharedMem, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridDim = hipFuncArgs.gridDim;
		blockDim = hipFuncArgs.blockDim;
		sharedMem = hipFuncArgs.sharedMem;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(dim3 *,dim3 *,size_t *,hipStream_t *)>("__hipPopCallConfiguration");
		return hipFunc(gridDim, blockDim, sharedMem, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID___hipPushCallConfiguration;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipPushCallConfiguration_api_args_t hipFuncArgs{gridDim, blockDim, sharedMem, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(dim3,dim3,size_t,hipStream_t)>("__hipPushCallConfiguration");
			out = hipFunc(hipFuncArgs.gridDim, hipFuncArgs.blockDim, hipFuncArgs.sharedMem, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridDim = hipFuncArgs.gridDim;
		blockDim = hipFuncArgs.blockDim;
		sharedMem = hipFuncArgs.sharedMem;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(dim3,dim3,size_t,hipStream_t)>("__hipPushCallConfiguration");
		return hipFunc(gridDim, blockDim, sharedMem, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hip::FatBinaryInfo * * __hipRegisterFatBinary(const void * data) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID___hipRegisterFatBinary;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipRegisterFatBinary_api_args_t hipFuncArgs{data};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hip::FatBinaryInfo * *(*)(const void *)>("__hipRegisterFatBinary");
			out = hipFunc(hipFuncArgs.data);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hip::FatBinaryInfo * *>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hip::FatBinaryInfo * *(*)(const void *)>("__hipRegisterFatBinary");
		return hipFunc(data);
	};
}

extern "C" __attribute__((visibility("default")))
void __hipRegisterFunction(hip::FatBinaryInfo * * modules, const void * hostFunction, char * deviceFunction, const char * deviceName, unsigned int threadLimit, uint3 * tid, uint3 * bid, dim3 * blockDim, dim3 * gridDim, int * wSize) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID___hipRegisterFunction;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipRegisterFunction_api_args_t hipFuncArgs{modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim, wSize};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *,const void *,char *,const char *,unsigned int,uint3 *,uint3 *,dim3 *,dim3 *,int *)>("__hipRegisterFunction");
			hipFunc(hipFuncArgs.modules, hipFuncArgs.hostFunction, hipFuncArgs.deviceFunction, hipFuncArgs.deviceName, hipFuncArgs.threadLimit, hipFuncArgs.tid, hipFuncArgs.bid, hipFuncArgs.blockDim, hipFuncArgs.gridDim, hipFuncArgs.wSize);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		modules = hipFuncArgs.modules;
		deviceFunction = hipFuncArgs.deviceFunction;
		threadLimit = hipFuncArgs.threadLimit;
		tid = hipFuncArgs.tid;
		bid = hipFuncArgs.bid;
		blockDim = hipFuncArgs.blockDim;
		gridDim = hipFuncArgs.gridDim;
		wSize = hipFuncArgs.wSize;
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *,const void *,char *,const char *,unsigned int,uint3 *,uint3 *,dim3 *,dim3 *,int *)>("__hipRegisterFunction");
		hipFunc(modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim, wSize);
	};
}

extern "C" __attribute__((visibility("default")))
void __hipRegisterManagedVar(void * hipModule, void * * pointer, void * init_value, const char * name, size_t size, unsigned align) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID___hipRegisterManagedVar;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipRegisterManagedVar_api_args_t hipFuncArgs{hipModule, pointer, init_value, name, size, align};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<void(*)(void *,void * *,void *,const char *,size_t,unsigned)>("__hipRegisterManagedVar");
			hipFunc(hipFuncArgs.hipModule, hipFuncArgs.pointer, hipFuncArgs.init_value, hipFuncArgs.name, hipFuncArgs.size, hipFuncArgs.align);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hipModule = hipFuncArgs.hipModule;
		pointer = hipFuncArgs.pointer;
		init_value = hipFuncArgs.init_value;
		size = hipFuncArgs.size;
		align = hipFuncArgs.align;
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<void(*)(void *,void * *,void *,const char *,size_t,unsigned)>("__hipRegisterManagedVar");
		hipFunc(hipModule, pointer, init_value, name, size, align);
	};
}

extern "C" __attribute__((visibility("default")))
void __hipRegisterSurface(hip::FatBinaryInfo * * modules, void * var, char * hostVar, char * deviceVar, int type, int ext) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID___hipRegisterSurface;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipRegisterSurface_api_args_t hipFuncArgs{modules, var, hostVar, deviceVar, type, ext};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,int)>("__hipRegisterSurface");
			hipFunc(hipFuncArgs.modules, hipFuncArgs.var, hipFuncArgs.hostVar, hipFuncArgs.deviceVar, hipFuncArgs.type, hipFuncArgs.ext);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		modules = hipFuncArgs.modules;
		var = hipFuncArgs.var;
		hostVar = hipFuncArgs.hostVar;
		deviceVar = hipFuncArgs.deviceVar;
		type = hipFuncArgs.type;
		ext = hipFuncArgs.ext;
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,int)>("__hipRegisterSurface");
		hipFunc(modules, var, hostVar, deviceVar, type, ext);
	};
}

extern "C" __attribute__((visibility("default")))
void __hipRegisterTexture(hip::FatBinaryInfo * * modules, void * var, char * hostVar, char * deviceVar, int type, int norm, int ext) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID___hipRegisterTexture;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipRegisterTexture_api_args_t hipFuncArgs{modules, var, hostVar, deviceVar, type, norm, ext};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,int,int)>("__hipRegisterTexture");
			hipFunc(hipFuncArgs.modules, hipFuncArgs.var, hipFuncArgs.hostVar, hipFuncArgs.deviceVar, hipFuncArgs.type, hipFuncArgs.norm, hipFuncArgs.ext);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		modules = hipFuncArgs.modules;
		var = hipFuncArgs.var;
		hostVar = hipFuncArgs.hostVar;
		deviceVar = hipFuncArgs.deviceVar;
		type = hipFuncArgs.type;
		norm = hipFuncArgs.norm;
		ext = hipFuncArgs.ext;
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,int,int)>("__hipRegisterTexture");
		hipFunc(modules, var, hostVar, deviceVar, type, norm, ext);
	};
}

extern "C" __attribute__((visibility("default")))
void __hipRegisterVar(hip::FatBinaryInfo * * modules, void * var, char * hostVar, char * deviceVar, int ext, size_t size, int constant, int global) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID___hipRegisterVar;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipRegisterVar_api_args_t hipFuncArgs{modules, var, hostVar, deviceVar, ext, size, constant, global};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,size_t,int,int)>("__hipRegisterVar");
			hipFunc(hipFuncArgs.modules, hipFuncArgs.var, hipFuncArgs.hostVar, hipFuncArgs.deviceVar, hipFuncArgs.ext, hipFuncArgs.size, hipFuncArgs.constant, hipFuncArgs.global);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		modules = hipFuncArgs.modules;
		var = hipFuncArgs.var;
		hostVar = hipFuncArgs.hostVar;
		deviceVar = hipFuncArgs.deviceVar;
		ext = hipFuncArgs.ext;
		size = hipFuncArgs.size;
		constant = hipFuncArgs.constant;
		global = hipFuncArgs.global;
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *,void *,char *,char *,int,size_t,int,int)>("__hipRegisterVar");
		hipFunc(modules, var, hostVar, deviceVar, ext, size, constant, global);
	};
}

extern "C" __attribute__((visibility("default")))
void __hipUnregisterFatBinary(hip::FatBinaryInfo * * modules) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID___hipUnregisterFatBinary;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip___hipUnregisterFatBinary_api_args_t hipFuncArgs{modules};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *)>("__hipUnregisterFatBinary");
			hipFunc(hipFuncArgs.modules);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		modules = hipFuncArgs.modules;
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<void(*)(hip::FatBinaryInfo * *)>("__hipUnregisterFatBinary");
		hipFunc(modules);
	};
}

extern "C" __attribute__((visibility("default")))
const char * hipApiName(uint32_t id) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipApiName;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipApiName_api_args_t hipFuncArgs{id};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(uint32_t)>("hipApiName");
			out = hipFunc(hipFuncArgs.id);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		id = hipFuncArgs.id;

		return std::any_cast<const char *>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(uint32_t)>("hipApiName");
		return hipFunc(id);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipArray3DCreate(hipArray_t * array, const HIP_ARRAY3D_DESCRIPTOR * pAllocateArray) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipArray3DCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipArray3DCreate_api_args_t hipFuncArgs{array, pAllocateArray};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const HIP_ARRAY3D_DESCRIPTOR *)>("hipArray3DCreate");
			out = hipFunc(hipFuncArgs.array, hipFuncArgs.pAllocateArray);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		array = hipFuncArgs.array;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const HIP_ARRAY3D_DESCRIPTOR *)>("hipArray3DCreate");
		return hipFunc(array, pAllocateArray);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR * pArrayDescriptor, hipArray_t array) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipArray3DGetDescriptor;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipArray3DGetDescriptor_api_args_t hipFuncArgs{pArrayDescriptor, array};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_ARRAY3D_DESCRIPTOR *,hipArray_t)>("hipArray3DGetDescriptor");
			out = hipFunc(hipFuncArgs.pArrayDescriptor, hipFuncArgs.array);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pArrayDescriptor = hipFuncArgs.pArrayDescriptor;
		array = hipFuncArgs.array;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_ARRAY3D_DESCRIPTOR *,hipArray_t)>("hipArray3DGetDescriptor");
		return hipFunc(pArrayDescriptor, array);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipArrayCreate(hipArray_t * pHandle, const HIP_ARRAY_DESCRIPTOR * pAllocateArray) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipArrayCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipArrayCreate_api_args_t hipFuncArgs{pHandle, pAllocateArray};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const HIP_ARRAY_DESCRIPTOR *)>("hipArrayCreate");
			out = hipFunc(hipFuncArgs.pHandle, hipFuncArgs.pAllocateArray);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pHandle = hipFuncArgs.pHandle;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const HIP_ARRAY_DESCRIPTOR *)>("hipArrayCreate");
		return hipFunc(pHandle, pAllocateArray);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipArrayDestroy(hipArray_t array) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipArrayDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipArrayDestroy_api_args_t hipFuncArgs{array};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t)>("hipArrayDestroy");
			out = hipFunc(hipFuncArgs.array);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		array = hipFuncArgs.array;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t)>("hipArrayDestroy");
		return hipFunc(array);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR * pArrayDescriptor, hipArray_t array) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipArrayGetDescriptor;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipArrayGetDescriptor_api_args_t hipFuncArgs{pArrayDescriptor, array};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_ARRAY_DESCRIPTOR *,hipArray_t)>("hipArrayGetDescriptor");
			out = hipFunc(hipFuncArgs.pArrayDescriptor, hipFuncArgs.array);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pArrayDescriptor = hipFuncArgs.pArrayDescriptor;
		array = hipFuncArgs.array;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_ARRAY_DESCRIPTOR *,hipArray_t)>("hipArrayGetDescriptor");
		return hipFunc(pArrayDescriptor, array);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipArrayGetInfo(hipChannelFormatDesc * desc, hipExtent * extent, unsigned int * flags, hipArray_t array) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipArrayGetInfo;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipArrayGetInfo_api_args_t hipFuncArgs{desc, extent, flags, array};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipChannelFormatDesc *,hipExtent *,unsigned int *,hipArray_t)>("hipArrayGetInfo");
			out = hipFunc(hipFuncArgs.desc, hipFuncArgs.extent, hipFuncArgs.flags, hipFuncArgs.array);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		desc = hipFuncArgs.desc;
		extent = hipFuncArgs.extent;
		flags = hipFuncArgs.flags;
		array = hipFuncArgs.array;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipChannelFormatDesc *,hipExtent *,unsigned int *,hipArray_t)>("hipArrayGetInfo");
		return hipFunc(desc, extent, flags, array);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipBindTexture(size_t * offset, const textureReference * tex, const void * devPtr, const hipChannelFormatDesc * desc, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipBindTexture;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipBindTexture_api_args_t hipFuncArgs{offset, tex, devPtr, desc, size};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const textureReference *,const void *,const hipChannelFormatDesc *,size_t)>("hipBindTexture");
			out = hipFunc(hipFuncArgs.offset, hipFuncArgs.tex, hipFuncArgs.devPtr, hipFuncArgs.desc, hipFuncArgs.size);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		offset = hipFuncArgs.offset;
		size = hipFuncArgs.size;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const textureReference *,const void *,const hipChannelFormatDesc *,size_t)>("hipBindTexture");
		return hipFunc(offset, tex, devPtr, desc, size);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipBindTexture2D(size_t * offset, const textureReference * tex, const void * devPtr, const hipChannelFormatDesc * desc, size_t width, size_t height, size_t pitch) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipBindTexture2D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipBindTexture2D_api_args_t hipFuncArgs{offset, tex, devPtr, desc, width, height, pitch};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const textureReference *,const void *,const hipChannelFormatDesc *,size_t,size_t,size_t)>("hipBindTexture2D");
			out = hipFunc(hipFuncArgs.offset, hipFuncArgs.tex, hipFuncArgs.devPtr, hipFuncArgs.desc, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.pitch);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		offset = hipFuncArgs.offset;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		pitch = hipFuncArgs.pitch;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const textureReference *,const void *,const hipChannelFormatDesc *,size_t,size_t,size_t)>("hipBindTexture2D");
		return hipFunc(offset, tex, devPtr, desc, width, height, pitch);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipBindTextureToArray(const textureReference * tex, hipArray_const_t array, const hipChannelFormatDesc * desc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipBindTextureToArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipBindTextureToArray_api_args_t hipFuncArgs{tex, array, desc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const textureReference *,hipArray_const_t,const hipChannelFormatDesc *)>("hipBindTextureToArray");
			out = hipFunc(hipFuncArgs.tex, hipFuncArgs.array, hipFuncArgs.desc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const textureReference *,hipArray_const_t,const hipChannelFormatDesc *)>("hipBindTextureToArray");
		return hipFunc(tex, array, desc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipBindTextureToMipmappedArray(const textureReference * tex, hipMipmappedArray_const_t mipmappedArray, const hipChannelFormatDesc * desc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipBindTextureToMipmappedArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipBindTextureToMipmappedArray_api_args_t hipFuncArgs{tex, mipmappedArray, desc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const textureReference *,hipMipmappedArray_const_t,const hipChannelFormatDesc *)>("hipBindTextureToMipmappedArray");
			out = hipFunc(hipFuncArgs.tex, hipFuncArgs.mipmappedArray, hipFuncArgs.desc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const textureReference *,hipMipmappedArray_const_t,const hipChannelFormatDesc *)>("hipBindTextureToMipmappedArray");
		return hipFunc(tex, mipmappedArray, desc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipChooseDevice(int * device, const hipDeviceProp_t * prop) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipChooseDevice;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipChooseDevice_api_args_t hipFuncArgs{device, prop};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const hipDeviceProp_t *)>("hipChooseDevice");
			out = hipFunc(hipFuncArgs.device, hipFuncArgs.prop);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const hipDeviceProp_t *)>("hipChooseDevice");
		return hipFunc(device, prop);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipConfigureCall;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipConfigureCall_api_args_t hipFuncArgs{gridDim, blockDim, sharedMem, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(dim3,dim3,size_t,hipStream_t)>("hipConfigureCall");
			out = hipFunc(hipFuncArgs.gridDim, hipFuncArgs.blockDim, hipFuncArgs.sharedMem, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridDim = hipFuncArgs.gridDim;
		blockDim = hipFuncArgs.blockDim;
		sharedMem = hipFuncArgs.sharedMem;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(dim3,dim3,size_t,hipStream_t)>("hipConfigureCall");
		return hipFunc(gridDim, blockDim, sharedMem, stream);
	};
}

__attribute__((visibility("default")))
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t * pSurfObject, const hipResourceDesc * pResDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCreateSurfaceObject;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCreateSurfaceObject_api_args_t hipFuncArgs{pSurfObject, pResDesc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSurfaceObject_t *,const hipResourceDesc *)>("hipCreateSurfaceObject");
			out = hipFunc(hipFuncArgs.pSurfObject, hipFuncArgs.pResDesc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pSurfObject = hipFuncArgs.pSurfObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSurfaceObject_t *,const hipResourceDesc *)>("hipCreateSurfaceObject");
		return hipFunc(pSurfObject, pResDesc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCreateTextureObject(hipTextureObject_t * pTexObject, const hipResourceDesc * pResDesc, const hipTextureDesc * pTexDesc, const struct hipResourceViewDesc * pResViewDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCreateTextureObject;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCreateTextureObject_api_args_t hipFuncArgs{pTexObject, pResDesc, pTexDesc, pResViewDesc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureObject_t *,const hipResourceDesc *,const hipTextureDesc *,const struct hipResourceViewDesc *)>("hipCreateTextureObject");
			out = hipFunc(hipFuncArgs.pTexObject, hipFuncArgs.pResDesc, hipFuncArgs.pTexDesc, hipFuncArgs.pResViewDesc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pTexObject = hipFuncArgs.pTexObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureObject_t *,const hipResourceDesc *,const hipTextureDesc *,const struct hipResourceViewDesc *)>("hipCreateTextureObject");
		return hipFunc(pTexObject, pResDesc, pTexDesc, pResViewDesc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxCreate(hipCtx_t * ctx, unsigned int flags, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxCreate_api_args_t hipFuncArgs{ctx, flags, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t *,unsigned int,hipDevice_t)>("hipCtxCreate");
			out = hipFunc(hipFuncArgs.ctx, hipFuncArgs.flags, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ctx = hipFuncArgs.ctx;
		flags = hipFuncArgs.flags;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t *,unsigned int,hipDevice_t)>("hipCtxCreate");
		return hipFunc(ctx, flags, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxDestroy(hipCtx_t ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxDestroy_api_args_t hipFuncArgs{ctx};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxDestroy");
			out = hipFunc(hipFuncArgs.ctx);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ctx = hipFuncArgs.ctx;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxDestroy");
		return hipFunc(ctx);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxDisablePeerAccess;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxDisablePeerAccess_api_args_t hipFuncArgs{peerCtx};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxDisablePeerAccess");
			out = hipFunc(hipFuncArgs.peerCtx);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		peerCtx = hipFuncArgs.peerCtx;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxDisablePeerAccess");
		return hipFunc(peerCtx);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxEnablePeerAccess;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxEnablePeerAccess_api_args_t hipFuncArgs{peerCtx, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t,unsigned int)>("hipCtxEnablePeerAccess");
			out = hipFunc(hipFuncArgs.peerCtx, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		peerCtx = hipFuncArgs.peerCtx;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t,unsigned int)>("hipCtxEnablePeerAccess");
		return hipFunc(peerCtx, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int * apiVersion) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxGetApiVersion;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxGetApiVersion_api_args_t hipFuncArgs{ctx, apiVersion};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t,int *)>("hipCtxGetApiVersion");
			out = hipFunc(hipFuncArgs.ctx, hipFuncArgs.apiVersion);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ctx = hipFuncArgs.ctx;
		apiVersion = hipFuncArgs.apiVersion;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t,int *)>("hipCtxGetApiVersion");
		return hipFunc(ctx, apiVersion);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetCacheConfig(hipFuncCache_t * cacheConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxGetCacheConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxGetCacheConfig_api_args_t hipFuncArgs{cacheConfig};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFuncCache_t *)>("hipCtxGetCacheConfig");
			out = hipFunc(hipFuncArgs.cacheConfig);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		cacheConfig = hipFuncArgs.cacheConfig;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFuncCache_t *)>("hipCtxGetCacheConfig");
		return hipFunc(cacheConfig);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetCurrent(hipCtx_t * ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxGetCurrent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxGetCurrent_api_args_t hipFuncArgs{ctx};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t *)>("hipCtxGetCurrent");
			out = hipFunc(hipFuncArgs.ctx);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ctx = hipFuncArgs.ctx;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t *)>("hipCtxGetCurrent");
		return hipFunc(ctx);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetDevice(hipDevice_t * device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxGetDevice;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxGetDevice_api_args_t hipFuncArgs{device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t *)>("hipCtxGetDevice");
			out = hipFunc(hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t *)>("hipCtxGetDevice");
		return hipFunc(device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetFlags(unsigned int * flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxGetFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxGetFlags_api_args_t hipFuncArgs{flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *)>("hipCtxGetFlags");
			out = hipFunc(hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *)>("hipCtxGetFlags");
		return hipFunc(flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig * pConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxGetSharedMemConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxGetSharedMemConfig_api_args_t hipFuncArgs{pConfig};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSharedMemConfig *)>("hipCtxGetSharedMemConfig");
			out = hipFunc(hipFuncArgs.pConfig);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pConfig = hipFuncArgs.pConfig;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSharedMemConfig *)>("hipCtxGetSharedMemConfig");
		return hipFunc(pConfig);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxPopCurrent(hipCtx_t * ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxPopCurrent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxPopCurrent_api_args_t hipFuncArgs{ctx};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t *)>("hipCtxPopCurrent");
			out = hipFunc(hipFuncArgs.ctx);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ctx = hipFuncArgs.ctx;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t *)>("hipCtxPopCurrent");
		return hipFunc(ctx);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxPushCurrent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxPushCurrent_api_args_t hipFuncArgs{ctx};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxPushCurrent");
			out = hipFunc(hipFuncArgs.ctx);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ctx = hipFuncArgs.ctx;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxPushCurrent");
		return hipFunc(ctx);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxSetCacheConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxSetCacheConfig_api_args_t hipFuncArgs{cacheConfig};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFuncCache_t)>("hipCtxSetCacheConfig");
			out = hipFunc(hipFuncArgs.cacheConfig);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		cacheConfig = hipFuncArgs.cacheConfig;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFuncCache_t)>("hipCtxSetCacheConfig");
		return hipFunc(cacheConfig);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxSetCurrent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxSetCurrent_api_args_t hipFuncArgs{ctx};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxSetCurrent");
			out = hipFunc(hipFuncArgs.ctx);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ctx = hipFuncArgs.ctx;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t)>("hipCtxSetCurrent");
		return hipFunc(ctx);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxSetSharedMemConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipCtxSetSharedMemConfig_api_args_t hipFuncArgs{config};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSharedMemConfig)>("hipCtxSetSharedMemConfig");
			out = hipFunc(hipFuncArgs.config);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		config = hipFuncArgs.config;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSharedMemConfig)>("hipCtxSetSharedMemConfig");
		return hipFunc(config);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipCtxSynchronize() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipCtxSynchronize;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipCtxSynchronize");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipCtxSynchronize");
		return hipFunc();
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDestroyExternalMemory;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDestroyExternalMemory_api_args_t hipFuncArgs{extMem};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipExternalMemory_t)>("hipDestroyExternalMemory");
			out = hipFunc(hipFuncArgs.extMem);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		extMem = hipFuncArgs.extMem;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipExternalMemory_t)>("hipDestroyExternalMemory");
		return hipFunc(extMem);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDestroyExternalSemaphore;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDestroyExternalSemaphore_api_args_t hipFuncArgs{extSem};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipExternalSemaphore_t)>("hipDestroyExternalSemaphore");
			out = hipFunc(hipFuncArgs.extSem);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		extSem = hipFuncArgs.extSem;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipExternalSemaphore_t)>("hipDestroyExternalSemaphore");
		return hipFunc(extSem);
	};
}

__attribute__((visibility("default")))
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDestroySurfaceObject;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDestroySurfaceObject_api_args_t hipFuncArgs{surfaceObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSurfaceObject_t)>("hipDestroySurfaceObject");
			out = hipFunc(hipFuncArgs.surfaceObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		surfaceObject = hipFuncArgs.surfaceObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSurfaceObject_t)>("hipDestroySurfaceObject");
		return hipFunc(surfaceObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDestroyTextureObject;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDestroyTextureObject_api_args_t hipFuncArgs{textureObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureObject_t)>("hipDestroyTextureObject");
			out = hipFunc(hipFuncArgs.textureObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		textureObject = hipFuncArgs.textureObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureObject_t)>("hipDestroyTextureObject");
		return hipFunc(textureObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceCanAccessPeer(int * canAccessPeer, int deviceId, int peerDeviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceCanAccessPeer;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceCanAccessPeer_api_args_t hipFuncArgs{canAccessPeer, deviceId, peerDeviceId};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int,int)>("hipDeviceCanAccessPeer");
			out = hipFunc(hipFuncArgs.canAccessPeer, hipFuncArgs.deviceId, hipFuncArgs.peerDeviceId);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		canAccessPeer = hipFuncArgs.canAccessPeer;
		deviceId = hipFuncArgs.deviceId;
		peerDeviceId = hipFuncArgs.peerDeviceId;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int,int)>("hipDeviceCanAccessPeer");
		return hipFunc(canAccessPeer, deviceId, peerDeviceId);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceComputeCapability(int * major, int * minor, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceComputeCapability;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceComputeCapability_api_args_t hipFuncArgs{major, minor, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *,hipDevice_t)>("hipDeviceComputeCapability");
			out = hipFunc(hipFuncArgs.major, hipFuncArgs.minor, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		major = hipFuncArgs.major;
		minor = hipFuncArgs.minor;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *,hipDevice_t)>("hipDeviceComputeCapability");
		return hipFunc(major, minor, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceDisablePeerAccess;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceDisablePeerAccess_api_args_t hipFuncArgs{peerDeviceId};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int)>("hipDeviceDisablePeerAccess");
			out = hipFunc(hipFuncArgs.peerDeviceId);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		peerDeviceId = hipFuncArgs.peerDeviceId;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int)>("hipDeviceDisablePeerAccess");
		return hipFunc(peerDeviceId);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceEnablePeerAccess;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceEnablePeerAccess_api_args_t hipFuncArgs{peerDeviceId, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,unsigned int)>("hipDeviceEnablePeerAccess");
			out = hipFunc(hipFuncArgs.peerDeviceId, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		peerDeviceId = hipFuncArgs.peerDeviceId;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,unsigned int)>("hipDeviceEnablePeerAccess");
		return hipFunc(peerDeviceId, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGet(hipDevice_t * device, int ordinal) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGet;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGet_api_args_t hipFuncArgs{device, ordinal};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t *,int)>("hipDeviceGet");
			out = hipFunc(hipFuncArgs.device, hipFuncArgs.ordinal);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device = hipFuncArgs.device;
		ordinal = hipFuncArgs.ordinal;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t *,int)>("hipDeviceGet");
		return hipFunc(device, ordinal);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetAttribute(int * pi, hipDeviceAttribute_t attr, int deviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetAttribute_api_args_t hipFuncArgs{pi, attr, deviceId};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipDeviceAttribute_t,int)>("hipDeviceGetAttribute");
			out = hipFunc(hipFuncArgs.pi, hipFuncArgs.attr, hipFuncArgs.deviceId);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pi = hipFuncArgs.pi;
		attr = hipFuncArgs.attr;
		deviceId = hipFuncArgs.deviceId;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipDeviceAttribute_t,int)>("hipDeviceGetAttribute");
		return hipFunc(pi, attr, deviceId);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetByPCIBusId(int * device, const char * pciBusId) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetByPCIBusId;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetByPCIBusId_api_args_t hipFuncArgs{device, pciBusId};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const char *)>("hipDeviceGetByPCIBusId");
			out = hipFunc(hipFuncArgs.device, hipFuncArgs.pciBusId);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const char *)>("hipDeviceGetByPCIBusId");
		return hipFunc(device, pciBusId);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t * cacheConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetCacheConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetCacheConfig_api_args_t hipFuncArgs{cacheConfig};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFuncCache_t *)>("hipDeviceGetCacheConfig");
			out = hipFunc(hipFuncArgs.cacheConfig);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		cacheConfig = hipFuncArgs.cacheConfig;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFuncCache_t *)>("hipDeviceGetCacheConfig");
		return hipFunc(cacheConfig);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetCount(int * count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetCount;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetCount_api_args_t hipFuncArgs{count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipDeviceGetCount");
			out = hipFunc(hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipDeviceGetCount");
		return hipFunc(count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t * mem_pool, int device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetDefaultMemPool;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetDefaultMemPool_api_args_t hipFuncArgs{mem_pool, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t *,int)>("hipDeviceGetDefaultMemPool");
			out = hipFunc(hipFuncArgs.mem_pool, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t *,int)>("hipDeviceGetDefaultMemPool");
		return hipFunc(mem_pool, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void * value) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetGraphMemAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetGraphMemAttribute_api_args_t hipFuncArgs{device, attr, value};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipGraphMemAttributeType,void *)>("hipDeviceGetGraphMemAttribute");
			out = hipFunc(hipFuncArgs.device, hipFuncArgs.attr, hipFuncArgs.value);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device = hipFuncArgs.device;
		attr = hipFuncArgs.attr;
		value = hipFuncArgs.value;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipGraphMemAttributeType,void *)>("hipDeviceGetGraphMemAttribute");
		return hipFunc(device, attr, value);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetLimit(size_t * pValue, enum hipLimit_t limit) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetLimit;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetLimit_api_args_t hipFuncArgs{pValue, limit};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,enum hipLimit_t)>("hipDeviceGetLimit");
			out = hipFunc(hipFuncArgs.pValue, hipFuncArgs.limit);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pValue = hipFuncArgs.pValue;
		limit = hipFuncArgs.limit;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,enum hipLimit_t)>("hipDeviceGetLimit");
		return hipFunc(pValue, limit);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetMemPool(hipMemPool_t * mem_pool, int device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetMemPool;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetMemPool_api_args_t hipFuncArgs{mem_pool, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t *,int)>("hipDeviceGetMemPool");
			out = hipFunc(hipFuncArgs.mem_pool, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t *,int)>("hipDeviceGetMemPool");
		return hipFunc(mem_pool, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetName(char * name, int len, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetName;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetName_api_args_t hipFuncArgs{name, len, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(char *,int,hipDevice_t)>("hipDeviceGetName");
			out = hipFunc(hipFuncArgs.name, hipFuncArgs.len, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		name = hipFuncArgs.name;
		len = hipFuncArgs.len;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(char *,int,hipDevice_t)>("hipDeviceGetName");
		return hipFunc(name, len, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetP2PAttribute(int * value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetP2PAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetP2PAttribute_api_args_t hipFuncArgs{value, attr, srcDevice, dstDevice};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipDeviceP2PAttr,int,int)>("hipDeviceGetP2PAttribute");
			out = hipFunc(hipFuncArgs.value, hipFuncArgs.attr, hipFuncArgs.srcDevice, hipFuncArgs.dstDevice);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		value = hipFuncArgs.value;
		attr = hipFuncArgs.attr;
		srcDevice = hipFuncArgs.srcDevice;
		dstDevice = hipFuncArgs.dstDevice;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipDeviceP2PAttr,int,int)>("hipDeviceGetP2PAttribute");
		return hipFunc(value, attr, srcDevice, dstDevice);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetPCIBusId(char * pciBusId, int len, int device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetPCIBusId;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetPCIBusId_api_args_t hipFuncArgs{pciBusId, len, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(char *,int,int)>("hipDeviceGetPCIBusId");
			out = hipFunc(hipFuncArgs.pciBusId, hipFuncArgs.len, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pciBusId = hipFuncArgs.pciBusId;
		len = hipFuncArgs.len;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(char *,int,int)>("hipDeviceGetPCIBusId");
		return hipFunc(pciBusId, len, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig * pConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetSharedMemConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetSharedMemConfig_api_args_t hipFuncArgs{pConfig};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSharedMemConfig *)>("hipDeviceGetSharedMemConfig");
			out = hipFunc(hipFuncArgs.pConfig);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pConfig = hipFuncArgs.pConfig;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSharedMemConfig *)>("hipDeviceGetSharedMemConfig");
		return hipFunc(pConfig);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetStreamPriorityRange;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetStreamPriorityRange_api_args_t hipFuncArgs{leastPriority, greatestPriority};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *)>("hipDeviceGetStreamPriorityRange");
			out = hipFunc(hipFuncArgs.leastPriority, hipFuncArgs.greatestPriority);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		leastPriority = hipFuncArgs.leastPriority;
		greatestPriority = hipFuncArgs.greatestPriority;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *)>("hipDeviceGetStreamPriorityRange");
		return hipFunc(leastPriority, greatestPriority);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGetUuid(hipUUID * uuid, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGetUuid;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGetUuid_api_args_t hipFuncArgs{uuid, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipUUID *,hipDevice_t)>("hipDeviceGetUuid");
			out = hipFunc(hipFuncArgs.uuid, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		uuid = hipFuncArgs.uuid;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipUUID *,hipDevice_t)>("hipDeviceGetUuid");
		return hipFunc(uuid, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceGraphMemTrim(int device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceGraphMemTrim;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceGraphMemTrim_api_args_t hipFuncArgs{device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int)>("hipDeviceGraphMemTrim");
			out = hipFunc(hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int)>("hipDeviceGraphMemTrim");
		return hipFunc(device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int * flags, int * active) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDevicePrimaryCtxGetState;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDevicePrimaryCtxGetState_api_args_t hipFuncArgs{dev, flags, active};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t,unsigned int *,int *)>("hipDevicePrimaryCtxGetState");
			out = hipFunc(hipFuncArgs.dev, hipFuncArgs.flags, hipFuncArgs.active);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev = hipFuncArgs.dev;
		flags = hipFuncArgs.flags;
		active = hipFuncArgs.active;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t,unsigned int *,int *)>("hipDevicePrimaryCtxGetState");
		return hipFunc(dev, flags, active);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDevicePrimaryCtxRelease;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDevicePrimaryCtxRelease_api_args_t hipFuncArgs{dev};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t)>("hipDevicePrimaryCtxRelease");
			out = hipFunc(hipFuncArgs.dev);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev = hipFuncArgs.dev;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t)>("hipDevicePrimaryCtxRelease");
		return hipFunc(dev);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDevicePrimaryCtxReset;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDevicePrimaryCtxReset_api_args_t hipFuncArgs{dev};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t)>("hipDevicePrimaryCtxReset");
			out = hipFunc(hipFuncArgs.dev);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev = hipFuncArgs.dev;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t)>("hipDevicePrimaryCtxReset");
		return hipFunc(dev);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t * pctx, hipDevice_t dev) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDevicePrimaryCtxRetain;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDevicePrimaryCtxRetain_api_args_t hipFuncArgs{pctx, dev};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t *,hipDevice_t)>("hipDevicePrimaryCtxRetain");
			out = hipFunc(hipFuncArgs.pctx, hipFuncArgs.dev);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pctx = hipFuncArgs.pctx;
		dev = hipFuncArgs.dev;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipCtx_t *,hipDevice_t)>("hipDevicePrimaryCtxRetain");
		return hipFunc(pctx, dev);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDevicePrimaryCtxSetFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDevicePrimaryCtxSetFlags_api_args_t hipFuncArgs{dev, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t,unsigned int)>("hipDevicePrimaryCtxSetFlags");
			out = hipFunc(hipFuncArgs.dev, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev = hipFuncArgs.dev;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDevice_t,unsigned int)>("hipDevicePrimaryCtxSetFlags");
		return hipFunc(dev, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceReset() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceReset;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipDeviceReset");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipDeviceReset");
		return hipFunc();
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceSetCacheConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceSetCacheConfig_api_args_t hipFuncArgs{cacheConfig};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFuncCache_t)>("hipDeviceSetCacheConfig");
			out = hipFunc(hipFuncArgs.cacheConfig);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		cacheConfig = hipFuncArgs.cacheConfig;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFuncCache_t)>("hipDeviceSetCacheConfig");
		return hipFunc(cacheConfig);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void * value) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceSetGraphMemAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceSetGraphMemAttribute_api_args_t hipFuncArgs{device, attr, value};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipGraphMemAttributeType,void *)>("hipDeviceSetGraphMemAttribute");
			out = hipFunc(hipFuncArgs.device, hipFuncArgs.attr, hipFuncArgs.value);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device = hipFuncArgs.device;
		attr = hipFuncArgs.attr;
		value = hipFuncArgs.value;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipGraphMemAttributeType,void *)>("hipDeviceSetGraphMemAttribute");
		return hipFunc(device, attr, value);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceSetLimit;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceSetLimit_api_args_t hipFuncArgs{limit, value};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(enum hipLimit_t,size_t)>("hipDeviceSetLimit");
			out = hipFunc(hipFuncArgs.limit, hipFuncArgs.value);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		limit = hipFuncArgs.limit;
		value = hipFuncArgs.value;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(enum hipLimit_t,size_t)>("hipDeviceSetLimit");
		return hipFunc(limit, value);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceSetMemPool;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceSetMemPool_api_args_t hipFuncArgs{device, mem_pool};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipMemPool_t)>("hipDeviceSetMemPool");
			out = hipFunc(hipFuncArgs.device, hipFuncArgs.mem_pool);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device = hipFuncArgs.device;
		mem_pool = hipFuncArgs.mem_pool;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipMemPool_t)>("hipDeviceSetMemPool");
		return hipFunc(device, mem_pool);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceSetSharedMemConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceSetSharedMemConfig_api_args_t hipFuncArgs{config};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSharedMemConfig)>("hipDeviceSetSharedMemConfig");
			out = hipFunc(hipFuncArgs.config);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		config = hipFuncArgs.config;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipSharedMemConfig)>("hipDeviceSetSharedMemConfig");
		return hipFunc(config);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceSynchronize() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceSynchronize;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipDeviceSynchronize");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipDeviceSynchronize");
		return hipFunc();
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDeviceTotalMem(size_t * bytes, hipDevice_t device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDeviceTotalMem;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDeviceTotalMem_api_args_t hipFuncArgs{bytes, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,hipDevice_t)>("hipDeviceTotalMem");
			out = hipFunc(hipFuncArgs.bytes, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		bytes = hipFuncArgs.bytes;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,hipDevice_t)>("hipDeviceTotalMem");
		return hipFunc(bytes, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDriverGetVersion(int * driverVersion) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDriverGetVersion;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDriverGetVersion_api_args_t hipFuncArgs{driverVersion};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipDriverGetVersion");
			out = hipFunc(hipFuncArgs.driverVersion);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		driverVersion = hipFuncArgs.driverVersion;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipDriverGetVersion");
		return hipFunc(driverVersion);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvGetErrorName(hipError_t hipError, const char * * errorString) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipDrvGetErrorName;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDrvGetErrorName_api_args_t hipFuncArgs{hipError, errorString};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipError_t,const char * *)>("hipDrvGetErrorName");
			out = hipFunc(hipFuncArgs.hipError, hipFuncArgs.errorString);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hipError = hipFuncArgs.hipError;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipError_t,const char * *)>("hipDrvGetErrorName");
		return hipFunc(hipError, errorString);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvGetErrorString(hipError_t hipError, const char * * errorString) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipDrvGetErrorString;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDrvGetErrorString_api_args_t hipFuncArgs{hipError, errorString};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipError_t,const char * *)>("hipDrvGetErrorString");
			out = hipFunc(hipFuncArgs.hipError, hipFuncArgs.errorString);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hipError = hipFuncArgs.hipError;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipError_t,const char * *)>("hipDrvGetErrorString");
		return hipFunc(hipError, errorString);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t * phGraphNode, hipGraph_t hGraph, const hipGraphNode_t * dependencies, size_t numDependencies, const HIP_MEMCPY3D * copyParams, hipCtx_t ctx) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipDrvGraphAddMemcpyNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDrvGraphAddMemcpyNode_api_args_t hipFuncArgs{phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const HIP_MEMCPY3D *,hipCtx_t)>("hipDrvGraphAddMemcpyNode");
			out = hipFunc(hipFuncArgs.phGraphNode, hipFuncArgs.hGraph, hipFuncArgs.dependencies, hipFuncArgs.numDependencies, hipFuncArgs.copyParams, hipFuncArgs.ctx);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		phGraphNode = hipFuncArgs.phGraphNode;
		hGraph = hipFuncArgs.hGraph;
		numDependencies = hipFuncArgs.numDependencies;
		ctx = hipFuncArgs.ctx;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const HIP_MEMCPY3D *,hipCtx_t)>("hipDrvGraphAddMemcpyNode");
		return hipFunc(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D * pCopy) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDrvMemcpy2DUnaligned;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDrvMemcpy2DUnaligned_api_args_t hipFuncArgs{pCopy};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hip_Memcpy2D *)>("hipDrvMemcpy2DUnaligned");
			out = hipFunc(hipFuncArgs.pCopy);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hip_Memcpy2D *)>("hipDrvMemcpy2DUnaligned");
		return hipFunc(pCopy);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D * pCopy) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDrvMemcpy3D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDrvMemcpy3D_api_args_t hipFuncArgs{pCopy};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const HIP_MEMCPY3D *)>("hipDrvMemcpy3D");
			out = hipFunc(hipFuncArgs.pCopy);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const HIP_MEMCPY3D *)>("hipDrvMemcpy3D");
		return hipFunc(pCopy);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D * pCopy, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDrvMemcpy3DAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDrvMemcpy3DAsync_api_args_t hipFuncArgs{pCopy, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const HIP_MEMCPY3D *,hipStream_t)>("hipDrvMemcpy3DAsync");
			out = hipFunc(hipFuncArgs.pCopy, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const HIP_MEMCPY3D *,hipStream_t)>("hipDrvMemcpy3DAsync");
		return hipFunc(pCopy, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute * attributes, void * * data, hipDeviceptr_t ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipDrvPointerGetAttributes;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipDrvPointerGetAttributes_api_args_t hipFuncArgs{numAttributes, attributes, data, ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int,hipPointer_attribute *,void * *,hipDeviceptr_t)>("hipDrvPointerGetAttributes");
			out = hipFunc(hipFuncArgs.numAttributes, hipFuncArgs.attributes, hipFuncArgs.data, hipFuncArgs.ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numAttributes = hipFuncArgs.numAttributes;
		attributes = hipFuncArgs.attributes;
		data = hipFuncArgs.data;
		ptr = hipFuncArgs.ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int,hipPointer_attribute *,void * *,hipDeviceptr_t)>("hipDrvPointerGetAttributes");
		return hipFunc(numAttributes, attributes, data, ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipEventCreate(hipEvent_t * event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipEventCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipEventCreate_api_args_t hipFuncArgs{event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t *)>("hipEventCreate");
			out = hipFunc(hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t *)>("hipEventCreate");
		return hipFunc(event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipEventCreateWithFlags(hipEvent_t * event, unsigned flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipEventCreateWithFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipEventCreateWithFlags_api_args_t hipFuncArgs{event, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t *,unsigned)>("hipEventCreateWithFlags");
			out = hipFunc(hipFuncArgs.event, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		event = hipFuncArgs.event;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t *,unsigned)>("hipEventCreateWithFlags");
		return hipFunc(event, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipEventDestroy(hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipEventDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipEventDestroy_api_args_t hipFuncArgs{event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t)>("hipEventDestroy");
			out = hipFunc(hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t)>("hipEventDestroy");
		return hipFunc(event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipEventElapsedTime(float * ms, hipEvent_t start, hipEvent_t stop) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipEventElapsedTime;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipEventElapsedTime_api_args_t hipFuncArgs{ms, start, stop};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(float *,hipEvent_t,hipEvent_t)>("hipEventElapsedTime");
			out = hipFunc(hipFuncArgs.ms, hipFuncArgs.start, hipFuncArgs.stop);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ms = hipFuncArgs.ms;
		start = hipFuncArgs.start;
		stop = hipFuncArgs.stop;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(float *,hipEvent_t,hipEvent_t)>("hipEventElapsedTime");
		return hipFunc(ms, start, stop);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipEventQuery(hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipEventQuery;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipEventQuery_api_args_t hipFuncArgs{event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t)>("hipEventQuery");
			out = hipFunc(hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t)>("hipEventQuery");
		return hipFunc(event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipEventRecord;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipEventRecord_api_args_t hipFuncArgs{event, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t,hipStream_t)>("hipEventRecord");
			out = hipFunc(hipFuncArgs.event, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		event = hipFuncArgs.event;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t,hipStream_t)>("hipEventRecord");
		return hipFunc(event, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipEventSynchronize(hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipEventSynchronize;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipEventSynchronize_api_args_t hipFuncArgs{event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t)>("hipEventSynchronize");
			out = hipFunc(hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t)>("hipEventSynchronize");
		return hipFunc(event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExtGetLastError() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipExtGetLastError;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipExtGetLastError");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipExtGetLastError");
		return hipFunc();
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t * linktype, uint32_t * hopcount) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipExtGetLinkTypeAndHopCount;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExtGetLinkTypeAndHopCount_api_args_t hipFuncArgs{device1, device2, linktype, hopcount};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,int,uint32_t *,uint32_t *)>("hipExtGetLinkTypeAndHopCount");
			out = hipFunc(hipFuncArgs.device1, hipFuncArgs.device2, hipFuncArgs.linktype, hipFuncArgs.hopcount);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device1 = hipFuncArgs.device1;
		device2 = hipFuncArgs.device2;
		linktype = hipFuncArgs.linktype;
		hopcount = hipFuncArgs.hopcount;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,int,uint32_t *,uint32_t *)>("hipExtGetLinkTypeAndHopCount");
		return hipFunc(device1, device2, linktype, hopcount);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExtLaunchKernel(const void * function_address, dim3 numBlocks, dim3 dimBlocks, void * * args, size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent, hipEvent_t stopEvent, int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipExtLaunchKernel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExtLaunchKernel_api_args_t hipFuncArgs{function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t,hipEvent_t,hipEvent_t,int)>("hipExtLaunchKernel");
			out = hipFunc(hipFuncArgs.function_address, hipFuncArgs.numBlocks, hipFuncArgs.dimBlocks, hipFuncArgs.args, hipFuncArgs.sharedMemBytes, hipFuncArgs.stream, hipFuncArgs.startEvent, hipFuncArgs.stopEvent, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numBlocks = hipFuncArgs.numBlocks;
		dimBlocks = hipFuncArgs.dimBlocks;
		args = hipFuncArgs.args;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		stream = hipFuncArgs.stream;
		startEvent = hipFuncArgs.startEvent;
		stopEvent = hipFuncArgs.stopEvent;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t,hipEvent_t,hipEvent_t,int)>("hipExtLaunchKernel");
		return hipFunc(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams * launchParamsList, int numDevices, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipExtLaunchMultiKernelMultiDevice;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExtLaunchMultiKernelMultiDevice_api_args_t hipFuncArgs{launchParamsList, numDevices, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipLaunchParams *,int,unsigned int)>("hipExtLaunchMultiKernelMultiDevice");
			out = hipFunc(hipFuncArgs.launchParamsList, hipFuncArgs.numDevices, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		launchParamsList = hipFuncArgs.launchParamsList;
		numDevices = hipFuncArgs.numDevices;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipLaunchParams *,int,unsigned int)>("hipExtLaunchMultiKernelMultiDevice");
		return hipFunc(launchParamsList, numDevices, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExtMallocWithFlags(void * * ptr, size_t sizeBytes, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipExtMallocWithFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExtMallocWithFlags_api_args_t hipFuncArgs{ptr, sizeBytes, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipExtMallocWithFlags");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.sizeBytes, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		sizeBytes = hipFuncArgs.sizeBytes;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipExtMallocWithFlags");
		return hipFunc(ptr, sizeBytes, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ, size_t sharedMemBytes, hipStream_t hStream, void * * kernelParams, void * * extra, hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipExtModuleLaunchKernel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExtModuleLaunchKernel_api_args_t hipFuncArgs{f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t,uint32_t)>("hipExtModuleLaunchKernel");
			out = hipFunc(hipFuncArgs.f, hipFuncArgs.globalWorkSizeX, hipFuncArgs.globalWorkSizeY, hipFuncArgs.globalWorkSizeZ, hipFuncArgs.localWorkSizeX, hipFuncArgs.localWorkSizeY, hipFuncArgs.localWorkSizeZ, hipFuncArgs.sharedMemBytes, hipFuncArgs.hStream, hipFuncArgs.kernelParams, hipFuncArgs.extra, hipFuncArgs.startEvent, hipFuncArgs.stopEvent, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		f = hipFuncArgs.f;
		globalWorkSizeX = hipFuncArgs.globalWorkSizeX;
		globalWorkSizeY = hipFuncArgs.globalWorkSizeY;
		globalWorkSizeZ = hipFuncArgs.globalWorkSizeZ;
		localWorkSizeX = hipFuncArgs.localWorkSizeX;
		localWorkSizeY = hipFuncArgs.localWorkSizeY;
		localWorkSizeZ = hipFuncArgs.localWorkSizeZ;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		hStream = hipFuncArgs.hStream;
		kernelParams = hipFuncArgs.kernelParams;
		extra = hipFuncArgs.extra;
		startEvent = hipFuncArgs.startEvent;
		stopEvent = hipFuncArgs.stopEvent;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t,uint32_t)>("hipExtModuleLaunchKernel");
		return hipFunc(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExtStreamCreateWithCUMask(hipStream_t * stream, uint32_t cuMaskSize, const uint32_t * cuMask) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipExtStreamCreateWithCUMask;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExtStreamCreateWithCUMask_api_args_t hipFuncArgs{stream, cuMaskSize, cuMask};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t *,uint32_t,const uint32_t *)>("hipExtStreamCreateWithCUMask");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.cuMaskSize, hipFuncArgs.cuMask);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		cuMaskSize = hipFuncArgs.cuMaskSize;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t *,uint32_t,const uint32_t *)>("hipExtStreamCreateWithCUMask");
		return hipFunc(stream, cuMaskSize, cuMask);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t * cuMask) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipExtStreamGetCUMask;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExtStreamGetCUMask_api_args_t hipFuncArgs{stream, cuMaskSize, cuMask};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,uint32_t,uint32_t *)>("hipExtStreamGetCUMask");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.cuMaskSize, hipFuncArgs.cuMask);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		cuMaskSize = hipFuncArgs.cuMaskSize;
		cuMask = hipFuncArgs.cuMask;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,uint32_t,uint32_t *)>("hipExtStreamGetCUMask");
		return hipFunc(stream, cuMaskSize, cuMask);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExternalMemoryGetMappedBuffer(void * * devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc * bufferDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipExternalMemoryGetMappedBuffer;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExternalMemoryGetMappedBuffer_api_args_t hipFuncArgs{devPtr, extMem, bufferDesc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,hipExternalMemory_t,const hipExternalMemoryBufferDesc *)>("hipExternalMemoryGetMappedBuffer");
			out = hipFunc(hipFuncArgs.devPtr, hipFuncArgs.extMem, hipFuncArgs.bufferDesc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		devPtr = hipFuncArgs.devPtr;
		extMem = hipFuncArgs.extMem;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,hipExternalMemory_t,const hipExternalMemoryBufferDesc *)>("hipExternalMemoryGetMappedBuffer");
		return hipFunc(devPtr, extMem, bufferDesc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipExternalMemoryGetMappedMipmappedArray(hipMipmappedArray_t * mipmap, hipExternalMemory_t extMem, const hipExternalMemoryMipmappedArrayDesc * mipmapDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipExternalMemoryGetMappedMipmappedArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipExternalMemoryGetMappedMipmappedArray_api_args_t hipFuncArgs{mipmap, extMem, mipmapDesc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,hipExternalMemory_t,const hipExternalMemoryMipmappedArrayDesc *)>("hipExternalMemoryGetMappedMipmappedArray");
			out = hipFunc(hipFuncArgs.mipmap, hipFuncArgs.extMem, hipFuncArgs.mipmapDesc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mipmap = hipFuncArgs.mipmap;
		extMem = hipFuncArgs.extMem;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,hipExternalMemory_t,const hipExternalMemoryMipmappedArrayDesc *)>("hipExternalMemoryGetMappedMipmappedArray");
		return hipFunc(mipmap, extMem, mipmapDesc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFree(void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFree;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFree_api_args_t hipFuncArgs{ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipFree");
			out = hipFunc(hipFuncArgs.ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipFree");
		return hipFunc(ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFreeArray(hipArray_t array) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFreeArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFreeArray_api_args_t hipFuncArgs{array};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t)>("hipFreeArray");
			out = hipFunc(hipFuncArgs.array);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		array = hipFuncArgs.array;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t)>("hipFreeArray");
		return hipFunc(array);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFreeAsync(void * dev_ptr, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFreeAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFreeAsync_api_args_t hipFuncArgs{dev_ptr, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipStream_t)>("hipFreeAsync");
			out = hipFunc(hipFuncArgs.dev_ptr, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev_ptr = hipFuncArgs.dev_ptr;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipStream_t)>("hipFreeAsync");
		return hipFunc(dev_ptr, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFreeHost(void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFreeHost;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFreeHost_api_args_t hipFuncArgs{ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipFreeHost");
			out = hipFunc(hipFuncArgs.ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipFreeHost");
		return hipFunc(ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFreeMipmappedArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFreeMipmappedArray_api_args_t hipFuncArgs{mipmappedArray};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t)>("hipFreeMipmappedArray");
			out = hipFunc(hipFuncArgs.mipmappedArray);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mipmappedArray = hipFuncArgs.mipmappedArray;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t)>("hipFreeMipmappedArray");
		return hipFunc(mipmappedArray);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncGetAttribute(int * value, hipFunction_attribute attrib, hipFunction_t hfunc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFuncGetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFuncGetAttribute_api_args_t hipFuncArgs{value, attrib, hfunc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipFunction_attribute,hipFunction_t)>("hipFuncGetAttribute");
			out = hipFunc(hipFuncArgs.value, hipFuncArgs.attrib, hipFuncArgs.hfunc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		value = hipFuncArgs.value;
		attrib = hipFuncArgs.attrib;
		hfunc = hipFuncArgs.hfunc;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipFunction_attribute,hipFunction_t)>("hipFuncGetAttribute");
		return hipFunc(value, attrib, hfunc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncGetAttributes(struct hipFuncAttributes * attr, const void * func) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFuncGetAttributes;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFuncGetAttributes_api_args_t hipFuncArgs{attr, func};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(struct hipFuncAttributes *,const void *)>("hipFuncGetAttributes");
			out = hipFunc(hipFuncArgs.attr, hipFuncArgs.func);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		attr = hipFuncArgs.attr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(struct hipFuncAttributes *,const void *)>("hipFuncGetAttributes");
		return hipFunc(attr, func);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncSetAttribute(const void * func, hipFuncAttribute attr, int value) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFuncSetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFuncSetAttribute_api_args_t hipFuncArgs{func, attr, value};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,hipFuncAttribute,int)>("hipFuncSetAttribute");
			out = hipFunc(hipFuncArgs.func, hipFuncArgs.attr, hipFuncArgs.value);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		attr = hipFuncArgs.attr;
		value = hipFuncArgs.value;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,hipFuncAttribute,int)>("hipFuncSetAttribute");
		return hipFunc(func, attr, value);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncSetCacheConfig(const void * func, hipFuncCache_t config) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFuncSetCacheConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFuncSetCacheConfig_api_args_t hipFuncArgs{func, config};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,hipFuncCache_t)>("hipFuncSetCacheConfig");
			out = hipFunc(hipFuncArgs.func, hipFuncArgs.config);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		config = hipFuncArgs.config;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,hipFuncCache_t)>("hipFuncSetCacheConfig");
		return hipFunc(func, config);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipFuncSetSharedMemConfig(const void * func, hipSharedMemConfig config) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipFuncSetSharedMemConfig;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipFuncSetSharedMemConfig_api_args_t hipFuncArgs{func, config};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,hipSharedMemConfig)>("hipFuncSetSharedMemConfig");
			out = hipFunc(hipFuncArgs.func, hipFuncArgs.config);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		config = hipFuncArgs.config;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,hipSharedMemConfig)>("hipFuncSetSharedMemConfig");
		return hipFunc(func, config);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGLGetDevices(unsigned int * pHipDeviceCount, int * pHipDevices, unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGLGetDevices;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGLGetDevices_api_args_t hipFuncArgs{pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *,int *,unsigned int,hipGLDeviceList)>("hipGLGetDevices");
			out = hipFunc(hipFuncArgs.pHipDeviceCount, hipFuncArgs.pHipDevices, hipFuncArgs.hipDeviceCount, hipFuncArgs.deviceList);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pHipDeviceCount = hipFuncArgs.pHipDeviceCount;
		pHipDevices = hipFuncArgs.pHipDevices;
		hipDeviceCount = hipFuncArgs.hipDeviceCount;
		deviceList = hipFuncArgs.deviceList;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *,int *,unsigned int,hipGLDeviceList)>("hipGLGetDevices");
		return hipFunc(pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetChannelDesc(hipChannelFormatDesc * desc, hipArray_const_t array) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetChannelDesc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetChannelDesc_api_args_t hipFuncArgs{desc, array};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipChannelFormatDesc *,hipArray_const_t)>("hipGetChannelDesc");
			out = hipFunc(hipFuncArgs.desc, hipFuncArgs.array);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		desc = hipFuncArgs.desc;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipChannelFormatDesc *,hipArray_const_t)>("hipGetChannelDesc");
		return hipFunc(desc, array);
	};
}

extern "C" __attribute__((visibility("default")))
const char * hipGetCmdName(unsigned op) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipGetCmdName;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetCmdName_api_args_t hipFuncArgs{op};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(unsigned)>("hipGetCmdName");
			out = hipFunc(hipFuncArgs.op);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		op = hipFuncArgs.op;

		return std::any_cast<const char *>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(unsigned)>("hipGetCmdName");
		return hipFunc(op);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetDevice(int * deviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetDevice;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetDevice_api_args_t hipFuncArgs{deviceId};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipGetDevice");
			out = hipFunc(hipFuncArgs.deviceId);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		deviceId = hipFuncArgs.deviceId;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipGetDevice");
		return hipFunc(deviceId);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetDeviceCount(int * count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetDeviceCount;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetDeviceCount_api_args_t hipFuncArgs{count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipGetDeviceCount");
			out = hipFunc(hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipGetDeviceCount");
		return hipFunc(count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetDeviceFlags(unsigned int * flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetDeviceFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetDeviceFlags_api_args_t hipFuncArgs{flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *)>("hipGetDeviceFlags");
			out = hipFunc(hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *)>("hipGetDeviceFlags");
		return hipFunc(flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetDeviceProperties(hipDeviceProp_t * prop, int deviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetDeviceProperties;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetDeviceProperties_api_args_t hipFuncArgs{prop, deviceId};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceProp_t *,int)>("hipGetDeviceProperties");
			out = hipFunc(hipFuncArgs.prop, hipFuncArgs.deviceId);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		prop = hipFuncArgs.prop;
		deviceId = hipFuncArgs.deviceId;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceProp_t *,int)>("hipGetDeviceProperties");
		return hipFunc(prop, deviceId);
	};
}

extern "C" __attribute__((visibility("default")))
const char * hipGetErrorName(hipError_t hip_error) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipGetErrorName;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetErrorName_api_args_t hipFuncArgs{hip_error};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(hipError_t)>("hipGetErrorName");
			out = hipFunc(hipFuncArgs.hip_error);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hip_error = hipFuncArgs.hip_error;

		return std::any_cast<const char *>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(hipError_t)>("hipGetErrorName");
		return hipFunc(hip_error);
	};
}

extern "C" __attribute__((visibility("default")))
const char * hipGetErrorString(hipError_t hipError) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetErrorString;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetErrorString_api_args_t hipFuncArgs{hipError};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(hipError_t)>("hipGetErrorString");
			out = hipFunc(hipFuncArgs.hipError);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hipError = hipFuncArgs.hipError;

		return std::any_cast<const char *>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(hipError_t)>("hipGetErrorString");
		return hipFunc(hipError);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetLastError() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetLastError;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipGetLastError");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipGetLastError");
		return hipFunc();
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetMipmappedArrayLevel(hipArray_t * levelArray, hipMipmappedArray_const_t mipmappedArray, unsigned int level) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetMipmappedArrayLevel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetMipmappedArrayLevel_api_args_t hipFuncArgs{levelArray, mipmappedArray, level};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,hipMipmappedArray_const_t,unsigned int)>("hipGetMipmappedArrayLevel");
			out = hipFunc(hipFuncArgs.levelArray, hipFuncArgs.mipmappedArray, hipFuncArgs.level);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		levelArray = hipFuncArgs.levelArray;
		level = hipFuncArgs.level;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,hipMipmappedArray_const_t,unsigned int)>("hipGetMipmappedArrayLevel");
		return hipFunc(levelArray, mipmappedArray, level);
	};
}

extern "C" __attribute__((visibility("default")))
int hipGetStreamDeviceId(hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipGetStreamDeviceId;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetStreamDeviceId_api_args_t hipFuncArgs{stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<int(*)(hipStream_t)>("hipGetStreamDeviceId");
			out = hipFunc(hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<int>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<int(*)(hipStream_t)>("hipGetStreamDeviceId");
		return hipFunc(stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetSymbolAddress(void * * devPtr, const void * symbol) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetSymbolAddress;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetSymbolAddress_api_args_t hipFuncArgs{devPtr, symbol};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,const void *)>("hipGetSymbolAddress");
			out = hipFunc(hipFuncArgs.devPtr, hipFuncArgs.symbol);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		devPtr = hipFuncArgs.devPtr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,const void *)>("hipGetSymbolAddress");
		return hipFunc(devPtr, symbol);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetSymbolSize(size_t * size, const void * symbol) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetSymbolSize;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetSymbolSize_api_args_t hipFuncArgs{size, symbol};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const void *)>("hipGetSymbolSize");
			out = hipFunc(hipFuncArgs.size, hipFuncArgs.symbol);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		size = hipFuncArgs.size;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const void *)>("hipGetSymbolSize");
		return hipFunc(size, symbol);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureAlignmentOffset(size_t * offset, const textureReference * texref) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetTextureAlignmentOffset;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetTextureAlignmentOffset_api_args_t hipFuncArgs{offset, texref};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const textureReference *)>("hipGetTextureAlignmentOffset");
			out = hipFunc(hipFuncArgs.offset, hipFuncArgs.texref);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		offset = hipFuncArgs.offset;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const textureReference *)>("hipGetTextureAlignmentOffset");
		return hipFunc(offset, texref);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc * pResDesc, hipTextureObject_t textureObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetTextureObjectResourceDesc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetTextureObjectResourceDesc_api_args_t hipFuncArgs{pResDesc, textureObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipResourceDesc *,hipTextureObject_t)>("hipGetTextureObjectResourceDesc");
			out = hipFunc(hipFuncArgs.pResDesc, hipFuncArgs.textureObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pResDesc = hipFuncArgs.pResDesc;
		textureObject = hipFuncArgs.textureObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipResourceDesc *,hipTextureObject_t)>("hipGetTextureObjectResourceDesc");
		return hipFunc(pResDesc, textureObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc * pResViewDesc, hipTextureObject_t textureObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetTextureObjectResourceViewDesc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetTextureObjectResourceViewDesc_api_args_t hipFuncArgs{pResViewDesc, textureObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(struct hipResourceViewDesc *,hipTextureObject_t)>("hipGetTextureObjectResourceViewDesc");
			out = hipFunc(hipFuncArgs.pResViewDesc, hipFuncArgs.textureObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pResViewDesc = hipFuncArgs.pResViewDesc;
		textureObject = hipFuncArgs.textureObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(struct hipResourceViewDesc *,hipTextureObject_t)>("hipGetTextureObjectResourceViewDesc");
		return hipFunc(pResViewDesc, textureObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc * pTexDesc, hipTextureObject_t textureObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetTextureObjectTextureDesc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetTextureObjectTextureDesc_api_args_t hipFuncArgs{pTexDesc, textureObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureDesc *,hipTextureObject_t)>("hipGetTextureObjectTextureDesc");
			out = hipFunc(hipFuncArgs.pTexDesc, hipFuncArgs.textureObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pTexDesc = hipFuncArgs.pTexDesc;
		textureObject = hipFuncArgs.textureObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureDesc *,hipTextureObject_t)>("hipGetTextureObjectTextureDesc");
		return hipFunc(pTexDesc, textureObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGetTextureReference(const textureReference * * texref, const void * symbol) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGetTextureReference;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGetTextureReference_api_args_t hipFuncArgs{texref, symbol};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const textureReference * *,const void *)>("hipGetTextureReference");
			out = hipFunc(hipFuncArgs.texref, hipFuncArgs.symbol);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const textureReference * *,const void *)>("hipGetTextureReference");
		return hipFunc(texref, symbol);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipGraph_t childGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddChildGraphNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddChildGraphNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, childGraph};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipGraph_t)>("hipGraphAddChildGraphNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.childGraph);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;
		childGraph = hipFuncArgs.childGraph;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipGraph_t)>("hipGraphAddChildGraphNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, childGraph);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t * from, const hipGraphNode_t * to, size_t numDependencies) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddDependencies;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddDependencies_api_args_t hipFuncArgs{graph, from, to, numDependencies};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,const hipGraphNode_t *,const hipGraphNode_t *,size_t)>("hipGraphAddDependencies");
			out = hipFunc(hipFuncArgs.graph, hipFuncArgs.from, hipFuncArgs.to, hipFuncArgs.numDependencies);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,const hipGraphNode_t *,const hipGraphNode_t *,size_t)>("hipGraphAddDependencies");
		return hipFunc(graph, from, to, numDependencies);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddEmptyNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddEmptyNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddEmptyNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t)>("hipGraphAddEmptyNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t)>("hipGraphAddEmptyNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddEventRecordNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddEventRecordNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipEvent_t)>("hipGraphAddEventRecordNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipEvent_t)>("hipGraphAddEventRecordNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddEventWaitNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddEventWaitNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipEvent_t)>("hipGraphAddEventWaitNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipEvent_t)>("hipGraphAddEventWaitNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddHostNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipHostNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddHostNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddHostNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipHostNodeParams *)>("hipGraphAddHostNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipHostNodeParams *)>("hipGraphAddHostNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddKernelNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipKernelNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddKernelNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddKernelNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipKernelNodeParams *)>("hipGraphAddKernelNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipKernelNodeParams *)>("hipGraphAddKernelNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemAllocNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, hipMemAllocNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddMemAllocNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddMemAllocNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipMemAllocNodeParams *)>("hipGraphAddMemAllocNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;
		pNodeParams = hipFuncArgs.pNodeParams;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,hipMemAllocNodeParams *)>("hipGraphAddMemAllocNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemFreeNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, void * dev_ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddMemFreeNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddMemFreeNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, dev_ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,void *)>("hipGraphAddMemFreeNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.dev_ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;
		dev_ptr = hipFuncArgs.dev_ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,void *)>("hipGraphAddMemFreeNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, dev_ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipMemcpy3DParms * pCopyParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddMemcpyNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddMemcpyNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, pCopyParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipMemcpy3DParms *)>("hipGraphAddMemcpyNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.pCopyParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipMemcpy3DParms *)>("hipGraphAddMemcpyNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, pCopyParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddMemcpyNode1D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddMemcpyNode1D_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNode1D");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.count, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;
		dst = hipFuncArgs.dst;
		count = hipFuncArgs.count;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNode1D");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddMemcpyNodeFromSymbol;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddMemcpyNodeFromSymbol_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNodeFromSymbol");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.dst, hipFuncArgs.symbol, hipFuncArgs.count, hipFuncArgs.offset, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;
		dst = hipFuncArgs.dst;
		count = hipFuncArgs.count;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNodeFromSymbol");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddMemcpyNodeToSymbol;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddMemcpyNodeToSymbol_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNodeToSymbol");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.symbol, hipFuncArgs.src, hipFuncArgs.count, hipFuncArgs.offset, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;
		count = hipFuncArgs.count;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphAddMemcpyNodeToSymbol");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphAddMemsetNode(hipGraphNode_t * pGraphNode, hipGraph_t graph, const hipGraphNode_t * pDependencies, size_t numDependencies, const hipMemsetParams * pMemsetParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphAddMemsetNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphAddMemsetNode_api_args_t hipFuncArgs{pGraphNode, graph, pDependencies, numDependencies, pMemsetParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipMemsetParams *)>("hipGraphAddMemsetNode");
			out = hipFunc(hipFuncArgs.pGraphNode, hipFuncArgs.graph, hipFuncArgs.pDependencies, hipFuncArgs.numDependencies, hipFuncArgs.pMemsetParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphNode = hipFuncArgs.pGraphNode;
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraph_t,const hipGraphNode_t *,size_t,const hipMemsetParams *)>("hipGraphAddMemsetNode");
		return hipFunc(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t * pGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphChildGraphNodeGetGraph;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphChildGraphNodeGetGraph_api_args_t hipFuncArgs{node, pGraph};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraph_t *)>("hipGraphChildGraphNodeGetGraph");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pGraph);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pGraph = hipFuncArgs.pGraph;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraph_t *)>("hipGraphChildGraphNodeGetGraph");
		return hipFunc(node, pGraph);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphClone(hipGraph_t * pGraphClone, hipGraph_t originalGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphClone;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphClone_api_args_t hipFuncArgs{pGraphClone, originalGraph};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t *,hipGraph_t)>("hipGraphClone");
			out = hipFunc(hipFuncArgs.pGraphClone, hipFuncArgs.originalGraph);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphClone = hipFuncArgs.pGraphClone;
		originalGraph = hipFuncArgs.originalGraph;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t *,hipGraph_t)>("hipGraphClone");
		return hipFunc(pGraphClone, originalGraph);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphCreate(hipGraph_t * pGraph, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphCreate_api_args_t hipFuncArgs{pGraph, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t *,unsigned int)>("hipGraphCreate");
			out = hipFunc(hipFuncArgs.pGraph, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraph = hipFuncArgs.pGraph;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t *,unsigned int)>("hipGraphCreate");
		return hipFunc(pGraph, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char * path, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphDebugDotPrint;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphDebugDotPrint_api_args_t hipFuncArgs{graph, path, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,const char *,unsigned int)>("hipGraphDebugDotPrint");
			out = hipFunc(hipFuncArgs.graph, hipFuncArgs.path, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,const char *,unsigned int)>("hipGraphDebugDotPrint");
		return hipFunc(graph, path, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphDestroy(hipGraph_t graph) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphDestroy_api_args_t hipFuncArgs{graph};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t)>("hipGraphDestroy");
			out = hipFunc(hipFuncArgs.graph);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t)>("hipGraphDestroy");
		return hipFunc(graph);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphDestroyNode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphDestroyNode_api_args_t hipFuncArgs{node};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t)>("hipGraphDestroyNode");
			out = hipFunc(hipFuncArgs.node);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t)>("hipGraphDestroyNode");
		return hipFunc(node);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t * event_out) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphEventRecordNodeGetEvent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphEventRecordNodeGetEvent_api_args_t hipFuncArgs{node, event_out};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t *)>("hipGraphEventRecordNodeGetEvent");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.event_out);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		event_out = hipFuncArgs.event_out;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t *)>("hipGraphEventRecordNodeGetEvent");
		return hipFunc(node, event_out);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphEventRecordNodeSetEvent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphEventRecordNodeSetEvent_api_args_t hipFuncArgs{node, event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t)>("hipGraphEventRecordNodeSetEvent");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t)>("hipGraphEventRecordNodeSetEvent");
		return hipFunc(node, event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t * event_out) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphEventWaitNodeGetEvent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphEventWaitNodeGetEvent_api_args_t hipFuncArgs{node, event_out};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t *)>("hipGraphEventWaitNodeGetEvent");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.event_out);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		event_out = hipFuncArgs.event_out;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t *)>("hipGraphEventWaitNodeGetEvent");
		return hipFunc(node, event_out);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphEventWaitNodeSetEvent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphEventWaitNodeSetEvent_api_args_t hipFuncArgs{node, event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t)>("hipGraphEventWaitNodeSetEvent");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipEvent_t)>("hipGraphEventWaitNodeSetEvent");
		return hipFunc(node, event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipGraph_t childGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecChildGraphNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecChildGraphNodeSetParams_api_args_t hipFuncArgs{hGraphExec, node, childGraph};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipGraph_t)>("hipGraphExecChildGraphNodeSetParams");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.node, hipFuncArgs.childGraph);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		node = hipFuncArgs.node;
		childGraph = hipFuncArgs.childGraph;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipGraph_t)>("hipGraphExecChildGraphNodeSetParams");
		return hipFunc(hGraphExec, node, childGraph);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecDestroy_api_args_t hipFuncArgs{graphExec};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t)>("hipGraphExecDestroy");
			out = hipFunc(hipFuncArgs.graphExec);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graphExec = hipFuncArgs.graphExec;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t)>("hipGraphExecDestroy");
		return hipFunc(graphExec);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecEventRecordNodeSetEvent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecEventRecordNodeSetEvent_api_args_t hipFuncArgs{hGraphExec, hNode, event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t)>("hipGraphExecEventRecordNodeSetEvent");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.hNode, hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		hNode = hipFuncArgs.hNode;
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t)>("hipGraphExecEventRecordNodeSetEvent");
		return hipFunc(hGraphExec, hNode, event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecEventWaitNodeSetEvent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecEventWaitNodeSetEvent_api_args_t hipFuncArgs{hGraphExec, hNode, event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t)>("hipGraphExecEventWaitNodeSetEvent");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.hNode, hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		hNode = hipFuncArgs.hNode;
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipEvent_t)>("hipGraphExecEventWaitNodeSetEvent");
		return hipFunc(hGraphExec, hNode, event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipHostNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecHostNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecHostNodeSetParams_api_args_t hipFuncArgs{hGraphExec, node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipHostNodeParams *)>("hipGraphExecHostNodeSetParams");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		node = hipFuncArgs.node;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipHostNodeParams *)>("hipGraphExecHostNodeSetParams");
		return hipFunc(hGraphExec, node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipKernelNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecKernelNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecKernelNodeSetParams_api_args_t hipFuncArgs{hGraphExec, node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipKernelNodeParams *)>("hipGraphExecKernelNodeSetParams");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		node = hipFuncArgs.node;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipKernelNodeParams *)>("hipGraphExecKernelNodeSetParams");
		return hipFunc(hGraphExec, node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipMemcpy3DParms * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecMemcpyNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecMemcpyNodeSetParams_api_args_t hipFuncArgs{hGraphExec, node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipMemcpy3DParms *)>("hipGraphExecMemcpyNodeSetParams");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		node = hipFuncArgs.node;
		pNodeParams = hipFuncArgs.pNodeParams;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,hipMemcpy3DParms *)>("hipGraphExecMemcpyNodeSetParams");
		return hipFunc(hGraphExec, node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node, void * dst, const void * src, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecMemcpyNodeSetParams1D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecMemcpyNodeSetParams1D_api_args_t hipFuncArgs{hGraphExec, node, dst, src, count, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParams1D");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.node, hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.count, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		node = hipFuncArgs.node;
		dst = hipFuncArgs.dst;
		count = hipFuncArgs.count;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParams1D");
		return hipFunc(hGraphExec, node, dst, src, count, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecMemcpyNodeSetParamsFromSymbol;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecMemcpyNodeSetParamsFromSymbol_api_args_t hipFuncArgs{hGraphExec, node, dst, symbol, count, offset, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParamsFromSymbol");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.node, hipFuncArgs.dst, hipFuncArgs.symbol, hipFuncArgs.count, hipFuncArgs.offset, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		node = hipFuncArgs.node;
		dst = hipFuncArgs.dst;
		count = hipFuncArgs.count;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParamsFromSymbol");
		return hipFunc(hGraphExec, node, dst, symbol, count, offset, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecMemcpyNodeSetParamsToSymbol;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecMemcpyNodeSetParamsToSymbol_api_args_t hipFuncArgs{hGraphExec, node, symbol, src, count, offset, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParamsToSymbol");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.node, hipFuncArgs.symbol, hipFuncArgs.src, hipFuncArgs.count, hipFuncArgs.offset, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		node = hipFuncArgs.node;
		count = hipFuncArgs.count;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphExecMemcpyNodeSetParamsToSymbol");
		return hipFunc(hGraphExec, node, symbol, src, count, offset, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipMemsetParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecMemsetNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecMemsetNodeSetParams_api_args_t hipFuncArgs{hGraphExec, node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipMemsetParams *)>("hipGraphExecMemsetNodeSetParams");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		node = hipFuncArgs.node;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,const hipMemsetParams *)>("hipGraphExecMemsetNodeSetParams");
		return hipFunc(hGraphExec, node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph, hipGraphNode_t * hErrorNode_out, hipGraphExecUpdateResult * updateResult_out) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphExecUpdate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphExecUpdate_api_args_t hipFuncArgs{hGraphExec, hGraph, hErrorNode_out, updateResult_out};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraph_t,hipGraphNode_t *,hipGraphExecUpdateResult *)>("hipGraphExecUpdate");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.hGraph, hipFuncArgs.hErrorNode_out, hipFuncArgs.updateResult_out);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		hGraph = hipFuncArgs.hGraph;
		hErrorNode_out = hipFuncArgs.hErrorNode_out;
		updateResult_out = hipFuncArgs.updateResult_out;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraph_t,hipGraphNode_t *,hipGraphExecUpdateResult *)>("hipGraphExecUpdate");
		return hipFunc(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t * from, hipGraphNode_t * to, size_t * numEdges) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphGetEdges;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphGetEdges_api_args_t hipFuncArgs{graph, from, to, numEdges};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,hipGraphNode_t *,size_t *)>("hipGraphGetEdges");
			out = hipFunc(hipFuncArgs.graph, hipFuncArgs.from, hipFuncArgs.to, hipFuncArgs.numEdges);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;
		from = hipFuncArgs.from;
		to = hipFuncArgs.to;
		numEdges = hipFuncArgs.numEdges;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,hipGraphNode_t *,size_t *)>("hipGraphGetEdges");
		return hipFunc(graph, from, to, numEdges);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t * nodes, size_t * numNodes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphGetNodes;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphGetNodes_api_args_t hipFuncArgs{graph, nodes, numNodes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,size_t *)>("hipGraphGetNodes");
			out = hipFunc(hipFuncArgs.graph, hipFuncArgs.nodes, hipFuncArgs.numNodes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;
		nodes = hipFuncArgs.nodes;
		numNodes = hipFuncArgs.numNodes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,size_t *)>("hipGraphGetNodes");
		return hipFunc(graph, nodes, numNodes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t * pRootNodes, size_t * pNumRootNodes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphGetRootNodes;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphGetRootNodes_api_args_t hipFuncArgs{graph, pRootNodes, pNumRootNodes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,size_t *)>("hipGraphGetRootNodes");
			out = hipFunc(hipFuncArgs.graph, hipFuncArgs.pRootNodes, hipFuncArgs.pNumRootNodes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;
		pRootNodes = hipFuncArgs.pRootNodes;
		pNumRootNodes = hipFuncArgs.pNumRootNodes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipGraphNode_t *,size_t *)>("hipGraphGetRootNodes");
		return hipFunc(graph, pRootNodes, pNumRootNodes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphHostNodeGetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphHostNodeGetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipHostNodeParams *)>("hipGraphHostNodeGetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pNodeParams = hipFuncArgs.pNodeParams;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipHostNodeParams *)>("hipGraphHostNodeGetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphHostNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphHostNodeSetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const hipHostNodeParams *)>("hipGraphHostNodeSetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const hipHostNodeParams *)>("hipGraphHostNodeSetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphInstantiate(hipGraphExec_t * pGraphExec, hipGraph_t graph, hipGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphInstantiate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphInstantiate_api_args_t hipFuncArgs{pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t *,hipGraph_t,hipGraphNode_t *,char *,size_t)>("hipGraphInstantiate");
			out = hipFunc(hipFuncArgs.pGraphExec, hipFuncArgs.graph, hipFuncArgs.pErrorNode, hipFuncArgs.pLogBuffer, hipFuncArgs.bufferSize);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphExec = hipFuncArgs.pGraphExec;
		graph = hipFuncArgs.graph;
		pErrorNode = hipFuncArgs.pErrorNode;
		pLogBuffer = hipFuncArgs.pLogBuffer;
		bufferSize = hipFuncArgs.bufferSize;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t *,hipGraph_t,hipGraphNode_t *,char *,size_t)>("hipGraphInstantiate");
		return hipFunc(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t * pGraphExec, hipGraph_t graph, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphInstantiateWithFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphInstantiateWithFlags_api_args_t hipFuncArgs{pGraphExec, graph, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t *,hipGraph_t,unsigned long long)>("hipGraphInstantiateWithFlags");
			out = hipFunc(hipFuncArgs.pGraphExec, hipFuncArgs.graph, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pGraphExec = hipFuncArgs.pGraphExec;
		graph = hipFuncArgs.graph;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t *,hipGraph_t,unsigned long long)>("hipGraphInstantiateWithFlags");
		return hipFunc(pGraphExec, graph, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphKernelNodeCopyAttributes;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphKernelNodeCopyAttributes_api_args_t hipFuncArgs{hSrc, hDst};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t)>("hipGraphKernelNodeCopyAttributes");
			out = hipFunc(hipFuncArgs.hSrc, hipFuncArgs.hDst);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hSrc = hipFuncArgs.hSrc;
		hDst = hipFuncArgs.hDst;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t)>("hipGraphKernelNodeCopyAttributes");
		return hipFunc(hSrc, hDst);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, hipKernelNodeAttrValue * value) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphKernelNodeGetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphKernelNodeGetAttribute_api_args_t hipFuncArgs{hNode, attr, value};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeAttrID,hipKernelNodeAttrValue *)>("hipGraphKernelNodeGetAttribute");
			out = hipFunc(hipFuncArgs.hNode, hipFuncArgs.attr, hipFuncArgs.value);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hNode = hipFuncArgs.hNode;
		attr = hipFuncArgs.attr;
		value = hipFuncArgs.value;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeAttrID,hipKernelNodeAttrValue *)>("hipGraphKernelNodeGetAttribute");
		return hipFunc(hNode, attr, value);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphKernelNodeGetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphKernelNodeGetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeParams *)>("hipGraphKernelNodeGetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pNodeParams = hipFuncArgs.pNodeParams;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeParams *)>("hipGraphKernelNodeGetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, const hipKernelNodeAttrValue * value) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphKernelNodeSetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphKernelNodeSetAttribute_api_args_t hipFuncArgs{hNode, attr, value};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeAttrID,const hipKernelNodeAttrValue *)>("hipGraphKernelNodeSetAttribute");
			out = hipFunc(hipFuncArgs.hNode, hipFuncArgs.attr, hipFuncArgs.value);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hNode = hipFuncArgs.hNode;
		attr = hipFuncArgs.attr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipKernelNodeAttrID,const hipKernelNodeAttrValue *)>("hipGraphKernelNodeSetAttribute");
		return hipFunc(hNode, attr, value);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphKernelNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphKernelNodeSetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const hipKernelNodeParams *)>("hipGraphKernelNodeSetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const hipKernelNodeParams *)>("hipGraphKernelNodeSetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphLaunch;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphLaunch_api_args_t hipFuncArgs{graphExec, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipStream_t)>("hipGraphLaunch");
			out = hipFunc(hipFuncArgs.graphExec, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graphExec = hipFuncArgs.graphExec;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipStream_t)>("hipGraphLaunch");
		return hipFunc(graphExec, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemAllocNodeGetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemAllocNodeGetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipMemAllocNodeParams *)>("hipGraphMemAllocNodeGetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pNodeParams = hipFuncArgs.pNodeParams;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipMemAllocNodeParams *)>("hipGraphMemAllocNodeGetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void * dev_ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemFreeNodeGetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemFreeNodeGetParams_api_args_t hipFuncArgs{node, dev_ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,void *)>("hipGraphMemFreeNodeGetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.dev_ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		dev_ptr = hipFuncArgs.dev_ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,void *)>("hipGraphMemFreeNodeGetParams");
		return hipFunc(node, dev_ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemcpyNodeGetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemcpyNodeGetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipMemcpy3DParms *)>("hipGraphMemcpyNodeGetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pNodeParams = hipFuncArgs.pNodeParams;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipMemcpy3DParms *)>("hipGraphMemcpyNodeGetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemcpyNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemcpyNodeSetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const hipMemcpy3DParms *)>("hipGraphMemcpyNodeSetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const hipMemcpy3DParms *)>("hipGraphMemcpyNodeSetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void * dst, const void * src, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemcpyNodeSetParams1D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemcpyNodeSetParams1D_api_args_t hipFuncArgs{node, dst, src, count, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParams1D");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.count, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		dst = hipFuncArgs.dst;
		count = hipFuncArgs.count;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,void *,const void *,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParams1D");
		return hipFunc(node, dst, src, count, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemcpyNodeSetParamsFromSymbol;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemcpyNodeSetParamsFromSymbol_api_args_t hipFuncArgs{node, dst, symbol, count, offset, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParamsFromSymbol");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.dst, hipFuncArgs.symbol, hipFuncArgs.count, hipFuncArgs.offset, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		dst = hipFuncArgs.dst;
		count = hipFuncArgs.count;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParamsFromSymbol");
		return hipFunc(node, dst, symbol, count, offset, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemcpyNodeSetParamsToSymbol;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemcpyNodeSetParamsToSymbol_api_args_t hipFuncArgs{node, symbol, src, count, offset, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParamsToSymbol");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.symbol, hipFuncArgs.src, hipFuncArgs.count, hipFuncArgs.offset, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		count = hipFuncArgs.count;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipGraphMemcpyNodeSetParamsToSymbol");
		return hipFunc(node, symbol, src, count, offset, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemsetNodeGetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemsetNodeGetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipMemsetParams *)>("hipGraphMemsetNodeGetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pNodeParams = hipFuncArgs.pNodeParams;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipMemsetParams *)>("hipGraphMemsetNodeGetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams * pNodeParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphMemsetNodeSetParams;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphMemsetNodeSetParams_api_args_t hipFuncArgs{node, pNodeParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const hipMemsetParams *)>("hipGraphMemsetNodeSetParams");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pNodeParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,const hipMemsetParams *)>("hipGraphMemsetNodeSetParams");
		return hipFunc(node, pNodeParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeFindInClone(hipGraphNode_t * pNode, hipGraphNode_t originalNode, hipGraph_t clonedGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphNodeFindInClone;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphNodeFindInClone_api_args_t hipFuncArgs{pNode, originalNode, clonedGraph};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraphNode_t,hipGraph_t)>("hipGraphNodeFindInClone");
			out = hipFunc(hipFuncArgs.pNode, hipFuncArgs.originalNode, hipFuncArgs.clonedGraph);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pNode = hipFuncArgs.pNode;
		originalNode = hipFuncArgs.originalNode;
		clonedGraph = hipFuncArgs.clonedGraph;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t *,hipGraphNode_t,hipGraph_t)>("hipGraphNodeFindInClone");
		return hipFunc(pNode, originalNode, clonedGraph);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t * pDependencies, size_t * pNumDependencies) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphNodeGetDependencies;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphNodeGetDependencies_api_args_t hipFuncArgs{node, pDependencies, pNumDependencies};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t *,size_t *)>("hipGraphNodeGetDependencies");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pDependencies, hipFuncArgs.pNumDependencies);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pDependencies = hipFuncArgs.pDependencies;
		pNumDependencies = hipFuncArgs.pNumDependencies;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t *,size_t *)>("hipGraphNodeGetDependencies");
		return hipFunc(node, pDependencies, pNumDependencies);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t * pDependentNodes, size_t * pNumDependentNodes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphNodeGetDependentNodes;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphNodeGetDependentNodes_api_args_t hipFuncArgs{node, pDependentNodes, pNumDependentNodes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t *,size_t *)>("hipGraphNodeGetDependentNodes");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pDependentNodes, hipFuncArgs.pNumDependentNodes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pDependentNodes = hipFuncArgs.pDependentNodes;
		pNumDependentNodes = hipFuncArgs.pNumDependentNodes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNode_t *,size_t *)>("hipGraphNodeGetDependentNodes");
		return hipFunc(node, pDependentNodes, pNumDependentNodes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int * isEnabled) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphNodeGetEnabled;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphNodeGetEnabled_api_args_t hipFuncArgs{hGraphExec, hNode, isEnabled};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,unsigned int *)>("hipGraphNodeGetEnabled");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.hNode, hipFuncArgs.isEnabled);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		hNode = hipFuncArgs.hNode;
		isEnabled = hipFuncArgs.isEnabled;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,unsigned int *)>("hipGraphNodeGetEnabled");
		return hipFunc(hGraphExec, hNode, isEnabled);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType * pType) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphNodeGetType;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphNodeGetType_api_args_t hipFuncArgs{node, pType};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNodeType *)>("hipGraphNodeGetType");
			out = hipFunc(hipFuncArgs.node, hipFuncArgs.pType);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		node = hipFuncArgs.node;
		pType = hipFuncArgs.pType;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphNode_t,hipGraphNodeType *)>("hipGraphNodeGetType");
		return hipFunc(node, pType);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int isEnabled) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphNodeSetEnabled;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphNodeSetEnabled_api_args_t hipFuncArgs{hGraphExec, hNode, isEnabled};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,unsigned int)>("hipGraphNodeSetEnabled");
			out = hipFunc(hipFuncArgs.hGraphExec, hipFuncArgs.hNode, hipFuncArgs.isEnabled);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hGraphExec = hipFuncArgs.hGraphExec;
		hNode = hipFuncArgs.hNode;
		isEnabled = hipFuncArgs.isEnabled;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipGraphNode_t,unsigned int)>("hipGraphNodeSetEnabled");
		return hipFunc(hGraphExec, hNode, isEnabled);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphReleaseUserObject;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphReleaseUserObject_api_args_t hipFuncArgs{graph, object, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipUserObject_t,unsigned int)>("hipGraphReleaseUserObject");
			out = hipFunc(hipFuncArgs.graph, hipFuncArgs.object, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;
		object = hipFuncArgs.object;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipUserObject_t,unsigned int)>("hipGraphReleaseUserObject");
		return hipFunc(graph, object, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t * from, const hipGraphNode_t * to, size_t numDependencies) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphRemoveDependencies;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphRemoveDependencies_api_args_t hipFuncArgs{graph, from, to, numDependencies};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,const hipGraphNode_t *,const hipGraphNode_t *,size_t)>("hipGraphRemoveDependencies");
			out = hipFunc(hipFuncArgs.graph, hipFuncArgs.from, hipFuncArgs.to, hipFuncArgs.numDependencies);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;
		numDependencies = hipFuncArgs.numDependencies;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,const hipGraphNode_t *,const hipGraphNode_t *,size_t)>("hipGraphRemoveDependencies");
		return hipFunc(graph, from, to, numDependencies);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphRetainUserObject;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphRetainUserObject_api_args_t hipFuncArgs{graph, object, count, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipUserObject_t,unsigned int,unsigned int)>("hipGraphRetainUserObject");
			out = hipFunc(hipFuncArgs.graph, hipFuncArgs.object, hipFuncArgs.count, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graph = hipFuncArgs.graph;
		object = hipFuncArgs.object;
		count = hipFuncArgs.count;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraph_t,hipUserObject_t,unsigned int,unsigned int)>("hipGraphRetainUserObject");
		return hipFunc(graph, object, count, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphUpload;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphUpload_api_args_t hipFuncArgs{graphExec, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipStream_t)>("hipGraphUpload");
			out = hipFunc(hipFuncArgs.graphExec, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		graphExec = hipFuncArgs.graphExec;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphExec_t,hipStream_t)>("hipGraphUpload");
		return hipFunc(graphExec, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource * * resource, GLuint buffer, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphicsGLRegisterBuffer;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphicsGLRegisterBuffer_api_args_t hipFuncArgs{resource, buffer, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphicsResource * *,GLuint,unsigned int)>("hipGraphicsGLRegisterBuffer");
			out = hipFunc(hipFuncArgs.resource, hipFuncArgs.buffer, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		resource = hipFuncArgs.resource;
		buffer = hipFuncArgs.buffer;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphicsResource * *,GLuint,unsigned int)>("hipGraphicsGLRegisterBuffer");
		return hipFunc(resource, buffer, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource * * resource, GLuint image, GLenum target, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphicsGLRegisterImage;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphicsGLRegisterImage_api_args_t hipFuncArgs{resource, image, target, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphicsResource * *,GLuint,GLenum,unsigned int)>("hipGraphicsGLRegisterImage");
			out = hipFunc(hipFuncArgs.resource, hipFuncArgs.image, hipFuncArgs.target, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		resource = hipFuncArgs.resource;
		image = hipFuncArgs.image;
		target = hipFuncArgs.target;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphicsResource * *,GLuint,GLenum,unsigned int)>("hipGraphicsGLRegisterImage");
		return hipFunc(resource, image, target, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t * resources, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphicsMapResources;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphicsMapResources_api_args_t hipFuncArgs{count, resources, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipGraphicsResource_t *,hipStream_t)>("hipGraphicsMapResources");
			out = hipFunc(hipFuncArgs.count, hipFuncArgs.resources, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		count = hipFuncArgs.count;
		resources = hipFuncArgs.resources;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipGraphicsResource_t *,hipStream_t)>("hipGraphicsMapResources");
		return hipFunc(count, resources, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsResourceGetMappedPointer(void * * devPtr, size_t * size, hipGraphicsResource_t resource) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphicsResourceGetMappedPointer;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphicsResourceGetMappedPointer_api_args_t hipFuncArgs{devPtr, size, resource};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t *,hipGraphicsResource_t)>("hipGraphicsResourceGetMappedPointer");
			out = hipFunc(hipFuncArgs.devPtr, hipFuncArgs.size, hipFuncArgs.resource);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		devPtr = hipFuncArgs.devPtr;
		size = hipFuncArgs.size;
		resource = hipFuncArgs.resource;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t *,hipGraphicsResource_t)>("hipGraphicsResourceGetMappedPointer");
		return hipFunc(devPtr, size, resource);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t * array, hipGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphicsSubResourceGetMappedArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphicsSubResourceGetMappedArray_api_args_t hipFuncArgs{array, resource, arrayIndex, mipLevel};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,hipGraphicsResource_t,unsigned int,unsigned int)>("hipGraphicsSubResourceGetMappedArray");
			out = hipFunc(hipFuncArgs.array, hipFuncArgs.resource, hipFuncArgs.arrayIndex, hipFuncArgs.mipLevel);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		array = hipFuncArgs.array;
		resource = hipFuncArgs.resource;
		arrayIndex = hipFuncArgs.arrayIndex;
		mipLevel = hipFuncArgs.mipLevel;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,hipGraphicsResource_t,unsigned int,unsigned int)>("hipGraphicsSubResourceGetMappedArray");
		return hipFunc(array, resource, arrayIndex, mipLevel);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t * resources, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphicsUnmapResources;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphicsUnmapResources_api_args_t hipFuncArgs{count, resources, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipGraphicsResource_t *,hipStream_t)>("hipGraphicsUnmapResources");
			out = hipFunc(hipFuncArgs.count, hipFuncArgs.resources, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		count = hipFuncArgs.count;
		resources = hipFuncArgs.resources;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int,hipGraphicsResource_t *,hipStream_t)>("hipGraphicsUnmapResources");
		return hipFunc(count, resources, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipGraphicsUnregisterResource;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipGraphicsUnregisterResource_api_args_t hipFuncArgs{resource};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphicsResource_t)>("hipGraphicsUnregisterResource");
			out = hipFunc(hipFuncArgs.resource);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		resource = hipFuncArgs.resource;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipGraphicsResource_t)>("hipGraphicsUnregisterResource");
		return hipFunc(resource);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, size_t sharedMemBytes, hipStream_t hStream, void * * kernelParams, void * * extra, hipEvent_t startEvent, hipEvent_t stopEvent) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipHccModuleLaunchKernel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipHccModuleLaunchKernel_api_args_t hipFuncArgs{f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t)>("hipHccModuleLaunchKernel");
			out = hipFunc(hipFuncArgs.f, hipFuncArgs.globalWorkSizeX, hipFuncArgs.globalWorkSizeY, hipFuncArgs.globalWorkSizeZ, hipFuncArgs.blockDimX, hipFuncArgs.blockDimY, hipFuncArgs.blockDimZ, hipFuncArgs.sharedMemBytes, hipFuncArgs.hStream, hipFuncArgs.kernelParams, hipFuncArgs.extra, hipFuncArgs.startEvent, hipFuncArgs.stopEvent);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		f = hipFuncArgs.f;
		globalWorkSizeX = hipFuncArgs.globalWorkSizeX;
		globalWorkSizeY = hipFuncArgs.globalWorkSizeY;
		globalWorkSizeZ = hipFuncArgs.globalWorkSizeZ;
		blockDimX = hipFuncArgs.blockDimX;
		blockDimY = hipFuncArgs.blockDimY;
		blockDimZ = hipFuncArgs.blockDimZ;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		hStream = hipFuncArgs.hStream;
		kernelParams = hipFuncArgs.kernelParams;
		extra = hipFuncArgs.extra;
		startEvent = hipFuncArgs.startEvent;
		stopEvent = hipFuncArgs.stopEvent;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t)>("hipHccModuleLaunchKernel");
		return hipFunc(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipHostAlloc(void * * ptr, size_t size, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipHostAlloc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipHostAlloc_api_args_t hipFuncArgs{ptr, size, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipHostAlloc");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipHostAlloc");
		return hipFunc(ptr, size, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipHostFree(void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipHostFree;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipHostFree_api_args_t hipFuncArgs{ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipHostFree");
			out = hipFunc(hipFuncArgs.ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipHostFree");
		return hipFunc(ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipHostGetDevicePointer(void * * devPtr, void * hstPtr, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipHostGetDevicePointer;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipHostGetDevicePointer_api_args_t hipFuncArgs{devPtr, hstPtr, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,void *,unsigned int)>("hipHostGetDevicePointer");
			out = hipFunc(hipFuncArgs.devPtr, hipFuncArgs.hstPtr, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		devPtr = hipFuncArgs.devPtr;
		hstPtr = hipFuncArgs.hstPtr;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,void *,unsigned int)>("hipHostGetDevicePointer");
		return hipFunc(devPtr, hstPtr, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipHostGetFlags(unsigned int * flagsPtr, void * hostPtr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipHostGetFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipHostGetFlags_api_args_t hipFuncArgs{flagsPtr, hostPtr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *,void *)>("hipHostGetFlags");
			out = hipFunc(hipFuncArgs.flagsPtr, hipFuncArgs.hostPtr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		flagsPtr = hipFuncArgs.flagsPtr;
		hostPtr = hipFuncArgs.hostPtr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *,void *)>("hipHostGetFlags");
		return hipFunc(flagsPtr, hostPtr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipHostMalloc(void * * ptr, size_t size, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipHostMalloc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipHostMalloc_api_args_t hipFuncArgs{ptr, size, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipHostMalloc");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipHostMalloc");
		return hipFunc(ptr, size, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipHostRegister(void * hostPtr, size_t sizeBytes, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipHostRegister;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipHostRegister_api_args_t hipFuncArgs{hostPtr, sizeBytes, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,unsigned int)>("hipHostRegister");
			out = hipFunc(hipFuncArgs.hostPtr, hipFuncArgs.sizeBytes, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hostPtr = hipFuncArgs.hostPtr;
		sizeBytes = hipFuncArgs.sizeBytes;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,unsigned int)>("hipHostRegister");
		return hipFunc(hostPtr, sizeBytes, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipHostUnregister(void * hostPtr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipHostUnregister;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipHostUnregister_api_args_t hipFuncArgs{hostPtr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipHostUnregister");
			out = hipFunc(hipFuncArgs.hostPtr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hostPtr = hipFuncArgs.hostPtr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipHostUnregister");
		return hipFunc(hostPtr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipImportExternalMemory(hipExternalMemory_t * extMem_out, const hipExternalMemoryHandleDesc * memHandleDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipImportExternalMemory;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipImportExternalMemory_api_args_t hipFuncArgs{extMem_out, memHandleDesc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipExternalMemory_t *,const hipExternalMemoryHandleDesc *)>("hipImportExternalMemory");
			out = hipFunc(hipFuncArgs.extMem_out, hipFuncArgs.memHandleDesc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		extMem_out = hipFuncArgs.extMem_out;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipExternalMemory_t *,const hipExternalMemoryHandleDesc *)>("hipImportExternalMemory");
		return hipFunc(extMem_out, memHandleDesc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t * extSem_out, const hipExternalSemaphoreHandleDesc * semHandleDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipImportExternalSemaphore;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipImportExternalSemaphore_api_args_t hipFuncArgs{extSem_out, semHandleDesc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipExternalSemaphore_t *,const hipExternalSemaphoreHandleDesc *)>("hipImportExternalSemaphore");
			out = hipFunc(hipFuncArgs.extSem_out, hipFuncArgs.semHandleDesc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		extSem_out = hipFuncArgs.extSem_out;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipExternalSemaphore_t *,const hipExternalSemaphoreHandleDesc *)>("hipImportExternalSemaphore");
		return hipFunc(extSem_out, semHandleDesc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipInit(unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipInit;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipInit_api_args_t hipFuncArgs{flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int)>("hipInit");
			out = hipFunc(hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int)>("hipInit");
		return hipFunc(flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcCloseMemHandle(void * devPtr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipIpcCloseMemHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipIpcCloseMemHandle_api_args_t hipFuncArgs{devPtr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipIpcCloseMemHandle");
			out = hipFunc(hipFuncArgs.devPtr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		devPtr = hipFuncArgs.devPtr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *)>("hipIpcCloseMemHandle");
		return hipFunc(devPtr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t * handle, hipEvent_t event) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipIpcGetEventHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipIpcGetEventHandle_api_args_t hipFuncArgs{handle, event};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipIpcEventHandle_t *,hipEvent_t)>("hipIpcGetEventHandle");
			out = hipFunc(hipFuncArgs.handle, hipFuncArgs.event);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		handle = hipFuncArgs.handle;
		event = hipFuncArgs.event;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipIpcEventHandle_t *,hipEvent_t)>("hipIpcGetEventHandle");
		return hipFunc(handle, event);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t * handle, void * devPtr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipIpcGetMemHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipIpcGetMemHandle_api_args_t hipFuncArgs{handle, devPtr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipIpcMemHandle_t *,void *)>("hipIpcGetMemHandle");
			out = hipFunc(hipFuncArgs.handle, hipFuncArgs.devPtr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		handle = hipFuncArgs.handle;
		devPtr = hipFuncArgs.devPtr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipIpcMemHandle_t *,void *)>("hipIpcGetMemHandle");
		return hipFunc(handle, devPtr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcOpenEventHandle(hipEvent_t * event, hipIpcEventHandle_t handle) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipIpcOpenEventHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipIpcOpenEventHandle_api_args_t hipFuncArgs{event, handle};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t *,hipIpcEventHandle_t)>("hipIpcOpenEventHandle");
			out = hipFunc(hipFuncArgs.event, hipFuncArgs.handle);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		event = hipFuncArgs.event;
		handle = hipFuncArgs.handle;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipEvent_t *,hipIpcEventHandle_t)>("hipIpcOpenEventHandle");
		return hipFunc(event, handle);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipIpcOpenMemHandle(void * * devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipIpcOpenMemHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipIpcOpenMemHandle_api_args_t hipFuncArgs{devPtr, handle, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,hipIpcMemHandle_t,unsigned int)>("hipIpcOpenMemHandle");
			out = hipFunc(hipFuncArgs.devPtr, hipFuncArgs.handle, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		devPtr = hipFuncArgs.devPtr;
		handle = hipFuncArgs.handle;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,hipIpcMemHandle_t,unsigned int)>("hipIpcOpenMemHandle");
		return hipFunc(devPtr, handle, flags);
	};
}

extern "C" __attribute__((visibility("default")))
const char * hipKernelNameRef(const hipFunction_t f) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipKernelNameRef;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipKernelNameRef_api_args_t hipFuncArgs{f};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(const hipFunction_t)>("hipKernelNameRef");
			out = hipFunc(hipFuncArgs.f);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<const char *>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(const hipFunction_t)>("hipKernelNameRef");
		return hipFunc(f);
	};
}

extern "C" __attribute__((visibility("default")))
const char * hipKernelNameRefByPtr(const void * hostFunction, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipKernelNameRefByPtr;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipKernelNameRefByPtr_api_args_t hipFuncArgs{hostFunction, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(const void *,hipStream_t)>("hipKernelNameRefByPtr");
			out = hipFunc(hipFuncArgs.hostFunction, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<const char *>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<const char *(*)(const void *,hipStream_t)>("hipKernelNameRefByPtr");
		return hipFunc(hostFunction, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchByPtr(const void * func) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipLaunchByPtr;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipLaunchByPtr_api_args_t hipFuncArgs{func};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *)>("hipLaunchByPtr");
			out = hipFunc(hipFuncArgs.func);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *)>("hipLaunchByPtr");
		return hipFunc(func);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchCooperativeKernel(const void * f, dim3 gridDim, dim3 blockDimX, void * * kernelParams, unsigned int sharedMemBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipLaunchCooperativeKernel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipLaunchCooperativeKernel_api_args_t hipFuncArgs{f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,unsigned int,hipStream_t)>("hipLaunchCooperativeKernel");
			out = hipFunc(hipFuncArgs.f, hipFuncArgs.gridDim, hipFuncArgs.blockDimX, hipFuncArgs.kernelParams, hipFuncArgs.sharedMemBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridDim = hipFuncArgs.gridDim;
		blockDimX = hipFuncArgs.blockDimX;
		kernelParams = hipFuncArgs.kernelParams;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,unsigned int,hipStream_t)>("hipLaunchCooperativeKernel");
		return hipFunc(f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams * launchParamsList, int numDevices, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipLaunchCooperativeKernelMultiDevice;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipLaunchCooperativeKernelMultiDevice_api_args_t hipFuncArgs{launchParamsList, numDevices, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipLaunchParams *,int,unsigned int)>("hipLaunchCooperativeKernelMultiDevice");
			out = hipFunc(hipFuncArgs.launchParamsList, hipFuncArgs.numDevices, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		launchParamsList = hipFuncArgs.launchParamsList;
		numDevices = hipFuncArgs.numDevices;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipLaunchParams *,int,unsigned int)>("hipLaunchCooperativeKernelMultiDevice");
		return hipFunc(launchParamsList, numDevices, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void * userData) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipLaunchHostFunc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipLaunchHostFunc_api_args_t hipFuncArgs{stream, fn, userData};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipHostFn_t,void *)>("hipLaunchHostFunc");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.fn, hipFuncArgs.userData);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		fn = hipFuncArgs.fn;
		userData = hipFuncArgs.userData;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipHostFn_t,void *)>("hipLaunchHostFunc");
		return hipFunc(stream, fn, userData);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchKernel(const void * function_address, dim3 numBlocks, dim3 dimBlocks, void * * args, size_t sharedMemBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipLaunchKernel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipLaunchKernel_api_args_t hipFuncArgs{function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel");
			out = hipFunc(hipFuncArgs.function_address, hipFuncArgs.numBlocks, hipFuncArgs.dimBlocks, hipFuncArgs.args, hipFuncArgs.sharedMemBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numBlocks = hipFuncArgs.numBlocks;
		dimBlocks = hipFuncArgs.dimBlocks;
		args = hipFuncArgs.args;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel");
		return hipFunc(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchKernel_common(const void * hostFunction, dim3 gridDim, dim3 blockDim, void * * args, size_t sharedMemBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipLaunchKernel_common;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipLaunchKernel_common_api_args_t hipFuncArgs{hostFunction, gridDim, blockDim, args, sharedMemBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel_common");
			out = hipFunc(hipFuncArgs.hostFunction, hipFuncArgs.gridDim, hipFuncArgs.blockDim, hipFuncArgs.args, hipFuncArgs.sharedMemBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridDim = hipFuncArgs.gridDim;
		blockDim = hipFuncArgs.blockDim;
		args = hipFuncArgs.args;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel_common");
		return hipFunc(hostFunction, gridDim, blockDim, args, sharedMemBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipLaunchKernel_spt(const void * hostFunction, dim3 gridDim, dim3 blockDim, void * * args, size_t sharedMemBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hipLaunchKernel_spt;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipLaunchKernel_spt_api_args_t hipFuncArgs{hostFunction, gridDim, blockDim, args, sharedMemBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel_spt");
			out = hipFunc(hipFuncArgs.hostFunction, hipFuncArgs.gridDim, hipFuncArgs.blockDim, hipFuncArgs.args, hipFuncArgs.sharedMemBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridDim = hipFuncArgs.gridDim;
		blockDim = hipFuncArgs.blockDim;
		args = hipFuncArgs.args;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,dim3,dim3,void * *,size_t,hipStream_t)>("hipLaunchKernel_spt");
		return hipFunc(hostFunction, gridDim, blockDim, args, sharedMemBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMalloc(void * * ptr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMalloc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMalloc_api_args_t hipFuncArgs{ptr, size};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t)>("hipMalloc");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t)>("hipMalloc");
		return hipFunc(ptr, size);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMalloc3D(hipPitchedPtr * pitchedDevPtr, hipExtent extent) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMalloc3D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMalloc3D_api_args_t hipFuncArgs{pitchedDevPtr, extent};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipPitchedPtr *,hipExtent)>("hipMalloc3D");
			out = hipFunc(hipFuncArgs.pitchedDevPtr, hipFuncArgs.extent);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pitchedDevPtr = hipFuncArgs.pitchedDevPtr;
		extent = hipFuncArgs.extent;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipPitchedPtr *,hipExtent)>("hipMalloc3D");
		return hipFunc(pitchedDevPtr, extent);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMalloc3DArray(hipArray_t * array, const struct hipChannelFormatDesc * desc, struct hipExtent extent, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMalloc3DArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMalloc3DArray_api_args_t hipFuncArgs{array, desc, extent, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const struct hipChannelFormatDesc *,struct hipExtent,unsigned int)>("hipMalloc3DArray");
			out = hipFunc(hipFuncArgs.array, hipFuncArgs.desc, hipFuncArgs.extent, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		array = hipFuncArgs.array;
		extent = hipFuncArgs.extent;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const struct hipChannelFormatDesc *,struct hipExtent,unsigned int)>("hipMalloc3DArray");
		return hipFunc(array, desc, extent, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocArray(hipArray_t * array, const hipChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMallocArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMallocArray_api_args_t hipFuncArgs{array, desc, width, height, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const hipChannelFormatDesc *,size_t,size_t,unsigned int)>("hipMallocArray");
			out = hipFunc(hipFuncArgs.array, hipFuncArgs.desc, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		array = hipFuncArgs.array;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const hipChannelFormatDesc *,size_t,size_t,unsigned int)>("hipMallocArray");
		return hipFunc(array, desc, width, height, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocAsync(void * * dev_ptr, size_t size, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMallocAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMallocAsync_api_args_t hipFuncArgs{dev_ptr, size, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,hipStream_t)>("hipMallocAsync");
			out = hipFunc(hipFuncArgs.dev_ptr, hipFuncArgs.size, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev_ptr = hipFuncArgs.dev_ptr;
		size = hipFuncArgs.size;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,hipStream_t)>("hipMallocAsync");
		return hipFunc(dev_ptr, size, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocFromPoolAsync(void * * dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMallocFromPoolAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMallocFromPoolAsync_api_args_t hipFuncArgs{dev_ptr, size, mem_pool, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,hipMemPool_t,hipStream_t)>("hipMallocFromPoolAsync");
			out = hipFunc(hipFuncArgs.dev_ptr, hipFuncArgs.size, hipFuncArgs.mem_pool, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev_ptr = hipFuncArgs.dev_ptr;
		size = hipFuncArgs.size;
		mem_pool = hipFuncArgs.mem_pool;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,hipMemPool_t,hipStream_t)>("hipMallocFromPoolAsync");
		return hipFunc(dev_ptr, size, mem_pool, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocHost(void * * ptr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMallocHost;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMallocHost_api_args_t hipFuncArgs{ptr, size};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t)>("hipMallocHost");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t)>("hipMallocHost");
		return hipFunc(ptr, size);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocManaged(void * * dev_ptr, size_t size, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMallocManaged;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMallocManaged_api_args_t hipFuncArgs{dev_ptr, size, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipMallocManaged");
			out = hipFunc(hipFuncArgs.dev_ptr, hipFuncArgs.size, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev_ptr = hipFuncArgs.dev_ptr;
		size = hipFuncArgs.size;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,unsigned int)>("hipMallocManaged");
		return hipFunc(dev_ptr, size, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocMipmappedArray(hipMipmappedArray_t * mipmappedArray, const struct hipChannelFormatDesc * desc, struct hipExtent extent, unsigned int numLevels, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMallocMipmappedArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMallocMipmappedArray_api_args_t hipFuncArgs{mipmappedArray, desc, extent, numLevels, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,const struct hipChannelFormatDesc *,struct hipExtent,unsigned int,unsigned int)>("hipMallocMipmappedArray");
			out = hipFunc(hipFuncArgs.mipmappedArray, hipFuncArgs.desc, hipFuncArgs.extent, hipFuncArgs.numLevels, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mipmappedArray = hipFuncArgs.mipmappedArray;
		extent = hipFuncArgs.extent;
		numLevels = hipFuncArgs.numLevels;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,const struct hipChannelFormatDesc *,struct hipExtent,unsigned int,unsigned int)>("hipMallocMipmappedArray");
		return hipFunc(mipmappedArray, desc, extent, numLevels, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMallocPitch(void * * ptr, size_t * pitch, size_t width, size_t height) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMallocPitch;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMallocPitch_api_args_t hipFuncArgs{ptr, pitch, width, height};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t *,size_t,size_t)>("hipMallocPitch");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.pitch, hipFuncArgs.width, hipFuncArgs.height);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		pitch = hipFuncArgs.pitch;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t *,size_t,size_t)>("hipMallocPitch");
		return hipFunc(ptr, pitch, width, height);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAddressFree(void * devPtr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemAddressFree;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemAddressFree_api_args_t hipFuncArgs{devPtr, size};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t)>("hipMemAddressFree");
			out = hipFunc(hipFuncArgs.devPtr, hipFuncArgs.size);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		devPtr = hipFuncArgs.devPtr;
		size = hipFuncArgs.size;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t)>("hipMemAddressFree");
		return hipFunc(devPtr, size);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAddressReserve(void * * ptr, size_t size, size_t alignment, void * addr, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemAddressReserve;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemAddressReserve_api_args_t hipFuncArgs{ptr, size, alignment, addr, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,size_t,void *,unsigned long long)>("hipMemAddressReserve");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size, hipFuncArgs.alignment, hipFuncArgs.addr, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;
		alignment = hipFuncArgs.alignment;
		addr = hipFuncArgs.addr;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t,size_t,void *,unsigned long long)>("hipMemAddressReserve");
		return hipFunc(ptr, size, alignment, addr, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAdvise(const void * dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemAdvise;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemAdvise_api_args_t hipFuncArgs{dev_ptr, count, advice, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,size_t,hipMemoryAdvise,int)>("hipMemAdvise");
			out = hipFunc(hipFuncArgs.dev_ptr, hipFuncArgs.count, hipFuncArgs.advice, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		count = hipFuncArgs.count;
		advice = hipFuncArgs.advice;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,size_t,hipMemoryAdvise,int)>("hipMemAdvise");
		return hipFunc(dev_ptr, count, advice, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAllocHost(void * * ptr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemAllocHost;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemAllocHost_api_args_t hipFuncArgs{ptr, size};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t)>("hipMemAllocHost");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t)>("hipMemAllocHost");
		return hipFunc(ptr, size);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemAllocPitch(hipDeviceptr_t * dptr, size_t * pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemAllocPitch;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemAllocPitch_api_args_t hipFuncArgs{dptr, pitch, widthInBytes, height, elementSizeBytes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,size_t,size_t,unsigned int)>("hipMemAllocPitch");
			out = hipFunc(hipFuncArgs.dptr, hipFuncArgs.pitch, hipFuncArgs.widthInBytes, hipFuncArgs.height, hipFuncArgs.elementSizeBytes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dptr = hipFuncArgs.dptr;
		pitch = hipFuncArgs.pitch;
		widthInBytes = hipFuncArgs.widthInBytes;
		height = hipFuncArgs.height;
		elementSizeBytes = hipFuncArgs.elementSizeBytes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,size_t,size_t,unsigned int)>("hipMemAllocPitch");
		return hipFunc(dptr, pitch, widthInBytes, height, elementSizeBytes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t * handle, size_t size, const hipMemAllocationProp * prop, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemCreate_api_args_t hipFuncArgs{handle, size, prop, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,size_t,const hipMemAllocationProp *,unsigned long long)>("hipMemCreate");
			out = hipFunc(hipFuncArgs.handle, hipFuncArgs.size, hipFuncArgs.prop, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		handle = hipFuncArgs.handle;
		size = hipFuncArgs.size;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,size_t,const hipMemAllocationProp *,unsigned long long)>("hipMemCreate");
		return hipFunc(handle, size, prop, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemExportToShareableHandle(void * shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemExportToShareableHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemExportToShareableHandle_api_args_t hipFuncArgs{shareableHandle, handle, handleType, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipMemGenericAllocationHandle_t,hipMemAllocationHandleType,unsigned long long)>("hipMemExportToShareableHandle");
			out = hipFunc(hipFuncArgs.shareableHandle, hipFuncArgs.handle, hipFuncArgs.handleType, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		shareableHandle = hipFuncArgs.shareableHandle;
		handle = hipFuncArgs.handle;
		handleType = hipFuncArgs.handleType;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipMemGenericAllocationHandle_t,hipMemAllocationHandleType,unsigned long long)>("hipMemExportToShareableHandle");
		return hipFunc(shareableHandle, handle, handleType, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetAccess(unsigned long long * flags, const hipMemLocation * location, void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemGetAccess;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemGetAccess_api_args_t hipFuncArgs{flags, location, ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned long long *,const hipMemLocation *,void *)>("hipMemGetAccess");
			out = hipFunc(hipFuncArgs.flags, hipFuncArgs.location, hipFuncArgs.ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		flags = hipFuncArgs.flags;
		ptr = hipFuncArgs.ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned long long *,const hipMemLocation *,void *)>("hipMemGetAccess");
		return hipFunc(flags, location, ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetAddressRange(hipDeviceptr_t * pbase, size_t * psize, hipDeviceptr_t dptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemGetAddressRange;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemGetAddressRange_api_args_t hipFuncArgs{pbase, psize, dptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,hipDeviceptr_t)>("hipMemGetAddressRange");
			out = hipFunc(hipFuncArgs.pbase, hipFuncArgs.psize, hipFuncArgs.dptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pbase = hipFuncArgs.pbase;
		psize = hipFuncArgs.psize;
		dptr = hipFuncArgs.dptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,hipDeviceptr_t)>("hipMemGetAddressRange");
		return hipFunc(pbase, psize, dptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetAllocationGranularity(size_t * granularity, const hipMemAllocationProp * prop, hipMemAllocationGranularity_flags option) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemGetAllocationGranularity;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemGetAllocationGranularity_api_args_t hipFuncArgs{granularity, prop, option};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const hipMemAllocationProp *,hipMemAllocationGranularity_flags)>("hipMemGetAllocationGranularity");
			out = hipFunc(hipFuncArgs.granularity, hipFuncArgs.prop, hipFuncArgs.option);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		granularity = hipFuncArgs.granularity;
		option = hipFuncArgs.option;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,const hipMemAllocationProp *,hipMemAllocationGranularity_flags)>("hipMemGetAllocationGranularity");
		return hipFunc(granularity, prop, option);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp * prop, hipMemGenericAllocationHandle_t handle) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemGetAllocationPropertiesFromHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemGetAllocationPropertiesFromHandle_api_args_t hipFuncArgs{prop, handle};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemAllocationProp *,hipMemGenericAllocationHandle_t)>("hipMemGetAllocationPropertiesFromHandle");
			out = hipFunc(hipFuncArgs.prop, hipFuncArgs.handle);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		prop = hipFuncArgs.prop;
		handle = hipFuncArgs.handle;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemAllocationProp *,hipMemGenericAllocationHandle_t)>("hipMemGetAllocationPropertiesFromHandle");
		return hipFunc(prop, handle);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemGetInfo(size_t * free, size_t * total) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemGetInfo;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemGetInfo_api_args_t hipFuncArgs{free, total};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,size_t *)>("hipMemGetInfo");
			out = hipFunc(hipFuncArgs.free, hipFuncArgs.total);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		free = hipFuncArgs.free;
		total = hipFuncArgs.total;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,size_t *)>("hipMemGetInfo");
		return hipFunc(free, total);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t * handle, void * osHandle, hipMemAllocationHandleType shHandleType) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemImportFromShareableHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemImportFromShareableHandle_api_args_t hipFuncArgs{handle, osHandle, shHandleType};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,void *,hipMemAllocationHandleType)>("hipMemImportFromShareableHandle");
			out = hipFunc(hipFuncArgs.handle, hipFuncArgs.osHandle, hipFuncArgs.shHandleType);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		handle = hipFuncArgs.handle;
		osHandle = hipFuncArgs.osHandle;
		shHandleType = hipFuncArgs.shHandleType;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,void *,hipMemAllocationHandleType)>("hipMemImportFromShareableHandle");
		return hipFunc(handle, osHandle, shHandleType);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemMap(void * ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemMap;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemMap_api_args_t hipFuncArgs{ptr, size, offset, handle, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,size_t,hipMemGenericAllocationHandle_t,unsigned long long)>("hipMemMap");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size, hipFuncArgs.offset, hipFuncArgs.handle, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;
		offset = hipFuncArgs.offset;
		handle = hipFuncArgs.handle;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,size_t,hipMemGenericAllocationHandle_t,unsigned long long)>("hipMemMap");
		return hipFunc(ptr, size, offset, handle, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemMapArrayAsync(hipArrayMapInfo * mapInfoList, unsigned int count, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemMapArrayAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemMapArrayAsync_api_args_t hipFuncArgs{mapInfoList, count, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArrayMapInfo *,unsigned int,hipStream_t)>("hipMemMapArrayAsync");
			out = hipFunc(hipFuncArgs.mapInfoList, hipFuncArgs.count, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mapInfoList = hipFuncArgs.mapInfoList;
		count = hipFuncArgs.count;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArrayMapInfo *,unsigned int,hipStream_t)>("hipMemMapArrayAsync");
		return hipFunc(mapInfoList, count, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolCreate(hipMemPool_t * mem_pool, const hipMemPoolProps * pool_props) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolCreate_api_args_t hipFuncArgs{mem_pool, pool_props};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t *,const hipMemPoolProps *)>("hipMemPoolCreate");
			out = hipFunc(hipFuncArgs.mem_pool, hipFuncArgs.pool_props);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t *,const hipMemPoolProps *)>("hipMemPoolCreate");
		return hipFunc(mem_pool, pool_props);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolDestroy_api_args_t hipFuncArgs{mem_pool};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t)>("hipMemPoolDestroy");
			out = hipFunc(hipFuncArgs.mem_pool);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t)>("hipMemPoolDestroy");
		return hipFunc(mem_pool);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData * export_data, void * dev_ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolExportPointer;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolExportPointer_api_args_t hipFuncArgs{export_data, dev_ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPoolPtrExportData *,void *)>("hipMemPoolExportPointer");
			out = hipFunc(hipFuncArgs.export_data, hipFuncArgs.dev_ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		export_data = hipFuncArgs.export_data;
		dev_ptr = hipFuncArgs.dev_ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPoolPtrExportData *,void *)>("hipMemPoolExportPointer");
		return hipFunc(export_data, dev_ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolExportToShareableHandle(void * shared_handle, hipMemPool_t mem_pool, hipMemAllocationHandleType handle_type, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolExportToShareableHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolExportToShareableHandle_api_args_t hipFuncArgs{shared_handle, mem_pool, handle_type, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipMemPool_t,hipMemAllocationHandleType,unsigned int)>("hipMemPoolExportToShareableHandle");
			out = hipFunc(hipFuncArgs.shared_handle, hipFuncArgs.mem_pool, hipFuncArgs.handle_type, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		shared_handle = hipFuncArgs.shared_handle;
		mem_pool = hipFuncArgs.mem_pool;
		handle_type = hipFuncArgs.handle_type;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipMemPool_t,hipMemAllocationHandleType,unsigned int)>("hipMemPoolExportToShareableHandle");
		return hipFunc(shared_handle, mem_pool, handle_type, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolGetAccess(hipMemAccessFlags * flags, hipMemPool_t mem_pool, hipMemLocation * location) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolGetAccess;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolGetAccess_api_args_t hipFuncArgs{flags, mem_pool, location};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemAccessFlags *,hipMemPool_t,hipMemLocation *)>("hipMemPoolGetAccess");
			out = hipFunc(hipFuncArgs.flags, hipFuncArgs.mem_pool, hipFuncArgs.location);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		flags = hipFuncArgs.flags;
		mem_pool = hipFuncArgs.mem_pool;
		location = hipFuncArgs.location;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemAccessFlags *,hipMemPool_t,hipMemLocation *)>("hipMemPoolGetAccess");
		return hipFunc(flags, mem_pool, location);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void * value) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolGetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolGetAttribute_api_args_t hipFuncArgs{mem_pool, attr, value};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t,hipMemPoolAttr,void *)>("hipMemPoolGetAttribute");
			out = hipFunc(hipFuncArgs.mem_pool, hipFuncArgs.attr, hipFuncArgs.value);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;
		attr = hipFuncArgs.attr;
		value = hipFuncArgs.value;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t,hipMemPoolAttr,void *)>("hipMemPoolGetAttribute");
		return hipFunc(mem_pool, attr, value);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t * mem_pool, void * shared_handle, hipMemAllocationHandleType handle_type, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolImportFromShareableHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolImportFromShareableHandle_api_args_t hipFuncArgs{mem_pool, shared_handle, handle_type, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t *,void *,hipMemAllocationHandleType,unsigned int)>("hipMemPoolImportFromShareableHandle");
			out = hipFunc(hipFuncArgs.mem_pool, hipFuncArgs.shared_handle, hipFuncArgs.handle_type, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;
		shared_handle = hipFuncArgs.shared_handle;
		handle_type = hipFuncArgs.handle_type;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t *,void *,hipMemAllocationHandleType,unsigned int)>("hipMemPoolImportFromShareableHandle");
		return hipFunc(mem_pool, shared_handle, handle_type, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolImportPointer(void * * dev_ptr, hipMemPool_t mem_pool, hipMemPoolPtrExportData * export_data) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolImportPointer;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolImportPointer_api_args_t hipFuncArgs{dev_ptr, mem_pool, export_data};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,hipMemPool_t,hipMemPoolPtrExportData *)>("hipMemPoolImportPointer");
			out = hipFunc(hipFuncArgs.dev_ptr, hipFuncArgs.mem_pool, hipFuncArgs.export_data);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev_ptr = hipFuncArgs.dev_ptr;
		mem_pool = hipFuncArgs.mem_pool;
		export_data = hipFuncArgs.export_data;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,hipMemPool_t,hipMemPoolPtrExportData *)>("hipMemPoolImportPointer");
		return hipFunc(dev_ptr, mem_pool, export_data);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc * desc_list, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolSetAccess;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolSetAccess_api_args_t hipFuncArgs{mem_pool, desc_list, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t,const hipMemAccessDesc *,size_t)>("hipMemPoolSetAccess");
			out = hipFunc(hipFuncArgs.mem_pool, hipFuncArgs.desc_list, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t,const hipMemAccessDesc *,size_t)>("hipMemPoolSetAccess");
		return hipFunc(mem_pool, desc_list, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void * value) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolSetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolSetAttribute_api_args_t hipFuncArgs{mem_pool, attr, value};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t,hipMemPoolAttr,void *)>("hipMemPoolSetAttribute");
			out = hipFunc(hipFuncArgs.mem_pool, hipFuncArgs.attr, hipFuncArgs.value);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;
		attr = hipFuncArgs.attr;
		value = hipFuncArgs.value;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t,hipMemPoolAttr,void *)>("hipMemPoolSetAttribute");
		return hipFunc(mem_pool, attr, value);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPoolTrimTo;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPoolTrimTo_api_args_t hipFuncArgs{mem_pool, min_bytes_to_hold};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t,size_t)>("hipMemPoolTrimTo");
			out = hipFunc(hipFuncArgs.mem_pool, hipFuncArgs.min_bytes_to_hold);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mem_pool = hipFuncArgs.mem_pool;
		min_bytes_to_hold = hipFuncArgs.min_bytes_to_hold;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemPool_t,size_t)>("hipMemPoolTrimTo");
		return hipFunc(mem_pool, min_bytes_to_hold);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPrefetchAsync(const void * dev_ptr, size_t count, int device, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPrefetchAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPrefetchAsync_api_args_t hipFuncArgs{dev_ptr, count, device, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,size_t,int,hipStream_t)>("hipMemPrefetchAsync");
			out = hipFunc(hipFuncArgs.dev_ptr, hipFuncArgs.count, hipFuncArgs.device, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		count = hipFuncArgs.count;
		device = hipFuncArgs.device;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,size_t,int,hipStream_t)>("hipMemPrefetchAsync");
		return hipFunc(dev_ptr, count, device, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemPtrGetInfo(void * ptr, size_t * size) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemPtrGetInfo;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemPtrGetInfo_api_args_t hipFuncArgs{ptr, size};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t *)>("hipMemPtrGetInfo");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t *)>("hipMemPtrGetInfo");
		return hipFunc(ptr, size);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemRangeGetAttribute(void * data, size_t data_size, hipMemRangeAttribute attribute, const void * dev_ptr, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemRangeGetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemRangeGetAttribute_api_args_t hipFuncArgs{data, data_size, attribute, dev_ptr, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,hipMemRangeAttribute,const void *,size_t)>("hipMemRangeGetAttribute");
			out = hipFunc(hipFuncArgs.data, hipFuncArgs.data_size, hipFuncArgs.attribute, hipFuncArgs.dev_ptr, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		data = hipFuncArgs.data;
		data_size = hipFuncArgs.data_size;
		attribute = hipFuncArgs.attribute;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,hipMemRangeAttribute,const void *,size_t)>("hipMemRangeGetAttribute");
		return hipFunc(data, data_size, attribute, dev_ptr, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemRangeGetAttributes(void * * data, size_t * data_sizes, hipMemRangeAttribute * attributes, size_t num_attributes, const void * dev_ptr, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemRangeGetAttributes;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemRangeGetAttributes_api_args_t hipFuncArgs{data, data_sizes, attributes, num_attributes, dev_ptr, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t *,hipMemRangeAttribute *,size_t,const void *,size_t)>("hipMemRangeGetAttributes");
			out = hipFunc(hipFuncArgs.data, hipFuncArgs.data_sizes, hipFuncArgs.attributes, hipFuncArgs.num_attributes, hipFuncArgs.dev_ptr, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		data = hipFuncArgs.data;
		data_sizes = hipFuncArgs.data_sizes;
		attributes = hipFuncArgs.attributes;
		num_attributes = hipFuncArgs.num_attributes;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void * *,size_t *,hipMemRangeAttribute *,size_t,const void *,size_t)>("hipMemRangeGetAttributes");
		return hipFunc(data, data_sizes, attributes, num_attributes, dev_ptr, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemRelease;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemRelease_api_args_t hipFuncArgs{handle};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t)>("hipMemRelease");
			out = hipFunc(hipFuncArgs.handle);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		handle = hipFuncArgs.handle;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t)>("hipMemRelease");
		return hipFunc(handle);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t * handle, void * addr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemRetainAllocationHandle;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemRetainAllocationHandle_api_args_t hipFuncArgs{handle, addr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,void *)>("hipMemRetainAllocationHandle");
			out = hipFunc(hipFuncArgs.handle, hipFuncArgs.addr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		handle = hipFuncArgs.handle;
		addr = hipFuncArgs.addr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMemGenericAllocationHandle_t *,void *)>("hipMemRetainAllocationHandle");
		return hipFunc(handle, addr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemSetAccess(void * ptr, size_t size, const hipMemAccessDesc * desc, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemSetAccess;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemSetAccess_api_args_t hipFuncArgs{ptr, size, desc, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,const hipMemAccessDesc *,size_t)>("hipMemSetAccess");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size, hipFuncArgs.desc, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,const hipMemAccessDesc *,size_t)>("hipMemSetAccess");
		return hipFunc(ptr, size, desc, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemUnmap(void * ptr, size_t size) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemUnmap;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemUnmap_api_args_t hipFuncArgs{ptr, size};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t)>("hipMemUnmap");
			out = hipFunc(hipFuncArgs.ptr, hipFuncArgs.size);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ptr = hipFuncArgs.ptr;
		size = hipFuncArgs.size;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t)>("hipMemUnmap");
		return hipFunc(ptr, size);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy_api_args_t hipFuncArgs{dst, src, sizeBytes, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind)>("hipMemcpy");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		sizeBytes = hipFuncArgs.sizeBytes;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind)>("hipMemcpy");
		return hipFunc(dst, src, sizeBytes, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy2D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy2D_api_args_t hipFuncArgs{dst, dpitch, src, spitch, width, height, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2D");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.dpitch, hipFuncArgs.src, hipFuncArgs.spitch, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		dpitch = hipFuncArgs.dpitch;
		spitch = hipFuncArgs.spitch;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2D");
		return hipFunc(dst, dpitch, src, spitch, width, height, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy2DAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy2DAsync_api_args_t hipFuncArgs{dst, dpitch, src, spitch, width, height, kind, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.dpitch, hipFuncArgs.src, hipFuncArgs.spitch, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.kind, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		dpitch = hipFuncArgs.dpitch;
		spitch = hipFuncArgs.spitch;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		kind = hipFuncArgs.kind;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DAsync");
		return hipFunc(dst, dpitch, src, spitch, width, height, kind, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DFromArray(void * dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy2DFromArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy2DFromArray_api_args_t hipFuncArgs{dst, dpitch, src, wOffset, hOffset, width, height, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,hipArray_const_t,size_t,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2DFromArray");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.dpitch, hipFuncArgs.src, hipFuncArgs.wOffset, hipFuncArgs.hOffset, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		dpitch = hipFuncArgs.dpitch;
		wOffset = hipFuncArgs.wOffset;
		hOffset = hipFuncArgs.hOffset;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,hipArray_const_t,size_t,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2DFromArray");
		return hipFunc(dst, dpitch, src, wOffset, hOffset, width, height, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DFromArrayAsync(void * dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy2DFromArrayAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy2DFromArrayAsync_api_args_t hipFuncArgs{dst, dpitch, src, wOffset, hOffset, width, height, kind, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,hipArray_const_t,size_t,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DFromArrayAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.dpitch, hipFuncArgs.src, hipFuncArgs.wOffset, hipFuncArgs.hOffset, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.kind, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		dpitch = hipFuncArgs.dpitch;
		wOffset = hipFuncArgs.wOffset;
		hOffset = hipFuncArgs.hOffset;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		kind = hipFuncArgs.kind;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,hipArray_const_t,size_t,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DFromArrayAsync");
		return hipFunc(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy2DToArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy2DToArray_api_args_t hipFuncArgs{dst, wOffset, hOffset, src, spitch, width, height, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t,size_t,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2DToArray");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.wOffset, hipFuncArgs.hOffset, hipFuncArgs.src, hipFuncArgs.spitch, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		wOffset = hipFuncArgs.wOffset;
		hOffset = hipFuncArgs.hOffset;
		spitch = hipFuncArgs.spitch;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t,size_t,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpy2DToArray");
		return hipFunc(dst, wOffset, hOffset, src, spitch, width, height, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy2DToArrayAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy2DToArrayAsync_api_args_t hipFuncArgs{dst, wOffset, hOffset, src, spitch, width, height, kind, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t,size_t,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DToArrayAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.wOffset, hipFuncArgs.hOffset, hipFuncArgs.src, hipFuncArgs.spitch, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.kind, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		wOffset = hipFuncArgs.wOffset;
		hOffset = hipFuncArgs.hOffset;
		spitch = hipFuncArgs.spitch;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		kind = hipFuncArgs.kind;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t,size_t,size_t,const void *,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DToArrayAsync");
		return hipFunc(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy3D(const struct hipMemcpy3DParms * p) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy3D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy3D_api_args_t hipFuncArgs{p};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const struct hipMemcpy3DParms *)>("hipMemcpy3D");
			out = hipFunc(hipFuncArgs.p);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const struct hipMemcpy3DParms *)>("hipMemcpy3D");
		return hipFunc(p);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms * p, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpy3DAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpy3DAsync_api_args_t hipFuncArgs{p, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const struct hipMemcpy3DParms *,hipStream_t)>("hipMemcpy3DAsync");
			out = hipFunc(hipFuncArgs.p, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const struct hipMemcpy3DParms *,hipStream_t)>("hipMemcpy3DAsync");
		return hipFunc(p, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyAsync(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyAsync_api_args_t hipFuncArgs{dst, src, sizeBytes, kind, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes, hipFuncArgs.kind, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		sizeBytes = hipFuncArgs.sizeBytes;
		kind = hipFuncArgs.kind;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyAsync");
		return hipFunc(dst, src, sizeBytes, kind, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyAtoH(void * dst, hipArray_t srcArray, size_t srcOffset, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyAtoH;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyAtoH_api_args_t hipFuncArgs{dst, srcArray, srcOffset, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipArray_t,size_t,size_t)>("hipMemcpyAtoH");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.srcArray, hipFuncArgs.srcOffset, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		srcArray = hipFuncArgs.srcArray;
		srcOffset = hipFuncArgs.srcOffset;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipArray_t,size_t,size_t)>("hipMemcpyAtoH");
		return hipFunc(dst, srcArray, srcOffset, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyDtoD;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyDtoD_api_args_t hipFuncArgs{dst, src, sizeBytes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,hipDeviceptr_t,size_t)>("hipMemcpyDtoD");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		src = hipFuncArgs.src;
		sizeBytes = hipFuncArgs.sizeBytes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,hipDeviceptr_t,size_t)>("hipMemcpyDtoD");
		return hipFunc(dst, src, sizeBytes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyDtoDAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyDtoDAsync_api_args_t hipFuncArgs{dst, src, sizeBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,hipDeviceptr_t,size_t,hipStream_t)>("hipMemcpyDtoDAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		src = hipFuncArgs.src;
		sizeBytes = hipFuncArgs.sizeBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,hipDeviceptr_t,size_t,hipStream_t)>("hipMemcpyDtoDAsync");
		return hipFunc(dst, src, sizeBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyDtoH(void * dst, hipDeviceptr_t src, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyDtoH;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyDtoH_api_args_t hipFuncArgs{dst, src, sizeBytes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipDeviceptr_t,size_t)>("hipMemcpyDtoH");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		src = hipFuncArgs.src;
		sizeBytes = hipFuncArgs.sizeBytes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipDeviceptr_t,size_t)>("hipMemcpyDtoH");
		return hipFunc(dst, src, sizeBytes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyDtoHAsync(void * dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyDtoHAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyDtoHAsync_api_args_t hipFuncArgs{dst, src, sizeBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipDeviceptr_t,size_t,hipStream_t)>("hipMemcpyDtoHAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		src = hipFuncArgs.src;
		sizeBytes = hipFuncArgs.sizeBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipDeviceptr_t,size_t,hipStream_t)>("hipMemcpyDtoHAsync");
		return hipFunc(dst, src, sizeBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyFromArray(void * dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyFromArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyFromArray_api_args_t hipFuncArgs{dst, srcArray, wOffset, hOffset, count, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipArray_const_t,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpyFromArray");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.srcArray, hipFuncArgs.wOffset, hipFuncArgs.hOffset, hipFuncArgs.count, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		wOffset = hipFuncArgs.wOffset;
		hOffset = hipFuncArgs.hOffset;
		count = hipFuncArgs.count;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipArray_const_t,size_t,size_t,size_t,hipMemcpyKind)>("hipMemcpyFromArray");
		return hipFunc(dst, srcArray, wOffset, hOffset, count, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyFromSymbol(void * dst, const void * symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyFromSymbol;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyFromSymbol_api_args_t hipFuncArgs{dst, symbol, sizeBytes, offset, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,size_t,hipMemcpyKind)>("hipMemcpyFromSymbol");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.symbol, hipFuncArgs.sizeBytes, hipFuncArgs.offset, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		sizeBytes = hipFuncArgs.sizeBytes;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,size_t,hipMemcpyKind)>("hipMemcpyFromSymbol");
		return hipFunc(dst, symbol, sizeBytes, offset, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyFromSymbolAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyFromSymbolAsync_api_args_t hipFuncArgs{dst, symbol, sizeBytes, offset, kind, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyFromSymbolAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.symbol, hipFuncArgs.sizeBytes, hipFuncArgs.offset, hipFuncArgs.kind, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		sizeBytes = hipFuncArgs.sizeBytes;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyFromSymbolAsync");
		return hipFunc(dst, symbol, sizeBytes, offset, kind, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void * srcHost, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyHtoA;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyHtoA_api_args_t hipFuncArgs{dstArray, dstOffset, srcHost, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t,size_t,const void *,size_t)>("hipMemcpyHtoA");
			out = hipFunc(hipFuncArgs.dstArray, hipFuncArgs.dstOffset, hipFuncArgs.srcHost, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dstArray = hipFuncArgs.dstArray;
		dstOffset = hipFuncArgs.dstOffset;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t,size_t,const void *,size_t)>("hipMemcpyHtoA");
		return hipFunc(dstArray, dstOffset, srcHost, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void * src, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyHtoD;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyHtoD_api_args_t hipFuncArgs{dst, src, sizeBytes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,void *,size_t)>("hipMemcpyHtoD");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		src = hipFuncArgs.src;
		sizeBytes = hipFuncArgs.sizeBytes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,void *,size_t)>("hipMemcpyHtoD");
		return hipFunc(dst, src, sizeBytes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void * src, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyHtoDAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyHtoDAsync_api_args_t hipFuncArgs{dst, src, sizeBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,void *,size_t,hipStream_t)>("hipMemcpyHtoDAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		src = hipFuncArgs.src;
		sizeBytes = hipFuncArgs.sizeBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,void *,size_t,hipStream_t)>("hipMemcpyHtoDAsync");
		return hipFunc(dst, src, sizeBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyParam2D(const hip_Memcpy2D * pCopy) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyParam2D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyParam2D_api_args_t hipFuncArgs{pCopy};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hip_Memcpy2D *)>("hipMemcpyParam2D");
			out = hipFunc(hipFuncArgs.pCopy);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hip_Memcpy2D *)>("hipMemcpyParam2D");
		return hipFunc(pCopy);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D * pCopy, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyParam2DAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyParam2DAsync_api_args_t hipFuncArgs{pCopy, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hip_Memcpy2D *,hipStream_t)>("hipMemcpyParam2DAsync");
			out = hipFunc(hipFuncArgs.pCopy, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hip_Memcpy2D *,hipStream_t)>("hipMemcpyParam2DAsync");
		return hipFunc(pCopy, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyPeer(void * dst, int dstDeviceId, const void * src, int srcDeviceId, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyPeer;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyPeer_api_args_t hipFuncArgs{dst, dstDeviceId, src, srcDeviceId, sizeBytes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,int,const void *,int,size_t)>("hipMemcpyPeer");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.dstDeviceId, hipFuncArgs.src, hipFuncArgs.srcDeviceId, hipFuncArgs.sizeBytes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		dstDeviceId = hipFuncArgs.dstDeviceId;
		srcDeviceId = hipFuncArgs.srcDeviceId;
		sizeBytes = hipFuncArgs.sizeBytes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,int,const void *,int,size_t)>("hipMemcpyPeer");
		return hipFunc(dst, dstDeviceId, src, srcDeviceId, sizeBytes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyPeerAsync(void * dst, int dstDeviceId, const void * src, int srcDevice, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyPeerAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyPeerAsync_api_args_t hipFuncArgs{dst, dstDeviceId, src, srcDevice, sizeBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,int,const void *,int,size_t,hipStream_t)>("hipMemcpyPeerAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.dstDeviceId, hipFuncArgs.src, hipFuncArgs.srcDevice, hipFuncArgs.sizeBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		dstDeviceId = hipFuncArgs.dstDeviceId;
		srcDevice = hipFuncArgs.srcDevice;
		sizeBytes = hipFuncArgs.sizeBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,int,const void *,int,size_t,hipStream_t)>("hipMemcpyPeerAsync");
		return hipFunc(dst, dstDeviceId, src, srcDevice, sizeBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyToArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyToArray_api_args_t hipFuncArgs{dst, wOffset, hOffset, src, count, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t,size_t,size_t,const void *,size_t,hipMemcpyKind)>("hipMemcpyToArray");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.wOffset, hipFuncArgs.hOffset, hipFuncArgs.src, hipFuncArgs.count, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		wOffset = hipFuncArgs.wOffset;
		hOffset = hipFuncArgs.hOffset;
		count = hipFuncArgs.count;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t,size_t,size_t,const void *,size_t,hipMemcpyKind)>("hipMemcpyToArray");
		return hipFunc(dst, wOffset, hOffset, src, count, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyToSymbol(const void * symbol, const void * src, size_t sizeBytes, size_t offset, hipMemcpyKind kind) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyToSymbol;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyToSymbol_api_args_t hipFuncArgs{symbol, src, sizeBytes, offset, kind};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipMemcpyToSymbol");
			out = hipFunc(hipFuncArgs.symbol, hipFuncArgs.src, hipFuncArgs.sizeBytes, hipFuncArgs.offset, hipFuncArgs.kind);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		sizeBytes = hipFuncArgs.sizeBytes;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,const void *,size_t,size_t,hipMemcpyKind)>("hipMemcpyToSymbol");
		return hipFunc(symbol, src, sizeBytes, offset, kind);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyToSymbolAsync(const void * symbol, const void * src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyToSymbolAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyToSymbolAsync_api_args_t hipFuncArgs{symbol, src, sizeBytes, offset, kind, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,const void *,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyToSymbolAsync");
			out = hipFunc(hipFuncArgs.symbol, hipFuncArgs.src, hipFuncArgs.sizeBytes, hipFuncArgs.offset, hipFuncArgs.kind, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		sizeBytes = hipFuncArgs.sizeBytes;
		offset = hipFuncArgs.offset;
		kind = hipFuncArgs.kind;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,const void *,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyToSymbolAsync");
		return hipFunc(symbol, src, sizeBytes, offset, kind, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemcpyWithStream(void * dst, const void * src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemcpyWithStream;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemcpyWithStream_api_args_t hipFuncArgs{dst, src, sizeBytes, kind, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyWithStream");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.src, hipFuncArgs.sizeBytes, hipFuncArgs.kind, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		sizeBytes = hipFuncArgs.sizeBytes;
		kind = hipFuncArgs.kind;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,const void *,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyWithStream");
		return hipFunc(dst, src, sizeBytes, kind, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset(void * dst, int value, size_t sizeBytes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemset;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemset_api_args_t hipFuncArgs{dst, value, sizeBytes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,int,size_t)>("hipMemset");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.value, hipFuncArgs.sizeBytes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		value = hipFuncArgs.value;
		sizeBytes = hipFuncArgs.sizeBytes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,int,size_t)>("hipMemset");
		return hipFunc(dst, value, sizeBytes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset2D(void * dst, size_t pitch, int value, size_t width, size_t height) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemset2D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemset2D_api_args_t hipFuncArgs{dst, pitch, value, width, height};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,int,size_t,size_t)>("hipMemset2D");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.pitch, hipFuncArgs.value, hipFuncArgs.width, hipFuncArgs.height);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		pitch = hipFuncArgs.pitch;
		value = hipFuncArgs.value;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,int,size_t,size_t)>("hipMemset2D");
		return hipFunc(dst, pitch, value, width, height);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset2DAsync(void * dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemset2DAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemset2DAsync_api_args_t hipFuncArgs{dst, pitch, value, width, height, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,int,size_t,size_t,hipStream_t)>("hipMemset2DAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.pitch, hipFuncArgs.value, hipFuncArgs.width, hipFuncArgs.height, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		pitch = hipFuncArgs.pitch;
		value = hipFuncArgs.value;
		width = hipFuncArgs.width;
		height = hipFuncArgs.height;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,size_t,int,size_t,size_t,hipStream_t)>("hipMemset2DAsync");
		return hipFunc(dst, pitch, value, width, height, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemset3D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemset3D_api_args_t hipFuncArgs{pitchedDevPtr, value, extent};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipPitchedPtr,int,hipExtent)>("hipMemset3D");
			out = hipFunc(hipFuncArgs.pitchedDevPtr, hipFuncArgs.value, hipFuncArgs.extent);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pitchedDevPtr = hipFuncArgs.pitchedDevPtr;
		value = hipFuncArgs.value;
		extent = hipFuncArgs.extent;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipPitchedPtr,int,hipExtent)>("hipMemset3D");
		return hipFunc(pitchedDevPtr, value, extent);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemset3DAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemset3DAsync_api_args_t hipFuncArgs{pitchedDevPtr, value, extent, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipPitchedPtr,int,hipExtent,hipStream_t)>("hipMemset3DAsync");
			out = hipFunc(hipFuncArgs.pitchedDevPtr, hipFuncArgs.value, hipFuncArgs.extent, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pitchedDevPtr = hipFuncArgs.pitchedDevPtr;
		value = hipFuncArgs.value;
		extent = hipFuncArgs.extent;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipPitchedPtr,int,hipExtent,hipStream_t)>("hipMemset3DAsync");
		return hipFunc(pitchedDevPtr, value, extent, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetAsync(void * dst, int value, size_t sizeBytes, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemsetAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemsetAsync_api_args_t hipFuncArgs{dst, value, sizeBytes, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,int,size_t,hipStream_t)>("hipMemsetAsync");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.value, hipFuncArgs.sizeBytes, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		value = hipFuncArgs.value;
		sizeBytes = hipFuncArgs.sizeBytes;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,int,size_t,hipStream_t)>("hipMemsetAsync");
		return hipFunc(dst, value, sizeBytes, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemsetD16;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemsetD16_api_args_t hipFuncArgs{dest, value, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned short,size_t)>("hipMemsetD16");
			out = hipFunc(hipFuncArgs.dest, hipFuncArgs.value, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dest = hipFuncArgs.dest;
		value = hipFuncArgs.value;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned short,size_t)>("hipMemsetD16");
		return hipFunc(dest, value, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemsetD16Async;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemsetD16Async_api_args_t hipFuncArgs{dest, value, count, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned short,size_t,hipStream_t)>("hipMemsetD16Async");
			out = hipFunc(hipFuncArgs.dest, hipFuncArgs.value, hipFuncArgs.count, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dest = hipFuncArgs.dest;
		value = hipFuncArgs.value;
		count = hipFuncArgs.count;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned short,size_t,hipStream_t)>("hipMemsetD16Async");
		return hipFunc(dest, value, count, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemsetD32;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemsetD32_api_args_t hipFuncArgs{dest, value, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,int,size_t)>("hipMemsetD32");
			out = hipFunc(hipFuncArgs.dest, hipFuncArgs.value, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dest = hipFuncArgs.dest;
		value = hipFuncArgs.value;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,int,size_t)>("hipMemsetD32");
		return hipFunc(dest, value, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemsetD32Async;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemsetD32Async_api_args_t hipFuncArgs{dst, value, count, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,int,size_t,hipStream_t)>("hipMemsetD32Async");
			out = hipFunc(hipFuncArgs.dst, hipFuncArgs.value, hipFuncArgs.count, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dst = hipFuncArgs.dst;
		value = hipFuncArgs.value;
		count = hipFuncArgs.count;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,int,size_t,hipStream_t)>("hipMemsetD32Async");
		return hipFunc(dst, value, count, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemsetD8;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemsetD8_api_args_t hipFuncArgs{dest, value, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned char,size_t)>("hipMemsetD8");
			out = hipFunc(hipFuncArgs.dest, hipFuncArgs.value, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dest = hipFuncArgs.dest;
		value = hipFuncArgs.value;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned char,size_t)>("hipMemsetD8");
		return hipFunc(dest, value, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMemsetD8Async;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMemsetD8Async_api_args_t hipFuncArgs{dest, value, count, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned char,size_t,hipStream_t)>("hipMemsetD8Async");
			out = hipFunc(hipFuncArgs.dest, hipFuncArgs.value, hipFuncArgs.count, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dest = hipFuncArgs.dest;
		value = hipFuncArgs.value;
		count = hipFuncArgs.count;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t,unsigned char,size_t,hipStream_t)>("hipMemsetD8Async");
		return hipFunc(dest, value, count, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t * pHandle, HIP_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int numMipmapLevels) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMipmappedArrayCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMipmappedArrayCreate_api_args_t hipFuncArgs{pHandle, pMipmappedArrayDesc, numMipmapLevels};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,HIP_ARRAY3D_DESCRIPTOR *,unsigned int)>("hipMipmappedArrayCreate");
			out = hipFunc(hipFuncArgs.pHandle, hipFuncArgs.pMipmappedArrayDesc, hipFuncArgs.numMipmapLevels);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pHandle = hipFuncArgs.pHandle;
		pMipmappedArrayDesc = hipFuncArgs.pMipmappedArrayDesc;
		numMipmapLevels = hipFuncArgs.numMipmapLevels;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,HIP_ARRAY3D_DESCRIPTOR *,unsigned int)>("hipMipmappedArrayCreate");
		return hipFunc(pHandle, pMipmappedArrayDesc, numMipmapLevels);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMipmappedArrayDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMipmappedArrayDestroy_api_args_t hipFuncArgs{hMipmappedArray};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t)>("hipMipmappedArrayDestroy");
			out = hipFunc(hipFuncArgs.hMipmappedArray);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		hMipmappedArray = hipFuncArgs.hMipmappedArray;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t)>("hipMipmappedArrayDestroy");
		return hipFunc(hMipmappedArray);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipMipmappedArrayGetLevel(hipArray_t * pLevelArray, hipMipmappedArray_t hMipMappedArray, unsigned int level) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipMipmappedArrayGetLevel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipMipmappedArrayGetLevel_api_args_t hipFuncArgs{pLevelArray, hMipMappedArray, level};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,hipMipmappedArray_t,unsigned int)>("hipMipmappedArrayGetLevel");
			out = hipFunc(hipFuncArgs.pLevelArray, hipFuncArgs.hMipMappedArray, hipFuncArgs.level);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pLevelArray = hipFuncArgs.pLevelArray;
		hMipMappedArray = hipFuncArgs.hMipMappedArray;
		level = hipFuncArgs.level;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,hipMipmappedArray_t,unsigned int)>("hipMipmappedArrayGetLevel");
		return hipFunc(pLevelArray, hMipMappedArray, level);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleGetFunction(hipFunction_t * function, hipModule_t module, const char * kname) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleGetFunction;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleGetFunction_api_args_t hipFuncArgs{function, module, kname};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t *,hipModule_t,const char *)>("hipModuleGetFunction");
			out = hipFunc(hipFuncArgs.function, hipFuncArgs.module, hipFuncArgs.kname);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		function = hipFuncArgs.function;
		module = hipFuncArgs.module;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t *,hipModule_t,const char *)>("hipModuleGetFunction");
		return hipFunc(function, module, kname);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleGetGlobal(hipDeviceptr_t * dptr, size_t * bytes, hipModule_t hmod, const char * name) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleGetGlobal;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleGetGlobal_api_args_t hipFuncArgs{dptr, bytes, hmod, name};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,hipModule_t,const char *)>("hipModuleGetGlobal");
			out = hipFunc(hipFuncArgs.dptr, hipFuncArgs.bytes, hipFuncArgs.hmod, hipFuncArgs.name);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dptr = hipFuncArgs.dptr;
		bytes = hipFuncArgs.bytes;
		hmod = hipFuncArgs.hmod;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t *,size_t *,hipModule_t,const char *)>("hipModuleGetGlobal");
		return hipFunc(dptr, bytes, hmod, name);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleGetTexRef(textureReference * * texRef, hipModule_t hmod, const char * name) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleGetTexRef;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleGetTexRef_api_args_t hipFuncArgs{texRef, hmod, name};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference * *,hipModule_t,const char *)>("hipModuleGetTexRef");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.hmod, hipFuncArgs.name);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		hmod = hipFuncArgs.hmod;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference * *,hipModule_t,const char *)>("hipModuleGetTexRef");
		return hipFunc(texRef, hmod, name);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void * * kernelParams) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleLaunchCooperativeKernel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleLaunchCooperativeKernel_api_args_t hipFuncArgs{f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,hipStream_t,void * *)>("hipModuleLaunchCooperativeKernel");
			out = hipFunc(hipFuncArgs.f, hipFuncArgs.gridDimX, hipFuncArgs.gridDimY, hipFuncArgs.gridDimZ, hipFuncArgs.blockDimX, hipFuncArgs.blockDimY, hipFuncArgs.blockDimZ, hipFuncArgs.sharedMemBytes, hipFuncArgs.stream, hipFuncArgs.kernelParams);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		f = hipFuncArgs.f;
		gridDimX = hipFuncArgs.gridDimX;
		gridDimY = hipFuncArgs.gridDimY;
		gridDimZ = hipFuncArgs.gridDimZ;
		blockDimX = hipFuncArgs.blockDimX;
		blockDimY = hipFuncArgs.blockDimY;
		blockDimZ = hipFuncArgs.blockDimZ;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		stream = hipFuncArgs.stream;
		kernelParams = hipFuncArgs.kernelParams;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,hipStream_t,void * *)>("hipModuleLaunchCooperativeKernel");
		return hipFunc(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams * launchParamsList, unsigned int numDevices, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleLaunchCooperativeKernelMultiDevice;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleLaunchCooperativeKernelMultiDevice_api_args_t hipFuncArgs{launchParamsList, numDevices, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunctionLaunchParams *,unsigned int,unsigned int)>("hipModuleLaunchCooperativeKernelMultiDevice");
			out = hipFunc(hipFuncArgs.launchParamsList, hipFuncArgs.numDevices, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		launchParamsList = hipFuncArgs.launchParamsList;
		numDevices = hipFuncArgs.numDevices;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunctionLaunchParams *,unsigned int,unsigned int)>("hipModuleLaunchCooperativeKernelMultiDevice");
		return hipFunc(launchParamsList, numDevices, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void * * kernelParams, void * * extra) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleLaunchKernel;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleLaunchKernel_api_args_t hipFuncArgs{f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,hipStream_t,void * *,void * *)>("hipModuleLaunchKernel");
			out = hipFunc(hipFuncArgs.f, hipFuncArgs.gridDimX, hipFuncArgs.gridDimY, hipFuncArgs.gridDimZ, hipFuncArgs.blockDimX, hipFuncArgs.blockDimY, hipFuncArgs.blockDimZ, hipFuncArgs.sharedMemBytes, hipFuncArgs.stream, hipFuncArgs.kernelParams, hipFuncArgs.extra);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		f = hipFuncArgs.f;
		gridDimX = hipFuncArgs.gridDimX;
		gridDimY = hipFuncArgs.gridDimY;
		gridDimZ = hipFuncArgs.gridDimZ;
		blockDimX = hipFuncArgs.blockDimX;
		blockDimY = hipFuncArgs.blockDimY;
		blockDimZ = hipFuncArgs.blockDimZ;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		stream = hipFuncArgs.stream;
		kernelParams = hipFuncArgs.kernelParams;
		extra = hipFuncArgs.extra;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,hipStream_t,void * *,void * *)>("hipModuleLaunchKernel");
		return hipFunc(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLaunchKernelExt(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, size_t sharedMemBytes, hipStream_t hStream, void * * kernelParams, void * * extra, hipEvent_t startEvent, hipEvent_t stopEvent) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleLaunchKernelExt;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleLaunchKernelExt_api_args_t hipFuncArgs{f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t)>("hipModuleLaunchKernelExt");
			out = hipFunc(hipFuncArgs.f, hipFuncArgs.globalWorkSizeX, hipFuncArgs.globalWorkSizeY, hipFuncArgs.globalWorkSizeZ, hipFuncArgs.blockDimX, hipFuncArgs.blockDimY, hipFuncArgs.blockDimZ, hipFuncArgs.sharedMemBytes, hipFuncArgs.hStream, hipFuncArgs.kernelParams, hipFuncArgs.extra, hipFuncArgs.startEvent, hipFuncArgs.stopEvent);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		f = hipFuncArgs.f;
		globalWorkSizeX = hipFuncArgs.globalWorkSizeX;
		globalWorkSizeY = hipFuncArgs.globalWorkSizeY;
		globalWorkSizeZ = hipFuncArgs.globalWorkSizeZ;
		blockDimX = hipFuncArgs.blockDimX;
		blockDimY = hipFuncArgs.blockDimY;
		blockDimZ = hipFuncArgs.blockDimZ;
		sharedMemBytes = hipFuncArgs.sharedMemBytes;
		hStream = hipFuncArgs.hStream;
		kernelParams = hipFuncArgs.kernelParams;
		extra = hipFuncArgs.extra;
		startEvent = hipFuncArgs.startEvent;
		stopEvent = hipFuncArgs.stopEvent;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void * *,void * *,hipEvent_t,hipEvent_t)>("hipModuleLaunchKernelExt");
		return hipFunc(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLoad(hipModule_t * module, const char * fname) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleLoad;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleLoad_api_args_t hipFuncArgs{module, fname};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipModule_t *,const char *)>("hipModuleLoad");
			out = hipFunc(hipFuncArgs.module, hipFuncArgs.fname);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		module = hipFuncArgs.module;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipModule_t *,const char *)>("hipModuleLoad");
		return hipFunc(module, fname);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLoadData(hipModule_t * module, const void * image) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleLoadData;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleLoadData_api_args_t hipFuncArgs{module, image};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipModule_t *,const void *)>("hipModuleLoadData");
			out = hipFunc(hipFuncArgs.module, hipFuncArgs.image);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		module = hipFuncArgs.module;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipModule_t *,const void *)>("hipModuleLoadData");
		return hipFunc(module, image);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleLoadDataEx(hipModule_t * module, const void * image, unsigned int numOptions, hipJitOption * options, void * * optionValues) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleLoadDataEx;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleLoadDataEx_api_args_t hipFuncArgs{module, image, numOptions, options, optionValues};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipModule_t *,const void *,unsigned int,hipJitOption *,void * *)>("hipModuleLoadDataEx");
			out = hipFunc(hipFuncArgs.module, hipFuncArgs.image, hipFuncArgs.numOptions, hipFuncArgs.options, hipFuncArgs.optionValues);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		module = hipFuncArgs.module;
		numOptions = hipFuncArgs.numOptions;
		options = hipFuncArgs.options;
		optionValues = hipFuncArgs.optionValues;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipModule_t *,const void *,unsigned int,hipJitOption *,void * *)>("hipModuleLoadDataEx");
		return hipFunc(module, image, numOptions, options, optionValues);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_api_args_t hipFuncArgs{numBlocks, f, blockSize, dynSharedMemPerBlk};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipFunction_t,int,size_t)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
			out = hipFunc(hipFuncArgs.numBlocks, hipFuncArgs.f, hipFuncArgs.blockSize, hipFuncArgs.dynSharedMemPerBlk);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numBlocks = hipFuncArgs.numBlocks;
		f = hipFuncArgs.f;
		blockSize = hipFuncArgs.blockSize;
		dynSharedMemPerBlk = hipFuncArgs.dynSharedMemPerBlk;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipFunction_t,int,size_t)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
		return hipFunc(numBlocks, f, blockSize, dynSharedMemPerBlk);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_api_args_t hipFuncArgs{numBlocks, f, blockSize, dynSharedMemPerBlk, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipFunction_t,int,size_t,unsigned int)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
			out = hipFunc(hipFuncArgs.numBlocks, hipFuncArgs.f, hipFuncArgs.blockSize, hipFuncArgs.dynSharedMemPerBlk, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numBlocks = hipFuncArgs.numBlocks;
		f = hipFuncArgs.f;
		blockSize = hipFuncArgs.blockSize;
		dynSharedMemPerBlk = hipFuncArgs.dynSharedMemPerBlk;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,hipFunction_t,int,size_t,unsigned int)>("hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
		return hipFunc(numBlocks, f, blockSize, dynSharedMemPerBlk, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int * gridSize, int * blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleOccupancyMaxPotentialBlockSize_api_args_t hipFuncArgs{gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *,hipFunction_t,size_t,int)>("hipModuleOccupancyMaxPotentialBlockSize");
			out = hipFunc(hipFuncArgs.gridSize, hipFuncArgs.blockSize, hipFuncArgs.f, hipFuncArgs.dynSharedMemPerBlk, hipFuncArgs.blockSizeLimit);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridSize = hipFuncArgs.gridSize;
		blockSize = hipFuncArgs.blockSize;
		f = hipFuncArgs.f;
		dynSharedMemPerBlk = hipFuncArgs.dynSharedMemPerBlk;
		blockSizeLimit = hipFuncArgs.blockSizeLimit;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *,hipFunction_t,size_t,int)>("hipModuleOccupancyMaxPotentialBlockSize");
		return hipFunc(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int * gridSize, int * blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_api_args_t hipFuncArgs{gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *,hipFunction_t,size_t,int,unsigned int)>("hipModuleOccupancyMaxPotentialBlockSizeWithFlags");
			out = hipFunc(hipFuncArgs.gridSize, hipFuncArgs.blockSize, hipFuncArgs.f, hipFuncArgs.dynSharedMemPerBlk, hipFuncArgs.blockSizeLimit, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridSize = hipFuncArgs.gridSize;
		blockSize = hipFuncArgs.blockSize;
		f = hipFuncArgs.f;
		dynSharedMemPerBlk = hipFuncArgs.dynSharedMemPerBlk;
		blockSizeLimit = hipFuncArgs.blockSizeLimit;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *,hipFunction_t,size_t,int,unsigned int)>("hipModuleOccupancyMaxPotentialBlockSizeWithFlags");
		return hipFunc(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipModuleUnload(hipModule_t module) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipModuleUnload;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipModuleUnload_api_args_t hipFuncArgs{module};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipModule_t)>("hipModuleUnload");
			out = hipFunc(hipFuncArgs.module);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		module = hipFuncArgs.module;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipModule_t)>("hipModuleUnload");
		return hipFunc(module);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * f, int blockSize, size_t dynSharedMemPerBlk) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipOccupancyMaxActiveBlocksPerMultiprocessor_api_args_t hipFuncArgs{numBlocks, f, blockSize, dynSharedMemPerBlk};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const void *,int,size_t)>("hipOccupancyMaxActiveBlocksPerMultiprocessor");
			out = hipFunc(hipFuncArgs.numBlocks, hipFuncArgs.f, hipFuncArgs.blockSize, hipFuncArgs.dynSharedMemPerBlk);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numBlocks = hipFuncArgs.numBlocks;
		blockSize = hipFuncArgs.blockSize;
		dynSharedMemPerBlk = hipFuncArgs.dynSharedMemPerBlk;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const void *,int,size_t)>("hipOccupancyMaxActiveBlocksPerMultiprocessor");
		return hipFunc(numBlocks, f, blockSize, dynSharedMemPerBlk);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_api_args_t hipFuncArgs{numBlocks, f, blockSize, dynSharedMemPerBlk, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const void *,int,size_t,unsigned int)>("hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
			out = hipFunc(hipFuncArgs.numBlocks, hipFuncArgs.f, hipFuncArgs.blockSize, hipFuncArgs.dynSharedMemPerBlk, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numBlocks = hipFuncArgs.numBlocks;
		blockSize = hipFuncArgs.blockSize;
		dynSharedMemPerBlk = hipFuncArgs.dynSharedMemPerBlk;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const void *,int,size_t,unsigned int)>("hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
		return hipFunc(numBlocks, f, blockSize, dynSharedMemPerBlk, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipOccupancyMaxPotentialBlockSize(int * gridSize, int * blockSize, const void * f, size_t dynSharedMemPerBlk, int blockSizeLimit) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipOccupancyMaxPotentialBlockSize;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipOccupancyMaxPotentialBlockSize_api_args_t hipFuncArgs{gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *,const void *,size_t,int)>("hipOccupancyMaxPotentialBlockSize");
			out = hipFunc(hipFuncArgs.gridSize, hipFuncArgs.blockSize, hipFuncArgs.f, hipFuncArgs.dynSharedMemPerBlk, hipFuncArgs.blockSizeLimit);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		gridSize = hipFuncArgs.gridSize;
		blockSize = hipFuncArgs.blockSize;
		dynSharedMemPerBlk = hipFuncArgs.dynSharedMemPerBlk;
		blockSizeLimit = hipFuncArgs.blockSizeLimit;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int *,const void *,size_t,int)>("hipOccupancyMaxPotentialBlockSize");
		return hipFunc(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipPeekAtLastError() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipPeekAtLastError;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipPeekAtLastError");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipPeekAtLastError");
		return hipFunc();
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipPointerGetAttribute(void * data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipPointerGetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipPointerGetAttribute_api_args_t hipFuncArgs{data, attribute, ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipPointer_attribute,hipDeviceptr_t)>("hipPointerGetAttribute");
			out = hipFunc(hipFuncArgs.data, hipFuncArgs.attribute, hipFuncArgs.ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		data = hipFuncArgs.data;
		attribute = hipFuncArgs.attribute;
		ptr = hipFuncArgs.ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(void *,hipPointer_attribute,hipDeviceptr_t)>("hipPointerGetAttribute");
		return hipFunc(data, attribute, ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipPointerGetAttributes(hipPointerAttribute_t * attributes, const void * ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipPointerGetAttributes;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipPointerGetAttributes_api_args_t hipFuncArgs{attributes, ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipPointerAttribute_t *,const void *)>("hipPointerGetAttributes");
			out = hipFunc(hipFuncArgs.attributes, hipFuncArgs.ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		attributes = hipFuncArgs.attributes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipPointerAttribute_t *,const void *)>("hipPointerGetAttributes");
		return hipFunc(attributes, ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipPointerSetAttribute(const void * value, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipPointerSetAttribute;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipPointerSetAttribute_api_args_t hipFuncArgs{value, attribute, ptr};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,hipPointer_attribute,hipDeviceptr_t)>("hipPointerSetAttribute");
			out = hipFunc(hipFuncArgs.value, hipFuncArgs.attribute, hipFuncArgs.ptr);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		attribute = hipFuncArgs.attribute;
		ptr = hipFuncArgs.ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,hipPointer_attribute,hipDeviceptr_t)>("hipPointerSetAttribute");
		return hipFunc(value, attribute, ptr);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipProfilerStart() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipProfilerStart;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipProfilerStart");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipProfilerStart");
		return hipFunc();
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipProfilerStop() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipProfilerStop;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipProfilerStop");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hipProfilerStop");
		return hipFunc();
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipRuntimeGetVersion(int * runtimeVersion) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipRuntimeGetVersion;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipRuntimeGetVersion_api_args_t hipFuncArgs{runtimeVersion};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipRuntimeGetVersion");
			out = hipFunc(hipFuncArgs.runtimeVersion);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		runtimeVersion = hipFuncArgs.runtimeVersion;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *)>("hipRuntimeGetVersion");
		return hipFunc(runtimeVersion);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipSetDevice(int deviceId) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipSetDevice;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipSetDevice_api_args_t hipFuncArgs{deviceId};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int)>("hipSetDevice");
			out = hipFunc(hipFuncArgs.deviceId);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		deviceId = hipFuncArgs.deviceId;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int)>("hipSetDevice");
		return hipFunc(deviceId);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipSetDeviceFlags(unsigned flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipSetDeviceFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipSetDeviceFlags_api_args_t hipFuncArgs{flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned)>("hipSetDeviceFlags");
			out = hipFunc(hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned)>("hipSetDeviceFlags");
		return hipFunc(flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipSetValidDevices(int * device_arr, int len) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipSetValidDevices;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipSetValidDevices_api_args_t hipFuncArgs{device_arr, len};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int)>("hipSetValidDevices");
			out = hipFunc(hipFuncArgs.device_arr, hipFuncArgs.len);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		device_arr = hipFuncArgs.device_arr;
		len = hipFuncArgs.len;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,int)>("hipSetValidDevices");
		return hipFunc(device_arr, len);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipSetupArgument(const void * arg, size_t size, size_t offset) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipSetupArgument;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipSetupArgument_api_args_t hipFuncArgs{arg, size, offset};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,size_t,size_t)>("hipSetupArgument");
			out = hipFunc(hipFuncArgs.arg, hipFuncArgs.size, hipFuncArgs.offset);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		size = hipFuncArgs.size;
		offset = hipFuncArgs.offset;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const void *,size_t,size_t)>("hipSetupArgument");
		return hipFunc(arg, size, offset);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t * extSemArray, const hipExternalSemaphoreSignalParams * paramsArray, unsigned int numExtSems, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipSignalExternalSemaphoresAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipSignalExternalSemaphoresAsync_api_args_t hipFuncArgs{extSemArray, paramsArray, numExtSems, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hipExternalSemaphore_t *,const hipExternalSemaphoreSignalParams *,unsigned int,hipStream_t)>("hipSignalExternalSemaphoresAsync");
			out = hipFunc(hipFuncArgs.extSemArray, hipFuncArgs.paramsArray, hipFuncArgs.numExtSems, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numExtSems = hipFuncArgs.numExtSems;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hipExternalSemaphore_t *,const hipExternalSemaphoreSignalParams *,unsigned int,hipStream_t)>("hipSignalExternalSemaphoresAsync");
		return hipFunc(extSemArray, paramsArray, numExtSems, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void * userData, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamAddCallback;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamAddCallback_api_args_t hipFuncArgs{stream, callback, userData, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCallback_t,void *,unsigned int)>("hipStreamAddCallback");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.callback, hipFuncArgs.userData, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		callback = hipFuncArgs.callback;
		userData = hipFuncArgs.userData;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCallback_t,void *,unsigned int)>("hipStreamAddCallback");
		return hipFunc(stream, callback, userData, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void * dev_ptr, size_t length, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamAttachMemAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamAttachMemAsync_api_args_t hipFuncArgs{stream, dev_ptr, length, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,size_t,unsigned int)>("hipStreamAttachMemAsync");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.dev_ptr, hipFuncArgs.length, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		dev_ptr = hipFuncArgs.dev_ptr;
		length = hipFuncArgs.length;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,size_t,unsigned int)>("hipStreamAttachMemAsync");
		return hipFunc(stream, dev_ptr, length, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamBeginCapture;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamBeginCapture_api_args_t hipFuncArgs{stream, mode};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureMode)>("hipStreamBeginCapture");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.mode);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		mode = hipFuncArgs.mode;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureMode)>("hipStreamBeginCapture");
		return hipFunc(stream, mode);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamCreate(hipStream_t * stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamCreate_api_args_t hipFuncArgs{stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t *)>("hipStreamCreate");
			out = hipFunc(hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t *)>("hipStreamCreate");
		return hipFunc(stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamCreateWithFlags(hipStream_t * stream, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamCreateWithFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamCreateWithFlags_api_args_t hipFuncArgs{stream, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t *,unsigned int)>("hipStreamCreateWithFlags");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t *,unsigned int)>("hipStreamCreateWithFlags");
		return hipFunc(stream, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamCreateWithPriority(hipStream_t * stream, unsigned int flags, int priority) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamCreateWithPriority;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamCreateWithPriority_api_args_t hipFuncArgs{stream, flags, priority};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t *,unsigned int,int)>("hipStreamCreateWithPriority");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.flags, hipFuncArgs.priority);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		flags = hipFuncArgs.flags;
		priority = hipFuncArgs.priority;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t *,unsigned int,int)>("hipStreamCreateWithPriority");
		return hipFunc(stream, flags, priority);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamDestroy(hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamDestroy_api_args_t hipFuncArgs{stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t)>("hipStreamDestroy");
			out = hipFunc(hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t)>("hipStreamDestroy");
		return hipFunc(stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t * pGraph) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamEndCapture;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamEndCapture_api_args_t hipFuncArgs{stream, pGraph};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipGraph_t *)>("hipStreamEndCapture");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.pGraph);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		pGraph = hipFuncArgs.pGraph;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipGraph_t *)>("hipStreamEndCapture");
		return hipFunc(stream, pGraph);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus * pCaptureStatus, unsigned long long * pId) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamGetCaptureInfo;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamGetCaptureInfo_api_args_t hipFuncArgs{stream, pCaptureStatus, pId};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *)>("hipStreamGetCaptureInfo");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.pCaptureStatus, hipFuncArgs.pId);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		pCaptureStatus = hipFuncArgs.pCaptureStatus;
		pId = hipFuncArgs.pId;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *)>("hipStreamGetCaptureInfo");
		return hipFunc(stream, pCaptureStatus, pId);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, hipGraph_t * graph_out, const hipGraphNode_t * * dependencies_out, size_t * numDependencies_out) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamGetCaptureInfo_v2;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamGetCaptureInfo_v2_api_args_t hipFuncArgs{stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *,hipGraph_t *,const hipGraphNode_t * *,size_t *)>("hipStreamGetCaptureInfo_v2");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.captureStatus_out, hipFuncArgs.id_out, hipFuncArgs.graph_out, hipFuncArgs.dependencies_out, hipFuncArgs.numDependencies_out);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		captureStatus_out = hipFuncArgs.captureStatus_out;
		id_out = hipFuncArgs.id_out;
		graph_out = hipFuncArgs.graph_out;
		numDependencies_out = hipFuncArgs.numDependencies_out;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *,unsigned long long *,hipGraph_t *,const hipGraphNode_t * *,size_t *)>("hipStreamGetCaptureInfo_v2");
		return hipFunc(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t * device) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamGetDevice;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamGetDevice_api_args_t hipFuncArgs{stream, device};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipDevice_t *)>("hipStreamGetDevice");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.device);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		device = hipFuncArgs.device;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipDevice_t *)>("hipStreamGetDevice");
		return hipFunc(stream, device);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int * flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamGetFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamGetFlags_api_args_t hipFuncArgs{stream, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,unsigned int *)>("hipStreamGetFlags");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,unsigned int *)>("hipStreamGetFlags");
		return hipFunc(stream, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamGetPriority(hipStream_t stream, int * priority) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamGetPriority;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamGetPriority_api_args_t hipFuncArgs{stream, priority};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,int *)>("hipStreamGetPriority");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.priority);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		priority = hipFuncArgs.priority;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,int *)>("hipStreamGetPriority");
		return hipFunc(stream, priority);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus * pCaptureStatus) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamIsCapturing;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamIsCapturing_api_args_t hipFuncArgs{stream, pCaptureStatus};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *)>("hipStreamIsCapturing");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.pCaptureStatus);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		pCaptureStatus = hipFuncArgs.pCaptureStatus;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipStreamCaptureStatus *)>("hipStreamIsCapturing");
		return hipFunc(stream, pCaptureStatus);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamQuery(hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamQuery;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamQuery_api_args_t hipFuncArgs{stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t)>("hipStreamQuery");
			out = hipFunc(hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t)>("hipStreamQuery");
		return hipFunc(stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamSynchronize(hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamSynchronize;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamSynchronize_api_args_t hipFuncArgs{stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t)>("hipStreamSynchronize");
			out = hipFunc(hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t)>("hipStreamSynchronize");
		return hipFunc(stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t * dependencies, size_t numDependencies, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamUpdateCaptureDependencies;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamUpdateCaptureDependencies_api_args_t hipFuncArgs{stream, dependencies, numDependencies, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipGraphNode_t *,size_t,unsigned int)>("hipStreamUpdateCaptureDependencies");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.dependencies, hipFuncArgs.numDependencies, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		dependencies = hipFuncArgs.dependencies;
		numDependencies = hipFuncArgs.numDependencies;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipGraphNode_t *,size_t,unsigned int)>("hipStreamUpdateCaptureDependencies");
		return hipFunc(stream, dependencies, numDependencies, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamWaitEvent;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamWaitEvent_api_args_t hipFuncArgs{stream, event, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipEvent_t,unsigned int)>("hipStreamWaitEvent");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.event, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		event = hipFuncArgs.event;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,hipEvent_t,unsigned int)>("hipStreamWaitEvent");
		return hipFunc(stream, event, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWaitValue32(hipStream_t stream, void * ptr, uint32_t value, unsigned int flags, uint32_t mask) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamWaitValue32;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamWaitValue32_api_args_t hipFuncArgs{stream, ptr, value, flags, mask};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,uint32_t,unsigned int,uint32_t)>("hipStreamWaitValue32");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.ptr, hipFuncArgs.value, hipFuncArgs.flags, hipFuncArgs.mask);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		ptr = hipFuncArgs.ptr;
		value = hipFuncArgs.value;
		flags = hipFuncArgs.flags;
		mask = hipFuncArgs.mask;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,uint32_t,unsigned int,uint32_t)>("hipStreamWaitValue32");
		return hipFunc(stream, ptr, value, flags, mask);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWaitValue64(hipStream_t stream, void * ptr, uint64_t value, unsigned int flags, uint64_t mask) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamWaitValue64;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamWaitValue64_api_args_t hipFuncArgs{stream, ptr, value, flags, mask};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,uint64_t,unsigned int,uint64_t)>("hipStreamWaitValue64");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.ptr, hipFuncArgs.value, hipFuncArgs.flags, hipFuncArgs.mask);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		ptr = hipFuncArgs.ptr;
		value = hipFuncArgs.value;
		flags = hipFuncArgs.flags;
		mask = hipFuncArgs.mask;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,uint64_t,unsigned int,uint64_t)>("hipStreamWaitValue64");
		return hipFunc(stream, ptr, value, flags, mask);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWriteValue32(hipStream_t stream, void * ptr, uint32_t value, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamWriteValue32;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamWriteValue32_api_args_t hipFuncArgs{stream, ptr, value, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,uint32_t,unsigned int)>("hipStreamWriteValue32");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.ptr, hipFuncArgs.value, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		ptr = hipFuncArgs.ptr;
		value = hipFuncArgs.value;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,uint32_t,unsigned int)>("hipStreamWriteValue32");
		return hipFunc(stream, ptr, value, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipStreamWriteValue64(hipStream_t stream, void * ptr, uint64_t value, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipStreamWriteValue64;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipStreamWriteValue64_api_args_t hipFuncArgs{stream, ptr, value, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,uint64_t,unsigned int)>("hipStreamWriteValue64");
			out = hipFunc(hipFuncArgs.stream, hipFuncArgs.ptr, hipFuncArgs.value, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		stream = hipFuncArgs.stream;
		ptr = hipFuncArgs.ptr;
		value = hipFuncArgs.value;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStream_t,void *,uint64_t,unsigned int)>("hipStreamWriteValue64");
		return hipFunc(stream, ptr, value, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectCreate(hipTextureObject_t * pTexObject, const HIP_RESOURCE_DESC * pResDesc, const HIP_TEXTURE_DESC * pTexDesc, const HIP_RESOURCE_VIEW_DESC * pResViewDesc) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexObjectCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexObjectCreate_api_args_t hipFuncArgs{pTexObject, pResDesc, pTexDesc, pResViewDesc};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureObject_t *,const HIP_RESOURCE_DESC *,const HIP_TEXTURE_DESC *,const HIP_RESOURCE_VIEW_DESC *)>("hipTexObjectCreate");
			out = hipFunc(hipFuncArgs.pTexObject, hipFuncArgs.pResDesc, hipFuncArgs.pTexDesc, hipFuncArgs.pResViewDesc);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pTexObject = hipFuncArgs.pTexObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureObject_t *,const HIP_RESOURCE_DESC *,const HIP_TEXTURE_DESC *,const HIP_RESOURCE_VIEW_DESC *)>("hipTexObjectCreate");
		return hipFunc(pTexObject, pResDesc, pTexDesc, pResViewDesc);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexObjectDestroy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexObjectDestroy_api_args_t hipFuncArgs{texObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureObject_t)>("hipTexObjectDestroy");
			out = hipFunc(hipFuncArgs.texObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texObject = hipFuncArgs.texObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipTextureObject_t)>("hipTexObjectDestroy");
		return hipFunc(texObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC * pResDesc, hipTextureObject_t texObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexObjectGetResourceDesc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexObjectGetResourceDesc_api_args_t hipFuncArgs{pResDesc, texObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_RESOURCE_DESC *,hipTextureObject_t)>("hipTexObjectGetResourceDesc");
			out = hipFunc(hipFuncArgs.pResDesc, hipFuncArgs.texObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pResDesc = hipFuncArgs.pResDesc;
		texObject = hipFuncArgs.texObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_RESOURCE_DESC *,hipTextureObject_t)>("hipTexObjectGetResourceDesc");
		return hipFunc(pResDesc, texObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC * pResViewDesc, hipTextureObject_t texObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexObjectGetResourceViewDesc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexObjectGetResourceViewDesc_api_args_t hipFuncArgs{pResViewDesc, texObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_RESOURCE_VIEW_DESC *,hipTextureObject_t)>("hipTexObjectGetResourceViewDesc");
			out = hipFunc(hipFuncArgs.pResViewDesc, hipFuncArgs.texObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pResViewDesc = hipFuncArgs.pResViewDesc;
		texObject = hipFuncArgs.texObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_RESOURCE_VIEW_DESC *,hipTextureObject_t)>("hipTexObjectGetResourceViewDesc");
		return hipFunc(pResViewDesc, texObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC * pTexDesc, hipTextureObject_t texObject) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexObjectGetTextureDesc;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexObjectGetTextureDesc_api_args_t hipFuncArgs{pTexDesc, texObject};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_TEXTURE_DESC *,hipTextureObject_t)>("hipTexObjectGetTextureDesc");
			out = hipFunc(hipFuncArgs.pTexDesc, hipFuncArgs.texObject);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pTexDesc = hipFuncArgs.pTexDesc;
		texObject = hipFuncArgs.texObject;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(HIP_TEXTURE_DESC *,hipTextureObject_t)>("hipTexObjectGetTextureDesc");
		return hipFunc(pTexDesc, texObject);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetAddress(hipDeviceptr_t * dev_ptr, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetAddress;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetAddress_api_args_t hipFuncArgs{dev_ptr, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t *,const textureReference *)>("hipTexRefGetAddress");
			out = hipFunc(hipFuncArgs.dev_ptr, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		dev_ptr = hipFuncArgs.dev_ptr;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipDeviceptr_t *,const textureReference *)>("hipTexRefGetAddress");
		return hipFunc(dev_ptr, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode * pam, const textureReference * texRef, int dim) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetAddressMode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetAddressMode_api_args_t hipFuncArgs{pam, texRef, dim};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(enum hipTextureAddressMode *,const textureReference *,int)>("hipTexRefGetAddressMode");
			out = hipFunc(hipFuncArgs.pam, hipFuncArgs.texRef, hipFuncArgs.dim);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pam = hipFuncArgs.pam;
		dim = hipFuncArgs.dim;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(enum hipTextureAddressMode *,const textureReference *,int)>("hipTexRefGetAddressMode");
		return hipFunc(pam, texRef, dim);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetArray(hipArray_t * pArray, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetArray_api_args_t hipFuncArgs{pArray, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const textureReference *)>("hipTexRefGetArray");
			out = hipFunc(hipFuncArgs.pArray, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pArray = hipFuncArgs.pArray;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_t *,const textureReference *)>("hipTexRefGetArray");
		return hipFunc(pArray, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetBorderColor(float * pBorderColor, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetBorderColor;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetBorderColor_api_args_t hipFuncArgs{pBorderColor, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(float *,const textureReference *)>("hipTexRefGetBorderColor");
			out = hipFunc(hipFuncArgs.pBorderColor, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pBorderColor = hipFuncArgs.pBorderColor;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(float *,const textureReference *)>("hipTexRefGetBorderColor");
		return hipFunc(pBorderColor, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode * pfm, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetFilterMode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetFilterMode_api_args_t hipFuncArgs{pfm, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(enum hipTextureFilterMode *,const textureReference *)>("hipTexRefGetFilterMode");
			out = hipFunc(hipFuncArgs.pfm, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pfm = hipFuncArgs.pfm;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(enum hipTextureFilterMode *,const textureReference *)>("hipTexRefGetFilterMode");
		return hipFunc(pfm, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetFlags(unsigned int * pFlags, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetFlags_api_args_t hipFuncArgs{pFlags, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *,const textureReference *)>("hipTexRefGetFlags");
			out = hipFunc(hipFuncArgs.pFlags, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pFlags = hipFuncArgs.pFlags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(unsigned int *,const textureReference *)>("hipTexRefGetFlags");
		return hipFunc(pFlags, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetFormat(hipArray_Format * pFormat, int * pNumChannels, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetFormat;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetFormat_api_args_t hipFuncArgs{pFormat, pNumChannels, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_Format *,int *,const textureReference *)>("hipTexRefGetFormat");
			out = hipFunc(hipFuncArgs.pFormat, hipFuncArgs.pNumChannels, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pFormat = hipFuncArgs.pFormat;
		pNumChannels = hipFuncArgs.pNumChannels;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipArray_Format *,int *,const textureReference *)>("hipTexRefGetFormat");
		return hipFunc(pFormat, pNumChannels, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMaxAnisotropy(int * pmaxAnsio, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetMaxAnisotropy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetMaxAnisotropy_api_args_t hipFuncArgs{pmaxAnsio, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const textureReference *)>("hipTexRefGetMaxAnisotropy");
			out = hipFunc(hipFuncArgs.pmaxAnsio, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pmaxAnsio = hipFuncArgs.pmaxAnsio;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(int *,const textureReference *)>("hipTexRefGetMaxAnisotropy");
		return hipFunc(pmaxAnsio, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t * pArray, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetMipMappedArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetMipMappedArray_api_args_t hipFuncArgs{pArray, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,const textureReference *)>("hipTexRefGetMipMappedArray");
			out = hipFunc(hipFuncArgs.pArray, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pArray = hipFuncArgs.pArray;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,const textureReference *)>("hipTexRefGetMipMappedArray");
		return hipFunc(pArray, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode * pfm, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetMipmapFilterMode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetMipmapFilterMode_api_args_t hipFuncArgs{pfm, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(enum hipTextureFilterMode *,const textureReference *)>("hipTexRefGetMipmapFilterMode");
			out = hipFunc(hipFuncArgs.pfm, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pfm = hipFuncArgs.pfm;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(enum hipTextureFilterMode *,const textureReference *)>("hipTexRefGetMipmapFilterMode");
		return hipFunc(pfm, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipmapLevelBias(float * pbias, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetMipmapLevelBias;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetMipmapLevelBias_api_args_t hipFuncArgs{pbias, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(float *,const textureReference *)>("hipTexRefGetMipmapLevelBias");
			out = hipFunc(hipFuncArgs.pbias, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pbias = hipFuncArgs.pbias;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(float *,const textureReference *)>("hipTexRefGetMipmapLevelBias");
		return hipFunc(pbias, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetMipmapLevelClamp;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetMipmapLevelClamp_api_args_t hipFuncArgs{pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(float *,float *,const textureReference *)>("hipTexRefGetMipmapLevelClamp");
			out = hipFunc(hipFuncArgs.pminMipmapLevelClamp, hipFuncArgs.pmaxMipmapLevelClamp, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pminMipmapLevelClamp = hipFuncArgs.pminMipmapLevelClamp;
		pmaxMipmapLevelClamp = hipFuncArgs.pmaxMipmapLevelClamp;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(float *,float *,const textureReference *)>("hipTexRefGetMipmapLevelClamp");
		return hipFunc(pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefGetMipmappedArray(hipMipmappedArray_t * pArray, const textureReference * texRef) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefGetMipmappedArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefGetMipmappedArray_api_args_t hipFuncArgs{pArray, texRef};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,const textureReference *)>("hipTexRefGetMipmappedArray");
			out = hipFunc(hipFuncArgs.pArray, hipFuncArgs.texRef);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		pArray = hipFuncArgs.pArray;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipMipmappedArray_t *,const textureReference *)>("hipTexRefGetMipmappedArray");
		return hipFunc(pArray, texRef);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetAddress(size_t * ByteOffset, textureReference * texRef, hipDeviceptr_t dptr, size_t bytes) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetAddress;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetAddress_api_args_t hipFuncArgs{ByteOffset, texRef, dptr, bytes};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,textureReference *,hipDeviceptr_t,size_t)>("hipTexRefSetAddress");
			out = hipFunc(hipFuncArgs.ByteOffset, hipFuncArgs.texRef, hipFuncArgs.dptr, hipFuncArgs.bytes);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		ByteOffset = hipFuncArgs.ByteOffset;
		texRef = hipFuncArgs.texRef;
		dptr = hipFuncArgs.dptr;
		bytes = hipFuncArgs.bytes;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(size_t *,textureReference *,hipDeviceptr_t,size_t)>("hipTexRefSetAddress");
		return hipFunc(ByteOffset, texRef, dptr, bytes);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetAddress2D(textureReference * texRef, const HIP_ARRAY_DESCRIPTOR * desc, hipDeviceptr_t dptr, size_t Pitch) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetAddress2D;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetAddress2D_api_args_t hipFuncArgs{texRef, desc, dptr, Pitch};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,const HIP_ARRAY_DESCRIPTOR *,hipDeviceptr_t,size_t)>("hipTexRefSetAddress2D");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.desc, hipFuncArgs.dptr, hipFuncArgs.Pitch);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		dptr = hipFuncArgs.dptr;
		Pitch = hipFuncArgs.Pitch;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,const HIP_ARRAY_DESCRIPTOR *,hipDeviceptr_t,size_t)>("hipTexRefSetAddress2D");
		return hipFunc(texRef, desc, dptr, Pitch);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetAddressMode(textureReference * texRef, int dim, enum hipTextureAddressMode am) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetAddressMode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetAddressMode_api_args_t hipFuncArgs{texRef, dim, am};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,int,enum hipTextureAddressMode)>("hipTexRefSetAddressMode");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.dim, hipFuncArgs.am);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		dim = hipFuncArgs.dim;
		am = hipFuncArgs.am;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,int,enum hipTextureAddressMode)>("hipTexRefSetAddressMode");
		return hipFunc(texRef, dim, am);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetArray(textureReference * tex, hipArray_const_t array, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetArray_api_args_t hipFuncArgs{tex, array, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,hipArray_const_t,unsigned int)>("hipTexRefSetArray");
			out = hipFunc(hipFuncArgs.tex, hipFuncArgs.array, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		tex = hipFuncArgs.tex;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,hipArray_const_t,unsigned int)>("hipTexRefSetArray");
		return hipFunc(tex, array, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetBorderColor(textureReference * texRef, float * pBorderColor) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetBorderColor;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetBorderColor_api_args_t hipFuncArgs{texRef, pBorderColor};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,float *)>("hipTexRefSetBorderColor");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.pBorderColor);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		pBorderColor = hipFuncArgs.pBorderColor;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,float *)>("hipTexRefSetBorderColor");
		return hipFunc(texRef, pBorderColor);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetFilterMode(textureReference * texRef, enum hipTextureFilterMode fm) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetFilterMode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetFilterMode_api_args_t hipFuncArgs{texRef, fm};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,enum hipTextureFilterMode)>("hipTexRefSetFilterMode");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.fm);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		fm = hipFuncArgs.fm;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,enum hipTextureFilterMode)>("hipTexRefSetFilterMode");
		return hipFunc(texRef, fm);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetFlags(textureReference * texRef, unsigned int Flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetFlags;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetFlags_api_args_t hipFuncArgs{texRef, Flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,unsigned int)>("hipTexRefSetFlags");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.Flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		Flags = hipFuncArgs.Flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,unsigned int)>("hipTexRefSetFlags");
		return hipFunc(texRef, Flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetFormat(textureReference * texRef, hipArray_Format fmt, int NumPackedComponents) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetFormat;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetFormat_api_args_t hipFuncArgs{texRef, fmt, NumPackedComponents};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,hipArray_Format,int)>("hipTexRefSetFormat");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.fmt, hipFuncArgs.NumPackedComponents);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		fmt = hipFuncArgs.fmt;
		NumPackedComponents = hipFuncArgs.NumPackedComponents;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,hipArray_Format,int)>("hipTexRefSetFormat");
		return hipFunc(texRef, fmt, NumPackedComponents);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMaxAnisotropy(textureReference * texRef, unsigned int maxAniso) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetMaxAnisotropy;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetMaxAnisotropy_api_args_t hipFuncArgs{texRef, maxAniso};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,unsigned int)>("hipTexRefSetMaxAnisotropy");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.maxAniso);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		maxAniso = hipFuncArgs.maxAniso;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,unsigned int)>("hipTexRefSetMaxAnisotropy");
		return hipFunc(texRef, maxAniso);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMipmapFilterMode(textureReference * texRef, enum hipTextureFilterMode fm) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetMipmapFilterMode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetMipmapFilterMode_api_args_t hipFuncArgs{texRef, fm};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,enum hipTextureFilterMode)>("hipTexRefSetMipmapFilterMode");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.fm);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		fm = hipFuncArgs.fm;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,enum hipTextureFilterMode)>("hipTexRefSetMipmapFilterMode");
		return hipFunc(texRef, fm);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMipmapLevelBias(textureReference * texRef, float bias) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetMipmapLevelBias;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetMipmapLevelBias_api_args_t hipFuncArgs{texRef, bias};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,float)>("hipTexRefSetMipmapLevelBias");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.bias);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		bias = hipFuncArgs.bias;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,float)>("hipTexRefSetMipmapLevelBias");
		return hipFunc(texRef, bias);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMipmapLevelClamp(textureReference * texRef, float minMipMapLevelClamp, float maxMipMapLevelClamp) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetMipmapLevelClamp;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetMipmapLevelClamp_api_args_t hipFuncArgs{texRef, minMipMapLevelClamp, maxMipMapLevelClamp};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,float,float)>("hipTexRefSetMipmapLevelClamp");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.minMipMapLevelClamp, hipFuncArgs.maxMipMapLevelClamp);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		minMipMapLevelClamp = hipFuncArgs.minMipMapLevelClamp;
		maxMipMapLevelClamp = hipFuncArgs.maxMipMapLevelClamp;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,float,float)>("hipTexRefSetMipmapLevelClamp");
		return hipFunc(texRef, minMipMapLevelClamp, maxMipMapLevelClamp);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipTexRefSetMipmappedArray(textureReference * texRef, struct hipMipmappedArray * mipmappedArray, unsigned int Flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipTexRefSetMipmappedArray;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipTexRefSetMipmappedArray_api_args_t hipFuncArgs{texRef, mipmappedArray, Flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,struct hipMipmappedArray *,unsigned int)>("hipTexRefSetMipmappedArray");
			out = hipFunc(hipFuncArgs.texRef, hipFuncArgs.mipmappedArray, hipFuncArgs.Flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		texRef = hipFuncArgs.texRef;
		mipmappedArray = hipFuncArgs.mipmappedArray;
		Flags = hipFuncArgs.Flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(textureReference *,struct hipMipmappedArray *,unsigned int)>("hipTexRefSetMipmappedArray");
		return hipFunc(texRef, mipmappedArray, Flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode * mode) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipThreadExchangeStreamCaptureMode;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipThreadExchangeStreamCaptureMode_api_args_t hipFuncArgs{mode};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStreamCaptureMode *)>("hipThreadExchangeStreamCaptureMode");
			out = hipFunc(hipFuncArgs.mode);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		mode = hipFuncArgs.mode;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipStreamCaptureMode *)>("hipThreadExchangeStreamCaptureMode");
		return hipFunc(mode);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipUnbindTexture(const textureReference * tex) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipUnbindTexture;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipUnbindTexture_api_args_t hipFuncArgs{tex};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const textureReference *)>("hipUnbindTexture");
			out = hipFunc(hipFuncArgs.tex);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const textureReference *)>("hipUnbindTexture");
		return hipFunc(tex);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipUserObjectCreate(hipUserObject_t * object_out, void * ptr, hipHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipUserObjectCreate;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipUserObjectCreate_api_args_t hipFuncArgs{object_out, ptr, destroy, initialRefcount, flags};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipUserObject_t *,void *,hipHostFn_t,unsigned int,unsigned int)>("hipUserObjectCreate");
			out = hipFunc(hipFuncArgs.object_out, hipFuncArgs.ptr, hipFuncArgs.destroy, hipFuncArgs.initialRefcount, hipFuncArgs.flags);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		object_out = hipFuncArgs.object_out;
		ptr = hipFuncArgs.ptr;
		destroy = hipFuncArgs.destroy;
		initialRefcount = hipFuncArgs.initialRefcount;
		flags = hipFuncArgs.flags;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipUserObject_t *,void *,hipHostFn_t,unsigned int,unsigned int)>("hipUserObjectCreate");
		return hipFunc(object_out, ptr, destroy, initialRefcount, flags);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipUserObjectRelease;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipUserObjectRelease_api_args_t hipFuncArgs{object, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipUserObject_t,unsigned int)>("hipUserObjectRelease");
			out = hipFunc(hipFuncArgs.object, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		object = hipFuncArgs.object;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipUserObject_t,unsigned int)>("hipUserObjectRelease");
		return hipFunc(object, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipUserObjectRetain;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipUserObjectRetain_api_args_t hipFuncArgs{object, count};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipUserObject_t,unsigned int)>("hipUserObjectRetain");
			out = hipFunc(hipFuncArgs.object, hipFuncArgs.count);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		object = hipFuncArgs.object;
		count = hipFuncArgs.count;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(hipUserObject_t,unsigned int)>("hipUserObjectRetain");
		return hipFunc(object, count);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t * extSemArray, const hipExternalSemaphoreWaitParams * paramsArray, unsigned int numExtSems, hipStream_t stream) {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_API_ID_hipWaitExternalSemaphoresAsync;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		hip_hipWaitExternalSemaphoresAsync_api_args_t hipFuncArgs{extSemArray, paramsArray, numExtSems, stream};
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hipExternalSemaphore_t *,const hipExternalSemaphoreWaitParams *,unsigned int,hipStream_t)>("hipWaitExternalSemaphoresAsync");
			out = hipFunc(hipFuncArgs.extSemArray, hipFuncArgs.paramsArray, hipFuncArgs.numExtSems, hipFuncArgs.stream);
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(static_cast<void*>(&hipFuncArgs), LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);
		// Copy the modified arguments back to the original arguments (if non-const)
		numExtSems = hipFuncArgs.numExtSems;
		stream = hipFuncArgs.stream;

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)(const hipExternalSemaphore_t *,const hipExternalSemaphoreWaitParams *,unsigned int,hipStream_t)>("hipWaitExternalSemaphoresAsync");
		return hipFunc(extSemArray, paramsArray, numExtSems, stream);
	};
}

extern "C" __attribute__((visibility("default")))
hipError_t hip_init() {
	auto& hipInterceptor = luthier::HipInterceptor::instance();
	auto apiId = HIP_PRIVATE_API_ID_hip_init;
	bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
	bool isInternalCallbackEnabled = hipInterceptor.isInternalCallbackEnabled(apiId);
	if (isUserCallbackEnabled || isInternalCallbackEnabled) {		auto& hipUserCallback = hipInterceptor.getUserCallback();
		auto& hipInternalCallback = hipInterceptor.getInternalCallback();
		// Copy Arguments for PHASE_ENTER
		// Flag to skip calling the original function
		bool skipFunction{false};
		std::optional<std::any> out{std::nullopt};
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction, &out);
		if (!skipFunction) {
			static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hip_init");
			out = hipFunc();
		};
		// Exit Callback
		if (isUserCallbackEnabled) hipUserCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId);
		if (isInternalCallbackEnabled) hipInternalCallback(nullptr, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction, &out);

		return std::any_cast<hipError_t>(*out);
	} else {
		static auto hipFunc = hipInterceptor.getHipFunction<hipError_t(*)()>("hip_init");
		return hipFunc();
	};
}

