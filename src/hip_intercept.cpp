#include "hip_intercept.hpp"
#include <link.h>

luthier::hip::Interceptor::Interceptor() {
  // Iterate through the process' loaded shared objects and try to dlopen the
  // first entry with a file name starting with the given 'pattern'. This allows
  // the loader to acquire a handle to the target library iff it is already
  // loaded. The handle is used to query symbols exported by that library.
  auto Callback = [this](dl_phdr_info *Info) {
    if (Handle == nullptr && fs::path(Info->dlpi_name)
                                     .filename()
                                     .string()
                                     .rfind("libamdhip64.so", 0) == 0)
      Handle = ::dlopen(Info->dlpi_name, RTLD_LAZY);
  };
  dl_iterate_phdr(
      [](dl_phdr_info *Info, size_t Size, void *Data) {
        (*reinterpret_cast<decltype(Callback) *>(Data))(Info);
        return 0;
      },
      &Callback);
};

extern "C" __attribute__((visibility("default"))) void
__hipRegisterFunction(hip::FatBinaryInfo **Modules, const void *HostFunction,
                      char *DeviceFunction, const char *DeviceName,
                      unsigned int ThreadLimit, uint3 *Tid, uint3 *Bid,
                      dim3 *BlockDim, dim3 *GridDim, int *WSize) {
  auto &HipInterceptor = luthier::hip::Interceptor::instance();
  auto ApiId = luthier::hip::HIP_API_ID___hipRegisterFunction;
  bool IsInternalCallbackEnabled =
      HipInterceptor.isInternalCallbackEnabled(ApiId);
  static auto HipFunc = HipInterceptor.getHipFunction<void (*)(
      hip::FatBinaryInfo **, const void *, char *, const char *, unsigned int,
      uint3 *, uint3 *, dim3 *, dim3 *, int *)>("__hipRegisterFunction");
  if (IsInternalCallbackEnabled) {
    auto &HipInternalCallback = HipInterceptor.getInternalCallback();
    // Copy Arguments for PHASE_ENTER
    luthier::hip::ApiArgs Args;
    Args.__hipRegisterFunction = {
        Modules, HostFunction, DeviceFunction, DeviceName, ThreadLimit,
        Tid,     Bid,          BlockDim,       GridDim,    WSize};
    HipInternalCallback(Args, nullptr, luthier::API_EVT_PHASE_ENTER, ApiId);
    HipFunc(
        Args.__hipRegisterFunction.modules,
        Args.__hipRegisterFunction.hostFunction,
        Args.__hipRegisterFunction.deviceFunction,
        Args.__hipRegisterFunction.deviceName,
        Args.__hipRegisterFunction.threadLimit, Args.__hipRegisterFunction.tid,
        Args.__hipRegisterFunction.bid, Args.__hipRegisterFunction.blockDim,
        Args.__hipRegisterFunction.gridDim, Args.__hipRegisterFunction.wSize);
    // Exit Callback
    HipInternalCallback(Args, nullptr, luthier::API_EVT_PHASE_EXIT, ApiId);
    // Copy the modified arguments back to the original arguments (if non-const)
    Modules = Args.__hipRegisterFunction.modules;
    DeviceFunction = Args.__hipRegisterFunction.deviceFunction;
    ThreadLimit = Args.__hipRegisterFunction.threadLimit;
    Tid = Args.__hipRegisterFunction.tid;
    Bid = Args.__hipRegisterFunction.bid;
    BlockDim = Args.__hipRegisterFunction.blockDim;
    GridDim = Args.__hipRegisterFunction.gridDim;
    WSize = Args.__hipRegisterFunction.wSize;
  } else {
    HipFunc(Modules, HostFunction, DeviceFunction, DeviceName, ThreadLimit, Tid,
            Bid, BlockDim, GridDim, WSize);
  };
}

extern "C" __attribute__((visibility("default"))) void
__hipRegisterManagedVar(void *hipModule, void **pointer, void *init_value,
                        const char *name, size_t size, unsigned align) {
  auto &HipInterceptor = luthier::hip::Interceptor::instance();
  auto ApiId = luthier::hip::HIP_API_ID___hipRegisterManagedVar;
  bool IsInternalCallbackEnabled =
      HipInterceptor.isInternalCallbackEnabled(ApiId);
  if (IsInternalCallbackEnabled) {
    auto &HipInternalCallback = HipInterceptor.getInternalCallback();
    // Copy Arguments for PHASE_ENTER
    luthier::hip::ApiArgs Args;
    Args.__hipRegisterManagedVar = {hipModule, pointer, init_value,
                                    name,      size,    align};
    HipInternalCallback(Args, nullptr, luthier::API_EVT_PHASE_ENTER, ApiId);
    static auto HipFunc = HipInterceptor.getHipFunction<void (*)(
        void *, void **, void *, const char *, size_t, unsigned)>(
        "__hipRegisterManagedVar");
    HipFunc(Args.__hipRegisterManagedVar.hipModule,
            Args.__hipRegisterManagedVar.pointer,
            Args.__hipRegisterManagedVar.init_value,
            Args.__hipRegisterManagedVar.name,
            Args.__hipRegisterManagedVar.size,
            Args.__hipRegisterManagedVar.align);
    // Exit Callback
    HipInternalCallback(Args, nullptr, luthier::API_EVT_PHASE_EXIT, ApiId);
    // Copy the modified arguments back to the original arguments (if non-const)
    hipModule = Args.__hipRegisterManagedVar.hipModule;
    pointer = Args.__hipRegisterManagedVar.pointer;
    init_value = Args.__hipRegisterManagedVar.init_value;
    name = Args.__hipRegisterManagedVar.name;
    size = Args.__hipRegisterManagedVar.size;
    align = Args.__hipRegisterManagedVar.align;
  } else {
    static auto HipFunc = HipInterceptor.getHipFunction<void (*)(
        void *, void **, void *, const char *, size_t, unsigned)>(
        "__hipRegisterManagedVar");
    HipFunc(hipModule, pointer, init_value, name, size, align);
  };
}

// extern "C" __attribute__((visibility("default"))) void
//__hipRegisterSurface(hip::FatBinaryInfo **modules, void *var, char *hostVar,
//                      char *deviceVar, int type, int ext) {
//   auto &HipInterceptor = luthier::hip::Interceptor::instance();
//   auto ApiId = luthier::hip::HIP_API_ID___hipRegisterSurface;
//   bool IsInternalCallbackEnabled =
//       HipInterceptor.isInternalCallbackEnabled(ApiId);
//   if (IsInternalCallbackEnabled) {
//     auto &HipInternalCallback = HipInterceptor.getInternalCallback();
//     // Copy Arguments for PHASE_ENTER
//     hip___hipRegisterSurface_api_args_t hipFuncArgs{modules,   var,  hostVar,
//                                                     deviceVar, type, ext};
//     if (isUserCallbackEnabled)
//       hipUserCallback(static_cast<void *>(&hipFuncArgs),
//                       LUTHIER_API_EVT_PHASE_ENTER, ApiId);
//     if (IsInternalCallbackEnabled)
//       HipInternalCallback(static_cast<void *>(&hipFuncArgs),
//                           LUTHIER_API_EVT_PHASE_ENTER, ApiId, &skipFunction,
//                           &out);
//     if (!skipFunction) {
//       static auto hipFunc = HipInterceptor.getHipFunction<void (*)(
//           hip::FatBinaryInfo **, void *, char *, char *, int, int)>(
//           "__hipRegisterSurface");
//       hipFunc(hipFuncArgs.modules, hipFuncArgs.var, hipFuncArgs.hostVar,
//               hipFuncArgs.deviceVar, hipFuncArgs.type, hipFuncArgs.ext);
//     };
//     // Exit Callback
//     if (isUserCallbackEnabled)
//       hipUserCallback(static_cast<void *>(&hipFuncArgs),
//                       LUTHIER_API_EVT_PHASE_EXIT, ApiId);
//     if (IsInternalCallbackEnabled)
//       HipInternalCallback(static_cast<void *>(&hipFuncArgs),
//                           LUTHIER_API_EVT_PHASE_EXIT, ApiId, &skipFunction,
//                           &out);
//     // Copy the modified arguments back to the original arguments (if
//     non-const) modules = hipFuncArgs.modules; var = hipFuncArgs.var; hostVar
//     = hipFuncArgs.hostVar; deviceVar = hipFuncArgs.deviceVar; type =
//     hipFuncArgs.type; ext = hipFuncArgs.ext;
//   } else {
//     static auto hipFunc = HipInterceptor.getHipFunction<void (*)(
//         hip::FatBinaryInfo **, void *, char *, char *, int, int)>(
//         "__hipRegisterSurface");
//     hipFunc(modules, var, hostVar, deviceVar, type, ext);
//   };
// }

// extern "C" __attribute__((visibility("default"))) void
//__hipRegisterTexture(hip::FatBinaryInfo **modules, void *var, char *hostVar,
//                      char *deviceVar, int type, int norm, int ext) {
//   auto &hipInterceptor = luthier::HipInterceptor::instance();
//   auto apiId = HIP_PRIVATE_API_ID___hipRegisterTexture;
//   bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
//   bool isInternalCallbackEnabled =
//       hipInterceptor.isInternalCallbackEnabled(apiId);
//   if (isUserCallbackEnabled || isInternalCallbackEnabled) {
//     auto &hipUserCallback = hipInterceptor.getUserCallback();
//     auto &hipInternalCallback = hipInterceptor.getInternalCallback();
//     // Copy Arguments for PHASE_ENTER
//     // Flag to skip calling the original function
//     bool skipFunction{false};
//     std::optional<std::any> out{std::nullopt};
//     hip___hipRegisterTexture_api_args_t hipFuncArgs{
//         modules, var, hostVar, deviceVar, type, norm, ext};
//     if (isUserCallbackEnabled)
//       hipUserCallback(static_cast<void *>(&hipFuncArgs),
//                       LUTHIER_API_EVT_PHASE_ENTER, apiId);
//     if (isInternalCallbackEnabled)
//       hipInternalCallback(static_cast<void *>(&hipFuncArgs),
//                           LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction,
//                           &out);
//     if (!skipFunction) {
//       static auto hipFunc = hipInterceptor.getHipFunction<void (*)(
//           hip::FatBinaryInfo **, void *, char *, char *, int, int, int)>(
//           "__hipRegisterTexture");
//       hipFunc(hipFuncArgs.modules, hipFuncArgs.var, hipFuncArgs.hostVar,
//               hipFuncArgs.deviceVar, hipFuncArgs.type, hipFuncArgs.norm,
//               hipFuncArgs.ext);
//     };
//     // Exit Callback
//     if (isUserCallbackEnabled)
//       hipUserCallback(static_cast<void *>(&hipFuncArgs),
//                       LUTHIER_API_EVT_PHASE_EXIT, apiId);
//     if (isInternalCallbackEnabled)
//       hipInternalCallback(static_cast<void *>(&hipFuncArgs),
//                           LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction,
//                           &out);
//     // Copy the modified arguments back to the original arguments (if
//     non-const) modules = hipFuncArgs.modules; var = hipFuncArgs.var; hostVar
//     = hipFuncArgs.hostVar; deviceVar = hipFuncArgs.deviceVar; type =
//     hipFuncArgs.type; norm = hipFuncArgs.norm; ext = hipFuncArgs.ext;
//   } else {
//     static auto hipFunc = hipInterceptor.getHipFunction<void (*)(
//         hip::FatBinaryInfo **, void *, char *, char *, int, int, int)>(
//         "__hipRegisterTexture");
//     hipFunc(modules, var, hostVar, deviceVar, type, norm, ext);
//   };
// }
//
// extern "C" __attribute__((visibility("default"))) void
//__hipRegisterVar(hip::FatBinaryInfo **modules, void *var, char *hostVar,
//                  char *deviceVar, int ext, size_t size, int constant,
//                  int global) {
//   auto &hipInterceptor = luthier::HipInterceptor::instance();
//   auto apiId = HIP_PRIVATE_API_ID___hipRegisterVar;
//   bool isUserCallbackEnabled = hipInterceptor.isUserCallbackEnabled(apiId);
//   bool isInternalCallbackEnabled =
//       hipInterceptor.isInternalCallbackEnabled(apiId);
//   if (isUserCallbackEnabled || isInternalCallbackEnabled) {
//     auto &hipUserCallback = hipInterceptor.getUserCallback();
//     auto &hipInternalCallback = hipInterceptor.getInternalCallback();
//     // Copy Arguments for PHASE_ENTER
//     // Flag to skip calling the original function
//     bool skipFunction{false};
//     std::optional<std::any> out{std::nullopt};
//     hip___hipRegisterVar_api_args_t hipFuncArgs{
//         modules, var, hostVar, deviceVar, ext, size, constant, global};
//     if (isUserCallbackEnabled)
//       hipUserCallback(static_cast<void *>(&hipFuncArgs),
//                       LUTHIER_API_EVT_PHASE_ENTER, apiId);
//     if (isInternalCallbackEnabled)
//       hipInternalCallback(static_cast<void *>(&hipFuncArgs),
//                           LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction,
//                           &out);
//     if (!skipFunction) {
//       static auto hipFunc =
//           hipInterceptor
//               .getHipFunction<void (*)(hip::FatBinaryInfo **, void *, char *,
//                                        char *, int, size_t, int, int)>(
//                   "__hipRegisterVar");
//       hipFunc(hipFuncArgs.modules, hipFuncArgs.var, hipFuncArgs.hostVar,
//               hipFuncArgs.deviceVar, hipFuncArgs.ext, hipFuncArgs.size,
//               hipFuncArgs.constant, hipFuncArgs.global);
//     };
//     // Exit Callback
//     if (isUserCallbackEnabled)
//       hipUserCallback(static_cast<void *>(&hipFuncArgs),
//                       LUTHIER_API_EVT_PHASE_EXIT, apiId);
//     if (isInternalCallbackEnabled)
//       hipInternalCallback(static_cast<void *>(&hipFuncArgs),
//                           LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction,
//                           &out);
//     // Copy the modified arguments back to the original arguments (if
//     non-const) modules = hipFuncArgs.modules; var = hipFuncArgs.var; hostVar
//     = hipFuncArgs.hostVar; deviceVar = hipFuncArgs.deviceVar; ext =
//     hipFuncArgs.ext; size = hipFuncArgs.size; constant =
//     hipFuncArgs.constant; global = hipFuncArgs.global;
//   } else {
//     static auto hipFunc = hipInterceptor.getHipFunction<void (*)(
//         hip::FatBinaryInfo **, void *, char *, char *, int, size_t, int,
//         int)>(
//         "__hipRegisterVar");
//     hipFunc(modules, var, hostVar, deviceVar, ext, size, constant, global);
//   };
// }