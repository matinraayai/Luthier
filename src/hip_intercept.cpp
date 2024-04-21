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
      hip::FatBinaryInfo **, const void *, char *, const char *,
      unsigned int, uint3 *, uint3 *, dim3 *, dim3 *, int *)>(
      "__hipRegisterFunction");
  if (IsInternalCallbackEnabled) {
    auto &HipInternalCallback = HipInterceptor.getInternalCallback();
    // Copy Arguments for PHASE_ENTER
    luthier::hip::ApiArgs Args;
    Args.__hipRegisterFunction = {
        Modules, HostFunction, DeviceFunction, DeviceName, ThreadLimit,
        Tid,     Bid,          BlockDim,       GridDim,    WSize};
    HipInternalCallback(Args, nullptr,
                        luthier::API_EVT_PHASE_ENTER, ApiId);
    HipFunc(Args.__hipRegisterFunction.modules,
            Args.__hipRegisterFunction.hostFunction,
            Args.__hipRegisterFunction.deviceFunction,
            Args.__hipRegisterFunction.deviceName,
            Args.__hipRegisterFunction.threadLimit,
            Args.__hipRegisterFunction.tid,
            Args.__hipRegisterFunction.bid,
            Args.__hipRegisterFunction.blockDim,
            Args.__hipRegisterFunction.gridDim,
            Args.__hipRegisterFunction.wSize);
    // Exit Callback
    HipInternalCallback(Args, nullptr,
                        luthier::API_EVT_PHASE_EXIT, ApiId);
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