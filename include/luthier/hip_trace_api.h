#ifndef HIP_ARGS
#define HIP_ARGS
#undef USE_PROF_API
#include <hip/hip_runtime_api.h>

namespace luthier::hip {


enum ApiID : unsigned int {
  HIP_API_ID_NONE = 0,
  HIP_API_ID_FIRST = 1,
  HIP_API_ID___hipRegisterFatBinary = 2,
  HIP_API_ID___hipRegisterFunction = 3,
  HIP_API_ID___hipRegisterManagedVar = 4,
  HIP_API_ID___hipRegisterSurface = 5,
  HIP_API_ID___hipRegisterTexture = 6,
  HIP_API_ID___hipRegisterVar = 7,
  HIP_API_ID___hipUnregisterFatBinary = 8,
  HIP_API_ID_LAST = 8
};

typedef union {
  struct {
    const void *data;
  } __hipRegisterFatBinary;
  struct {
    void **modules;
    const void *hostFunction;
    char *deviceFunction;
    const char *deviceName;
    unsigned int threadLimit;
    uint3 *tid;
    uint3 *bid;
    dim3 *blockDim;
    dim3 *gridDim;
    int *wSize;
  } __hipRegisterFunction;
  struct {
    void *hipModule;
    void **pointer;
    void *init_value;
    const char *name;
    size_t size;
    unsigned align;
  } __hipRegisterManagedVar;
  struct {
    void **modules;
    void *var;
    char *hostVar;
    char *deviceVar;
    int type;
    int ext;
  } __hipRegisterSurface;
  struct {
    void **modules;
    void *var;
    char *hostVar;
    char *deviceVar;
    int type;
    int norm;
    int ext;
  } __hipRegisterTexture;
  struct {
    void **modules;
    void *var;
    char *hostVar;
    char *deviceVar;
    int ext;
    size_t size;
    int constant;
    int global;
  } __hipRegisterVar;
  struct {
    void **modules;
  } __hipUnregisterFatBinary;
} ApiArgs;

typedef union {
  void **__hipRegisterFatBinary;
} ApiReturn;

} // namespace luthier::hip

#endif