#include "src/internal.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#include "src/util.h"

std::vector<hipModule_t>* call_original_hip_register_fat_binary(
    const void* data);

extern "C" std::vector<hipModule_t>* __hipRegisterFatBinary(const void* data) {
  // hipError_t err;

  printf("Here in %s\n", __FUNCTION__);

  // const __CudaFatBinaryWrapper* fbwrapper =
  //     reinterpret_cast<const __CudaFatBinaryWrapper*>(data);
  // __builtin_dump_struct(fbwrapper, &printf);

  // if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
  //   return call_original_hip_register_fat_binary(data);
  // }

  // const __ClangOffloadBundleHeader* header = fbwrapper->binary;
  // __builtin_dump_struct(header, &printf);
  // std::string magic(reinterpret_cast<const char*>(header),
  //                   sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  // if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
  //   return nullptr;
  // }

  // int device_count = 0;
  // err = hipGetDeviceCount(&device_count);
  // if (err != hipSuccess) {
  //   panic("cannot get device count.");
  // }

  // const __ClangOffloadBundleDesc* desc = &header->desc[0];

  // auto modules = new std::vector<hipModule_t>(device_count);
  // for (uint64_t i = 0; i < header->numBundles; i++, desc = desc->next()) {
  //   printf("%lu\n", i);
  //   printf("%s\n", desc->triple);

  //   std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};
  //   if (triple.compare(AMDGCN_AMDHSA_TRIPLE)) {
  //     continue;
  //   }

  //   std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)],
  //                      desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};
  //   printf("Found bundle for %s\n", target.c_str());

  //   for (int device_id = 1; device_id <= device_count; device_id++) {
  //     struct hipDeviceProp_t device_prop;
  //     err = hipGetDeviceProperties(&device_prop, device_id);
  //     if (err != hipSuccess) {
  //       panic("cannot get device properties.");
  //     }

  //     ihipModule_t* module = new ihipModule_t;

  //     auto image_ptr = reinterpret_cast<uintptr_t>(header) + desc->offset;

  //     std::string image{reinterpret_cast<const char*>(image_ptr),
  //     desc->size};
  //     // __hipDumpCodeObject(image);

  //     module->executable = hsa_executable_t();
  //     module->executable.handle = reinterpret_cast<uint64_t>(image_ptr);

  //     modules->at(device_id - 1) = module;
  //   }
  // }

  auto modules = call_original_hip_register_fat_binary(data);

  printf("Number of modules: %lu\n", modules->size());
  for (auto module : *modules) {
    __builtin_dump_struct(module, &printf);
  }

  return modules;
}

std::vector<hipModule_t>* call_original_hip_register_fat_binary(
    const void* data) {
  std::vector<hipModule_t>* (*func)(const void*);
  func = (decltype(func))dlsym(RTLD_NEXT, "__hipRegisterFatBinary");
  std::vector<hipModule_t>* ret = func(data);

  return ret;
}