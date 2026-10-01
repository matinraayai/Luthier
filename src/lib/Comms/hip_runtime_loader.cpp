// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "luthier/Comms/hip_runtime_loader.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>

namespace dh_comms {
namespace hip_runtime_loader {

hipHostMalloc_fn pfn_hipHostMalloc = nullptr;
hipMalloc_fn pfn_hipMalloc = nullptr;
hipMemcpy_fn pfn_hipMemcpy = nullptr;
hipFree_fn pfn_hipFree = nullptr;
hipGetErrorString_fn pfn_hipGetErrorString = nullptr;

static std::once_flag init_flag;

static void do_init() {
  // First try RTLD_DEFAULT — picks up whatever HIP the application loaded.
  auto try_resolve = [](const char* name) -> void* {
    return dlsym(RTLD_DEFAULT, name);
  };

  pfn_hipHostMalloc = reinterpret_cast<hipHostMalloc_fn>(try_resolve("hipHostMalloc"));
  pfn_hipMalloc = reinterpret_cast<hipMalloc_fn>(try_resolve("hipMalloc"));
  pfn_hipMemcpy = reinterpret_cast<hipMemcpy_fn>(try_resolve("hipMemcpy"));
  pfn_hipFree = reinterpret_cast<hipFree_fn>(try_resolve("hipFree"));
  pfn_hipGetErrorString = reinterpret_cast<hipGetErrorString_fn>(try_resolve("hipGetErrorString"));

  // If any symbol is missing, fall back to dlopen.
  if (!pfn_hipHostMalloc || !pfn_hipMalloc || !pfn_hipMemcpy ||
      !pfn_hipFree || !pfn_hipGetErrorString) {
    void* handle = dlopen("libamdhip64.so", RTLD_LAZY);
    if (!handle) {
      fprintf(stderr, "dh_comms: failed to load HIP runtime: %s\n", dlerror());
      abort();
    }

    auto resolve_or_die = [&](const char* name) -> void* {
      void* sym = dlsym(handle, name);
      if (!sym) {
        fprintf(stderr, "dh_comms: failed to resolve %s: %s\n", name, dlerror());
        abort();
      }
      return sym;
    };

    pfn_hipHostMalloc = reinterpret_cast<hipHostMalloc_fn>(resolve_or_die("hipHostMalloc"));
    pfn_hipMalloc = reinterpret_cast<hipMalloc_fn>(resolve_or_die("hipMalloc"));
    pfn_hipMemcpy = reinterpret_cast<hipMemcpy_fn>(resolve_or_die("hipMemcpy"));
    pfn_hipFree = reinterpret_cast<hipFree_fn>(resolve_or_die("hipFree"));
    pfn_hipGetErrorString = reinterpret_cast<hipGetErrorString_fn>(resolve_or_die("hipGetErrorString"));
  }
}

void init() {
  std::call_once(init_flag, do_init);
}

} // namespace hip_runtime_loader
} // namespace dh_comms
