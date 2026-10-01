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

#pragma once

#include <cstddef>
#include <cstdint>

// Lazy-loading HIP runtime function pointers.
//
// On first call, resolves HIP symbols via dlsym(RTLD_DEFAULT, ...) so we
// piggyback on whatever HIP the application already loaded (system HIP for
// regular HIP programs, PyTorch's bundled HIP for Triton workloads). Falls
// back to dlopen("libamdhip64.so", RTLD_LAZY) for standalone programs that
// haven't loaded HIP yet. Thread-safe via std::call_once.

namespace dh_comms {
namespace hip_runtime_loader {

// HIP error type (matches hipError_t)
using hipError_t = int;

// Function pointer types matching the HIP API signatures
using hipHostMalloc_fn = hipError_t (*)(void**, size_t, unsigned int);
using hipMalloc_fn = hipError_t (*)(void**, size_t);
using hipMemcpy_fn = hipError_t (*)(void*, const void*, size_t, int);
using hipFree_fn = hipError_t (*)(void*);
using hipGetErrorString_fn = const char* (*)(hipError_t);

// Resolved function pointers (valid after init())
extern hipHostMalloc_fn pfn_hipHostMalloc;
extern hipMalloc_fn pfn_hipMalloc;
extern hipMemcpy_fn pfn_hipMemcpy;
extern hipFree_fn pfn_hipFree;
extern hipGetErrorString_fn pfn_hipGetErrorString;

// Initialise the loader (idempotent, thread-safe). Called automatically by
// the first use of any function pointer, but may also be called explicitly.
void init();

} // namespace hip_runtime_loader
} // namespace dh_comms
