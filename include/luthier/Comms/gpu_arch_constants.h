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

#include <cstdint>
#include <string>

namespace dh_comms {
namespace gpu_arch_constants {

// L2 cache line sizes for each GCN architecture (indexed by gcnarch enum)
// Values are in bytes. Architecture enum is defined in message.h
constexpr uint32_t L2_CACHE_LINE_SIZES[] = {
    0,   // unsupported (gcnarch::unsupported = 0)
    64,  // gfx906 (gcnarch::gfx906 = 1)
    64,  // gfx908 (gcnarch::gfx908 = 2)
    128, // gfx90a (gcnarch::gfx90a = 3)
    128, // gfx940 (gcnarch::gfx940 = 4)
    128, // gfx941 (gcnarch::gfx941 = 5)
    128  // gfx942 (gcnarch::gfx942 = 6)
};

// Helper function to get L2 cache line size with bounds checking
// Returns 0 for invalid architecture values
inline uint32_t get_l2_cache_line_size(uint8_t arch) {
    constexpr size_t array_size = sizeof(L2_CACHE_LINE_SIZES) / sizeof(L2_CACHE_LINE_SIZES[0]);
    if (arch >= array_size) {
        return 0;
    }
    return L2_CACHE_LINE_SIZES[arch];
}

// Helper function to convert architecture string to gcnarch enum value
// Returns gcnarch::unsupported for unknown architectures
inline uint8_t arch_string_to_enum(const std::string& arch_str) {
    if (arch_str == "gfx906") return 1; // gcnarch::gfx906
    if (arch_str == "gfx908") return 2; // gcnarch::gfx908
    if (arch_str == "gfx90a") return 3; // gcnarch::gfx90a
    if (arch_str == "gfx940") return 4; // gcnarch::gfx940
    if (arch_str == "gfx941") return 5; // gcnarch::gfx941
    if (arch_str == "gfx942") return 6; // gcnarch::gfx942
    return 0; // gcnarch::unsupported
}

// Reverse mapping: convert gcnarch enum value to architecture string
// Returns "unknown" for unsupported or out-of-range values
inline const char* arch_enum_to_string(uint8_t arch) {
    switch (arch) {
        case 1: return "gfx906";
        case 2: return "gfx908";
        case 3: return "gfx90a";
        case 4: return "gfx940";
        case 5: return "gfx941";
        case 6: return "gfx942";
        default: return "unknown";
    }
}

// Device-side L2 cache line size (for use in device code)
// This is a placeholder for future device-side integration.
// When device code needs to know its own architecture's cache line size,
// this should be set at compile time based on the GPU target.
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx90a__)
constexpr uint32_t DEVICE_L2_CACHE_LINE_SIZE = 128;
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
constexpr uint32_t DEVICE_L2_CACHE_LINE_SIZE = 128;
#elif defined(__gfx906__) || defined(__gfx908__)
constexpr uint32_t DEVICE_L2_CACHE_LINE_SIZE = 64;
#else
constexpr uint32_t DEVICE_L2_CACHE_LINE_SIZE = 0; // unsupported
#endif
#endif // __HIP_DEVICE_COMPILE__

} // namespace gpu_arch_constants
} // namespace dh_comms
