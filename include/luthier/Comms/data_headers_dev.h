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

#include "data_headers.h"
#include "hip_utils.h"
#include "message.h"

namespace dh_comms {
__device__ inline wave_header_t::wave_header_t(uint64_t exec, uint64_t data_size, bool is_vector_message,
                                               bool has_lane_headers, uint64_t timestamp, uint32_t active_lane_count,
                                               uint64_t dwarf_fname_hash, uint32_t dwarf_line, uint32_t dwarf_column,
                                               uint32_t user_type, uint32_t user_data)
    : exec(exec),
      data_size(data_size),
      is_vector_message(is_vector_message),
      has_lane_headers(has_lane_headers),
      timestamp(timestamp),
      dwarf_fname_hash(dwarf_fname_hash),
      dwarf_line(dwarf_line),
      dwarf_column(dwarf_column),
      user_type(user_type),
      user_data(user_data),
      block_idx_x(blockIdx.x),
      block_idx_y(blockIdx.y),
      block_idx_z(blockIdx.z),
      wave_num(__wave_num()),
      active_lane_count(active_lane_count),
      arch(gcnarch::unsupported) { // We'll check if the arch is supported in the constructor body.
  // for pre-MI300 hardware that isn't partitioned into XCCx, we set the xcc_id to zero. For
  // the MI300 variants (gfx94[012], we query the hardware registers to find out where on the
  // device we are running. Documentation of the hardware registers can be found in Section
  // 3.12 of the AMD Instinct MI300 Instruction Set Architecture Reference Guide at
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf.
  // Note that a wave may be preempted from the XCC|SE|CU on which it started and resumed on
  // another XCC|SE|CU, so it's not safe to rely on these values.
  xcc_id = 0;
#if defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__)
  uint32_t xcc_reg;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc_reg));
  xcc_id = (xcc_reg & 0xf);
#endif
  uint32_t cu_se_reg;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(cu_se_reg));
  se_id = ((cu_se_reg >> 13) & 0x7);
  cu_id = (cu_se_reg >> 8) & 0xf;

#if defined(__gfx906__)
#pragma message("Building device code for gfx906")
  arch = gcnarch::gfx906;
#elif defined(__gfx908__)
#pragma message("Building device code for gfx908")
  arch = gcnarch::gfx908;
#elif defined(__gfx90a__)
#pragma message("Building device code for gfx90a")
  arch = gcnarch::gfx90a;
#elif defined(__gfx940__)
#pragma message("Building device code for gfx940")
  arch = gcnarch::gfx940;
#elif defined(__gfx941__)
#pragma message("Building device code for gfx941")
  arch = gcnarch::gfx941;
#elif defined(__gfx942__)
#pragma message("Building device code for gfx942")
  arch = gcnarch::gfx942;
#elif !defined(__host__)
#error Unsupported GPU architecture, update data_headers_dev.h
#endif
}

__device__ inline lane_header_t::lane_header_t() {
  thread_idx_x = threadIdx.x;
  thread_idx_y = threadIdx.y;
  thread_idx_z = threadIdx.z;
}

} // namespace dh_comms