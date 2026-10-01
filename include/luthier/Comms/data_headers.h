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
#include <vector>

namespace dh_comms {
//! \brief Messages start with a wave header containing information that partains to the whole wave.
//!
//! User device code does not use wave headers directly, but user host code may.
struct wave_header_t {
  uint64_t exec; //!< Execution mask of the wavefront submitting the message.

  uint64_t data_size : 62;        //!< \brief Size of the data following the wave header.
                                  //!<
                                  //!< This is computed as: (number of active lanes) * (lane header (4 bytes) +
                                  //!< number of data bytes per lane), rounded up to nearest multiple of 4 bytes.
  uint64_t is_vector_message : 1; //!< Indicates whether the message has one data item per lane (for vector messages),
                                  //!< or a single data item for the whole wave (for scalar messages).
  uint64_t has_lane_headers : 1;  //!< Indicates whether the message has lane headers (see lane_header_t) for the
                                  //!< active lanes.
  uint64_t timestamp;             //!< GPU timestamp of the moment the message was submitted.
  uint64_t
      dwarf_fname_hash;  //!< Hash of the source file containing the instruction that was instrumented with the message
  uint32_t dwarf_line;   //!< Line number for the instrumented instruction
  uint32_t dwarf_column; //!< Column for the instrumented instruction
  uint32_t user_type;    //!< \brief User-defined tag that indicates the content/interpretation of the data.
                         //!<
                         //!< Since kernels can submit any type of data that is valid on the device, such as
                         //!< an uint64_t (which could represent a memory address or a loop counter) or a
                         //!< struct containing several basic type), host code that processes the messages must
                         //!< be able to determine what kind of data is in the message, and how to process it.
                         //!< Note that the processing code on the host is to be provided by the user, although
  //!< some basic data processor classes are included with dh_comms. By using the user_type
  //!< tag, kernel code can indicate the type of data in the message, so that the host
  //!< processing code knows what to do with it.
  uint32_t user_data; //!< \brief user-defined field for additional data for the message.
                      //!<
                      //!< This field can for instance be used to distinguish between messages from different
                      //!< kernels or kernel dispatches, if we are using a single dh_comms object to pass
                      //!< messages from the device to the host, as opposed to having a separate dh_comms
  //!< object per kernel or kernel dispatch. Another use of the user_type tag is to distinguish
  //!< between memory reads vs writes, if our messages contain memory addresses (say, if
  //!< we are building a memory heatmap of the application).
  //!< different locations in the ISA or source code.
  uint16_t block_idx_x;  //!< blockIdx.x value of the workgroup to which the wave belongs.
  uint16_t block_idx_y;  //!< blockIdx.y value of the workgroup to which the wave belongs.
  uint16_t block_idx_z;  //!< blockIdx.z value of the workgroup to which the wave belongs.
  uint16_t wave_num : 4; //!< Wave number withing workgroup; there can be at most 16 waves per workgroup.
  uint16_t xcc_id : 4;   //!< \brief Number of the XCC on which the wavefront runs; zero for pre-MI300 hardware
                         //!<
                         //!< From MI300 on, CDNA hardware is partitioned into multiple XCCs (these are the individual
                         //!< dies). MI200 has two dies, but these are called XCDs, and opposed to MI300, the MI200
                         //!< dies are individual devices; the workgroups of a kernel can only run on one of them.
                         //!< MI300 can be booted into various partitioning schemes, including SPX, where all the dies
                         //!< combine into a single devices, and the waves of a kernel can be distributed over all
  //!< XCCs. Regardless of the CDNA generation, the dies are subdivided into SEs (Shader Engines),
  //!< and there can be up to 8 SEs per die. SEs are subdivided into CUs (Compute Units), of which
  //!< there can be up to 16 per SE. If as SE has fewer than 16 CUs, they need not be numbered
  //!< consecutively, nor does the lowest CU number need to be 0. In fact, the CU numbers may
  //!< be different from one SE on the device to the next SE.
  uint16_t se_id : 3;        //!< Number of the SE of the XCC on which the wavefront runs.
  uint16_t cu_id : 4;        //!< Number of the CU of the SE on which the wavefront runs.
  uint16_t unused16 : 1;     //!< Padding; reserved for future use.
  uint8_t active_lane_count; //!< number of active lanes in the wavefront, i.e., number of 1-bits in the execution mask
  uint8_t arch;              // architecture of the device we're running on

  //! Wave header constructor; wave header members for which there is no corresponding
  //! constructor argument are detected and assigned by the constructor.
  __device__ wave_header_t(uint64_t exec, uint64_t data_size, bool is_vector_message, bool has_lane_headers,
                           uint64_t timestamp, uint32_t active_lane_count, uint64_t dwarf_fname_hash,
                           uint32_t dwarf_line, uint32_t dwarf_column, uint32_t user_type, uint32_t user_data);

  //! Wave header constructor; creates a wave header from raw bytes
  wave_header_t(const char *wave_header_p);
};

//! \brief After the wave header, there's an optional lane header for each of the active lanes in the wavefront.
//!
//! The lane header contains the threadIdx.[x,y,z] values for the lane. Since the thread indices range from
//! 0 to at most 1023 in each dimension, they can be packed into a 4-byte struct.
//! User device code does not use lane headers directly, but user host code may.

struct lane_header_t {
  uint32_t thread_idx_x : 10;
  uint32_t thread_idx_y : 10;
  uint32_t thread_idx_z : 10;
  uint32_t unused : 2;

  //! The lane header default device constructor fills the thread_idx_[x,y,z] fields.
  __device__ lane_header_t();

  //! Lane header constructor; creates a lane header from raw bytes
  lane_header_t(const char *lane_header_p);
};

std::vector<size_t> get_lane_ids_of_active_lanes(const wave_header_t &wave_header);

} // namespace dh_comms