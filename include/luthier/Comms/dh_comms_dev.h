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

//! @file

#include "data_headers_dev.h"
#include "dh_comms.h"

// device functions
namespace dh_comms {
// Have Doxygen skip implementation functions that aren't used directly in user code
//! @cond

// Returns the index of the sub-buffer to which the calling wave should write.
__device__ inline size_t get_sub_buffer_idx(size_t no_sub_buffers) {
  // To balance the number of waves writing to the sub-buffers, this function
  // calculates the flattened workgroup/block index, and takes that value
  // modulo the number of sub-buffers.
  // So, we're computing (blockIdx.z * gridDim.y * grimDim.z +
  //                      blockIdx.y * gridDim.x +
  //                      blockIdx.x) % no_sub_buffers.
  // A slight problem here is that block dimensions and indices are
  // 32-bit values, so we'd need a 96-bit type to do our calculations
  // without risking overflow if we'd do the calculation as above.
  // Fortunately, the modulo operation has the property that
  // (a * b) % m = ((a % m) * (b % m)) % m and
  // (a + b) % m = ((a % m) + (b % m)) % m and
  // which means that we can apply the modulo operation on intermediate results,
  // and get the same final result without risking overflow.

  size_t grid_dim_x_m = gridDim.x % no_sub_buffers;
  size_t grid_dim_y_m = gridDim.y % no_sub_buffers;
  size_t grid_dim_xy_m = (grid_dim_x_m * grid_dim_y_m) % no_sub_buffers;
  size_t block_id_z_grid_dim_xy_m = (blockIdx.z * grid_dim_xy_m) % no_sub_buffers;

  size_t block_id_y_grid_dim_x_m = (blockIdx.y * grid_dim_x_m) % no_sub_buffers;

  size_t block_id_x_m = blockIdx.x % no_sub_buffers;

  return (block_id_z_grid_dim_xy_m + block_id_y_grid_dim_x_m + block_id_x_m) % no_sub_buffers;
}

// Have the calling wave spin on an atomic to get exclusive read/write access to a sub-buffer.
__device__ inline void wave_acquire(uint8_t *atomic_flags_d, size_t sub_buf_idx, unsigned int active_lane_id) {
  if (active_lane_id == 0) // only one lane acquires the lock on behalf of the wave
  {
    uint8_t expected = 0; // 0 -> sub-buffer is not locked
    uint8_t desired = 1;  // 1 -> sub-buffer is locked by (this) wave
    bool weak = false;
    while (not __atomic_compare_exchange(&atomic_flags_d[sub_buf_idx], &expected, &desired, weak, __ATOMIC_ACQUIRE,
                                         __ATOMIC_RELAXED)) {
      // __atomic_compare_exchange returns the value at the address (first argument) in
      // the second argument (expected), so we need to reset it in this spin-loop
      expected = 0;
    }
  }
}

// Have the calling wave release its exclusive read/write access to a sub-buffer
// Note: we should be able to do this with an __atomic_store, but in some cases, the
// compiler implemented the __atomic_store without a memory operation, leading to an
// infinite spin loop for the next call to wave_acquire. Implementing this function
// with an __atomic_compare_exchange instead resolves the issue for the cases we
// have seen so far.
__device__ inline void wave_release(uint8_t *atomic_flags_d, size_t sub_buf_idx, unsigned int active_lane_id) {
  if (active_lane_id == 0) // only one lane acquires the lock on behalf of the wave
  {
    uint8_t expected = 1; // 1 -> sub-buffer is locked by this wave
    uint8_t desired = 0;  // 0 -> sub-buffer is unlocked
    bool weak = false;
    while (not __atomic_compare_exchange(&atomic_flags_d[sub_buf_idx], &expected, &desired, weak, __ATOMIC_RELEASE,
                                         __ATOMIC_RELAXED)) {
      // __atomic_compare_exchange returns the value at the address (first argument) in
      // the second argument (expected), so we need to reset it in this spin-loop
      expected = 1;
    }
  }
}

// Have the calling wave, which has exclusive read/write access to a sub-buffer, pass control
// of the sub-buffer to the host by setting a flag to 1, and then spin on the flag until the host
// resets it to zero, indicating that it is done processing the sub-buffer.
__device__ inline void wave_signal_host(uint8_t *atomic_flags_hd, size_t sub_buf_idx, unsigned int active_lane_id) {
  if (active_lane_id == 0) // only one lane passes control of the sub-buffer to the host on behalf of the wave
  {
    uint8_t flag = 1;
    __atomic_store_n(&atomic_flags_hd[sub_buf_idx], flag, __ATOMIC_RELEASE);
    while (__atomic_load_n(&atomic_flags_hd[sub_buf_idx], __ATOMIC_ACQUIRE) != 0) {
    }
  }
}

// common function for submission of scalar and vector messages
__device__ inline void generic_submit_message(
    dh_comms_descriptor *rsrc, // Pointer to dh_comms device resources used for message submission.
                               // This pointer is acquired by host code by calling dh_comms::get_dev_rsrc_ptr(),
                               // and passed as a kernel argument to kernels that want to use v_submit_message().
    const void *message,       // Pointer to message to be submitted.
    size_t message_size,       // Size of the message in bytes
    uint64_t dwarf_fname_hash, // Hash for the source file name from which v_submit_message() is called
    uint32_t dwarf_line,       // Line number for which v_submit_message() is called
    uint32_t dwarf_column,     // Column for which v_submit_message() is called
    uint32_t user_type,        // Tag to distinguish between different kinds of messages, used by host
                               // code that processes the messages.
    uint32_t user_data,        // Auxiliary data field; it's use depends on the message type.
    bool is_vector_message,    // true if all active lanes are to submit a message, false for scalar submit, where only
                               // the first active lane is to submit
    bool submit_lane_headers)  // true if lane headers (with thread coordinates for the lane) are to be included after
                               // the wave header
{
  uint64_t timestamp = __clock64(); // Get the timestamp first so that it is minimally perturbed by the direct and
                                    // indirect instructions issued by v_submit_message()
  unsigned int active_lane_id =
      __active_lane_id(); // One lane is doing some operations on behalf of the whole wave. That must be
                          // a lane that is active, i.e., not masked out in the execution mask.
                          // Also: all active lanes write data, and they use the active lane id to determine
                          // the offset in the buffer for writing the lane data.
  size_t sub_buf_idx = get_sub_buffer_idx(rsrc->no_sub_buffers_); // Index of the sub-buffer the wave is going to use
  uint64_t exec =
      __builtin_amdgcn_read_exec(); // Execution mask is passed in the wave header; can be used for analysis purposes.
  unsigned int active_lane_count = __active_lane_count(
      exec); // Passed in the wave header, and used to compute the total size of the message for all lanes.
  size_t *sub_buffer_sizes = rsrc->sub_buffer_sizes_;      // Size of the data currently in our sub-buffer, in bytes
  size_t sub_buffer_capacity = rsrc->sub_buffer_capacity_; // Max number of bytes thatfit into the sub-buffer
  char *buffer = rsrc->buffer_; // Pointer to the main data buffer, of which our sub-buffer is a slice.

  // size of the data after the wave header. Data is written in dword units,
  // so round up to multiple of dword size
  uint64_t lane_header_size = submit_lane_headers ? sizeof(lane_header_t) : 0;
  uint64_t submitting_lane_count = is_vector_message ? active_lane_count : 1;
  uint64_t rounded_message_size = sizeof(uint32_t) * ((message_size + sizeof(uint32_t) - 1) / sizeof(uint32_t));
  uint64_t data_size = active_lane_count * lane_header_size + submitting_lane_count * rounded_message_size;

  wave_acquire(rsrc->atomic_flags_d_, sub_buf_idx,
               active_lane_id); // Get exclusive access to the sub-buffer and associated data
  if (sizeof(wave_header_t) + data_size >
      sub_buffer_capacity) // Sanity check: does the message even fit in an empty sub-buffer?
  {
    *rsrc->error_bits_ |= 1; // If not, set an error bit, don't write any data, and return
    wave_release(rsrc->atomic_flags_d_, sub_buf_idx, active_lane_id);
    return;
  }
  size_t current_size = sub_buffer_sizes[sub_buf_idx]; // Does the message fit in the sub-buffer, given the data already
                                                       // in the sub-buffer?
  if (current_size + sizeof(wave_header_t) + data_size >
      sub_buffer_capacity) { // If not, tell the host to clear the sub-buffer, and to return control to us when it's
                             // done.
    wave_signal_host(rsrc->atomic_flags_hd_, sub_buf_idx, active_lane_id);
    // The host clears the sub-buffer and resets its size to 0. Since it
    // returns control to us, no other wave can have changed the sub-buffer
    // size after wave_signal_host returns. So instead of re-reading the size
    // from memory, we can deduce that it must be 0. This saves a slow memory
    // read.
    current_size = 0;
  }
  // First write the wave header; only one wave takes care of that
  if (active_lane_id == 0) {
    wave_header_t wave_header(exec, data_size, is_vector_message, submit_lane_headers, timestamp, active_lane_count,
                              dwarf_fname_hash, dwarf_line, dwarf_column, user_type, user_data);
    size_t byte_offset = sub_buf_idx * sub_buffer_capacity + current_size;
    wave_header_t *wave_header_p = (wave_header_t *)(&buffer[byte_offset]);
    *wave_header_p = wave_header;
  }

  // Optionally, have the active lanes write their lane headers
  // Note that even for scalar messages, all active lanes write their lane header
  size_t byte_offset = sub_buf_idx * sub_buffer_capacity + current_size + sizeof(wave_header_t);
  if (submit_lane_headers) {
    byte_offset += active_lane_id * sizeof(lane_header_t);
    lane_header_t *lane_header_p = (lane_header_t *)(&buffer[byte_offset]);
    lane_header_t lane_header; // constructor fills in thread coordinates into lane header
    *lane_header_p = lane_header;
  }

  // Write the message one dword at a time in a coalesced fashion
  // All active lanes should start writing after the (optional) lane headers, with the offsets
  // of consecutive active lanes 4 bytes apart.
  if constexpr (sizeof(lane_header_t) == sizeof(uint32_t)) {
    // In the initial/current implementation, lane headers are 4 bytes,
    // which makes updating the offset simple and efficient, since
    // the offsets for consecutive active lanes are already four bytes apart.
    // Note: in the calculation, we use active_lane_count, since all active lanes
    // submit lanes headers if submit_lane_headers == true, and we use
    // lane_header_size which is either 0 if no lane headers are submitted, or
    // sizeof(lane_header_t otherwise)
    byte_offset += active_lane_count * lane_header_size;
  } else {
    // If we ever add data to the lane headers so that the size changes.
    // offsets for consecutive lanes are no longer four bytes apart after
    // writing the lane headers, so we need a more complicated/expensive
    // calculation.
    byte_offset = sub_buf_idx * sub_buffer_capacity      // start of our sub-buffer
                  + current_size                         // skipping past data that was already there
                  + sizeof(wave_header_t)                // skipping past the single wave header
                  + active_lane_count * lane_header_size // skipping past the (optional) lane headers
                  + active_lane_id * sizeof(uint32_t);   // four bytes between consecutive active lanes
  }

  // Write the message four bytes at a time, taking care of
  // the tail of the message, which may have fewer than
  // four byes.
  size_t remaining_bytes = message_size;
  char *src = (char *)message;
  char *dst = &(buffer)[byte_offset];
  while (remaining_bytes) {
    // For vector messages, all active lanes write (submitting_lane_count == active_lane_cout).
    // For scalar messages, only the first active lane writes (submitting_lane_count == 1).
    // Note that only the memcpy is conditional. In particular, the update of size must be
    // unconditional. Otherwise, in the case of scalar messages, where only the first active
    // lane submits, the remaining active lanes would get into an infinite loop (while(remaining_bytes))
    size_t size = min(remaining_bytes, sizeof(uint32_t));
    if (is_vector_message or (active_lane_id == 0)) {
      memcpy(dst, src, size);
    }
    src += sizeof(uint32_t);
    dst += submitting_lane_count * sizeof(uint32_t);
    remaining_bytes -= size;
  }

  // Update sub-buffer size and release lock
  if (active_lane_id == 0) // Only one wave must update the sub-buffer size
  {
    sub_buffer_sizes[sub_buf_idx] += sizeof(wave_header_t) + data_size;
  }
  wave_release(rsrc->atomic_flags_d_, sub_buf_idx, active_lane_id);
}

// End of section to be skipped by Doxygen
//! @endcond

//! \brief Submit a vector message of any type that is valid in device code from the device to the host.
//!
//! Messages are submitted on a per-wave basis, and only the active lanes in the wave submit.
__attribute__((used)) extern "C" __device__ inline void v_submit_message(
    dh_comms_descriptor *rsrc,     //!< Pointer to dh_comms device resources used for message submission.
                                   //!< This pointer is acquired by host code by calling dh_comms::get_dev_rsrc_ptr(),
                                   //!< and passed as a kernel argument to kernels that want to use v_submit_message().
    const void *message,           //!< Pointer to message to be submitted.
    size_t message_size,           //!< Size of the message in bytes
    uint64_t dwarf_fname_hash = 0, //!< Hash of the source file from which v_submit_message() is called.
    uint32_t dwarf_line =
        0xffffffff, //!< Line number of the instrumented instruction for which v_submit_message() is called.
    uint32_t dwarf_column =
        0xffffffff,                  //!< Column of the instrumented instruction for which v_submit_message() is called.
    uint32_t user_type = 0xffffffff, //!< Tag to distinguish between different kinds of messages, used by host
                                     //!< code that processes the messages.
    uint32_t user_data = 0xffffffff) //!< Auxiliary data; use depends on user_type.
{
  bool is_vector_message = true;
  bool submit_lane_headers = true;
  generic_submit_message(rsrc, message, message_size, dwarf_fname_hash, dwarf_line, dwarf_column, user_type, user_data,
                         is_vector_message, submit_lane_headers);
}

//! \brief Submit a scalar message of any type that is valid in device code from the device to the host.
//!
//! Messages are submitted on a per-wave basis, and only the first active lane in the wave submits.
__attribute__((used)) extern "C" __device__ inline void s_submit_message(
    dh_comms_descriptor *rsrc,     //!< Pointer to dh_comms device resources used for message submission.
                                   //!< This pointer is acquired by host code by calling dh_comms::get_dev_rsrc_ptr(),
                                   //!< and passed as a kernel argument to kernels that want to use v_submit_message().
    const void *message = nullptr, //!< Pointer to message to be submitted.
    size_t message_size = 0,       //!< Size of the message in bytes
    bool submit_lane_headers = false, //!< true if lane headers for active lanes (containing thread coordinates) are to
                                      //!< be submitted, false otherwise
    uint64_t dwarf_fname_hash = 0,    //!< Hash of the source file from which v_submit_message() is called.
    uint32_t dwarf_line =
        0xffffffff, //!< Line number of the instrumented instruction for which v_submit_message() is called.
    uint32_t dwarf_column =
        0xffffffff,                  //!< Column of the instrumented instruction for which v_submit_message() is called.
    uint32_t user_type = 0xffffffff, //!< Tag to distinguish between different kinds of messages, used by host
                                     //!< code that processes the messages.
    uint32_t user_data = 0xffffffff) //!< Auxiliary data; use depends on user_type.
{
  bool is_vector_message = false;
  generic_submit_message(rsrc, message, message_size, dwarf_fname_hash, dwarf_line, dwarf_column, user_type, user_data,
                         is_vector_message, submit_lane_headers);
}

//! \brief Submit a a single wave header; no lane headers, no message data.
//!
//! Messages are submitted on a per-wave basis, and only the first active lane in the wave submits.
__attribute__((used)) extern "C" __device__ inline void s_submit_wave_header(
    dh_comms_descriptor *rsrc,     //!< Pointer to dh_comms device resources used for message submission.
    uint64_t dwarf_fname_hash = 0, //!< Hash of the source file from which v_submit_message() is called.
    uint32_t dwarf_line =
        0xffffffff, //!< Line number of the instrumented instruction for which v_submit_message() is called.
    uint32_t dwarf_column =
        0xffffffff,                  //!< Column of the instrumented instruction for which v_submit_message() is called.
    uint32_t user_type = 0xffffffff, //!< Tag to distinguish between different kinds of messages, used by host
                                     //!< code that processes the messages.
    uint32_t user_data = 0xffffffff) //!< Auxiliary data; use depends on user_type.
{
  s_submit_message(rsrc, nullptr, 0, false, dwarf_fname_hash, dwarf_line, dwarf_column, user_type, user_data);
}

__attribute__((used)) extern "C" __device__ inline void v_submit_address(
    dh_comms_descriptor *rsrc, void *address,
    uint64_t dwarf_fname_hash, //!< Hash of the source file from which v_submit_message() is called.
    uint32_t dwarf_line,       //!< Line number of the instrumented instruction for which v_submit_message() is called.
    uint32_t dwarf_column,     //!< Column of the instrumented instruction for which v_submit_message() is called.
    uint8_t rw_kind,           // use 2 bits: 0b01 = read, 0b10 = write, 0b11 = modify (e.g., atomic add)
    uint8_t memory_space,      // use 4 bits (don't need that many on current hardware)
    uint16_t sizeof_pointee) { // use 16 bits (unlikely large for GPU code)
  uint32_t user_type = message_type::address;
  uint32_t user_data = rw_kind & 0b11;
  user_data |= ((memory_space & 0xf) << 2);
  user_data |= (sizeof_pointee << 6);
  v_submit_message(rsrc, &address, sizeof(void *), dwarf_fname_hash, dwarf_line, dwarf_column, user_type, user_data);
}

__attribute__((used)) extern "C" __device__ inline void s_submit_time_interval(
    dh_comms_descriptor *rsrc, void *time_interval,
    uint64_t dwarf_fname_hash = 0, //!< Hash of the source file from which v_submit_message() is called.
    uint32_t dwarf_line =
        0xffffffff, //!< Line number of the instrumented instruction for which v_submit_message() is called.
    uint32_t dwarf_column = 0xffffffff, //!
    uint32_t user_data = 0xffffffff)    //!< Auxiliary data; use depends on user_type.
{
  s_submit_message(rsrc, time_interval, 2 * sizeof(uint64_t), /* submit_lane_headers */ false, dwarf_fname_hash,
                   dwarf_line, dwarf_column, /* user_type */ 1, user_data);
}

__attribute__((used)) extern "C" __device__ inline uint64_t s_clock64() { return __clock64(); }

} // namespace dh_comms
