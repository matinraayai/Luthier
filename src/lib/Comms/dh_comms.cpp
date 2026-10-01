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

#include "luthier/Comms/dh_comms.h"

#include "luthier/Comms/data_headers.h"
#include "luthier/Comms/hip_runtime_loader.h"
#include "luthier/Comms/message.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// HIP enum constants (avoid pulling in hip_runtime.h for host-only code)
static constexpr unsigned int hipHostMallocCoherent = 0x4;
static constexpr int hipMemcpyHostToDevice = 1;

namespace {
using namespace dh_comms::hip_runtime_loader;

inline int checkHipError(hipError_t err, const char *cmd) {
  if (err) {
    init(); // ensure pfn_hipGetErrorString is resolved
    printf("HIP error in command '%s'\n", cmd);
    printf("Error message: %s\n", pfn_hipGetErrorString(err));
  }
  return err;
}
} // unnamed namespace

#define CHK_HIP_ERR(cmd) checkHipError(cmd, #cmd)

namespace dh_comms {

std::atomic<std::size_t> dh_comms::dh_comms_id_counter_{0};

dh_comms_mem_mgr::dh_comms_mem_mgr() { return; }

dh_comms_mem_mgr::~dh_comms_mem_mgr() {}

void *dh_comms_mem_mgr::calloc(std::size_t size) {
  hip_runtime_loader::init();
  void *buffer;
  CHK_HIP_ERR(hip_runtime_loader::pfn_hipHostMalloc(&buffer, size, hipHostMallocCoherent));
  zero((char *)buffer, size);
  return buffer;
}

void *dh_comms_mem_mgr::calloc_device_memory(std::size_t size) {
  hip_runtime_loader::init();
  void *result = NULL;
  CHK_HIP_ERR(hip_runtime_loader::pfn_hipMalloc(&result, size));
  zero_device_memory(result, size);
  return result;
}

void *dh_comms_mem_mgr::copy_to_device(void *dst, const void *src, std::size_t size) {
  hip_runtime_loader::init();
  CHK_HIP_ERR(hip_runtime_loader::pfn_hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
  return dst;
}

void dh_comms_mem_mgr::free(void *ptr) {
  hip_runtime_loader::init();
  CHK_HIP_ERR(hip_runtime_loader::pfn_hipFree(ptr));
  return;
}

void dh_comms_mem_mgr::free_device_memory(void *ptr) { this->free(ptr); }

void *dh_comms_mem_mgr::copy(void *dst, void *src, std::size_t size) {
  memcpy(dst, src, size);
  return dst;
}

void dh_comms_mem_mgr::zero(void *buffer, std::size_t size) {
  std::vector<char> zeros(size);
  std::copy(zeros.cbegin(), zeros.cend(), (char *)buffer);
}

void dh_comms_mem_mgr::zero_device_memory(void *buffer, std::size_t size) {
  std::vector<char> zeros(size);
  copy_to_device(buffer, zeros.data(), size);
}
} // namespace dh_comms

namespace {
constexpr bool shared_buffers_are_host_pinned = true;

template <typename T> T *clone_to_device(const T &host_data, dh_comms::dh_comms_mem_mgr &mgr) {
  T *device_data;
  device_data = reinterpret_cast<T *>(mgr.calloc_device_memory(sizeof(T)));
  mgr.copy_to_device(device_data, &host_data, sizeof(T));
  return device_data;
}

} // unnamed namespace

namespace dh_comms {
dh_comms_resources::dh_comms_resources(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity,
                                       dh_comms_mem_mgr &mgr)
    : desc_({no_sub_buffers, sub_buffer_capacity,
             (decltype(desc_.buffer_))mgr.calloc(no_sub_buffers * sub_buffer_capacity),
             (decltype(desc_.sub_buffer_sizes_))mgr.calloc(no_sub_buffers * sizeof(decltype(*desc_.sub_buffer_sizes_))),
             (decltype(desc_.error_bits_))mgr.calloc(sizeof(decltype(*desc_.error_bits_))),
             (decltype(desc_.atomic_flags_d_))mgr.calloc_device_memory(no_sub_buffers *
                                                                       sizeof(decltype(*desc_.atomic_flags_d_))),
             (decltype(desc_.atomic_flags_hd_))mgr.calloc(no_sub_buffers * sizeof(decltype(*desc_.atomic_flags_hd_)))}),
      mgr_(mgr) {}

dh_comms_resources::~dh_comms_resources() {
  mgr_.free(desc_.atomic_flags_hd_);
  mgr_.free(desc_.atomic_flags_d_);
  mgr_.free(desc_.error_bits_);
  mgr_.free(desc_.sub_buffer_sizes_);
  mgr_.free_device_memory(desc_.buffer_);
}

bool dh_comms::message_passes_filter(const wave_header_t &header) const {
  // Fast path: no filters enabled
  if (!any_filter_enabled_) {
    return true;
  }

  // Check each enabled filter
  if (filter_x_.enabled) {
    if (header.block_idx_x < filter_x_.min || header.block_idx_x >= filter_x_.max) {
      return false;
    }
  }
  if (filter_y_.enabled) {
    if (header.block_idx_y < filter_y_.min || header.block_idx_y >= filter_y_.max) {
      return false;
    }
  }
  if (filter_z_.enabled) {
    if (header.block_idx_z < filter_z_.min || header.block_idx_z >= filter_z_.max) {
      return false;
    }
  }

  return true;
}

block_idx_filter_t dh_comms::parse_filter_env(const char *env_value) {
  block_idx_filter_t filter;
  if (env_value == nullptr || env_value[0] == '\0') {
    return filter;  // Not set, filtering disabled
  }

  std::string value(env_value);
  try {
    size_t colon_pos = value.find(':');
    if (colon_pos != std::string::npos) {
      // Range format: "N:M"
      int min_val = std::stoi(value.substr(0, colon_pos));
      int max_val = std::stoi(value.substr(colon_pos + 1));

      if (min_val < 0 || max_val < 0) {
        std::cerr << "Warning: Invalid filter range '" << value << "' (negative values). Filter disabled." << std::endl;
        return filter;
      }
      if (min_val > max_val) {
        std::cerr << "Warning: Invalid filter range '" << value << "' (min > max). Filter disabled." << std::endl;
        return filter;
      }

      filter.enabled = true;
      filter.min = static_cast<uint16_t>(min_val);
      filter.max = static_cast<uint16_t>(max_val);
    } else {
      // Single value format: "N" -> range [N, N+1)
      int single_val = std::stoi(value);
      if (single_val < 0) {
        std::cerr << "Warning: Invalid filter value '" << value << "' (negative). Filter disabled." << std::endl;
        return filter;
      }

      filter.enabled = true;
      filter.min = static_cast<uint16_t>(single_val);
      filter.max = static_cast<uint16_t>(single_val + 1);
    }
  } catch (const std::exception &e) {
    std::cerr << "Warning: Failed to parse filter value '" << value << "': " << e.what() << ". Filter disabled."
              << std::endl;
  }

  return filter;
}

dh_comms::dh_comms(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity, bool verbose,
                   bool install_default_handlers, dh_comms_mem_mgr *mgr, bool handlers_pass_through)
    : mgr_(mgr ? mgr : &default_mgr_),
      rsrc_(no_sub_buffers, sub_buffer_capacity, *mgr_),
      dev_rsrc_p_(clone_to_device(rsrc_.desc_, *mgr_)),
      running_(false),
      verbose_(verbose),
      message_handler_chain_(handlers_pass_through),
      sub_buffer_processor_(),
      start_time_(),
      stop_time_(),
      dh_comms_id_(dh_comms_id_counter_.fetch_add(1, std::memory_order_relaxed)),
      filter_x_(parse_filter_env(std::getenv("DH_COMMS_GROUP_FILTER_X"))),
      filter_y_(parse_filter_env(std::getenv("DH_COMMS_GROUP_FILTER_Y"))),
      filter_z_(parse_filter_env(std::getenv("DH_COMMS_GROUP_FILTER_Z"))),
      any_filter_enabled_(filter_x_.enabled || filter_y_.enabled || filter_z_.enabled) {
  if (install_default_handlers) {
    install_default_message_handlers();
  }
  if (verbose_) {
    if constexpr (shared_buffers_are_host_pinned) {
      printf("%s:%d:\n\t Buffers accessed from both host and device are allocated in pinned host memory\n", __FILE__,
             __LINE__);
    } else {
      printf("%s:%d:\n\t Buffers accessed from both host and device are allocated in device memory\n", __FILE__,
             __LINE__);
    }
  }
  if (any_filter_enabled_ && verbose_) {
    std::cerr << "dh_comms: Block index filters active:";
    if (filter_x_.enabled)
      std::cerr << " X=[" << filter_x_.min << "," << filter_x_.max << ")";
    if (filter_y_.enabled)
      std::cerr << " Y=[" << filter_y_.min << "," << filter_y_.max << ")";
    if (filter_z_.enabled)
      std::cerr << " Z=[" << filter_z_.min << "," << filter_z_.max << ")";
    std::cerr << std::endl;
  }
}

dh_comms::~dh_comms() {
  if (running_) {
    // if processing threads are still running, stop/join them, to avoid the program
    // to hang.
    stop();
  }
  if (*rsrc_.desc_.error_bits_ & 1) {
    std::cerr << "Error detected: data from device dropped because message size was larger than sub-buffer size"
              << std::endl;
  }
  mgr_->free_device_memory(dev_rsrc_p_);
}

dh_comms_descriptor *dh_comms::get_dev_rsrc_ptr() { return dev_rsrc_p_; }

void dh_comms::start() {
  assert(not running_);
  running_ = true;
  start_time_ = std::chrono::steady_clock::now();
  bytes_processed_ = 0;
  sub_buffer_processor_ = std::thread(&dh_comms::process_sub_buffers, this);
}

void dh_comms::start(const std::string &kernel_name) {
  (void)kernel_name;
  start();
}

void dh_comms::stop() {
  assert(running_);
  running_ = false;
  stop_time_ = std::chrono::steady_clock::now();
  sub_buffer_processor_.join();
}

void dh_comms::clear_handler_states() {
  assert(not running_);
  message_handler_chain_.clear_handler_states();
}

void dh_comms::delete_handlers() {
  assert(not running_);
  message_handler_chain_.clear();
}

void dh_comms::report(bool auto_clear_states) {
  message_handler_chain_.report();

  const std::chrono::duration<double> processing_time = stop_time_ - start_time_;
  double MiBps = bytes_processed_ / processing_time.count() / 1.0e6;
  printf("%zu bytes processed in %lf seconds (%.1lf MiB/s)\n", bytes_processed_, processing_time.count(), MiBps);

  if (auto_clear_states) {
    clear_handler_states();
  }
}

void dh_comms::append_handler(std::unique_ptr<message_handler_base> &&message_handler) {
  assert(not running_);
  assert(message_handler);
  message_handler_chain_.add_handler(std::move(message_handler));
}

void dh_comms::install_default_message_handlers() {
  assert(not running_);
  // append_handler(std::make_unique<memory_heatmap_t>());
}

void dh_comms::processing_loop(bool is_final_loop) {
  for (size_t i = 0; i != rsrc_.desc_.no_sub_buffers_; ++i) {
    // when the sub-buffer for a wave on the device is full, it will
    // set the flag to either 3 (if it wants control back after the host
    // is done processing the sub-buffer) or 2 (if it doesn't want control back,
    // and instead allows any wave to take control of the sub-buffer)

    // in the final processing loop, sub-buffers are either empty or partially filled,
    // but not full, and we expect the flag to be zero.
    uint8_t flag = __atomic_load_n(&rsrc_.desc_.atomic_flags_hd_[i], __ATOMIC_ACQUIRE);
    if (is_final_loop or flag == 1) // process and reset
    {
      if (is_final_loop and flag != 0) // Should not happen, indicates a missing atomic release from device code
      {
        printf("Found non-zero flag for sub-buffer %lu in final processing loop\n", i);
      }
      // process data
      size_t size = rsrc_.desc_.sub_buffer_sizes_[i];
      bytes_processed_ += size;
      size_t byte_offset = i * rsrc_.desc_.sub_buffer_capacity_;
      char *message_p = &rsrc_.desc_.buffer_[byte_offset];
      while (size != 0 and message_handler_chain_.size() != 0) {
        message_t message(message_p);
        // Apply block index filter before invoking handlers
        if (message_passes_filter(message.wave_header())) {
          message_handler_chain_.handle(message);
        }
        assert(message.size() <= size);
        size -= message.size();
        message_p += message.size();
      }

      rsrc_.desc_.sub_buffer_sizes_[i] = 0;
      if (!is_final_loop) { // give control over the sub-buffer back to the wave that gave it to us
        flag = 0;
        __atomic_store_n(&rsrc_.desc_.atomic_flags_hd_[i], flag, __ATOMIC_RELEASE);
      }
    }
  }
}

void dh_comms::process_sub_buffers() {
  // process buffers when device code indicates they are full
  bool is_final_loop = false;
  while (__atomic_load_n(&running_, __ATOMIC_ACQUIRE)) {
    processing_loop(is_final_loop);
  }

  // after stopping dh_comms, process partially full buffers in a final pass
  is_final_loop = true;
  processing_loop(is_final_loop);
}

} // namespace dh_comms
