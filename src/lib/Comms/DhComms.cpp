//===-- DHComms.cpp -------------------------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//===----------------------------------------------------------------------===//

#include "luthier/Comms/DhComms.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Comms/DataHeaders.h"
#include "luthier/Comms/message.h"
#include "luthier/HSA/Memory.h"
#include "luthier/HSA/MemoryPool.h"
#include <llvm/Support/FormatVariadic.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace luthier {

std::atomic<std::size_t> dh_comms::dh_comms_id_counter_{0};

CommsMemManager::CommsMemManager(
    rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
    rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtApi,
    hsa_agent_t Agent)
    : CoreApi(CoreApi), AmdExtApi(AmdExtApi), Agent(Agent) {};

CommsMemManager::~CommsMemManager() = default;

llvm::Expected<void *> CommsMemManager::calloc(std::size_t size) {
  auto AmdExt = AmdExtApi.getTable();
  auto PoolOrErr = hsa::findHostFineGrainedPool(CoreApi.getTable(), AmdExt);
  LUTHIER_RETURN_ON_ERROR(PoolOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      PoolOrErr->has_value(),
      "No fine-grained memory pool was found for the host."));

  auto BufferOrErr = hsa::memoryPoolAllocate(AmdExt, **PoolOrErr, size);
  LUTHIER_RETURN_ON_ERROR(BufferOrErr.takeError());

  if (llvm::Error Err = hsa::agentsAllowAccess(AmdExt, {Agent}, *BufferOrErr))
    return llvm::joinErrors(std::move(Err),
                            hsa::memoryPoolFree(AmdExt, *BufferOrErr));

  zero((char *)*BufferOrErr, size);
  return *BufferOrErr;
}

llvm::Expected<void *> CommsMemManager::calloc_device_memory(std::size_t size) {
  auto AmdExt = AmdExtApi.getTable();
  auto PoolOrErr = hsa::agentFindCoarseGrainedPool(AmdExt, Agent);
  LUTHIER_RETURN_ON_ERROR(PoolOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      PoolOrErr->has_value(),
      llvm::formatv("Agent {0:x} exposes no coarse-grained memory pool",
                    Agent.handle)));

  auto ResultOrErr = hsa::memoryPoolAllocate(AmdExt, **PoolOrErr, size);
  LUTHIER_RETURN_ON_ERROR(ResultOrErr.takeError());

  if (llvm::Error Err = zero_device_memory(*ResultOrErr, size))
    return llvm::joinErrors(std::move(Err),
                            hsa::memoryPoolFree(AmdExt, *ResultOrErr));
  return *ResultOrErr;
}

llvm::Expected<void *> CommsMemManager::copy_to_device(void *dst,
                                                       const void *src,
                                                       std::size_t size) {
  LUTHIER_RETURN_ON_ERROR(hsa::memoryCopy(CoreApi.getTable(), dst, src, size));
  return dst;
}

llvm::Error CommsMemManager::free(void *ptr) {
  return hsa::memoryPoolFree(AmdExtApi.getTable(), ptr);
}

llvm::Error CommsMemManager::free_device_memory(void *ptr) {
  return this->free(ptr);
}

void *CommsMemManager::copy(void *dst, void *src, std::size_t size) {
  std::memcpy(dst, src, size);
  return dst;
}

void CommsMemManager::zero(void *buffer, std::size_t size) {
  std::vector<char> zeros(size);
  std::ranges::copy(std::as_const(zeros), (char *)buffer);
}

llvm::Error CommsMemManager::zero_device_memory(void *buffer,
                                                std::size_t size) {
  std::vector<char> zeros(size);
  return copy_to_device(buffer, zeros.data(), size).takeError();
}
} // namespace dh_comms

namespace {
constexpr bool shared_buffers_are_host_pinned = true;

template <typename T>
llvm::Expected<T *> clone_to_device(const T &host_data, luthier::CommsMemManager &mgr) {
  void *device_data = nullptr;
  LUTHIER_RETURN_ON_ERROR(
      mgr.calloc_device_memory(sizeof(T)).moveInto(device_data));
  LUTHIER_RETURN_ON_ERROR(
      mgr.copy_to_device(device_data, &host_data, sizeof(T)).takeError());
  return static_cast<T *>(device_data);
}

} // unnamed namespace

namespace luthier {
dh_comms_resources::dh_comms_resources(std::size_t no_sub_buffers,
                                       std::size_t sub_buffer_capacity,
                                       CommsMemManager &mgr)
    : desc_({no_sub_buffers, sub_buffer_capacity,
             (decltype(desc_.buffer_))mgr.calloc(no_sub_buffers *
                                                 sub_buffer_capacity),
             (decltype(desc_.sub_buffer_sizes_))mgr.calloc(
                 no_sub_buffers * sizeof(decltype(*desc_.sub_buffer_sizes_))),
             (decltype(desc_.error_bits_))mgr.calloc(
                 sizeof(decltype(*desc_.error_bits_))),
             (decltype(desc_.atomic_flags_d_))mgr.calloc_device_memory(
                 no_sub_buffers * sizeof(decltype(*desc_.atomic_flags_d_))),
             (decltype(desc_.atomic_flags_hd_))mgr.calloc(
                 no_sub_buffers * sizeof(decltype(*desc_.atomic_flags_hd_)))}),
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
    if (header.block_idx_x < filter_x_.min ||
        header.block_idx_x >= filter_x_.max) {
      return false;
    }
  }
  if (filter_y_.enabled) {
    if (header.block_idx_y < filter_y_.min ||
        header.block_idx_y >= filter_y_.max) {
      return false;
    }
  }
  if (filter_z_.enabled) {
    if (header.block_idx_z < filter_z_.min ||
        header.block_idx_z >= filter_z_.max) {
      return false;
    }
  }

  return true;
}

BlockIdxFilterT dh_comms::parse_filter_env(const char *env_value) {
  BlockIdxFilterT filter;
  if (env_value == nullptr || env_value[0] == '\0') {
    return filter; // Not set, filtering disabled
  }

  std::string value(env_value);
  try {
    size_t colon_pos = value.find(':');
    if (colon_pos != std::string::npos) {
      // Range format: "N:M"
      int min_val = std::stoi(value.substr(0, colon_pos));
      int max_val = std::stoi(value.substr(colon_pos + 1));

      if (min_val < 0 || max_val < 0) {
        std::cerr << "Warning: Invalid filter range '" << value
                  << "' (negative values). Filter disabled." << std::endl;
        return filter;
      }
      if (min_val > max_val) {
        std::cerr << "Warning: Invalid filter range '" << value
                  << "' (min > max). Filter disabled." << std::endl;
        return filter;
      }

      filter.enabled = true;
      filter.min = static_cast<uint16_t>(min_val);
      filter.max = static_cast<uint16_t>(max_val);
    } else {
      // Single value format: "N" -> range [N, N+1)
      int single_val = std::stoi(value);
      if (single_val < 0) {
        std::cerr << "Warning: Invalid filter value '" << value
                  << "' (negative). Filter disabled." << std::endl;
        return filter;
      }

      filter.enabled = true;
      filter.min = static_cast<uint16_t>(single_val);
      filter.max = static_cast<uint16_t>(single_val + 1);
    }
  } catch (const std::exception &e) {
    std::cerr << "Warning: Failed to parse filter value '" << value
              << "': " << e.what() << ". Filter disabled." << std::endl;
  }

  return filter;
}

dh_comms::dh_comms(std::size_t no_sub_buffers, std::size_t sub_buffer_capacity,
                   bool verbose, bool install_default_handlers,
                   CommsMemManager *mgr, bool handlers_pass_through)
    : mgr_(mgr ? mgr : &default_mgr_),
      rsrc_(no_sub_buffers, sub_buffer_capacity, *mgr_),
      dev_rsrc_p_(clone_to_device(rsrc_.desc_, *mgr_)), running_(false),
      verbose_(verbose), message_handler_chain_(handlers_pass_through),
      sub_buffer_processor_(), start_time_(), stop_time_(),
      dh_comms_id_(
          dh_comms_id_counter_.fetch_add(1, std::memory_order_relaxed)),
      filter_x_(parse_filter_env(std::getenv("DH_COMMS_GROUP_FILTER_X"))),
      filter_y_(parse_filter_env(std::getenv("DH_COMMS_GROUP_FILTER_Y"))),
      filter_z_(parse_filter_env(std::getenv("DH_COMMS_GROUP_FILTER_Z"))),
      any_filter_enabled_(filter_x_.enabled || filter_y_.enabled ||
                          filter_z_.enabled) {
  if (install_default_handlers) {
    install_default_message_handlers();
  }
  if (verbose_) {
    if constexpr (shared_buffers_are_host_pinned) {
      printf("%s:%d:\n\t Buffers accessed from both host and device are "
             "allocated in pinned host memory\n",
             __FILE__, __LINE__);
    } else {
      printf("%s:%d:\n\t Buffers accessed from both host and device are "
             "allocated in device memory\n",
             __FILE__, __LINE__);
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
    // if processing threads are still running, stop/join them, to avoid the
    // program to hang.
    stop();
  }
  if (*rsrc_.desc_.error_bits_ & 1) {
    std::cerr << "Error detected: data from device dropped because message "
                 "size was larger than sub-buffer size"
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

  const std::chrono::duration<double> processing_time =
      stop_time_ - start_time_;
  double MiBps = bytes_processed_ / processing_time.count() / 1.0e6;
  printf("%zu bytes processed in %lf seconds (%.1lf MiB/s)\n", bytes_processed_,
         processing_time.count(), MiBps);

  if (auto_clear_states) {
    clear_handler_states();
  }
}

void dh_comms::append_handler(
    std::unique_ptr<message_handler_base> &&message_handler) {
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

    // in the final processing loop, sub-buffers are either empty or partially
    // filled, but not full, and we expect the flag to be zero.
    uint8_t flag =
        __atomic_load_n(&rsrc_.desc_.atomic_flags_hd_[i], __ATOMIC_ACQUIRE);
    if (is_final_loop or flag == 1) // process and reset
    {
      if (is_final_loop and flag != 0) // Should not happen, indicates a missing
                                       // atomic release from device code
      {
        printf(
            "Found non-zero flag for sub-buffer %lu in final processing loop\n",
            i);
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
      if (!is_final_loop) { // give control over the sub-buffer back to the wave
                            // that gave it to us
        flag = 0;
        __atomic_store_n(&rsrc_.desc_.atomic_flags_hd_[i], flag,
                         __ATOMIC_RELEASE);
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
