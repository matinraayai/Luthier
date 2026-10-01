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

#include <vector>

namespace dh_comms {

namespace message_type {
enum : uint32_t { address = 0, time_interval = 1, basic_block_start = 2, undefined = 0xffffffff };
}

namespace memory_access {
enum : uint8_t { undefined = 0, read = 1, write = 2, read_write = 3 };
}

namespace address_space {
enum : uint8_t { flat = 0, global = 1, gds = 2, shared = 3, constant = 4, scratch = 5, undefined = 0xf };
}

namespace gcnarch {
enum : uint8_t { unsupported = 0, gfx906 = 1, gfx908 = 2, gfx90a = 3, gfx940 = 4, gfx941 = 5, gfx942 = 6 };
}

class message_t {
public:
  message_t(const char *message_p);
  size_t size() const;

  const wave_header_t &wave_header() const { return wave_header_; }
  size_t no_lane_headers() const { return lane_headers_.size(); }
  size_t no_data_items() const { return data_.size(); }
  size_t data_item_size() const { return data_.size() == 0 ? 0 : data_[0].size(); }
  const lane_header_t &lane_header(size_t i) const { return lane_headers_[i]; }
  const void *data_item(size_t i) const { return data_[i].data(); }

private:
  wave_header_t wave_header_;
  std::vector<lane_header_t> lane_headers_;
  std::vector<std::vector<char>> data_;
};
} // namespace dh_comms