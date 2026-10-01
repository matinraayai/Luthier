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

#include "luthier/Comms/data_headers.h"

#include <cstring>

namespace dh_comms {
wave_header_t::wave_header_t(const char *wave_header_p) {
  memcpy((char *)this, wave_header_p, sizeof(wave_header_t));
}

lane_header_t::lane_header_t(const char *lane_header_p) {
  memcpy((char *)this, lane_header_p, sizeof(lane_header_t));
}

//! Returns a vector with the lane ids of the active lanes based on the
//! execution mask in the wave header.
std::vector<size_t>
get_lane_ids_of_active_lanes(const wave_header_t &wave_header) {
  std::vector<size_t> lane_ids;
  std::size_t lane_mask = 1;
  for (std::size_t lane_id = 0; lane_id != 64; ++lane_id) {
    if (wave_header.exec & lane_mask) {
      lane_ids.push_back(lane_id);
    }
    lane_mask <<= 1;
  }
  return lane_ids;
}

} // namespace dh_comms