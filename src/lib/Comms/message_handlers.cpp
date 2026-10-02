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

#include "luthier/Comms/message_handlers.h"

#include <cassert>

namespace luthier {

message_handler_chain_t::message_handler_chain_t(bool pass_through)
    : pass_through_(pass_through) {}

size_t message_handler_chain_t::size() const {
  return message_handlers_.size();
}

bool message_handler_chain_t::handle(const message_t &message) {
  for (auto &mh : message_handlers_) {
    if (mh->handle(message) and not pass_through_) {
      return true;
    }
  }
  return false;
}

void message_handler_chain_t::add_handler(
    std::unique_ptr<message_handler_base> &&message_handler) {
  message_handlers_.push_back(std::move(message_handler));
}

void message_handler_chain_t::report() {
  for (auto &mh : message_handlers_) {
    mh->report();
  }
}

void message_handler_chain_t::clear_handler_states() {
  for (auto &mh : message_handlers_) {
    mh->clear();
  }
}

void message_handler_chain_t::clear() { message_handlers_.clear(); }

} // namespace luthier
