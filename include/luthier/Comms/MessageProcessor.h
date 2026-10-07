//===-- MessageProcessor.h - Host side: drain the shared buffers -*- C++ -*-===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// While a kernel runs, a background thread watches the host hand-off
/// flags. When the device marks a sub-buffer full, the thread parses every
/// message in it, passes each to the \c MessageHandler s, empties it and
/// clears the flag so the waiting wave continues. After the kernel ends,
/// \c stop() drains what is left in partially filled sub-buffers.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_MESSAGE_PROCESSOR_H
#define LUTHIER_COMMS_MESSAGE_PROCESSOR_H

#include "luthier/Comms/SharedBuffers.h"

#include <llvm/ADT/ArrayRef.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace luthier::comms {

/// Receives every message. Called from the processing thread only, one
/// message at a time, so implementations need no locking.
class MessageHandler {
public:
  virtual ~MessageHandler() = default;

  /// \p Data is the lane data as sent: dword \c D of active lane \c L is
  /// <tt>Data[D * Header.ActiveLanes + L]</tt>.
  virtual void handle(const WaveHeader &Header,
                      llvm::ArrayRef<uint32_t> Data) = 0;
};

class MessageProcessor {
public:
  MessageProcessor(SharedBuffers &Buffers, std::vector<MessageHandler *> H)
      : Buffers(Buffers), Handlers(std::move(H)) {}
  ~MessageProcessor() {
    if (Running.load())
      stop();
  }
  MessageProcessor(const MessageProcessor &) = delete;
  MessageProcessor &operator=(const MessageProcessor &) = delete;

  /// Starts draining. Call right before the kernel is dispatched.
  void start() {
    Running.store(true, std::memory_order_release);
    StartTime = std::chrono::steady_clock::now();
    Worker = std::thread(&MessageProcessor::pollLoop, this);
  }

  /// Stops the thread and drains what is left. Call once the kernel is done.
  void stop() {
    Running.store(false, std::memory_order_release);
    if (Worker.joinable())
      Worker.join();
    for (uint32_t Sb = 0; Sb < Buffers.numSubBuffers(); ++Sb)
      drain(Sb);
    Seconds += std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - StartTime)
                   .count();
  }

  uint64_t messages() const { return Messages; }
  uint64_t bytes() const { return Bytes; }
  uint64_t handOffs() const { return HandOffs; }
  uint64_t malformed() const { return Malformed; }
  double seconds() const { return Seconds; }

private:
  void pollLoop() {
    uint32_t *Flags = Buffers.hostFlags();
    while (Running.load(std::memory_order_acquire)) {
      bool Found = false;
      for (uint32_t Sb = 0; Sb < Buffers.numSubBuffers(); ++Sb) {
        if (__atomic_load_n(&Flags[Sb], __ATOMIC_ACQUIRE) != 1)
          continue;
        drain(Sb);
        ++HandOffs;
        Found = true;
        __atomic_store_n(&Flags[Sb], 0u, __ATOMIC_RELEASE); // resume the wave
      }
      if (!Found)
        std::this_thread::yield();
    }
  }

  /// Parses every message in sub-buffer \p Sb and empties it.
  void drain(uint32_t Sb) {
    uint64_t *Sizes = Buffers.subBufferSizes();
    const uint64_t Used = __atomic_load_n(&Sizes[Sb], __ATOMIC_ACQUIRE);
    if (Used > Buffers.subBufferCapacity()) {
      ++Malformed; // corrupt size: nothing here is safe to read
    } else {
      const char *Cursor = Buffers.subBuffer(Sb);
      const char *const End = Cursor + Used;
      while (Cursor + sizeof(WaveHeader) <= End) {
        WaveHeader H;
        std::memcpy(&H, Cursor, sizeof(H));
        const char *Data = Cursor + sizeof(WaveHeader);
        if (H.DataSize != uint32_t(H.ActiveLanes) * H.DwordsPerLane * 4 ||
            Data + H.DataSize > End) {
          ++Malformed;
          break;
        }
        Scratch.resize(H.DataSize / 4);
        std::memcpy(Scratch.data(), Data, H.DataSize);
        for (MessageHandler *Handler : Handlers)
          Handler->handle(H, Scratch);
        ++Messages;
        Cursor = Data + H.DataSize;
      }
      Bytes += Used;
    }
    __atomic_store_n(&Sizes[Sb], uint64_t(0), __ATOMIC_RELEASE);
  }

  SharedBuffers &Buffers;
  std::vector<MessageHandler *> Handlers;
  std::atomic<bool> Running{false};
  std::thread Worker;
  std::chrono::steady_clock::time_point StartTime;
  std::vector<uint32_t> Scratch;
  uint64_t Messages = 0, Bytes = 0, HandOffs = 0, Malformed = 0;
  double Seconds = 0;
};

} // namespace luthier::comms

#endif
