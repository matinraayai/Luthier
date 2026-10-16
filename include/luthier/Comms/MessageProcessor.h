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
/// The host side of the channel: a background thread that hands the device's
/// messages to the tool.
///
/// While a kernel runs, the thread watches the host hand-off flags. When a
/// wave finds its sub-buffer full, it sets the sub-buffer's flag to 1 and
/// spins (Device.h, wave_signal_host). The thread then parses every message
/// in that sub-buffer, passes each to the \c MessageHandler s, empties it and
/// clears the flag, which gives the sub-buffer back to the waiting wave.
/// After the kernel ends, \c stop() drains what is left in partially filled
/// sub-buffers.
///
/// The processing loop is taken from dh_comms
/// (https://github.com/AMDResearch/dh_comms, src/dh_comms.cpp,
/// dh_comms::process_sub_buffers and dh_comms::processing_loop); the
/// comments quoted below are dh_comms' own.
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
template <typename WaveT, unsigned N> class MessageHandler {
public:
  virtual ~MessageHandler() = default;

  /// \p Data holds the \p N dwords of each active lane, dword-major: with
  /// A active lanes, dword \c D of lane \c L is <tt>Data[D * A + L]</tt>,
  /// and lane \c L is the L-th set bit of \p Exec.
  virtual void handle(uint64_t Exec, const WaveT &Wave,
                      llvm::ArrayRef<uint32_t> Data) = 0;
};

template <typename WaveT, unsigned N> class MessageProcessor {
public:
  using Handler = MessageHandler<WaveT, N>;

  MessageProcessor(SharedBuffers &Buffers, std::vector<Handler *> H)
      : Buffers(Buffers), Handlers(std::move(H)) {}
  ~MessageProcessor() {
    if (Running.load())
      stop();
  }
  MessageProcessor(const MessageProcessor &) = delete;
  MessageProcessor &operator=(const MessageProcessor &) = delete;

  /// Starts draining. Call right before the kernel is dispatched.
  /// (dh_comms::start)
  void start() {
    Running.store(true, std::memory_order_release);
    StartTime = std::chrono::steady_clock::now();
    Worker = std::thread(&MessageProcessor::pollLoop, this);
  }

  /// Stops the thread and drains what is left. Call once the kernel is done.
  /// (dh_comms::stop and the final pass of dh_comms::process_sub_buffers)
  void stop() {
    Running.store(false, std::memory_order_release);
    if (Worker.joinable())
      Worker.join();
    // dh_comms: "after stopping dh_comms, process partially full buffers in
    // a final pass". In the final pass, sub-buffers are either empty or
    // partially filled, but not full, and no wave is waiting on a flag.
    for (uint32_t Sb = 0; Sb < Buffers.numSubBuffers(); ++Sb)
      drain(Sb);
    Seconds += std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - StartTime)
                   .count();
  }

  /// Statistics since the processor was created.
  uint64_t messages() const { return Messages; }
  uint64_t bytes() const { return Bytes; }
  uint64_t handOffs() const { return HandOffs; }
  uint64_t malformed() const { return Malformed; }
  double seconds() const { return Seconds; }

private:
  /// dh_comms: "process buffers when device code indicates they are full".
  /// (dh_comms::process_sub_buffers + dh_comms::processing_loop)
  void pollLoop() {
    uint32_t *Flags = Buffers.hostFlags();
    while (Running.load(std::memory_order_acquire)) {
      bool Found = false;
      for (uint32_t Sb = 0; Sb < Buffers.numSubBuffers(); ++Sb) {
        // dh_comms: "when the sub-buffer for a wave on the device is full, it
        // will set the flag" (here always to 1: the wave wants control back
        // after the host is done processing the sub-buffer).
        if (__atomic_load_n(&Flags[Sb], __ATOMIC_ACQUIRE) != 1)
          continue;
        drain(Sb);
        ++HandOffs;
        Found = true;
        // dh_comms: "give control over the sub-buffer back to the wave that
        // gave it to us".
        __atomic_store_n(&Flags[Sb], 0u, __ATOMIC_RELEASE);
      }
      if (!Found)
        std::this_thread::yield(); // nothing full: let the app's threads run
    }
  }

  /// Parses every message in sub-buffer \p Sb, hands each to the handlers,
  /// and empties the sub-buffer (dh_comms: "process data").
  void drain(uint32_t Sb) {
    // The device publishes the fill size after the message bytes (Device.h,
    // submit), so once it is read every message up to it is complete.
    uint64_t *Sizes = Buffers.subBufferSizes();
    const uint64_t Used = __atomic_load_n(&Sizes[Sb], __ATOMIC_ACQUIRE);
    if (Used > Buffers.subBufferCapacity()) {
      ++Malformed; // corrupt size: nothing here is safe to read
    } else {
      using Layout = MessageLayout<WaveT, N>;
      const char *Cursor = Buffers.subBuffer(Sb);
      const char *const End = Cursor + Used;
      // Messages are packed one after another (dh_comms: "size -=
      // message.size(); message_p += message.size();").
      while (Cursor + Layout::DataOffset <= End) {
        // The message's length follows from Exec (Protocol.h).
        uint64_t Exec;
        std::memcpy(&Exec, Cursor, sizeof(Exec));
        const uint32_t ActiveLanes = uint32_t(__builtin_popcountll(Exec));
        const uint32_t Size = Layout::bytes(ActiveLanes);
        if (ActiveLanes == 0 || Cursor + Size > End) {
          ++Malformed;
          break;
        }
        // Copy out of the shared buffer: the handlers get plain host data,
        // and the sub-buffer can be reused as soon as it is emptied.
        WaveT Wave;
        std::memcpy(&Wave, Cursor + Layout::WaveOffset, sizeof(Wave));
        Data.resize(ActiveLanes * N);
        std::memcpy(Data.data(), Cursor + Layout::DataOffset,
                    Data.size() * sizeof(uint32_t));
        for (Handler *H : Handlers)
          H->handle(Exec, Wave, Data);
        ++Messages;
        Cursor += Size;
      }
      Bytes += Used;
    }
    // Empty the sub-buffer (dh_comms: "sub_buffer_sizes_[i] = 0").
    __atomic_store_n(&Sizes[Sb], uint64_t(0), __ATOMIC_RELEASE);
  }

  SharedBuffers &Buffers;          ///< The memory the device writes into.
  std::vector<Handler *> Handlers; ///< Every message goes to each of them.
  std::atomic<bool> Running{false};
  std::thread Worker; ///< Runs pollLoop while a kernel is in flight.
  std::chrono::steady_clock::time_point StartTime;
  std::vector<uint32_t> Data; ///< Lane data of the current message.
  uint64_t Messages = 0, Bytes = 0, HandOffs = 0, Malformed = 0;
  double Seconds = 0;
};

} // namespace luthier::comms

#endif
