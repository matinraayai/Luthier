//===-- MessageProcessor.h - CPU side: drain the shared buffers -*- C++ -*-===//
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
/// The consumer half of dh_comms. While a kernel runs, a background thread
/// watches the host hand-off flags; when the GPU marks a sub-buffer full it
/// parses every message in it, passes each to the \c MessageHandler s, empties
/// the sub-buffer and clears the flag so the waiting wave can continue.
/// After the kernel finishes, \c stop() drains whatever is left in partially
/// filled sub-buffers. Port of dh_comms' \c processing_loop /
/// \c process_sub_buffers.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXAMPLES_DH_COMMS_MESSAGE_PROCESSOR_H
#define LUTHIER_EXAMPLES_DH_COMMS_MESSAGE_PROCESSOR_H

#include "dh_comms/Protocol.h"

#include <llvm/ADT/ArrayRef.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace luthier::dh_comms {

class SharedBuffers;

/// Receives every message the GPU sends. Called from the processing thread
/// only, one message at a time, so implementations need no locking.
class MessageHandler {
public:
  virtual ~MessageHandler() = default;
  /// \p Records holds one entry per active lane, in lane order: the i-th
  /// record belongs to the i-th set bit of \c Header.Exec.
  virtual void handle(const WaveHeader &Header,
                      llvm::ArrayRef<AccessRecord> Records) = 0;
};

class MessageProcessor {
public:
  /// Every message is passed to each of \p Handlers, in order (dh_comms'
  /// handler chain).
  MessageProcessor(SharedBuffers &Buffers,
                   std::vector<MessageHandler *> Handlers)
      : Buffers(Buffers), Handlers(std::move(Handlers)) {}
  ~MessageProcessor();
  MessageProcessor(const MessageProcessor &) = delete;
  MessageProcessor &operator=(const MessageProcessor &) = delete;

  /// Starts draining sub-buffers as the GPU fills them. Call right before
  /// the kernel is dispatched.
  void start();

  /// Stops the thread and drains partially filled sub-buffers. Call only
  /// once the kernel has completed.
  void stop();

  uint64_t messagesProcessed() const { return Messages; }
  uint64_t bytesProcessed() const { return Bytes; }
  uint64_t handOffs() const { return HandOffs; }
  uint64_t malformedSubBuffers() const { return Malformed; }
  double secondsActive() const { return Seconds; }

private:
  /// Thread body: poll the flags until \c stop().
  void pollLoop();
  /// Parses every message in sub-buffer \p Sb and empties it.
  void drain(uint32_t Sb);

  SharedBuffers &Buffers;
  std::vector<MessageHandler *> Handlers;
  std::atomic<bool> Running{false};
  std::thread Worker;
  std::chrono::steady_clock::time_point StartTime;
  std::vector<AccessRecord> Scratch; // reused per message

  uint64_t Messages = 0;
  uint64_t Bytes = 0;
  uint64_t HandOffs = 0;
  uint64_t Malformed = 0;
  double Seconds = 0;
};

} // namespace luthier::dh_comms

#endif
