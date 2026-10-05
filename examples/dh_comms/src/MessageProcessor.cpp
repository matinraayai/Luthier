//===-- MessageProcessor.cpp - CPU side: drain the shared buffers ---------===//
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
#include "dh_comms/MessageProcessor.h"

#include "dh_comms/SharedBuffers.h"

#include <cstring>

namespace luthier::dh_comms {

MessageProcessor::~MessageProcessor() {
  if (Running.load())
    stop();
}

void MessageProcessor::start() {
  Running.store(true, std::memory_order_release);
  StartTime = std::chrono::steady_clock::now();
  Worker = std::thread(&MessageProcessor::pollLoop, this);
}

void MessageProcessor::stop() {
  Running.store(false, std::memory_order_release);
  if (Worker.joinable())
    Worker.join();
  // The kernel has finished, so every flag is clear and the sub-buffers
  // hold only messages that never filled their sub-buffer.
  for (uint32_t Sb = 0; Sb < Buffers.numSubBuffers(); ++Sb)
    drain(Sb);
  Seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           StartTime)
                 .count();
}

void MessageProcessor::pollLoop() {
  uint32_t *Flags = Buffers.hostFlags();
  while (Running.load(std::memory_order_acquire)) {
    bool Found = false;
    for (uint32_t Sb = 0; Sb < Buffers.numSubBuffers(); ++Sb) {
      if (__atomic_load_n(&Flags[Sb], __ATOMIC_ACQUIRE) != 1)
        continue;
      drain(Sb);
      ++HandOffs;
      Found = true;
      // Give the sub-buffer back to the wave spinning on it.
      __atomic_store_n(&Flags[Sb], 0u, __ATOMIC_RELEASE);
    }
    if (!Found)
      std::this_thread::yield();
  }
}

void MessageProcessor::drain(uint32_t Sb) {
  uint64_t *Sizes = Buffers.subBufferSizes();
  const uint64_t Used = __atomic_load_n(&Sizes[Sb], __ATOMIC_ACQUIRE);
  const char *Cursor = Buffers.subBuffer(Sb);
  const char *const End = Cursor + Used;

  if (Used > Buffers.subBufferCapacity()) {
    ++Malformed; // a corrupt size: nothing in this sub-buffer is safe to read
  } else {
    while (Cursor + sizeof(WaveHeader) <= End) {
      WaveHeader H;
      std::memcpy(&H, Cursor, sizeof(H));
      const char *Data = Cursor + sizeof(WaveHeader);
      const uint64_t Lanes = H.ActiveLaneCount;
      // Reject anything that does not describe exactly one record per lane.
      if (H.UserType != message_type::MemoryIndex ||
          H.DataSize != Lanes * sizeof(AccessRecord) ||
          Data + H.DataSize > End) {
        ++Malformed;
        break;
      }
      // Undo the dword-major layout: dword d of lane l is at d * Lanes + l.
      const auto *Dwords = reinterpret_cast<const uint32_t *>(Data);
      Scratch.resize(Lanes);
      for (uint64_t L = 0; L < Lanes; ++L) {
        Scratch[L].ElementIndex = Dwords[0 * Lanes + L];
        Scratch[L].BufferId = Dwords[1 * Lanes + L];
      }
      Handler.handle(H, Scratch);
      ++Messages;
      Cursor = Data + H.DataSize;
    }
    Bytes += Used;
  }
  __atomic_store_n(&Sizes[Sb], uint64_t(0), __ATOMIC_RELEASE);
}

} // namespace luthier::dh_comms
