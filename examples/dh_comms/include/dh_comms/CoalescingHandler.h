//===-- CoalescingHandler.h - CPU side: count uncoalesced accesses -*- C++ -*-===//
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
/// A \c MessageHandler that decides, for every executed memory instruction of
/// every wave, whether the access was coalesced.
///
/// The hardware does not service a 64-lane VMEM instruction at once: CDNA
/// issues it in groups of \c LanesPerCycle lanes (16, a quarter wave). The
/// accesses issued in the same cycle are coalesced if they touch no more
/// cache lines than their bytes need, i.e. ceil(bytes / CacheLineBytes)
/// lines; for a group of 16 dword accesses that is one 128-byte line. A wave
/// instruction is uncoalesced if any of its groups wastes a line.
///
/// This is what the TD hardware counter measures:
///   uncoalesced wave-instructions = SQ_INSTS_VMEM - TD_COALESCABLE_WAVEFRONT_sum
/// so the totals printed by \c report() can be checked against rocprofv3.
///
/// Addresses are rebuilt on the CPU from (buffer, element index) and the
/// buffer bases of the current dispatch: base + index * element size.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXAMPLES_DH_COMMS_COALESCING_HANDLER_H
#define LUTHIER_EXAMPLES_DH_COMMS_COALESCING_HANDLER_H

#include "dh_comms/AccessIndexHandler.h"
#include "dh_comms/MessageProcessor.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

#include <array>
#include <map>

namespace luthier::dh_comms {

class CoalescingHandler final : public MessageHandler {
public:
  CoalescingHandler(uint32_t LanesPerCycle, uint32_t CacheLineBytes)
      : LanesPerCycle(LanesPerCycle), CacheLineBytes(CacheLineBytes) {}

  /// Called before each dispatch with that dispatch's buffers, so records
  /// can be turned back into addresses. \p D is the descriptor the GPU got.
  void beginDispatch(const DeviceDescriptor &D);

  void handle(const WaveHeader &Header,
              llvm::ArrayRef<AccessRecord> Records) override;

  /// Prints one line per instrumented instruction and the totals to compare
  /// with SQ_INSTS_VMEM - TD_COALESCABLE_WAVEFRONT_sum.
  void report(llvm::raw_ostream &OS,
              llvm::ArrayRef<InstrumentedAccess> Accesses) const;

private:
  struct Stats {
    uint64_t Waves = 0;       ///< executions of the instruction (one per wave)
    uint64_t Uncoalesced = 0; ///< executions with at least one wasted line
    uint64_t Lines = 0;       ///< cache lines touched, summed over groups
    uint64_t IdealLines = 0;  ///< lines the same bytes need when contiguous
    uint64_t Unresolved = 0;  ///< lanes whose address is in no tracked buffer
  };

  uint32_t LanesPerCycle;
  uint32_t CacheLineBytes;
  uint32_t ElementSizeLog2 = 2;
  /// Buffer start per kernel-argument slot for the current dispatch.
  std::array<uint64_t, MaxTrackedBuffers> BaseBySlot{};
  std::map<uint32_t, Stats> PerInstr; ///< keyed by WaveHeader::InstrId
};

} // namespace luthier::dh_comms

#endif
