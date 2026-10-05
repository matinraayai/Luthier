//===-- AccessIndexHandler.h - CPU side: aggregate memory indices -*- C++ -*-===//
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
/// A \c MessageHandler that summarises the memory indices it receives, per
/// instrumented instruction: which buffer, how many lanes, and the range of
/// the linear index and of its 2-D (row, column) decomposition. Used to check
/// that what reaches the CPU is a valid index into the right array.
///
/// This is the place to plug in real analyses (heat maps, reuse distance,
/// coalescing, ...): write another \c MessageHandler.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXAMPLES_DH_COMMS_ACCESS_INDEX_HANDLER_H
#define LUTHIER_EXAMPLES_DH_COMMS_ACCESS_INDEX_HANDLER_H

#include "dh_comms/MessageProcessor.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

#include <map>
#include <string>

namespace luthier::dh_comms {

/// How the tool described an instrumented instruction when it instrumented it.
struct InstrumentedAccess {
  std::string Kernel;
  std::string Opcode;
  uint32_t AccessInfo; ///< packAccessInfo()
};

class AccessIndexHandler final : public MessageHandler {
public:
  /// Indices are split as row = index / \p RowLength, col = index % RowLength,
  /// and a row or column is in range if it is below \p NumRows / RowLength.
  AccessIndexHandler(uint32_t RowLength, uint32_t NumRows)
      : RowLength(RowLength), NumRows(NumRows) {}

  void handle(const WaveHeader &Header,
              llvm::ArrayRef<AccessRecord> Records) override;

  /// Prints one line per instrumented instruction. Returns true iff every
  /// index landed in a tracked buffer and every row and column is in range.
  bool report(llvm::raw_ostream &OS,
              llvm::ArrayRef<InstrumentedAccess> Accesses) const;

private:
  struct Range {
    uint32_t Min = UINT32_MAX;
    uint32_t Max = 0;
    void add(uint32_t V) {
      Min = V < Min ? V : Min;
      Max = V > Max ? V : Max;
    }
    bool empty() const { return Min > Max; }
  };

  struct Stats {
    uint64_t Waves = 0;
    uint64_t Lanes = 0;
    uint64_t Unresolved = 0;
    uint32_t BufferMask = 0; ///< bit i: kernel-argument slot i was accessed
    Range LanesPerWave;      ///< active lanes per message
    Range Index, Row, Col;
  };

  uint32_t RowLength;
  uint32_t NumRows;
  std::map<uint32_t, Stats> PerInstr; ///< keyed by WaveHeader::InstrId
};

} // namespace luthier::dh_comms

#endif
