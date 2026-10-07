//===-- CoalescingHandler.h - Count uncoalesced accesses --------*- C++ -*-===//
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
/// Decides, for every executed memory instruction of every wave, whether the
/// access was coalesced. The hardware issues a 64-lane VMEM instruction in
/// groups of \c LanesPerCycle lanes; a group is coalesced if it touches no
/// more cache lines than its bytes need (ceil(bytes / line size)), and a
/// wave-instruction is uncoalesced if any group wastes a line.
///
/// Addresses are rebuilt from (buffer, element index) and the dispatch's
/// buffer bases.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_EXAMPLES_UNCOALESCED_MEM_ACCESS_COALESCING_HANDLER_H
#define LUTHIER_EXAMPLES_UNCOALESCED_MEM_ACCESS_COALESCING_HANDLER_H

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <luthier/Comms/MemoryTrace.h>

#include <algorithm>
#include <array>
#include <map>

class CoalescingHandler final : public luthier::comms::MemoryAccessHandler {
public:
  CoalescingHandler(uint32_t LanesPerCycle, uint32_t CacheLineBytes)
      : LanesPerCycle(LanesPerCycle), CacheLineBytes(CacheLineBytes) {}

  void beginDispatch(llvm::ArrayRef<luthier::comms::TrackedBuffer> Buffers,
                     uint32_t ElemSizeLog2) override {
    ElementSizeLog2 = ElemSizeLog2;
    BaseBySlot.fill(0);
    for (const auto &B : Buffers)
      if (B.ArgSlot < BaseBySlot.size())
        BaseBySlot[B.ArgSlot] = B.Base;
  }

  void handleAccesses(
      const luthier::comms::WaveHeader &H,
      llvm::ArrayRef<luthier::comms::AccessRecord> Records) override {
    Stats &S = PerSite[H.Site];
    ++S.Waves;
    uint64_t Width = luthier::comms::accessWidth(H.Info);
    if (Width == 0)
      Width = 4;

    // Lines touched and bytes moved by each group of lanes issued together.
    llvm::SmallVector<uint64_t, 16> GroupLines[64];
    uint64_t GroupBytes[64] = {};
    uint64_t Exec = H.Exec; // i-th record <-> i-th set bit
    for (const auto &R : Records) {
      const unsigned Lane = unsigned(__builtin_ctzll(Exec));
      Exec &= Exec - 1;
      if (R.BufferId >= BaseBySlot.size()) {
        ++S.Unresolved;
        continue;
      }
      const uint64_t Addr =
          BaseBySlot[R.BufferId] + (uint64_t(R.ElementIndex) << ElementSizeLog2);
      const unsigned G = Lane / LanesPerCycle;
      GroupBytes[G] += Width;
      for (uint64_t L = Addr / CacheLineBytes;
           L <= (Addr + Width - 1) / CacheLineBytes; ++L)
        GroupLines[G].push_back(L);
    }

    bool Wasted = false;
    for (unsigned G = 0; G < 64; ++G) {
      if (GroupBytes[G] == 0)
        continue;
      auto &L = GroupLines[G];
      std::sort(L.begin(), L.end());
      const uint64_t Lines = std::unique(L.begin(), L.end()) - L.begin();
      const uint64_t Ideal =
          (GroupBytes[G] + CacheLineBytes - 1) / CacheLineBytes;
      S.Lines += Lines;
      S.IdealLines += Ideal;
      Wasted |= Lines > Ideal;
    }
    S.Uncoalesced += Wasted;
  }

  /// One line per traced site, then the totals to compare with
  /// SQ_INSTS_VMEM - TD_COALESCABLE_WAVEFRONT_sum.
  void report(llvm::raw_ostream &OS,
              llvm::ArrayRef<luthier::comms::TracedSite> Sites) const {
    OS << llvm::formatv("[uncoalesced] {0}-lane groups, {1}-byte lines\n",
                        LanesPerCycle, CacheLineBytes);
    OS << llvm::formatv("{0,4}  {1,-24} {2,10} {3,12} {4,9} {5,9}\n", "site",
                        "instruction", "waves", "uncoalesced", "lines/wv",
                        "ideal/wv");
    uint64_t Waves = 0, Uncoalesced = 0, Lines = 0, Ideal = 0, Unresolved = 0;
    for (const auto &[Site, S] : PerSite) {
      OS << llvm::formatv("{0,4}  {1,-24} {2,10} {3,12} {4,9:F2} {5,9:F2}\n",
                          Site, Site < Sites.size() ? Sites[Site].Opcode : "?",
                          S.Waves, S.Uncoalesced,
                          double(S.Lines) / double(S.Waves),
                          double(S.IdealLines) / double(S.Waves));
      Waves += S.Waves;
      Uncoalesced += S.Uncoalesced;
      Lines += S.Lines;
      Ideal += S.IdealLines;
      Unresolved += S.Unresolved;
    }
    // Per kernel, to compare with per-dispatch counters.
    struct KernelTotals {
      uint64_t Waves = 0, Uncoalesced = 0, Lines = 0;
    };
    std::map<std::string, KernelTotals> PerKernel;
    for (const auto &[Site, S] : PerSite) {
      auto &K = PerKernel[Site < Sites.size() ? Sites[Site].Kernel : "?"];
      K.Waves += S.Waves;
      K.Uncoalesced += S.Uncoalesced;
      K.Lines += S.Lines;
    }
    for (const auto &[Kernel, K] : PerKernel)
      OS << llvm::formatv("[uncoalesced] kernel {0}: VMEM {1} coalesced {2} "
                          "uncoalesced {3} lines {4}\n",
                          Kernel, K.Waves, K.Waves - K.Uncoalesced,
                          K.Uncoalesced, K.Lines);
    OS << llvm::formatv("[uncoalesced] VMEM wave-instructions: {0}  "
                        "coalesced: {1}  uncoalesced: {2}\n",
                        Waves, Waves - Uncoalesced, Uncoalesced);
    OS << llvm::formatv("[uncoalesced] cache lines: {0} touched, {1} needed\n",
                        Lines, Ideal);
    if (Unresolved)
      OS << llvm::formatv("[uncoalesced] WARNING: {0} lanes accessed no "
                          "tracked buffer and were skipped\n",
                          Unresolved);
  }

private:
  struct Stats {
    uint64_t Waves = 0, Uncoalesced = 0, Lines = 0, IdealLines = 0,
             Unresolved = 0;
  };
  uint32_t LanesPerCycle;
  uint32_t CacheLineBytes;
  uint32_t ElementSizeLog2 = 2;
  std::array<uint64_t, luthier::comms::MaxTrackedBuffers> BaseBySlot{};
  std::map<uint32_t, Stats> PerSite;
};

#endif
